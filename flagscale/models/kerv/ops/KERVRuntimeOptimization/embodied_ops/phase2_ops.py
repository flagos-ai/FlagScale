# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Second-stage KERV operator paths.

These operators deliberately keep the model's exact control semantics in the
native implementation.  Triton is used only for layouts that are fixed by the
KERV runtime (batch one, contiguous tensors and short verification trees); all
other inputs go through the reference path.  The public schemas are small
enough to be reused by the KERV bridge and by downstream FlagOS backends.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .kerv_ops import _is_cuda_contiguous, _record_hit, _use_triton

try:  # Triton is optional for CPU-only installations.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only environments.
    triton = None
    tl = None


_DEF_LIBRARY: torch.library.Library | None = None
_IMPL_LIBRARY: torch.library.Library | None = None


if triton is not None:

    @triton.jit
    def _tree_embed_pack_kernel(
        token_ptr,
        kept_ptr,
        weight_ptr,
        output_ptr,
        source_nodes,
        kept_nodes,
        target_nodes,
        hidden_size: tl.constexpr,
        token_stride_batch: tl.constexpr,
        token_stride_node: tl.constexpr,
        kept_stride: tl.constexpr,
        weight_stride: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_node: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = target_nodes * hidden_size
        valid = linear < total
        node = linear // hidden_size
        dim = linear - node * hidden_size
        source_slot = tl.load(
            kept_ptr + tl.where(node < kept_nodes, node, 0) * kept_stride,
            mask=valid & (kept_nodes > 0),
            other=0,
        )
        source_slot = tl.minimum(source_slot, source_nodes - 1)
        token_id = tl.load(
            token_ptr + source_slot * token_stride_node,
            mask=valid & (source_nodes > 0),
            other=0,
        )
        value = tl.load(
            weight_ptr + token_id * weight_stride + dim,
            mask=valid & (dim < hidden_size),
            other=0.0,
        )
        tl.store(
            output_ptr + node * output_stride_node + dim,
            value,
            mask=valid,
        )

    @triton.jit
    def _rope_kv_store_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        cos_ptr,
        sin_ptr,
        query_out_ptr,
        key_cache_ptr,
        value_cache_ptr,
        sequence,
        q_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        write_start,
        q_stride_h: tl.constexpr,
        q_stride_s: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_s: tl.constexpr,
        v_stride_h: tl.constexpr,
        v_stride_s: tl.constexpr,
        oq_stride_h: tl.constexpr,
        oq_stride_s: tl.constexpr,
        cache_stride_h: tl.constexpr,
        cache_stride_s: tl.constexpr,
        cos_stride_s: tl.constexpr,
        sin_stride_s: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        dims = tl.arange(0, BLOCK)
        valid_dim = dims < head_dim
        valid_token = token < sequence
        half = head_dim // 2
        rotated = (dims + half) % head_dim
        cos = tl.load(
            cos_ptr + token * cos_stride_s + dims,
            mask=valid_token & valid_dim,
            other=0.0,
        )
        sin = tl.load(
            sin_ptr + token * sin_stride_s + dims,
            mask=valid_token & valid_dim,
            other=0.0,
        )
        sin = tl.where(dims < half, -sin, sin)
        q_valid = valid_token & valid_dim & (head < q_heads)
        q_base = head * q_stride_h + token * q_stride_s
        q = tl.load(query_ptr + q_base + dims, mask=q_valid, other=0.0)
        q_rot = tl.load(query_ptr + q_base + rotated, mask=q_valid, other=0.0)
        q_result = q * cos + q_rot * sin
        q_out_base = head * oq_stride_h + token * oq_stride_s
        tl.store(query_out_ptr + q_out_base + dims, q_result, mask=q_valid)

        kv_valid = valid_token & valid_dim & (head < kv_heads)
        k_base = head * k_stride_h + token * k_stride_s
        k = tl.load(key_ptr + k_base + dims, mask=kv_valid, other=0.0)
        k_rot = tl.load(key_ptr + k_base + rotated, mask=kv_valid, other=0.0)
        k_result = k * cos + k_rot * sin
        cache_base = head * cache_stride_h + (write_start + token) * cache_stride_s
        tl.store(key_cache_ptr + cache_base + dims, k_result, mask=kv_valid)
        v_base = head * v_stride_h + token * v_stride_s
        value = tl.load(value_ptr + v_base + dims, mask=kv_valid, other=0.0)
        tl.store(value_cache_ptr + cache_base + dims, value, mask=kv_valid)

    @triton.jit
    def _vision_add_layer_norm_kernel(
        hidden_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        rows,
        width: tl.constexpr,
        eps,
        has_bias: tl.constexpr,
        hidden_stride: tl.constexpr,
        residual_stride: tl.constexpr,
        output_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < rows) & (cols < width)
        value = tl.load(hidden_ptr + row * hidden_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        value += tl.load(residual_ptr + row * residual_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        mean = tl.sum(tl.where(mask, value, 0.0), axis=0) / width
        centered = value - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / width
        weight = tl.load(weight_ptr + cols, mask=cols < width, other=1.0).to(tl.float32)
        result = centered * tl.rsqrt(variance + eps) * weight
        if has_bias:
            result += tl.load(bias_ptr + cols, mask=cols < width, other=0.0).to(tl.float32)
        tl.store(output_ptr + row * output_stride + cols, result, mask=mask)

    @triton.jit
    def _vision_bias_gelu_kernel(
        input_ptr,
        bias_ptr,
        output_ptr,
        rows,
        width: tl.constexpr,
        input_stride: tl.constexpr,
        output_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = (row < rows) & (cols < width)
        value = tl.load(input_ptr + row * input_stride + cols, mask=mask, other=0.0).to(tl.float32)
        value += tl.load(bias_ptr + cols, mask=cols < width, other=0.0).to(tl.float32)
        # Exact GELU is used by the vision towers covered by this adapter.
        inv_sqrt2 = 0.7071067811865476
        result = value * 0.5 * (1.0 + tl.erf(value * inv_sqrt2))
        tl.store(output_ptr + row * output_stride + cols, result, mask=mask)

    @triton.jit
    def _kv_accept_commit_kernel(
        storage_ptr,
        indices_ptr,
        previous_length,
        accepted,
        row_stride: tl.constexpr,
        capacity: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        valid = offsets < accepted * head_dim
        accepted_index = offsets // head_dim
        feature = offsets - accepted_index * head_dim
        source_node = tl.load(indices_ptr + accepted_index, mask=valid, other=0)
        base = row * row_stride
        source = base + (previous_length + source_node) * head_dim + feature
        values = tl.load(storage_ptr + source, mask=valid, other=0.0)
        destination = base + (previous_length + accepted_index) * head_dim + feature
        tl.store(storage_ptr + destination, values, mask=valid)


def _check_matrix(hidden: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.shape[-1] != weight.shape[-1]:
        raise ValueError("hidden and weight dimensions are incompatible")
    flat = hidden.reshape(-1, hidden.shape[-1])
    return flat, int(flat.shape[0]), int(flat.shape[1])


def _tree_embed_pack_impl(
    draft_tokens: torch.Tensor,
    kept_indices: torch.Tensor,
    embedding_weight: torch.Tensor,
    target_nodes: int,
) -> torch.Tensor:
    if draft_tokens.ndim != 2 or kept_indices.ndim != 1 or embedding_weight.ndim != 2:
        raise ValueError("tree embed pack expects [B,N], [M] and [V,H] tensors")
    target_nodes = int(target_nodes)
    if draft_tokens.shape[0] != 1 or target_nodes <= 0 or kept_indices.numel() <= 0:
        raise ValueError("tree embed pack currently requires batch=1 and non-empty indices")
    if target_nodes < kept_indices.numel():
        raise ValueError("target_nodes must cover all kept nodes")
    if draft_tokens.device != embedding_weight.device or draft_tokens.device != kept_indices.device:
        raise ValueError("tree embed pack tensors must share a device")
    # Avoid a host synchronization on every CUDA tree replay.  Production
    # templates are validated when they are built; retain the range check for
    # CPU/debug calls where it is useful to report a malformed template.
    if not kept_indices.is_cuda:
        if int(kept_indices.max()) >= draft_tokens.shape[1] or int(kept_indices.min()) < 0:
            raise ValueError("kept_indices is outside draft_tokens")
    _record_hit("kerv_tree_embed_pack", draft_tokens, kept_indices, embedding_weight)
    # A static verifier template normally presents the already ordered full
    # node list. Avoid an otherwise redundant index_select in that common
    # case; padded/gathered templates still use the general path below.
    if (
        target_nodes == int(draft_tokens.shape[1])
        and kept_indices.numel() == draft_tokens.shape[1]
        and not kept_indices.is_cuda
        and torch.equal(
            kept_indices,
            torch.arange(
                draft_tokens.shape[1],
                device=kept_indices.device,
                dtype=kept_indices.dtype,
            ),
        )
    ):
        return F.embedding(draft_tokens, embedding_weight)
    output = torch.empty(
        (1, target_nodes, embedding_weight.shape[-1]),
        device=embedding_weight.device,
        dtype=embedding_weight.dtype,
    )
    if (
        _use_triton("kerv_tree_embed_pack")
        and triton is not None
        and _is_cuda_contiguous(draft_tokens, kept_indices, embedding_weight, output)
        and embedding_weight.shape[-1] <= 8192
    ):
        block = min(triton.next_power_of_2(int(embedding_weight.shape[-1])), 8192)
        _tree_embed_pack_kernel[(triton.cdiv(output.numel(), block),)](
            draft_tokens,
            kept_indices,
            embedding_weight,
            output,
            int(draft_tokens.shape[1]),
            int(kept_indices.numel()),
            target_nodes,
            int(embedding_weight.shape[-1]),
            draft_tokens.stride(0),
            draft_tokens.stride(1),
            kept_indices.stride(0),
            embedding_weight.stride(0),
            output.stride(0),
            output.stride(1),
            BLOCK=block,
            num_warps=4,
        )
        return output
    selected = draft_tokens.index_select(1, kept_indices)
    if selected.shape[1] < target_nodes:
        selected = torch.cat(
            (selected, selected[:, :1].expand(-1, target_nodes - selected.shape[1])),
            dim=1,
        )
    return F.embedding(selected, embedding_weight)


def _rope_kv_store_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    write_start: int,
) -> torch.Tensor:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("RoPE KV store expects [B,H,S,D] tensors")
    if query.shape[0] != key.shape[0] or key.shape != value.shape:
        raise ValueError("Q/K/V layouts are incompatible")
    if query.shape[-1] % 2 or cos.shape != sin.shape:
        raise ValueError("RoPE tables must have matching even-width shapes")
    if query.shape[0] != 1 or query.shape[-1] != 128:
        raise ValueError("KERV RoPE KV store currently supports batch=1, head_dim=128")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise ValueError("resident K/V destinations are incompatible")
    if key_cache.shape[1] != key.shape[1] or key_cache.shape[-1] != key.shape[-1]:
        raise ValueError("resident K/V head dimensions are incompatible")
    sequence = int(query.shape[2])
    write_start = int(write_start)
    if write_start < 0 or write_start + sequence > key_cache.shape[2]:
        raise ValueError("resident K/V write exceeds capacity")
    _record_hit("kerv_rope_kv_store", query, key, value, key_cache, value_cache)
    q_out = torch.empty_like(query)
    direct_tables = (
        cos.ndim == 3
        and sin.ndim == 3
        and cos.shape == (1, sequence, query.shape[-1])
        and sin.shape == cos.shape
    )
    if (
        _use_triton("kerv_rope_kv_store")
        and triton is not None
        and direct_tables
        and position_ids.numel() == 0
        and all(
            tensor.is_cuda and tensor.stride(-1) == 1
            for tensor in (query, key, value, cos, sin, q_out)
        )
        and key_cache.is_cuda
        and value_cache.is_cuda
        and key_cache.stride(-1) == value_cache.stride(-1) == 1
    ):
        block = triton.next_power_of_2(int(query.shape[-1]))
        _rope_kv_store_kernel[(sequence, max(int(query.shape[1]), int(key.shape[1])))](
            query,
            key,
            value,
            cos,
            sin,
            q_out,
            key_cache,
            value_cache,
            sequence,
            int(query.shape[1]),
            int(key.shape[1]),
            int(query.shape[-1]),
            write_start,
            query.stride(1),
            query.stride(2),
            key.stride(1),
            key.stride(2),
            value.stride(1),
            value.stride(2),
            q_out.stride(1),
            q_out.stride(2),
            key_cache.stride(1),
            key_cache.stride(2),
            cos.stride(1),
            sin.stride(1),
            BLOCK=block,
            num_warps=1,
        )
        return q_out
    if position_ids.numel() > 0:
        if position_ids.shape != (1, sequence):
            raise ValueError("position_ids must be [1,S]")
        cos_selected = (
            cos.reshape(-1, cos.shape[-1])
            .index_select(0, position_ids.reshape(-1))
            .reshape(1, sequence, -1)
        )
        sin_selected = (
            sin.reshape(-1, sin.shape[-1])
            .index_select(0, position_ids.reshape(-1))
            .reshape(1, sequence, -1)
        )
    else:
        cos_selected, sin_selected = cos, sin
    q_cos = cos_selected.unsqueeze(1)
    q_sin = sin_selected.unsqueeze(1)
    half = query.shape[-1] // 2

    def rotate(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q_out.copy_(query * q_cos + rotate(query) * q_sin)
    key_out = key * q_cos + rotate(key) * q_sin
    key_cache[..., write_start : write_start + sequence, :].copy_(key_out)
    value_cache[..., write_start : write_start + sequence, :].copy_(value)
    return q_out


def _action_verify_accept_impl(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    candidates: torch.Tensor,
    threshold: float,
    token_offset: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat, _, _ = _check_matrix(hidden, weight)
    if candidates.ndim != 2 or candidates.numel() == 0:
        raise ValueError("candidates must be a non-empty [paths, sequence] tensor")
    if flat.shape[0] != candidates.numel():
        raise ValueError("hidden rows must equal candidates.numel()")
    if int(weight.shape[0]) <= 0:
        raise ValueError("action vocabulary cannot be empty")
    _record_hit("kerv_action_verify_accept", hidden, weight, candidates)
    logits = F.linear(flat, weight, bias).reshape(*candidates.shape, int(weight.shape[0]))
    predicted = logits.argmax(dim=-1).to(torch.long) + int(token_offset)
    matches = (predicted[:, :-1] - candidates[:, 1:]).abs() <= float(threshold)
    accepted = torch.cumprod(matches.to(torch.int32), dim=1).sum(dim=1)
    best = accepted.argmax().to(torch.long)
    length = accepted[best].to(torch.long)
    next_position = length.clamp_max(int(predicted.shape[1] - 1))
    next_token = predicted[best, next_position]
    return best, length, next_token


def _draft_action_topk_impl(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    action_token_offset: int,
    k: int,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat, _, _ = _check_matrix(hidden, weight)
    k = int(k)
    if k <= 0 or k > weight.shape[0]:
        raise ValueError("invalid action Top-K")
    _record_hit("kerv_draft_action_topk", hidden, weight)
    logits = F.linear(flat, weight, bias)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Stable sorting gives the deterministic lower-token-id tie rule required
    # for reproducible tree templates.  The action vocabulary is only 256
    # rows in KERV, so this reduction is cheaper than the full model LM head;
    # replacing it with an unconstrained TopK would make equal BF16 scores
    # reorder candidates across backends.
    values, indices = torch.sort(log_probs, dim=-1, descending=True, stable=True)
    return (
        values[..., :k].to(logits.dtype),
        indices[..., :k].to(torch.long) + int(action_token_offset),
    )


def _kv_accept_commit_impl(
    storage: torch.Tensor,
    tree_indices: torch.Tensor,
    previous_length: int,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    if storage.ndim != 6 or tree_indices.ndim != 1 or not storage.is_contiguous():
        raise ValueError(
            "KV accept commit expects contiguous [L,2,B,H,T,D] storage and [A] indices"
        )
    previous_length = int(previous_length)
    if previous_length < 0 or tree_indices.numel() == 0:
        raise ValueError("invalid KV commit arguments")
    accepted = int(tree_indices.numel())
    if previous_length + accepted > storage.shape[-2]:
        raise ValueError("KV accept commit exceeds storage capacity")
    _record_hit("kerv_kv_accept_commit", storage, tree_indices)
    # The source and destination ranges can overlap when a tree path reuses a
    # prefix node.  The native path takes a clone and is therefore the exact
    # default.  Only use the one-pass kernel for a provably disjoint mapping;
    # this keeps explicit Triton benchmarking from changing cache semantics.
    source_indices = tree_indices.to(device=storage.device, dtype=torch.long)
    if scratch is not None:
        expected = (*storage.shape[:-2], accepted, storage.shape[-1])
        if (
            tuple(scratch.shape) != expected
            or scratch.device != storage.device
            or scratch.dtype != storage.dtype
        ):
            raise ValueError("KV commit scratch buffer has an incompatible layout")
        absolute_indices = source_indices + previous_length
        torch.index_select(storage, -2, absolute_indices, out=scratch)
        storage[..., previous_length : previous_length + accepted, :].copy_(scratch)
        return storage
    disjoint = bool(torch.all(source_indices >= accepted).item())
    if (
        _use_triton("kerv_kv_accept_commit")
        and triton is not None
        and _is_cuda_contiguous(storage, tree_indices)
        and accepted * storage.shape[-1] <= 8192
        and disjoint
    ):
        row_count = int(storage.numel() // (storage.shape[-2] * storage.shape[-1]))
        block = triton.next_power_of_2(accepted * int(storage.shape[-1]))
        _kv_accept_commit_kernel[(row_count,)](
            storage,
            tree_indices.to(dtype=torch.long),
            previous_length,
            accepted,
            int(storage.shape[-2] * storage.shape[-1]),
            int(storage.shape[-2]),
            int(storage.shape[-1]),
            BLOCK=block,
            num_warps=4,
        )
        return storage
    # Advanced indexing already materialises a gather result.  Cloning it
    # again doubled the temporary HBM traffic on every accepted branch; the
    # destination copy below consumes the gathered tensor directly.
    source = storage[..., previous_length + tree_indices, :]
    storage[..., previous_length : previous_length + accepted, :].copy_(source)
    return storage


def _vision_add_layer_norm_impl(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if hidden.shape != residual.shape or hidden.shape[-1] != weight.numel():
        raise ValueError("vision Add-LayerNorm shapes are incompatible")
    _record_hit("kerv_vision_add_layer_norm", hidden, residual, weight)
    output = torch.empty_like(hidden)
    width = int(hidden.shape[-1])
    if (
        _use_triton("kerv_vision_add_layer_norm")
        and triton is not None
        and _is_cuda_contiguous(hidden, residual, weight, output)
        and (bias is None or _is_cuda_contiguous(bias))
        and width <= 8192
    ):
        block = min(triton.next_power_of_2(width), 8192)
        _vision_add_layer_norm_kernel[(hidden.numel() // width,)](
            hidden,
            residual,
            weight,
            bias if bias is not None else weight.new_empty((1,)),
            output,
            hidden.numel() // width,
            width,
            float(eps),
            bias is not None,
            hidden.stride(-2) if hidden.ndim > 1 else width,
            residual.stride(-2) if residual.ndim > 1 else width,
            output.stride(-2) if output.ndim > 1 else width,
            BLOCK=block,
            num_warps=8 if width >= 2048 else 4,
        )
        return output
    return F.layer_norm(hidden + residual, (width,), weight, bias, float(eps))


def _vision_bias_gelu_impl(
    hidden: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if hidden.shape[-1] != bias.numel():
        raise ValueError("vision Bias-GELU width mismatch")
    _record_hit("kerv_vision_bias_gelu", hidden, bias)
    output = torch.empty_like(hidden)
    width = int(hidden.shape[-1])
    if (
        _use_triton("kerv_vision_bias_gelu")
        and triton is not None
        and _is_cuda_contiguous(hidden, bias, output)
        and width <= 8192
    ):
        block = min(triton.next_power_of_2(width), 8192)
        _vision_bias_gelu_kernel[(hidden.numel() // width,)](
            hidden,
            bias,
            output,
            hidden.numel() // width,
            width,
            hidden.stride(-2) if hidden.ndim > 1 else width,
            output.stride(-2) if output.ndim > 1 else width,
            BLOCK=block,
            num_warps=8 if width >= 2048 else 4,
        )
        return output
    return F.gelu(hidden + bias, approximate="none")


def _o_proj_residual_rms_norm_impl(
    hidden: torch.Tensor,
    weight_proj: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _record_hit("kerv_o_proj_residual_rms_norm", hidden, weight_proj, residual, norm_weight)
    projected = F.linear(hidden, weight_proj, bias)
    return F.rms_norm(projected + residual, (projected.shape[-1],), norm_weight, float(eps))


def _down_proj_residual_rms_norm_impl(
    hidden: torch.Tensor,
    weight_proj: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _record_hit("kerv_down_proj_residual_rms_norm", hidden, weight_proj, residual, norm_weight)
    projected = F.linear(hidden, weight_proj, bias)
    return F.rms_norm(projected + residual, (projected.shape[-1],), norm_weight, float(eps))


def _logical_kv_commit_impl(
    logical_map: torch.Tensor,
    tree_indices: torch.Tensor,
    previous_length: int,
) -> torch.Tensor:
    if logical_map.ndim != 1 or tree_indices.ndim != 1:
        raise ValueError("logical KV commit expects one-dimensional maps")
    previous_length = int(previous_length)
    end = previous_length + int(tree_indices.numel())
    if previous_length < 0 or end > logical_map.numel():
        raise ValueError("logical KV commit exceeds map capacity")
    _record_hit("kerv_logical_kv_commit", logical_map, tree_indices)
    logical_map[previous_length:end].copy_(tree_indices.to(logical_map.device, logical_map.dtype))
    return logical_map


def register_phase2_ops() -> None:
    """Register second-stage operators under the existing namespace."""

    global _DEF_LIBRARY, _IMPL_LIBRARY
    if _DEF_LIBRARY is not None:
        return
    _DEF_LIBRARY = torch.library.Library("flagos_embodied", "FRAGMENT")
    definitions = (
        "kerv_tree_embed_pack(Tensor draft_tokens, Tensor kept_indices, Tensor embedding_weight, int target_nodes) -> Tensor",
        "kerv_rope_kv_store(Tensor query, Tensor key, Tensor value, Tensor cos, Tensor sin, Tensor position_ids, Tensor(a!) key_cache, Tensor(b!) value_cache, int write_start) -> Tensor",
        "kerv_action_verify_accept(Tensor hidden, Tensor weight, Tensor candidates, float threshold, int token_offset, Tensor? bias=None) -> (Tensor, Tensor, Tensor)",
        "kerv_draft_action_topk(Tensor hidden, Tensor weight, int action_token_offset, int k, Tensor? bias=None) -> (Tensor, Tensor)",
        "kerv_kv_accept_commit(Tensor(a!) storage, Tensor tree_indices, int previous_length, Tensor? scratch=None) -> Tensor(a!)",
        "kerv_vision_add_layer_norm(Tensor hidden, Tensor residual, Tensor weight, Tensor? bias, float eps) -> Tensor",
        "kerv_vision_bias_gelu(Tensor hidden, Tensor bias) -> Tensor",
        "kerv_o_proj_residual_rms_norm(Tensor hidden, Tensor weight_proj, Tensor residual, Tensor norm_weight, float eps, Tensor? bias=None) -> Tensor",
        "kerv_down_proj_residual_rms_norm(Tensor hidden, Tensor weight_proj, Tensor residual, Tensor norm_weight, float eps, Tensor? bias=None) -> Tensor",
        "kerv_logical_kv_commit(Tensor(a!) logical_map, Tensor tree_indices, int previous_length) -> Tensor(a!)",
    )
    for definition in definitions:
        _DEF_LIBRARY.define(definition)
    _IMPL_LIBRARY = torch.library.Library("flagos_embodied", "IMPL", "CompositeExplicitAutograd")
    implementations = {
        "kerv_tree_embed_pack": _tree_embed_pack_impl,
        "kerv_rope_kv_store": _rope_kv_store_impl,
        "kerv_action_verify_accept": _action_verify_accept_impl,
        "kerv_draft_action_topk": _draft_action_topk_impl,
        "kerv_kv_accept_commit": _kv_accept_commit_impl,
        "kerv_vision_add_layer_norm": _vision_add_layer_norm_impl,
        "kerv_vision_bias_gelu": _vision_bias_gelu_impl,
        "kerv_o_proj_residual_rms_norm": _o_proj_residual_rms_norm_impl,
        "kerv_down_proj_residual_rms_norm": _down_proj_residual_rms_norm_impl,
        "kerv_logical_kv_commit": _logical_kv_commit_impl,
    }
    for name, implementation in implementations.items():
        _IMPL_LIBRARY.impl(name, implementation)


def kerv_tree_embed_pack(draft_tokens, kept_indices, embedding_weight, target_nodes):
    return torch.ops.flagos_embodied.kerv_tree_embed_pack(
        draft_tokens, kept_indices, embedding_weight, int(target_nodes)
    )


def kerv_rope_kv_store(
    query, key, value, cos, sin, position_ids, key_cache, value_cache, write_start
):
    return torch.ops.flagos_embodied.kerv_rope_kv_store(
        query, key, value, cos, sin, position_ids, key_cache, value_cache, int(write_start)
    )


def kerv_action_verify_accept(hidden, weight, candidates, threshold, token_offset, bias=None):
    return torch.ops.flagos_embodied.kerv_action_verify_accept(
        hidden, weight, candidates, float(threshold), int(token_offset), bias
    )


def kerv_draft_action_topk(hidden, weight, action_token_offset, k=8, bias=None):
    return torch.ops.flagos_embodied.kerv_draft_action_topk(
        hidden, weight, int(action_token_offset), int(k), bias
    )


def kerv_kv_accept_commit(storage, tree_indices, previous_length, scratch=None):
    return torch.ops.flagos_embodied.kerv_kv_accept_commit(
        storage, tree_indices, int(previous_length), scratch
    )


def kerv_vision_add_layer_norm(hidden, residual, weight, bias, eps):
    return torch.ops.flagos_embodied.kerv_vision_add_layer_norm(
        hidden, residual, weight, bias, float(eps)
    )


def kerv_vision_bias_gelu(hidden, bias):
    return torch.ops.flagos_embodied.kerv_vision_bias_gelu(hidden, bias)


def kerv_o_proj_residual_rms_norm(hidden, weight_proj, residual, norm_weight, eps, bias=None):
    return torch.ops.flagos_embodied.kerv_o_proj_residual_rms_norm(
        hidden, weight_proj, residual, norm_weight, float(eps), bias
    )


def kerv_down_proj_residual_rms_norm(hidden, weight_proj, residual, norm_weight, eps, bias=None):
    return torch.ops.flagos_embodied.kerv_down_proj_residual_rms_norm(
        hidden, weight_proj, residual, norm_weight, float(eps), bias
    )


def kerv_logical_kv_commit(logical_map, tree_indices, previous_length):
    return torch.ops.flagos_embodied.kerv_logical_kv_commit(
        logical_map, tree_indices, int(previous_length)
    )
