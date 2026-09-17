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

"""Inference kernels for KERV's batch-one, short-sequence execution path.

The kernels in this module are intentionally shape guarded.  KERV has many
small, irregular tensors for which a generic Triton implementation can be
slower than ATen/cuBLAS.  Unsupported layouts therefore use a transparent
native implementation instead of silently changing the model semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:  # Triton is optional for CPU-only FlagScale installations.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only without Triton.
    triton = None
    tl = None


_DEF_LIBRARY: torch.library.Library | None = None
_IMPL_LIBRARY: torch.library.Library | None = None
_RECORD_PATH: Path | None = None
_RECORD_ONCE = True
_RECORDED: set[tuple[object, ...]] = set()
_BACKEND = "auto"
_ENABLED_OPERATORS: set[str] | None = None
# Tiny KERV elementwise/reduction shapes are currently faster through ATen on
# A100.  Auto mode keeps those paths native; explicit ``backend=triton`` is
# available for tuning on another backend.
_AUTO_TRITON_OPERATORS = {
    "kerv_static_tree_attention": True,
    # The fused Q/K rotation plus resident K/V write passed the fixed
    # B=1,Hd=128 short-sequence gate on A100.  Other layouts are rejected by
    # phase2_ops and fall back to the native implementation.
    "kerv_rope_kv_store": True,
}


def configure_recording(path: str | None, record: bool, once: bool = True) -> None:
    """Configure lightweight JSONL hit recording for the public operators."""

    global _RECORD_PATH, _RECORD_ONCE
    _RECORD_PATH = Path(path) if record and path else None
    _RECORD_ONCE = bool(once)
    _RECORDED.clear()
    if _RECORD_PATH is not None:
        _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECORD_PATH.touch(exist_ok=True)


def configure_backend(backend: str) -> None:
    global _BACKEND
    backend = str(backend).lower()
    if backend not in {"auto", "triton", "native"}:
        raise ValueError(f"unsupported KERV embodied backend: {backend}")
    _BACKEND = backend


def configure_enabled_operators(names: set[str] | None) -> None:
    global _ENABLED_OPERATORS
    _ENABLED_OPERATORS = None if names is None else set(names)


def _use_triton(name: str) -> bool:
    if _ENABLED_OPERATORS is not None and name not in _ENABLED_OPERATORS:
        return False
    if _BACKEND == "native":
        return False
    if _BACKEND == "triton":
        return True
    return bool(_AUTO_TRITON_OPERATORS.get(name, False))


def _record_hit(name: str, *values: torch.Tensor) -> None:
    if _RECORD_PATH is None:
        return
    key = (name, *((tuple(value.shape), str(value.dtype), value.device.type) for value in values))
    if _RECORD_ONCE and key in _RECORDED:
        return
    _RECORDED.add(key)
    with _RECORD_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "operator": name,
                    "shapes": [list(value.shape) for value in values],
                    "dtypes": [str(value.dtype).removeprefix("torch.") for value in values],
                    "devices": [value.device.type for value in values],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _is_cuda_contiguous(*values: torch.Tensor) -> bool:
    return all(value.is_cuda and value.is_contiguous() for value in values)


if triton is not None:

    @triton.jit
    def _silu_mul_kernel(
        gate_ptr,
        up_ptr,
        rows,
        width: tl.constexpr,
        stride_gate_row: tl.constexpr,
        stride_up_row: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < width
        gate = tl.load(gate_ptr + row * stride_gate_row + cols, mask=mask, other=0.0)
        up = tl.load(up_ptr + row * stride_up_row + cols, mask=mask, other=0.0)
        gate_f = gate.to(tl.float32)
        result = gate_f * (1.0 / (1.0 + tl.exp(-gate_f))) * up.to(tl.float32)
        tl.store(gate_ptr + row * stride_gate_row + cols, result, mask=mask)

    @triton.jit
    def _add_rms_norm_kernel(
        hidden_ptr,
        residual_ptr,
        weight_ptr,
        rows,
        width: tl.constexpr,
        stride_hidden_row: tl.constexpr,
        stride_residual_row: tl.constexpr,
        eps,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < width
        hidden = tl.load(hidden_ptr + row * stride_hidden_row + cols, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + row * stride_residual_row + cols, mask=mask, other=0.0)
        value = hidden.to(tl.float32) + residual.to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(tl.where(mask, value * value, 0.0), axis=0) / width
        normalized = value * tl.rsqrt(mean_square + eps) * weight
        tl.store(hidden_ptr + row * stride_hidden_row + cols, normalized, mask=mask)

    @triton.jit
    def _verify_accept_control_kernel(
        logits_ptr,
        candidates_ptr,
        best_index_ptr,
        best_length_ptr,
        logits_stride_path: tl.constexpr,
        logits_stride_position: tl.constexpr,
        candidates_stride_path: tl.constexpr,
        candidates_stride_position: tl.constexpr,
        n_paths,
        n_positions,
        vocab_size,
        threshold,
        token_offset,
        BLOCK_PATHS: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
    ):
        paths = tl.arange(0, BLOCK_PATHS)
        valid_path = paths < n_paths
        vocab = tl.arange(0, BLOCK_VOCAB)
        accepted = tl.zeros((BLOCK_PATHS,), dtype=tl.int32)
        alive = tl.full((BLOCK_PATHS,), 1, dtype=tl.int32)
        # KERV's speculative draft is eight tokens.  The runtime falls back
        # to ATen for longer sequences rather than compiling an unbounded loop.
        for position in range(0, 32):
            position_valid = position < n_positions
            offsets = (
                paths[:, None] * logits_stride_path
                + position * logits_stride_position
                + vocab[None, :]
            )
            values = tl.load(
                logits_ptr + offsets,
                mask=valid_path[:, None] & (vocab[None, :] < vocab_size) & position_valid,
                other=-float("inf"),
            )
            predicted = tl.argmax(values, axis=1) + token_offset
            candidate = tl.load(
                candidates_ptr
                + paths * candidates_stride_path
                + (position + 1) * candidates_stride_position,
                mask=valid_path & position_valid,
                other=0,
            )
            match = tl.abs(candidate - predicted) <= threshold
            alive = tl.where(position_valid, alive & match.to(tl.int32), alive)
            accepted += tl.where(position_valid & (alive != 0), 1, 0)

        masked_lengths = tl.where(valid_path, accepted, -1)
        best_length = tl.max(masked_lengths, axis=0)
        # Negative path index makes ties deterministic and matches the first
        # occurrence behavior of torch.argmax.
        tie_score = tl.where(
            valid_path & (accepted == best_length), -paths.to(tl.int32), -2147483647
        )
        best_index = tl.argmax(tie_score, axis=0)
        tl.store(best_index_ptr, best_index.to(tl.int64))
        tl.store(best_length_ptr, best_length.to(tl.int64))

    @triton.jit
    def _pack_tree_kernel(
        draft_ptr,
        kept_ptr,
        output_ptr,
        batch,
        source_nodes,
        kept_nodes,
        target_nodes,
        draft_stride_batch: tl.constexpr,
        draft_stride_node: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_node: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = batch * target_nodes
        valid = linear < total
        batch_id = linear // target_nodes
        node_id = linear - batch_id * target_nodes
        has_source = node_id < kept_nodes
        kept = tl.load(kept_ptr + node_id, mask=has_source, other=0)
        kept = tl.minimum(kept, source_nodes - 1)
        source = tl.load(
            draft_ptr + batch_id * draft_stride_batch + kept * draft_stride_node,
            mask=valid,
            other=0,
        )
        tl.store(
            output_ptr + batch_id * output_stride_batch + node_id * output_stride_node,
            source,
            mask=valid,
        )

    @triton.jit
    def _action_argmax_kernel(
        hidden_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        rows,
        hidden_size: tl.constexpr,
        action_size: tl.constexpr,
        hidden_stride: tl.constexpr,
        weight_stride: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        dims = tl.arange(0, BLOCK_H)
        hidden = tl.load(
            hidden_ptr + row * hidden_stride + dims,
            mask=dims < hidden_size,
            other=0.0,
        ).to(tl.float32)
        best_value = tl.full((), -float("inf"), tl.float32)
        best_index = tl.zeros((), tl.int32)
        for start in range(0, action_size, BLOCK_V):
            ids = start + tl.arange(0, BLOCK_V)
            mask = ids < action_size
            weights = tl.load(
                weight_ptr + ids[:, None] * weight_stride + dims[None, :],
                mask=mask[:, None] & (dims[None, :] < hidden_size),
                other=0.0,
            ).to(tl.float32)
            values = tl.sum(weights * hidden[None, :], axis=1)
            if has_bias:
                values += tl.load(bias_ptr + ids, mask=mask, other=0.0).to(tl.float32)
            local_index = tl.argmax(values, axis=0)
            local_value = tl.max(values, axis=0)
            better = local_value > best_value
            best_value = tl.where(better, local_value, best_value)
            best_index = tl.where(better, start + local_index, best_index)
        tl.store(output_ptr + row, best_index.to(tl.int64))

    @triton.jit
    def _value_cache_store_kernel(
        source_ptr,
        destination_ptr,
        token_count,
        write_start,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        source_stride_head: tl.constexpr,
        source_stride_token: tl.constexpr,
        source_stride_dim: tl.constexpr,
        destination_stride_head: tl.constexpr,
        destination_stride_token: tl.constexpr,
        destination_stride_dim: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        head = tl.program_id(0)
        linear = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        token = linear // head_dim
        dim = linear - token * head_dim
        mask = (head < heads) & (token < token_count) & (dim < head_dim)
        source = head * source_stride_head + token * source_stride_token + dim * source_stride_dim
        destination = (
            head * destination_stride_head
            + (write_start + token) * destination_stride_token
            + dim * destination_stride_dim
        )
        value = tl.load(source_ptr + source, mask=mask, other=0.0)
        tl.store(destination_ptr + destination, value, mask=mask)


def _silu_mul_impl(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape or gate.device != up.device:
        raise ValueError("gate and up must have the same shape and device")
    if not gate.is_floating_point():
        raise TypeError("kerv_silu_mul requires a floating-point tensor")
    _record_hit("kerv_silu_mul", gate, up)
    if (
        _use_triton("kerv_silu_mul")
        and triton is not None
        and _is_cuda_contiguous(gate, up)
        and gate.ndim >= 1
    ):
        width = int(gate.shape[-1])
        if width <= 8192:
            rows = gate.numel() // width
            block = min(triton.next_power_of_2(width), 8192)
            _silu_mul_kernel[(rows,)](
                gate,
                up,
                rows,
                width,
                gate.stride(-2) if gate.ndim > 1 else width,
                up.stride(-2) if up.ndim > 1 else width,
                BLOCK=block,
                num_warps=8 if width >= 2048 else 4,
            )
            return gate
    gate.mul_(torch.sigmoid(gate))
    gate.mul_(up)
    return gate


def _add_rms_norm_impl(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if hidden.shape != residual.shape or hidden.device != residual.device:
        raise ValueError("hidden and residual must have the same shape and device")
    if weight.ndim != 1 or hidden.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight must match the hidden dimension")
    _record_hit("kerv_add_rms_norm", hidden, residual, weight)
    if (
        _use_triton("kerv_add_rms_norm")
        and triton is not None
        and _is_cuda_contiguous(hidden, residual, weight)
    ):
        width = int(hidden.shape[-1])
        if width <= 8192:
            rows = hidden.numel() // width
            block = min(triton.next_power_of_2(width), 8192)
            _add_rms_norm_kernel[(rows,)](
                hidden,
                residual,
                weight,
                rows,
                width,
                hidden.stride(-2) if hidden.ndim > 1 else width,
                residual.stride(-2) if residual.ndim > 1 else width,
                float(eps),
                BLOCK=block,
                num_warps=8 if width >= 2048 else 4,
            )
            return hidden
    hidden.add_(residual)
    hidden.copy_(F.rms_norm(hidden, (hidden.shape[-1],), weight, float(eps)))
    return hidden


def _verify_accept_impl(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    threshold: float,
    token_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or candidates.ndim != 2:
        raise ValueError("expected logits[P,S,V] and candidates[P,S]")
    if logits.shape[:2] != candidates.shape:
        raise ValueError("logits and candidates path/sequence dimensions must match")
    candidates_device = candidates.to(logits.device)
    n_paths, sequence, vocab = map(int, logits.shape)
    n_positions = sequence - 1
    if n_paths <= 0 or n_positions <= 0 or vocab <= 0:
        raise ValueError("Verify-Accept dimensions must be positive")
    _record_hit("kerv_verify_accept_control", logits, candidates_device)
    if (
        _use_triton("kerv_verify_accept_control")
        and triton is not None
        and _is_cuda_contiguous(logits, candidates_device)
        and n_paths <= 128
        and n_positions <= 32
        and vocab <= 65536
    ):
        block_paths = triton.next_power_of_2(n_paths)
        block_vocab = triton.next_power_of_2(vocab)
        # A single-program reduction is useful for the 256-action verifier
        # head.  Full-vocabulary logits would exceed Triton's maximum tensor
        # tile, so they intentionally use the native reduction.
        if block_paths * block_vocab > (1 << 20):
            block_paths = 0
            block_vocab = 0
        if block_paths and block_vocab:
            best_index = torch.empty((), dtype=torch.long, device=logits.device)
            best_length = torch.empty((), dtype=torch.long, device=logits.device)
            _verify_accept_control_kernel[(1,)](
                logits,
                candidates_device,
                best_index,
                best_length,
                logits.stride(0),
                logits.stride(1),
                candidates_device.stride(0),
                candidates_device.stride(1),
                n_paths,
                n_positions,
                vocab,
                float(threshold),
                int(token_offset),
                BLOCK_PATHS=block_paths,
                BLOCK_VOCAB=block_vocab,
                num_warps=8,
                num_stages=1,
            )
            return best_index, best_length

    predicted = torch.argmax(logits[:, :-1], dim=-1) + int(token_offset)
    matches = torch.abs(candidates_device[:, 1:] - predicted) <= float(threshold)
    accepted = torch.cumprod(matches.to(torch.int32), dim=1).sum(dim=1)
    best = torch.argmax(accepted)
    return best.to(torch.long), accepted[best].to(torch.long)


def _static_tree_pack_impl(
    draft_tokens: torch.Tensor,
    kept_indices: torch.Tensor,
    target_nodes: int,
) -> torch.Tensor:
    if draft_tokens.ndim != 2 or kept_indices.ndim != 1:
        raise ValueError("expected draft_tokens[B,N] and kept_indices[K]")
    target_nodes = int(target_nodes)
    kept_nodes = int(kept_indices.numel())
    if target_nodes < kept_nodes or target_nodes <= 0:
        raise ValueError("target_nodes must cover the selected tree nodes")
    if kept_nodes == 0 or draft_tokens.shape[1] <= 0:
        raise ValueError("tree inputs cannot be empty")
    _record_hit("kerv_static_tree_pack", draft_tokens, kept_indices)
    if (
        _use_triton("kerv_static_tree_pack")
        and triton is not None
        and _is_cuda_contiguous(draft_tokens, kept_indices)
        and kept_indices.dtype in (torch.int32, torch.int64)
    ):
        output = torch.empty(
            (draft_tokens.shape[0], target_nodes),
            device=draft_tokens.device,
            dtype=draft_tokens.dtype,
        )
        block = 256
        _pack_tree_kernel[(triton.cdiv(output.numel(), block),)](
            draft_tokens,
            kept_indices,
            output,
            int(draft_tokens.shape[0]),
            int(draft_tokens.shape[1]),
            kept_nodes,
            target_nodes,
            draft_tokens.stride(0),
            draft_tokens.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK=block,
            num_warps=4,
        )
        return output
    packed = torch.index_select(draft_tokens, 1, kept_indices.to(draft_tokens.device))
    if packed.shape[1] == target_nodes:
        return packed
    return torch.cat((packed, packed[:, :1].expand(-1, target_nodes - packed.shape[1])), dim=1)


def _action_projection_select_impl(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    action_token_offset: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if hidden.shape[-1] != weight.shape[-1] or weight.ndim != 2:
        raise ValueError("hidden and action weight dimensions are incompatible")
    if action_token_offset < 0:
        raise ValueError("action_token_offset must be non-negative")
    selected = weight
    if bias is not None and bias.numel() != weight.shape[0]:
        raise ValueError("action bias must match action weight rows")
    flat = hidden.reshape(-1, hidden.shape[-1])
    action_size = int(weight.shape[0])
    _record_hit("kerv_action_projection_select", hidden, weight)
    if (
        _use_triton("kerv_action_projection_select")
        and triton is not None
        and _is_cuda_contiguous(flat, weight)
        and (bias is None or bias.is_cuda and bias.is_contiguous())
        and action_size <= 512
        and hidden.shape[-1] <= 8192
    ):
        output = torch.empty((flat.shape[0],), dtype=torch.long, device=hidden.device)
        block_h = min(triton.next_power_of_2(int(hidden.shape[-1])), 8192)
        block_v = min(triton.next_power_of_2(action_size), 512)
        dummy_bias = bias if bias is not None else weight.new_empty((1,))
        _action_argmax_kernel[(flat.shape[0],)](
            flat,
            selected,
            dummy_bias,
            output,
            int(flat.shape[0]),
            int(hidden.shape[-1]),
            action_size,
            flat.stride(0),
            selected.stride(0),
            bias is not None,
            BLOCK_H=block_h,
            BLOCK_V=block_v,
            num_warps=8,
            num_stages=1,
        )
        return output.reshape(hidden.shape[:-1]) + int(action_token_offset)
    logits = F.linear(flat, selected, bias)
    return logits.argmax(dim=-1).reshape(hidden.shape[:-1]) + int(action_token_offset)


def _value_cache_store_impl(
    destination: torch.Tensor,
    source: torch.Tensor,
    write_start: int,
) -> torch.Tensor:
    if source.ndim != 4 or destination.ndim != 4:
        raise ValueError("value cache tensors must use [B,H,T,D]")
    if source.shape[0] != 1 or destination.shape[0] != 1:
        raise ValueError("KERV value cache store currently supports batch=1")
    if source.shape[1] != destination.shape[1] or source.shape[-1] != destination.shape[-1]:
        raise ValueError("source and destination cache layouts are incompatible")
    token_count = int(source.shape[-2])
    write_start = int(write_start)
    if write_start < 0 or write_start + token_count > destination.shape[-2]:
        raise ValueError("value cache write exceeds destination capacity")
    _record_hit("kerv_value_cache_store", destination, source)
    if (
        _use_triton("kerv_value_cache_store")
        and triton is not None
        and _is_cuda_contiguous(source, destination)
        and source.stride(-1) == destination.stride(-1) == 1
    ):
        block = 1024
        _value_cache_store_kernel[
            (source.shape[1], triton.cdiv(token_count * source.shape[-1], block))
        ](
            source,
            destination,
            token_count,
            write_start,
            source.shape[1],
            source.shape[-1],
            source.stride(1),
            source.stride(2),
            source.stride(3),
            destination.stride(1),
            destination.stride(2),
            destination.stride(3),
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        return destination
    destination[..., write_start : write_start + token_count, :].copy_(source)
    return destination


def _kv_commit_impl(
    destination: torch.Tensor,
    source: torch.Tensor,
    write_start: int,
) -> torch.Tensor:
    if source.ndim != destination.ndim or source.shape[-1] != destination.shape[-1]:
        raise ValueError("KV commit tensors are incompatible")
    end = int(write_start) + int(source.shape[-2])
    if int(write_start) < 0 or end > destination.shape[-2]:
        raise ValueError("KV commit exceeds destination capacity")
    _record_hit("kerv_kv_commit", destination, source)
    destination[..., int(write_start) : end, :].copy_(source)
    return destination


def register_kerv_ops() -> None:
    """Register all public operators once under ``torch.ops.flagos_embodied``."""

    global _DEF_LIBRARY, _IMPL_LIBRARY
    if _DEF_LIBRARY is not None:
        return
    _DEF_LIBRARY = torch.library.Library("flagos_embodied", "DEF")
    definitions = (
        "kerv_silu_mul(Tensor(a!) gate, Tensor up) -> Tensor(a!)",
        "kerv_add_rms_norm(Tensor(a!) hidden, Tensor residual, Tensor weight, float eps) -> Tensor(a!)",
        "kerv_verify_accept_control(Tensor logits, Tensor candidates, float threshold, int token_offset) -> (Tensor, Tensor)",
        "kerv_static_tree_pack(Tensor draft_tokens, Tensor kept_indices, int target_nodes) -> Tensor",
        "kerv_action_projection_select(Tensor hidden, Tensor weight, int action_token_offset, Tensor? bias=None) -> Tensor",
        "kerv_value_cache_store(Tensor(a!) destination, Tensor source, int write_start) -> Tensor(a!)",
        "kerv_kv_commit(Tensor(a!) destination, Tensor source, int write_start) -> Tensor(a!)",
    )
    for definition in definitions:
        _DEF_LIBRARY.define(definition)
    _IMPL_LIBRARY = torch.library.Library("flagos_embodied", "IMPL", "CompositeExplicitAutograd")
    _IMPL_LIBRARY.impl("kerv_silu_mul", _silu_mul_impl)
    _IMPL_LIBRARY.impl("kerv_add_rms_norm", _add_rms_norm_impl)
    _IMPL_LIBRARY.impl("kerv_verify_accept_control", _verify_accept_impl)
    _IMPL_LIBRARY.impl("kerv_static_tree_pack", _static_tree_pack_impl)
    _IMPL_LIBRARY.impl("kerv_action_projection_select", _action_projection_select_impl)
    _IMPL_LIBRARY.impl("kerv_value_cache_store", _value_cache_store_impl)
    _IMPL_LIBRARY.impl("kerv_kv_commit", _kv_commit_impl)

    # The second-stage KERV operators live in a separate module so that the
    # stable first-stage ABI remains importable on CPU-only installations.
    # Register them only after the base namespace has been defined; the
    # ``FRAGMENT`` library used by phase2_ops then safely extends it.
    from .phase2_ops import register_phase2_ops

    register_phase2_ops()


register_kerv_ops()


def kerv_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return torch.ops.flagos_embodied.kerv_silu_mul(gate, up)


def kerv_add_rms_norm(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.ops.flagos_embodied.kerv_add_rms_norm(hidden, residual, weight, float(eps))


def kerv_verify_accept_control(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    threshold: float,
    token_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.flagos_embodied.kerv_verify_accept_control(
        logits, candidates, float(threshold), int(token_offset)
    )


def kerv_static_tree_pack(
    draft_tokens: torch.Tensor,
    kept_indices: torch.Tensor,
    target_nodes: int,
) -> torch.Tensor:
    return torch.ops.flagos_embodied.kerv_static_tree_pack(
        draft_tokens, kept_indices, int(target_nodes)
    )


def kerv_action_projection_select(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    action_token_offset: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.flagos_embodied.kerv_action_projection_select(
        hidden, weight, int(action_token_offset), bias
    )


def kerv_value_cache_store(
    destination: torch.Tensor,
    source: torch.Tensor,
    write_start: int,
) -> torch.Tensor:
    return torch.ops.flagos_embodied.kerv_value_cache_store(destination, source, int(write_start))


def kerv_kv_commit(
    destination: torch.Tensor,
    source: torch.Tensor,
    write_start: int,
) -> torch.Tensor:
    return torch.ops.flagos_embodied.kerv_kv_commit(destination, source, int(write_start))
