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

"""Flash-style attention specialized for fixed KERV verification trees."""

from __future__ import annotations

import math
from types import MethodType

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _static_tree_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        ancestors_ptr,
        valid_prefix_ptr,
        output_ptr,
        stride_qh: tl.constexpr,
        stride_qm: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_am: tl.constexpr,
        stride_aa: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_om: tl.constexpr,
        stride_od: tl.constexpr,
        query_tokens: tl.constexpr,
        prefix_capacity: tl.constexpr,
        use_dynamic_prefix: tl.constexpr,
        head_dim: tl.constexpr,
        max_ancestors: tl.constexpr,
        ancestor_count,
        scale_log2: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        head = tl.program_id(1)
        rows = query_block * block_m + tl.arange(0, block_m)
        dims = tl.arange(0, head_dim)
        row_mask = rows < query_tokens
        q_offsets = head * stride_qh + rows[:, None] * stride_qm + dims[None, :] * stride_qd
        query = tl.load(q_ptr + q_offsets, mask=row_mask[:, None], other=0.0)

        running_max = tl.full((block_m,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_m,), tl.float32)
        accumulator = tl.zeros((block_m, head_dim), tl.float32)

        valid_prefix_tokens = prefix_capacity
        if use_dynamic_prefix:
            valid_prefix_tokens = tl.load(valid_prefix_ptr)
        for start in range(0, prefix_capacity, block_n):
            columns = start + tl.arange(0, block_n)
            column_mask = columns < valid_prefix_tokens
            k_offsets = head * stride_kh + columns[:, None] * stride_kn + dims[None, :] * stride_kd
            v_offsets = head * stride_vh + columns[:, None] * stride_vn + dims[None, :] * stride_vd
            keys = tl.load(k_ptr + k_offsets, mask=column_mask[:, None], other=0.0)
            scores = tl.dot(query, tl.trans(keys)) * scale_log2
            scores = tl.where(row_mask[:, None] & column_mask[None, :], scores, -float("inf"))
            block_max = tl.max(scores, axis=1)
            next_max = tl.maximum(running_max, block_max)
            correction = tl.exp2(running_max - next_max)
            probabilities = tl.exp2(scores - next_max[:, None])
            values = tl.load(v_ptr + v_offsets, mask=column_mask[:, None], other=0.0)
            accumulator = accumulator * correction[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
            running_max = next_max

        slots = tl.arange(0, max_ancestors)
        ancestor_offsets = rows[:, None] * stride_am + slots[None, :] * stride_aa
        ancestor_ids = tl.load(ancestors_ptr + ancestor_offsets, mask=row_mask[:, None], other=-1)
        ancestor_valid = (
            row_mask[:, None]
            & (slots[None, :] < ancestor_count)
            & (ancestor_ids >= prefix_capacity)
        )
        tree_k_offsets = (
            head * stride_kh
            + ancestor_ids[:, :, None] * stride_kn
            + dims[None, None, :] * stride_kd
        )
        tree_v_offsets = (
            head * stride_vh
            + ancestor_ids[:, :, None] * stride_vn
            + dims[None, None, :] * stride_vd
        )
        ancestor_keys = tl.load(k_ptr + tree_k_offsets, mask=ancestor_valid[:, :, None], other=0.0)
        ancestor_scores = tl.sum(query[:, None, :] * ancestor_keys, axis=2) * scale_log2
        ancestor_scores = tl.where(ancestor_valid, ancestor_scores, -float("inf"))
        ancestor_max = tl.max(ancestor_scores, axis=1)
        next_max = tl.maximum(running_max, ancestor_max)
        correction = tl.exp2(running_max - next_max)
        probabilities = tl.exp2(ancestor_scores - next_max[:, None])
        ancestor_values = tl.load(
            v_ptr + tree_v_offsets, mask=ancestor_valid[:, :, None], other=0.0
        )
        accumulator = accumulator * correction[:, None] + tl.sum(
            probabilities[:, :, None] * ancestor_values, axis=1
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)

        output = accumulator / running_sum[:, None]
        output_offsets = head * stride_oh + rows[:, None] * stride_om + dims[None, :] * stride_od
        tl.store(output_ptr + output_offsets, output, mask=row_mask[:, None])


def is_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ancestor_indices: torch.Tensor,
    prefix_tokens: int,
    valid_prefix_tokens: torch.Tensor | None = None,
) -> bool:
    dynamic_prefix_ok = bool(
        valid_prefix_tokens is None
        or (
            valid_prefix_tokens.is_cuda
            and valid_prefix_tokens.numel() == 1
            and valid_prefix_tokens.dtype in (torch.int32, torch.int64)
        )
    )
    return bool(
        triton is not None
        and query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and ancestor_indices.is_cuda
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[0] == key.shape[0] == value.shape[0] == 1
        and query.shape[1] == key.shape[1] == value.shape[1]
        and key.shape == value.shape
        and query.shape[-1] == key.shape[-1] == 128
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dtype == key.dtype == value.dtype
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
        and ancestor_indices.ndim == 2
        and ancestor_indices.shape[0] == query.shape[2]
        and ancestor_indices.shape[1] in (4, 5, 8)
        and ancestor_indices.dtype in (torch.int32, torch.int64)
        and ancestor_indices.is_contiguous()
        and 0 < int(prefix_tokens) <= key.shape[2]
        and dynamic_prefix_ok
    )


def static_tree_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ancestor_indices: torch.Tensor,
    prefix_tokens: int,
    scale: float | None = None,
    valid_prefix_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    query_tokens = query.shape[2]
    key_tokens = key.shape[2]
    visible = torch.zeros((query_tokens, key_tokens), device=query.device, dtype=torch.bool)
    valid_prefix = (
        int(prefix_tokens)
        if valid_prefix_tokens is None
        else int(valid_prefix_tokens.reshape(-1)[0].item())
    )
    visible[:, :valid_prefix] = True
    rows = torch.arange(query_tokens, device=query.device)[:, None]
    valid = ancestor_indices >= int(prefix_tokens)
    row_ids = rows.expand_as(ancestor_indices)[valid]
    col_ids = ancestor_indices[valid].long()
    visible[row_ids, col_ids] = True
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
    scores *= scale if scale is not None else query.shape[-1] ** -0.5
    scores.masked_fill_(~visible[None, None, :, :], -torch.inf)
    return torch.matmul(torch.softmax(scores, dim=-1), value.float()).to(query.dtype)


def static_tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ancestor_indices: torch.Tensor,
    prefix_tokens: int,
    scale: float | None = None,
    valid_prefix_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fast fixed-tree path, or the exact dense reference when needed."""

    if (
        not _ENABLED
        # This kernel has a stable local speedup on its fixed KERV layout, so
        # auto mode may use it.  Other embodied kernels remain conservative.
        or _BACKEND not in {"auto", "triton"}
        or not is_supported(
            query,
            key,
            value,
            ancestor_indices,
            prefix_tokens,
            valid_prefix_tokens,
        )
    ):
        return static_tree_attention_reference(
            query,
            key,
            value,
            ancestor_indices,
            prefix_tokens,
            scale,
            valid_prefix_tokens,
        )
    output = torch.empty_like(query)
    head_dim = int(query.shape[-1])
    # KERV's verification trees are wide (224--320 nodes) while each query
    # only follows at most eight ancestors.  A 16-row tile reduces register
    # pressure on A100; the previous 32-row tile measured about 9% slower.
    block_m, block_n = 16, 64
    attention_scale = (scale if scale is not None else head_dim**-0.5) * math.log2(math.e)
    grid = (triton.cdiv(query.shape[2], block_m), query.shape[1])
    ancestor_count = int(ancestor_indices.shape[1])
    ancestor_block = triton.next_power_of_2(ancestor_count)
    if ancestor_block != ancestor_count:
        padded = torch.full(
            (ancestor_indices.shape[0], ancestor_block),
            -1,
            device=ancestor_indices.device,
            dtype=ancestor_indices.dtype,
        )
        padded[:, :ancestor_count].copy_(ancestor_indices)
        ancestor_indices = padded
    _static_tree_attention_kernel[grid](
        query,
        key,
        value,
        ancestor_indices,
        valid_prefix_tokens if valid_prefix_tokens is not None else ancestor_indices,
        output,
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        ancestor_indices.stride(0),
        ancestor_indices.stride(1),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        query.shape[2],
        int(prefix_tokens),
        valid_prefix_tokens is not None,
        head_dim,
        ancestor_block,
        ancestor_count,
        attention_scale,
        block_m,
        block_n,
        num_warps=4,
        num_stages=1,
    )
    return output


_LIBRARY: torch.library.Library | None = None
_IMPL_LIBRARY: torch.library.Library | None = None
_BACKEND = "auto"
_ENABLED = True


def configure_backend(backend: str) -> None:
    global _BACKEND
    backend = str(backend).lower()
    if backend not in {"auto", "triton", "native"}:
        raise ValueError(f"unsupported KERV embodied backend: {backend}")
    _BACKEND = backend


def configure_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def install_static_tree_attention(model) -> list[str]:
    """Install an opt-in verifier hook on compatible Llama attention layers.

    The hook is deliberately disabled by the default ``auto`` KERV profile;
    callers select ``backend=triton`` after an exact closed-loop comparison.
    If a model uses an unsupported cache or mask layout, the original HF
    attention implementation is called unchanged.
    """

    if getattr(model, "_flagos_static_tree_attention_installed", False):
        return list(getattr(model, "_flagos_static_tree_attention_targets", ()))
    language_model = getattr(getattr(model, "base_model", None), "language_model", None)
    if language_model is None:
        return []
    try:
        from transformers.models.llama.modeling_llama import (
            repeat_kv as hf_repeat_kv,
        )
    except Exception:
        return []

    targets: list[str] = []
    for name, module in language_model.named_modules():
        if (
            not all(
                hasattr(module, attr)
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "rotary_emb")
            )
            or not hasattr(module, "num_heads")
            or getattr(module, "_flagos_static_tree_attention_wrapped", False)
        ):
            continue
        original = module.forward

        def static_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            _original=original,
            **kwargs,
        ):
            if (
                _BACKEND not in {"auto", "triton"}
                or not _ENABLED
                or not bool(
                    getattr(
                        language_model,
                        "_flagos_static_tree_attention_runtime_enabled",
                        False,
                    )
                )
                or self.training
                or output_attentions
                or not isinstance(attention_mask, torch.Tensor)
                or not attention_mask.dtype.is_floating_point
                or attention_mask.ndim != 4
                or attention_mask.shape[0] != 1
                or attention_mask.shape[1] != 1
                or not hidden_states.is_cuda
                or not attention_mask.is_cuda
                or hidden_states.ndim != 3
                or hidden_states.shape[0] != 1
                or hidden_states.shape[1] <= 1
            ):
                return _original(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
            bsz, query_tokens, _ = hidden_states.shape
            # Inspect the fixed mask before touching the cache.  This keeps
            # every unsupported case on the original HF path without a
            # partially updated DynamicCache.
            if self.config.pretraining_tp != 1:
                return _original(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
            past_key_value = getattr(self, "past_key_value", past_key_value)
            persistent = bool(
                getattr(past_key_value, "_flagos_persistent_tree_cache", False)
                and getattr(past_key_value, "fixed_workspace_layout", False)
            )
            ancestor_indices = getattr(past_key_value, "current_ancestor_indices", None)
            tree_start = int(getattr(past_key_value, "prefix_capacity", 0))
            key_tokens = int(attention_mask.shape[-1])
            if (
                not persistent
                or not isinstance(ancestor_indices, torch.Tensor)
                or ancestor_indices.shape != (query_tokens, 8)
                or tree_start <= 0
                or key_tokens != tree_start + query_tokens
            ):
                return _original(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
            try:
                query_states = (
                    self.q_proj(hidden_states)
                    .view(bsz, query_tokens, self.num_heads, self.head_dim)
                    .transpose(1, 2)
                )
                key_states = (
                    self.k_proj(hidden_states)
                    .view(bsz, query_tokens, self.num_key_value_heads, self.head_dim)
                    .transpose(1, 2)
                )
                value_states = (
                    self.v_proj(hidden_states)
                    .view(bsz, query_tokens, self.num_key_value_heads, self.head_dim)
                    .transpose(1, 2)
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
                key_destination = past_key_value.rotary_key_destination(
                    int(self.layer_idx), query_tokens
                )
                value_destination = past_key_value.rotary_value_destination(
                    int(self.layer_idx), query_tokens
                )
                query_states = torch.ops.flagos_embodied.kerv_rope_kv_store(
                    query_states,
                    key_states,
                    value_states,
                    cos,
                    sin,
                    past_key_value.empty_position_ids,
                    past_key_value.storage[int(self.layer_idx), 0],
                    past_key_value.storage[int(self.layer_idx), 1],
                    tree_start,
                )
                past_key_value.mark_rotary_value_written(int(self.layer_idx), query_tokens)
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_destination,
                    value_destination,
                    self.layer_idx,
                    cache_kwargs,
                )
                key_states = hf_repeat_kv(key_states, self.num_key_value_groups)
                value_states = hf_repeat_kv(value_states, self.num_key_value_groups)
                if int(key_states.shape[-2]) != key_tokens:
                    raise RuntimeError("static tree attention cache length changed")
                output = torch.ops.flagos_embodied.kerv_static_tree_attention(
                    query_states,
                    key_states,
                    value_states,
                    ancestor_indices,
                    tree_start,
                    self.head_dim**-0.5,
                    past_key_value.prefix_length_tensor,
                )
                output = output.transpose(1, 2).reshape(bsz, query_tokens, self.hidden_size)
                output = self.o_proj(output)
                language_model._flagos_static_tree_attention_hits = (
                    int(getattr(language_model, "_flagos_static_tree_attention_hits", 0)) + 1
                )
                return output, None, past_key_value
            except Exception as exc:
                raise RuntimeError("static tree attention execution failed") from exc

        module.forward = MethodType(static_forward, module)
        module._flagos_static_tree_attention_wrapped = True
        targets.append(name)
    model._flagos_static_tree_attention_installed = True
    model._flagos_static_tree_attention_targets = targets
    return targets


def register_static_tree_attention() -> None:
    global _LIBRARY, _IMPL_LIBRARY
    if _LIBRARY is not None:
        return
    # kerv_ops owns the namespace definition; a fragment lets this module add
    # the attention schema without creating a second TORCH_LIBRARY block.
    library = torch.library.Library("flagos_embodied", "FRAGMENT")
    library.define(
        "kerv_static_tree_attention(Tensor query, Tensor key, Tensor value, "
        "Tensor ancestor_indices, int prefix_tokens, float? scale=None, "
        "Tensor? valid_prefix_tokens=None) -> Tensor"
    )
    _IMPL_LIBRARY = torch.library.Library("flagos_embodied", "IMPL", "CompositeExplicitAutograd")
    _IMPL_LIBRARY.impl("kerv_static_tree_attention", static_tree_attention)
    _LIBRARY = library


register_static_tree_attention()
