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

"""KERV verifier/drafter RoPE fusion for FlagOS.

The stock path gathers cos/sin, materializes ``rotate_half`` twice, launches
four pointwise kernels, and writes intermediate tensors.  This inference-only
operator performs the position gather and both Q/K rotations in one Triton
launch while preserving KERV's native ``[B,H,S,D]`` layout.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

_TRITON_AVAILABLE = True
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on non-Triton platforms.
    _TRITON_AVAILABLE = False

    class _TritonStub:
        __version__ = None

        @staticmethod
        def jit(function=None, **_kwargs):
            if function is not None:
                return function
            return lambda candidate: candidate

    class _LanguageStub:
        constexpr = object()

    triton = _TritonStub()
    tl = _LanguageStub()


_RECORD_PATH: Path | None = None
_RECORD_ONCE = True
_RECORDED: set[tuple[object, ...]] = set()
_ORIGINALS: dict[object, Callable[..., object]] = {}
_ORIGINALS_L31: dict[object, Callable[..., object]] = {}
_ORIGINALS_DIRECT: dict[tuple[object, str], Callable[..., object]] = {}
_ORIGINAL_ATTENTION_FORWARDS: dict[type, Callable[..., object]] = {}
_ROTARY_KEY_DESTINATION: ContextVar[torch.Tensor | None] = ContextVar(
    "flagos_rotary_key_destination", default=None
)
_ROTARY_VALUE_DESTINATION: ContextVar[torch.Tensor | None] = ContextVar(
    "flagos_rotary_value_destination", default=None
)
_ROTARY_VALUE_SOURCE: ContextVar[torch.Tensor | None] = ContextVar(
    "flagos_rotary_value_source", default=None
)
_ROTARY_CACHE_CONTEXT: ContextVar[tuple[object, int, int] | None] = ContextVar(
    "flagos_rotary_cache_context", default=None
)
_ORIGINAL_INSTANCE_FORWARDS: dict[object, Callable[..., object]] = {}
_VALUE_CAPTURE_HANDLES: list[object] = []
_RESIDENT_KEY_WRITE_ENABLED = True


@triton.jit(do_not_specialize=["sequence"])
def _rotary_qk_bhsd_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    position_ptr,
    output_q_ptr,
    output_k_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_d: tl.constexpr,
    oq_stride_b: tl.constexpr,
    oq_stride_h: tl.constexpr,
    oq_stride_s: tl.constexpr,
    oq_stride_d: tl.constexpr,
    ok_stride_b: tl.constexpr,
    ok_stride_h: tl.constexpr,
    ok_stride_s: tl.constexpr,
    ok_stride_d: tl.constexpr,
    cos_stride_b: tl.constexpr,
    cos_stride_s: tl.constexpr,
    sin_stride_b: tl.constexpr,
    sin_stride_s: tl.constexpr,
    indexed: tl.constexpr,
    sequence,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    batch_id = token // sequence
    sequence_id = token % sequence
    if indexed:
        position = tl.load(position_ptr + batch_id * sequence + sequence_id)
        cos_offset = position * cos_stride_s
        sin_offset = position * sin_stride_s
    else:
        cos_offset = batch_id * cos_stride_b + sequence_id * cos_stride_s
        sin_offset = batch_id * sin_stride_b + sequence_id * sin_stride_s
    dims = tl.arange(0, block)
    mask = dims < head_dim
    half = head_dim // 2
    rotated_dims = (dims + half) % head_dim
    cos = tl.load(cos_ptr + cos_offset + dims, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + sin_offset + dims, mask=mask, other=0.0).to(tl.float32)
    sin = tl.where(dims < half, -sin, sin)

    for head in range(0, q_heads):
        source = batch_id * q_stride_b + head * q_stride_h + sequence_id * q_stride_s
        target = batch_id * oq_stride_b + head * oq_stride_h + sequence_id * oq_stride_s
        value = tl.load(q_ptr + source + dims * q_stride_d, mask=mask, other=0.0)
        rotated = tl.load(q_ptr + source + rotated_dims * q_stride_d, mask=mask, other=0.0)
        left = value.to(tl.float32) * cos
        right = rotated.to(tl.float32) * sin
        # PyTorch materializes both BF16/FP16 products before the final add.
        # Preserve that rounding order so fusion does not perturb verifier
        # hidden states near an acceptance boundary.
        if tl.constexpr(q_ptr.dtype.element_ty == tl.bfloat16):
            left = left.to(tl.bfloat16)
            right = right.to(tl.bfloat16)
        elif tl.constexpr(q_ptr.dtype.element_ty == tl.float16):
            left = left.to(tl.float16)
            right = right.to(tl.float16)
        tl.store(
            output_q_ptr + target + dims * oq_stride_d,
            left + right,
            mask=mask,
        )

    for head in range(0, k_heads):
        source = batch_id * k_stride_b + head * k_stride_h + sequence_id * k_stride_s
        target = batch_id * ok_stride_b + head * ok_stride_h + sequence_id * ok_stride_s
        value = tl.load(k_ptr + source + dims * k_stride_d, mask=mask, other=0.0)
        rotated = tl.load(k_ptr + source + rotated_dims * k_stride_d, mask=mask, other=0.0)
        left = value.to(tl.float32) * cos
        right = rotated.to(tl.float32) * sin
        if tl.constexpr(k_ptr.dtype.element_ty == tl.bfloat16):
            left = left.to(tl.bfloat16)
            right = right.to(tl.bfloat16)
        elif tl.constexpr(k_ptr.dtype.element_ty == tl.float16):
            left = left.to(tl.float16)
            right = right.to(tl.float16)
        tl.store(
            output_k_ptr + target + dims * ok_stride_d,
            left + right,
            mask=mask,
        )


@triton.jit(do_not_specialize=["sequence"])
def _rotary_qk_bhsd_head_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    position_ptr,
    output_q_ptr,
    output_k_ptr,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_d: tl.constexpr,
    oq_stride_b: tl.constexpr,
    oq_stride_h: tl.constexpr,
    oq_stride_s: tl.constexpr,
    oq_stride_d: tl.constexpr,
    ok_stride_b: tl.constexpr,
    ok_stride_h: tl.constexpr,
    ok_stride_s: tl.constexpr,
    ok_stride_d: tl.constexpr,
    cos_stride_b: tl.constexpr,
    cos_stride_s: tl.constexpr,
    sin_stride_b: tl.constexpr,
    sin_stride_s: tl.constexpr,
    indexed: tl.constexpr,
    sequence,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    """Expose heads as programs so batch=1, sequence=1 still fills the GPU."""

    token = tl.program_id(0)
    head = tl.program_id(1)
    batch_id = token // sequence
    sequence_id = token % sequence
    if indexed:
        position = tl.load(position_ptr + batch_id * sequence + sequence_id)
        cos_offset = position * cos_stride_s
        sin_offset = position * sin_stride_s
    else:
        cos_offset = batch_id * cos_stride_b + sequence_id * cos_stride_s
        sin_offset = batch_id * sin_stride_b + sequence_id * sin_stride_s
    dims = tl.arange(0, block)
    mask = dims < head_dim
    half = head_dim // 2
    rotated_dims = (dims + half) % head_dim
    cos = tl.load(cos_ptr + cos_offset + dims, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + sin_offset + dims, mask=mask, other=0.0).to(tl.float32)
    sin = tl.where(dims < half, -sin, sin)

    q_mask = mask & (head < q_heads)
    q_source = batch_id * q_stride_b + head * q_stride_h + sequence_id * q_stride_s
    q_target = batch_id * oq_stride_b + head * oq_stride_h + sequence_id * oq_stride_s
    q_value = tl.load(q_ptr + q_source + dims * q_stride_d, mask=q_mask, other=0.0)
    q_rotated = tl.load(
        q_ptr + q_source + rotated_dims * q_stride_d,
        mask=q_mask,
        other=0.0,
    )
    q_left = q_value.to(tl.float32) * cos
    q_right = q_rotated.to(tl.float32) * sin
    if tl.constexpr(q_ptr.dtype.element_ty == tl.bfloat16):
        q_left = q_left.to(tl.bfloat16)
        q_right = q_right.to(tl.bfloat16)
    elif tl.constexpr(q_ptr.dtype.element_ty == tl.float16):
        q_left = q_left.to(tl.float16)
        q_right = q_right.to(tl.float16)
    tl.store(
        output_q_ptr + q_target + dims * oq_stride_d,
        q_left + q_right,
        mask=q_mask,
    )

    k_mask = mask & (head < k_heads)
    k_source = batch_id * k_stride_b + head * k_stride_h + sequence_id * k_stride_s
    k_target = batch_id * ok_stride_b + head * ok_stride_h + sequence_id * ok_stride_s
    k_value = tl.load(k_ptr + k_source + dims * k_stride_d, mask=k_mask, other=0.0)
    k_rotated = tl.load(
        k_ptr + k_source + rotated_dims * k_stride_d,
        mask=k_mask,
        other=0.0,
    )
    k_left = k_value.to(tl.float32) * cos
    k_right = k_rotated.to(tl.float32) * sin
    if tl.constexpr(k_ptr.dtype.element_ty == tl.bfloat16):
        k_left = k_left.to(tl.bfloat16)
        k_right = k_right.to(tl.bfloat16)
    elif tl.constexpr(k_ptr.dtype.element_ty == tl.float16):
        k_left = k_left.to(tl.float16)
        k_right = k_right.to(tl.float16)
    tl.store(
        output_k_ptr + k_target + dims * ok_stride_d,
        k_left + k_right,
        mask=k_mask,
    )


def _supported(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> bool:
    return bool(
        _TRITON_AVAILABLE
        and query.is_cuda
        and key.is_cuda
        and query.ndim == 4
        and key.ndim == 4
        and query.shape[0] == key.shape[0] == 1
        and query.shape[2:] == key.shape[2:]
        and query.shape[-1] == 128
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and position_ids.is_cuda
        and position_ids.is_contiguous()
        and position_ids.shape == (query.shape[0], query.shape[2])
        and position_ids.dtype in (torch.int32, torch.int64)
        and cos.is_cuda
        and sin.is_cuda
    )


def fused_rotary_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse table gather, split-half rotation, and Q/K output writes."""

    if not _supported(query, key, cos, sin, position_ids):
        raise ValueError("unsupported KERV rotary layout")
    full_cos = cos.squeeze(0).squeeze(0).contiguous()
    full_sin = sin.squeeze(0).squeeze(0).contiguous()
    if full_cos.ndim != 2 or full_cos.shape[-1] != query.shape[-1]:
        raise ValueError("rotary tables must be [max_sequence, head_dim]")

    # Native RoPE arithmetic returns dense outputs even when Q/K are transposed
    # views.  Match that contract while accepting the verifier's strided input.
    output_q = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    output_k = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    block = triton.next_power_of_2(query.shape[-1])
    common_args = (
        query,
        key,
        full_cos,
        full_sin,
        position_ids,
        output_q,
        output_k,
        *query.stride(),
        *key.stride(),
        *output_q.stride(),
        *output_k.stride(),
        0,
        full_cos.stride(0),
        0,
        full_sin.stride(0),
        True,
        query.shape[2],
        query.shape[1],
        key.shape[1],
        query.shape[-1],
        block,
    )
    # Long prompts have enough token-level parallelism and benefit from loading
    # each cos/sin row once.  Decode/tree steps are short, so parallelize heads
    # to avoid one Triton program serially rotating all 32 attention heads.
    if query.shape[2] <= 8:
        _rotary_qk_bhsd_head_kernel[
            (query.shape[0] * query.shape[2], max(query.shape[1], key.shape[1]))
        ](
            *common_args,
            num_warps=1,
            num_stages=1,
        )
    else:
        _rotary_qk_bhsd_kernel[(query.shape[0] * query.shape[2],)](
            *common_args,
            num_warps=4,
            num_stages=1,
        )
    return output_q, output_k


def fused_rotary_qk_direct(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    output_key: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse LLaMA-3.1 RoPE when cos/sin are already selected as [B,S,D]."""

    supported = bool(
        _TRITON_AVAILABLE
        and query.is_cuda
        and key.is_cuda
        and query.ndim == 4
        and key.ndim == 4
        and query.shape[0] == key.shape[0] == 1
        and query.shape[2:] == key.shape[2:]
        and query.shape[-1] == 128
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.dtype == query.dtype
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and cos.is_cuda
        and sin.is_cuda
        and cos.dtype == query.dtype
        and sin.dtype == query.dtype
        and cos.shape == sin.shape == (query.shape[0], query.shape[2], query.shape[3])
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
    )
    if not supported:
        raise ValueError("unsupported KERV LLaMA-3.1 rotary layout")

    output_q = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    if output_key is None:
        output_k = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    else:
        if (
            output_key.shape != key.shape
            or output_key.device != key.device
            or output_key.dtype != key.dtype
            or output_key.stride(-1) != 1
        ):
            raise ValueError("resident rotary K destination is incompatible")
        output_k = output_key
    block = triton.next_power_of_2(query.shape[-1])
    common_args = (
        query,
        key,
        cos,
        sin,
        cos,  # Unused dummy pointer for the non-indexed specialization.
        output_q,
        output_k,
        *query.stride(),
        *key.stride(),
        *output_q.stride(),
        *output_k.stride(),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        False,
        query.shape[2],
        query.shape[1],
        key.shape[1],
        query.shape[-1],
        block,
    )
    if query.shape[2] <= 8:
        _rotary_qk_bhsd_head_kernel[
            (query.shape[0] * query.shape[2], max(query.shape[1], key.shape[1]))
        ](*common_args, num_warps=1, num_stages=1)
    else:
        _rotary_qk_bhsd_kernel[(query.shape[0] * query.shape[2],)](
            *common_args, num_warps=4, num_stages=1
        )
    return output_q, output_k


def _record(kind: str, query: torch.Tensor, key: torch.Tensor) -> None:
    if _RECORD_PATH is None:
        return
    record_key = (kind, tuple(query.shape), tuple(key.shape), str(query.dtype))
    if _RECORD_ONCE and record_key in _RECORDED:
        return
    _RECORDED.add(record_key)
    payload = {
        "operator": "fused_rotary_qk",
        "scope": kind,
        "implementation": "flagos_triton_adaptive_bhsd_rotary_qk",
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "dtype": str(query.dtype).removeprefix("torch."),
    }
    with _RECORD_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _install(module: object, scope: str) -> None:
    original = module.apply_rotary_pos_emb
    if module in _ORIGINALS:
        return
    _ORIGINALS[module] = original

    def adaptive(query, key, cos, sin, position_ids):
        if _supported(query, key, cos, sin, position_ids):
            _record(scope, query, key)
            return fused_rotary_qk(query, key, cos, sin, position_ids)
        return original(query, key, cos, sin, position_ids)

    module.apply_rotary_pos_emb = adaptive


def _install_l31(module: object, scope: str) -> None:
    if not hasattr(module, "apply_rotary_pos_emb_L31") or module in _ORIGINALS_L31:
        return
    original = module.apply_rotary_pos_emb_L31
    _ORIGINALS_L31[module] = original

    def adaptive(query, key, cos, sin, position_ids=None, unsqueeze_dim=1):
        del position_ids
        supported = bool(
            unsqueeze_dim == 1
            and query.is_cuda
            and key.is_cuda
            and query.ndim == 4
            and key.ndim == 4
            and cos.shape == sin.shape == (query.shape[0], query.shape[2], query.shape[3])
        )
        if supported:
            try:
                result = fused_rotary_qk_direct(query, key, cos, sin)
            except ValueError:
                pass
            else:
                _record(scope, query, key)
                return result
        return original(query, key, cos, sin, None, unsqueeze_dim)

    module.apply_rotary_pos_emb_L31 = adaptive


def _install_direct(module: object, attribute: str, scope: str) -> None:
    key = (module, attribute)
    if not hasattr(module, attribute) or key in _ORIGINALS_DIRECT:
        return
    original = getattr(module, attribute)
    _ORIGINALS_DIRECT[key] = original

    def adaptive(query, key_tensor, cos, sin, position_ids=None, unsqueeze_dim=1):
        supported = bool(
            unsqueeze_dim == 1
            and query.is_cuda
            and key_tensor.is_cuda
            and query.ndim == 4
            and key_tensor.ndim == 4
            and cos.shape == sin.shape == (query.shape[0], query.shape[2], query.shape[3])
        )
        if supported:
            resident_key = _ROTARY_KEY_DESTINATION.get()
            resident_value = _ROTARY_VALUE_DESTINATION.get()
            value_source = _ROTARY_VALUE_SOURCE.get()
            cache_context = _ROTARY_CACHE_CONTEXT.get()
            if (
                resident_key is not None
                and resident_value is not None
                and value_source is not None
                and cache_context is not None
            ):
                cache, layer_idx, token_count = cache_context
                try:
                    value_states = value_source.view(
                        query.shape[0],
                        query.shape[2],
                        key_tensor.shape[1],
                        query.shape[3],
                    ).transpose(1, 2)
                    query_output = torch.ops.flagos_embodied.kerv_rope_kv_store(
                        query,
                        key_tensor,
                        value_states,
                        cos,
                        sin,
                        cache.empty_position_ids,
                        resident_key,
                        resident_value,
                        0,
                    )
                    cache.mark_rotary_value_written(layer_idx, token_count)
                except (RuntimeError, ValueError, AttributeError):
                    pass
                else:
                    _record(f"{scope}_resident_kv", query, key_tensor)
                    return query_output, resident_key
            try:
                result = fused_rotary_qk_direct(
                    query,
                    key_tensor,
                    cos,
                    sin,
                    output_key=resident_key,
                )
            except ValueError:
                pass
            else:
                _record(
                    f"{scope}_resident_key" if resident_key is not None else scope,
                    query,
                    key_tensor,
                )
                return result
        return original(query, key_tensor, cos, sin, position_ids, unsqueeze_dim)

    setattr(module, attribute, adaptive)


def _install_resident_kv_instance(attention: object) -> None:
    """Expose both K and V resident slots around one attention invocation."""

    if attention in _ORIGINAL_INSTANCE_FORWARDS:
        return
    original = attention.forward
    _ORIGINAL_INSTANCE_FORWARDS[attention] = original

    def capture_value(_module, _inputs, output):
        if _ROTARY_VALUE_DESTINATION.get() is not None:
            _ROTARY_VALUE_SOURCE.set(output)

    _VALUE_CAPTURE_HANDLES.append(attention.v_proj.register_forward_hook(capture_value))

    def flagos_attention_forward(self, hidden_states, *args, **kwargs):
        cache = kwargs.get("past_key_value")
        if cache is None and len(args) >= 3:
            cache = args[2]
        cache = getattr(self, "past_key_value", cache)
        key_destination = None
        value_destination = None
        cache_context = None
        if (
            _RESIDENT_KEY_WRITE_ENABLED
            and getattr(cache, "_flagos_persistent_tree_cache", False)
            and hasattr(cache, "rotary_value_destination")
        ):
            token_count = int(hidden_states.shape[1])
            layer_idx = int(self.layer_idx)
            key_destination = cache.rotary_key_destination(layer_idx, token_count)
            value_destination = cache.rotary_value_destination(layer_idx, token_count)
            cache_context = (cache, layer_idx, token_count)
        key_token = _ROTARY_KEY_DESTINATION.set(key_destination)
        value_token = _ROTARY_VALUE_DESTINATION.set(value_destination)
        source_token = _ROTARY_VALUE_SOURCE.set(None)
        cache_token = _ROTARY_CACHE_CONTEXT.set(cache_context)
        try:
            return original(hidden_states, *args, **kwargs)
        finally:
            _ROTARY_CACHE_CONTEXT.reset(cache_token)
            _ROTARY_VALUE_SOURCE.reset(source_token)
            _ROTARY_VALUE_DESTINATION.reset(value_token)
            _ROTARY_KEY_DESTINATION.reset(key_token)

    attention.forward = flagos_attention_forward.__get__(attention, type(attention))


def _install_resident_key_context(attention_class: type) -> None:
    """Expose the current persistent-cache K slot to the fused RoPE call."""

    if attention_class in _ORIGINAL_ATTENTION_FORWARDS:
        return
    original = attention_class.forward
    _ORIGINAL_ATTENTION_FORWARDS[attention_class] = original

    def flagos_attention_forward(self, hidden_states, *args, **kwargs):
        cache = kwargs.get("past_key_value")
        # Positional order after hidden_states: attention_mask, position_ids,
        # past_key_value.  Most KERV calls use keywords, but retain both forms.
        if cache is None and len(args) >= 3:
            cache = args[2]
        cache = getattr(self, "past_key_value", cache)
        destination = None
        if (
            _RESIDENT_KEY_WRITE_ENABLED
            and getattr(cache, "_flagos_persistent_tree_cache", False)
            and hasattr(cache, "rotary_key_destination")
        ):
            destination = cache.rotary_key_destination(
                int(self.layer_idx), int(hidden_states.shape[1])
            )
        token = _ROTARY_KEY_DESTINATION.set(destination)
        try:
            return original(self, hidden_states, *args, **kwargs)
        finally:
            _ROTARY_KEY_DESTINATION.reset(token)

    attention_class.forward = flagos_attention_forward


def _restore() -> None:
    """Restore native module functions when the runtime switch is disabled."""

    for module, original in tuple(_ORIGINALS.items()):
        module.apply_rotary_pos_emb = original
    _ORIGINALS.clear()
    for module, original in tuple(_ORIGINALS_L31.items()):
        module.apply_rotary_pos_emb_L31 = original
    _ORIGINALS_L31.clear()
    for (module, attribute), original in tuple(_ORIGINALS_DIRECT.items()):
        setattr(module, attribute, original)
    _ORIGINALS_DIRECT.clear()
    for attention_class, original in tuple(_ORIGINAL_ATTENTION_FORWARDS.items()):
        attention_class.forward = original
    _ORIGINAL_ATTENTION_FORWARDS.clear()
    for attention, original in tuple(_ORIGINAL_INSTANCE_FORWARDS.items()):
        attention.forward = original
    _ORIGINAL_INSTANCE_FORWARDS.clear()
    for handle in _VALUE_CAPTURE_HANDLES:
        handle.remove()
    _VALUE_CAPTURE_HANDLES.clear()


def enable_rotary_fusion(
    enabled: bool,
    resident_key_write: bool = True,
    record: bool = True,
    record_once: bool = True,
    log_path: str | None = None,
    manifest_path: str | None = None,
    model=None,
) -> dict[str, Any]:
    """Install KERV verifier and drafter adapters after their modules import."""

    global _RECORD_PATH, _RECORD_ONCE, _RESIDENT_KEY_WRITE_ENABLED
    _RESIDENT_KEY_WRITE_ENABLED = bool(resident_key_write)
    if record and enabled and not log_path:
        raise ValueError("rotary fusion recording requires log_path")
    _RECORD_PATH = Path(log_path) if enabled and record and log_path else None
    _RECORD_ONCE = bool(record_once)
    _RECORDED.clear()
    if _RECORD_PATH is not None:
        _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECORD_PATH.write_text("", encoding="utf-8")

    scopes: list[str] = []
    if enabled:
        from openvla.specdecoding.model import cnets, modeling_llama_kv
        from transformers.models.llama import modeling_llama as hf_modeling_llama

        _install(modeling_llama_kv, "verifier")
        _install_l31(modeling_llama_kv, "verifier_l31")
        _install_direct(hf_modeling_llama, "apply_rotary_pos_emb", "verifier_hf")
        for attention_name in (
            "LlamaAttention",
            "LlamaFlashAttention2",
            "LlamaSdpaAttention",
        ):
            attention_class = getattr(hf_modeling_llama, attention_name, None)
            if attention_class is not None:
                _install_resident_key_context(attention_class)
        if model is not None:
            language_model = getattr(getattr(model, "base_model", None), "language_model", None)
            if language_model is not None:
                for module in language_model.modules():
                    if all(
                        hasattr(module, attribute)
                        for attribute in (
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "layer_idx",
                        )
                    ):
                        _install_resident_kv_instance(module)
        scopes.append("verifier")
        _install(cnets, "drafter")
        scopes.append("drafter")
    else:
        _restore()

    manifest: dict[str, Any] = {
        "enabled": bool(enabled),
        "operator": "fused_rotary_qk",
        "implementation": "flagos_triton_adaptive_bhsd_rotary_qk",
        "scopes": scopes,
        "supported_layout": "B,H,S,128",
        "supported_dtypes": ["float16", "bfloat16"],
        "native_fallback": True,
        "resident_key_write": bool(resident_key_write),
        "resident_value_write": bool(resident_key_write),
        "inference_only": True,
        "torch_version": torch.__version__,
        "triton_version": getattr(triton, "__version__", None),
    }
    if torch.cuda.is_available():
        manifest["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    if manifest_path:
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[FlagOS/RotaryFusion] {'enabled' if enabled else 'disabled'}; scopes={scopes}",
        flush=True,
    )
    return manifest
