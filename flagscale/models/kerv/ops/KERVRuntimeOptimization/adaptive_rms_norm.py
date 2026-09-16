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

"""Inference-only RMSNorm kernel for small-batch embodied-model workloads.

FlagGems provides the portable operator implementation.  This extension keeps
the same RMSNorm semantics while removing training-only state allocation from
the inference path.  Callers remain responsible for routing unsupported or
unprofitable shapes to their native implementation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

_TRITON_AVAILABLE = True
try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on non-Triton platforms.
    _TRITON_AVAILABLE = False

    class _TritonStub:
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
_RECORDED_SHAPES: set[tuple[object, ...]] = set()
_ORIGINAL_RMS_FORWARDS: dict[type, object] = {}


def configure_adaptive_recording(
    path: str | None,
    record: bool,
    once: bool = True,
) -> None:
    """Configure low-overhead, shape-level evidence for actual kernel hits."""
    global _RECORD_PATH, _RECORD_ONCE
    _RECORD_PATH = Path(path) if record and path else None
    _RECORD_ONCE = bool(once)
    _RECORDED_SHAPES.clear()
    if _RECORD_PATH is not None:
        _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECORD_PATH.write_text("", encoding="utf-8")


def _record_hit(input_tensor: torch.Tensor, columns: int) -> None:
    if _RECORD_PATH is None:
        return
    key = (
        tuple(input_tensor.shape),
        tuple(input_tensor.stride()),
        str(input_tensor.dtype),
        columns,
    )
    if _RECORD_ONCE and key in _RECORDED_SHAPES:
        return
    _RECORDED_SHAPES.add(key)
    record = {
        "operator": "rms_norm",
        "implementation": "flagos_adaptive_triton",
        "shape": list(input_tensor.shape),
        "stride": list(input_tensor.stride()),
        "dtype": str(input_tensor.dtype).removeprefix("torch."),
        "normalized_elements": columns,
    }
    with _RECORD_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


@triton.jit
def _rms_norm_inference_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    row_stride,
    columns: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
    round_before_weight: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < columns

    values = tl.load(
        input_ptr + row * row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / columns
    inverse_rms = 1.0 / tl.sqrt(variance + epsilon)

    # Handwritten Qwen/Llama modules round before applying the learned weight,
    # while aten::rms_norm keeps the calculation fused.  Both semantics occur
    # in the two embodied workloads, so the rounding point is explicit.
    if round_before_weight and tl.constexpr(input_ptr.dtype.element_ty == tl.bfloat16):
        normalized = (values * inverse_rms).to(tl.bfloat16)
    elif round_before_weight and tl.constexpr(input_ptr.dtype.element_ty == tl.float16):
        normalized = (values * inverse_rms).to(tl.float16)
    else:
        normalized = values * inverse_rms

    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        output_ptr + row * row_stride + offsets,
        normalized * weights,
        mask=mask,
    )


@triton.jit
def _add_rms_norm_inference_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    row_stride,
    columns: tl.constexpr,
    epsilon: tl.constexpr,
    block_size: tl.constexpr,
):
    """Residual add plus RMSNorm with the native BF16 rounding points."""
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < columns
    input_offsets = row * row_stride + offsets

    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    summed = values + residual

    # PyTorch produces the residual add in the input dtype before the
    # handwritten Llama/Qwen RMSNorm promotes it to FP32. Reintroducing both
    # BF16/FP16 rounding points avoids the token drift of generic fused kernels.
    if tl.constexpr(input_ptr.dtype.element_ty == tl.bfloat16):
        summed = summed.to(tl.bfloat16)
    elif tl.constexpr(input_ptr.dtype.element_ty == tl.float16):
        summed = summed.to(tl.float16)
    # The attention output is disposable; retain the residual input because
    # custom speculative decoders may keep aliases to it.
    tl.store(input_ptr + input_offsets, summed, mask=mask)

    summed_fp32 = summed.to(tl.float32)
    variance = tl.sum(summed_fp32 * summed_fp32, axis=0) / columns
    inverse_rms = tl.rsqrt(variance + epsilon)
    normalized = summed_fp32 * inverse_rms
    if tl.constexpr(input_ptr.dtype.element_ty == tl.bfloat16):
        normalized = normalized.to(tl.bfloat16)
    elif tl.constexpr(input_ptr.dtype.element_ty == tl.float16):
        normalized = normalized.to(tl.float16)

    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + input_offsets, normalized * weights, mask=mask)


def rms_norm_inference(
    input_tensor: torch.Tensor,
    normalized_shape: Iterable[int],
    weight: torch.Tensor,
    epsilon: float = 1e-6,
    round_before_weight: bool = True,
) -> torch.Tensor:
    """Run a forward-only, one-kernel RMSNorm on contiguous CUDA tensors."""
    if not _TRITON_AVAILABLE:
        raise ValueError("adaptive RMSNorm requires Triton")
    normalized_shape = tuple(int(value) for value in normalized_shape)
    columns = math.prod(normalized_shape)
    if not input_tensor.is_cuda or weight is None:
        raise ValueError("adaptive RMSNorm requires a CUDA tensor and a weight")
    if input_tensor.numel() % columns:
        raise ValueError(f"input elements {input_tensor.numel()} are not divisible by {columns}")
    if weight.numel() != columns:
        raise ValueError(
            f"weight elements {weight.numel()} do not match normalized shape {columns}"
        )
    if input_tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported adaptive RMSNorm dtype: {input_tensor.dtype}")

    _record_hit(input_tensor, columns)
    contiguous_input = input_tensor.contiguous()
    contiguous_weight = weight.contiguous()
    output = torch.empty_like(contiguous_input)
    rows = contiguous_input.numel() // columns
    block_size = triton.next_power_of_2(columns)
    num_warps = 8 if block_size >= 4096 else 4
    _rms_norm_inference_kernel[(rows,)](
        contiguous_input,
        contiguous_weight,
        output,
        columns,
        columns=columns,
        epsilon=float(epsilon),
        block_size=block_size,
        round_before_weight=bool(round_before_weight),
        num_warps=num_warps,
    )
    return output.reshape(input_tensor.shape)


def rms_norm_aten_inference(
    input_tensor: torch.Tensor,
    normalized_shape: Iterable[int],
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Inference implementation matching fused ``aten::rms_norm`` semantics."""
    return rms_norm_inference(
        input_tensor,
        normalized_shape,
        weight,
        epsilon,
        round_before_weight=False,
    )


def add_rms_norm_inference(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    normalized_shape: Iterable[int],
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse residual addition and handwritten Llama/Qwen RMSNorm in-place.

    The disposable attention ``input_tensor`` is reused for the residual sum;
    the original ``residual`` remains untouched because speculative decoders
    may retain aliases to it. A separate normalized output is returned first.
    """
    if not _TRITON_AVAILABLE:
        raise ValueError("adaptive Add-RMSNorm requires Triton")
    normalized_shape = tuple(int(value) for value in normalized_shape)
    columns = math.prod(normalized_shape)
    if input_tensor.shape != residual.shape:
        raise ValueError(
            f"add RMSNorm input shapes must match: {input_tensor.shape} != {residual.shape}"
        )
    if not input_tensor.is_cuda or not residual.is_cuda or weight is None:
        raise ValueError("adaptive Add-RMSNorm requires CUDA tensors and a weight")
    if input_tensor.dtype != residual.dtype or input_tensor.dtype != weight.dtype:
        raise ValueError("adaptive Add-RMSNorm requires a common input/weight dtype")
    if input_tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported adaptive Add-RMSNorm dtype: {input_tensor.dtype}")
    if input_tensor.numel() % columns or weight.numel() != columns:
        raise ValueError("adaptive Add-RMSNorm normalized shape is incompatible")

    contiguous_input = input_tensor.contiguous()
    contiguous_residual = residual.contiguous()
    contiguous_weight = weight.contiguous()
    output = torch.empty_like(contiguous_input)
    rows = contiguous_input.numel() // columns
    block_size = triton.next_power_of_2(columns)
    num_warps = 8 if block_size >= 4096 else 4
    _add_rms_norm_inference_kernel[(rows,)](
        contiguous_input,
        contiguous_residual,
        contiguous_weight,
        output,
        columns,
        columns=columns,
        epsilon=float(epsilon),
        block_size=block_size,
        num_warps=num_warps,
    )
    return (
        output.reshape(input_tensor.shape),
        contiguous_input.reshape(input_tensor.shape),
    )


def is_supported_rms_norm(
    input_tensor: torch.Tensor,
    normalized_shape: Iterable[int],
    weight: torch.Tensor,
) -> bool:
    """Cheap routing predicate shared by the VLN and KERV adapters."""
    normalized_shape = tuple(int(value) for value in normalized_shape)
    columns = math.prod(normalized_shape)
    return bool(
        _TRITON_AVAILABLE
        and input_tensor.is_cuda
        and input_tensor.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and input_tensor.stride(-1) == 1
        and input_tensor.numel() % columns == 0
        and weight is not None
        and weight.is_cuda
        and weight.numel() == columns
    )


def install_llama_rms_norm_bridge(enabled: bool = True) -> int:
    """Route handwritten verifier RMSNorm modules through the fused operator."""

    if not enabled:
        for module_class, original in tuple(_ORIGINAL_RMS_FORWARDS.items()):
            module_class.forward = original
        _ORIGINAL_RMS_FORWARDS.clear()
        return 0

    classes = []
    from transformers.models.llama import modeling_llama as hf_modeling_llama

    classes.append(hf_modeling_llama.LlamaRMSNorm)
    try:
        from openvla.specdecoding.model import modeling_llama_kv
    except ImportError:
        pass
    else:
        classes.append(modeling_llama_kv.LlamaRMSNorm)

    for module_class in classes:
        if module_class in _ORIGINAL_RMS_FORWARDS:
            continue
        original = module_class.forward
        _ORIGINAL_RMS_FORWARDS[module_class] = original

        def flagos_rms_forward(self, hidden_states, _original=original):
            weight = self.weight
            if is_supported_rms_norm(hidden_states, (weight.numel(),), weight):
                return rms_norm_inference(
                    hidden_states,
                    (weight.numel(),),
                    weight,
                    float(getattr(self, "variance_epsilon", 1e-6)),
                    round_before_weight=True,
                )
            return _original(self, hidden_states)

        module_class.forward = flagos_rms_forward
    return len(classes)
