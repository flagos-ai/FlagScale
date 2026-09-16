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

"""Single-kernel causal mask construction for KERV verification trees."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn

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


@triton.jit(do_not_specialize=["past_length", "tree_length", "total_length"])
def _tree_causal_mask_kernel(
    tree_mask,
    output,
    tree_stride_row: tl.constexpr,
    output_stride_row: tl.constexpr,
    past_length,
    tree_start,
    tree_length,
    total_length,
    min_value: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < tree_length) & (column < total_length)
    tree_column = column - tree_start
    in_tree = column >= tree_start
    tree_valid = valid & in_tree & (tree_column < tree_length)
    allowed = (
        tl.load(
            tree_mask + row * tree_stride_row + tree_column,
            mask=tree_valid,
            other=1,
        )
        != 0
    )
    future = tree_column > row
    prefix_gap = (column >= past_length) & (column < tree_start)
    value = tl.where(prefix_gap | (in_tree & (~allowed | future)), min_value, 0.0)
    tl.store(output + row * output_stride_row + column, value, mask=valid)


def build_tree_causal_mask(
    tree_mask: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    output: torch.Tensor | None = None,
    tree_start: int | None = None,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise ValueError("tree attention mask kernel requires Triton")
    if tree_mask.device.type != "cuda" or not tree_mask.is_contiguous():
        raise ValueError("tree attention mask kernel requires a contiguous CUDA tree mask")
    tree_length = int(tree_mask.shape[-1])
    if int(tree_mask.shape[-2]) != tree_length:
        raise ValueError("tree attention mask must be square")
    tree_start = int(past_length) if tree_start is None else int(tree_start)
    if tree_start < int(past_length):
        raise ValueError("tree_start cannot precede the valid prefix")
    total_length = tree_start + tree_length
    expected_shape = (1, 1, tree_length, total_length)
    if output is None:
        output = torch.empty(
            expected_shape,
            device=tree_mask.device,
            dtype=dtype,
        )
    elif (
        tuple(output.shape) != expected_shape
        or output.device != tree_mask.device
        or output.dtype != dtype
        or not output.is_contiguous()
    ):
        raise ValueError("resident tree-mask output has an incompatible layout")
    block = 256
    _tree_causal_mask_kernel[(tree_length, triton.cdiv(total_length, block))](
        tree_mask,
        output,
        tree_stride_row=tree_mask.stride(-2),
        output_stride_row=output.stride(-2),
        past_length=int(past_length),
        tree_start=tree_start,
        tree_length=tree_length,
        total_length=total_length,
        min_value=float(torch.finfo(dtype).min),
        BLOCK=block,
    )
    return output


def enable_tree_attention_mask(
    model: nn.Module,
    enabled: bool,
    record: bool,
    record_once: bool,
    log_path: str | None,
    manifest_path: str | None,
) -> dict[str, Any]:
    """Route KERV tree-only masks through one Triton construction kernel."""
    record_path = Path(log_path) if enabled and record and log_path else None
    if enabled and record and record_path is None:
        raise ValueError("tree attention mask recording requires a log path")
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.touch(exist_ok=True)
    recorded = set()
    targets = []

    selected = bool(enabled and _TRITON_AVAILABLE)
    if selected:
        for name, module in model.named_modules():
            if type(module).__name__ != "LlamaSpecForCausalLM" or not hasattr(
                module, "_update_causal_mask"
            ):
                continue
            original = module._update_causal_mask

            def fast_update_causal_mask(
                self,
                attention_mask,
                input_tensor,
                cache_position,
                past_seen_tokens,
                _original=original,
            ):
                tree_mask = getattr(self, "tree_mask", None)
                if (
                    attention_mask is None
                    and isinstance(tree_mask, torch.Tensor)
                    and tree_mask.device.type == "cuda"
                    and tree_mask.is_contiguous()
                    and int(tree_mask.shape[-1]) == int(input_tensor.shape[1])
                ):
                    result = build_tree_causal_mask(
                        tree_mask,
                        int(past_seen_tokens),
                        input_tensor.dtype,
                    )
                    self._flagos_tree_attention_mask_hits = (
                        int(getattr(self, "_flagos_tree_attention_mask_hits", 0)) + 1
                    )
                    if record_path is not None:
                        key = (
                            int(tree_mask.shape[-1]),
                            int(past_seen_tokens),
                            str(input_tensor.dtype),
                        )
                        if not record_once or key not in recorded:
                            recorded.add(key)
                            event = {
                                "operator": "tree_causal_attention_mask",
                                "implementation": "flagos_triton_dynamic_tree_mask",
                                "tree_length": key[0],
                                "past_length": key[1],
                                "dtype": str(input_tensor.dtype).removeprefix("torch."),
                            }
                            with record_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(event) + "\n")
                    return result
                return _original(
                    attention_mask,
                    input_tensor,
                    cache_position,
                    past_seen_tokens,
                )

            module._update_causal_mask = MethodType(fast_update_causal_mask, module)
            module._flagos_tree_attention_mask_installed = True
            targets.append(name)

        if not targets:
            raise RuntimeError("tree attention mask kernel found no compatible language model")

    manifest: dict[str, Any] = {
        "enabled": selected,
        "requested": bool(enabled),
        "implementation": "flagos_triton_dynamic_tree_mask",
        "model_class": type(model).__name__,
        "targets": targets,
        "native_fallback": True,
        "inference_only": True,
        "unavailable_reason": None if _TRITON_AVAILABLE else "Triton is unavailable",
    }
    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[FlagOS/TreeAttentionMask] {'enabled' if selected else 'disabled'}; targets={targets}",
        flush=True,
    )
    return manifest
