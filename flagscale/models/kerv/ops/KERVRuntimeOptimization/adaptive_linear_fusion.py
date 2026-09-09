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

"""Inference-only grouped Linear fusion for small-batch transformer models."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_RECORD_PATH: Path | None = None
_RECORD_ONCE = True
_RECORDED: set[tuple[object, ...]] = set()


def configure_linear_fusion_recording(
    path: str | None,
    record: bool,
    once: bool = True,
) -> None:
    global _RECORD_PATH, _RECORD_ONCE
    _RECORD_PATH = Path(path) if record and path else None
    _RECORD_ONCE = bool(once)
    _RECORDED.clear()
    if _RECORD_PATH is not None:
        _RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _RECORD_PATH.touch(exist_ok=True)


def _parse_rows(values: str | None) -> tuple[int, ...]:
    values = "" if values is None else str(values).strip()
    if values.lower() in ("", "none", "null"):
        return ()
    rows = tuple(sorted(set(int(value.strip()) for value in values.split(",") if value.strip())))
    if any(value <= 0 for value in rows):
        raise ValueError(f"fusion row counts must be positive: {rows}")
    return rows


def _record_hit(
    kind: str,
    value: torch.Tensor,
    output_sizes: Sequence[int],
    implementation: str = "flagos_grouped_cublas_linear",
) -> None:
    if _RECORD_PATH is None:
        return
    key = (kind, tuple(value.shape), tuple(output_sizes), str(value.dtype))
    if _RECORD_ONCE and key in _RECORDED:
        return
    _RECORDED.add(key)
    record = {
        "operator": kind,
        "implementation": implementation,
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "output_sizes": list(output_sizes),
    }
    with _RECORD_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class _LinearGroup:
    """Own one packed weight and serve sequential slices from one GEMM."""

    def __init__(
        self,
        linears: Sequence[nn.Linear],
        kind: str,
        fused_rows: Sequence[int],
    ):
        if len(linears) < 2:
            raise ValueError("a fused Linear group requires at least two projections")
        first = linears[0]
        if any(layer.in_features != first.in_features for layer in linears):
            raise ValueError("all fused Linear projections must share in_features")
        if any(layer.weight.device != first.weight.device for layer in linears):
            raise ValueError("all fused Linear projections must share a device")
        if any(layer.weight.dtype != first.weight.dtype for layer in linears):
            raise ValueError("all fused Linear projections must share a dtype")

        self.kind = kind
        self.in_features = first.in_features
        self.fused_rows = frozenset(int(value) for value in fused_rows)
        self.output_sizes = tuple(layer.out_features for layer in linears)
        self.offsets = tuple(
            sum(self.output_sizes[:index]) for index in range(len(self.output_sizes))
        )
        self.weight = torch.cat([layer.weight.detach() for layer in linears], dim=0).contiguous()
        if any(layer.bias is not None for layer in linears):
            biases = [
                layer.bias.detach()
                if layer.bias is not None
                else torch.zeros(
                    layer.out_features,
                    device=layer.weight.device,
                    dtype=layer.weight.dtype,
                )
                for layer in linears
            ]
            self.bias = torch.cat(biases, dim=0).contiguous()
        else:
            self.bias = None
        self._cached_input: torch.Tensor | None = None
        self._cached_outputs: tuple[torch.Tensor, ...] | None = None
        self._cached_fused = False

    @property
    def packed_bytes(self) -> int:
        total = self.weight.numel() * self.weight.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def weight_slice(self, index: int) -> torch.Tensor:
        start = self.offsets[index]
        return self.weight.narrow(0, start, self.output_sizes[index])

    def bias_slice(self, index: int) -> torch.Tensor | None:
        if self.bias is None:
            return None
        start = self.offsets[index]
        return self.bias.narrow(0, start, self.output_sizes[index])

    def forward(self, index: int, value: torch.Tensor) -> torch.Tensor:
        if index == 0:
            self._cached_input = value
            rows = value.numel() // value.shape[-1]
            self._cached_fused = rows in self.fused_rows
            if self._cached_fused:
                _record_hit(self.kind, value, self.output_sizes)
                combined = F.linear(value, self.weight, self.bias)
                self._cached_outputs = torch.split(combined, self.output_sizes, dim=-1)
            else:
                self._cached_outputs = None
        elif self._cached_input is not value:
            raise RuntimeError(
                f"{self.kind} projections must execute sequentially on the same tensor"
            )
        if self._cached_fused:
            if self._cached_outputs is None:
                raise RuntimeError(f"{self.kind} fused output cache is missing")
            result = self._cached_outputs[index]
        else:
            result = F.linear(value, self.weight_slice(index), self.bias_slice(index))
        if index == len(self.output_sizes) - 1:
            self._cached_input = None
            self._cached_outputs = None
            self._cached_fused = False
        return result


class FusedLinearSlice(nn.Module):
    """Drop-in ``nn.Linear`` view backed by a shared packed projection."""

    def __init__(self, group: _LinearGroup, index: int):
        super().__init__()
        object.__setattr__(self, "_group", group)
        self.index = int(index)
        self.in_features = group.in_features
        self.out_features = group.output_sizes[index]

    @property
    def weight(self) -> torch.Tensor:
        return self._group.weight_slice(self.index)

    @property
    def bias(self) -> torch.Tensor | None:
        return self._group.bias_slice(self.index)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._group.forward(self.index, value)


def _compatible_linears(values: Iterable[object]) -> bool:
    linears = list(values)
    return bool(linears) and all(
        isinstance(value, nn.Linear)
        and not isinstance(value, FusedLinearSlice)
        and value.weight.device.type == "cuda"
        and not value.weight.is_meta
        for value in linears
    )


def _compatible_swiglu_module(
    module: nn.Module,
    input_sizes: Sequence[int],
) -> bool:
    required = ("gate_proj", "up_proj", "down_proj", "act_fn")
    if not all(hasattr(module, name) for name in required):
        return False
    gate = module.gate_proj
    up = module.up_proj
    down = module.down_proj
    if not all(isinstance(value, nn.Linear | FusedLinearSlice) for value in (gate, up)):
        return False
    if not isinstance(down, nn.Linear):
        return False
    if gate.in_features != up.in_features or gate.out_features != up.out_features:
        return False
    if input_sizes and gate.in_features not in input_sizes:
        return False
    config = getattr(module, "config", None)
    if config is not None and int(getattr(config, "pretraining_tp", 1)) != 1:
        return False
    activation_name = type(module.act_fn).__name__.lower()
    return "silu" in activation_name or "swish" in activation_name


def _patch_swiglu(
    module: nn.Module,
    fused_rows: Sequence[int],
    backend: str,
) -> None:
    allowed_rows = frozenset(int(value) for value in fused_rows)
    if backend == "flag_gems_out":
        import flag_gems

        def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return flag_gems.silu_and_mul_out(gate, up, gate)

        implementation = "flag_gems_silu_and_mul_out"
    elif backend == "flag_gems":
        import flag_gems

        silu_and_mul = flag_gems.silu_and_mul
        implementation = "flag_gems_silu_and_mul"
    elif backend == "native_inplace":

        def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return F.silu(gate, inplace=True).mul_(up)

        implementation = "flagos_inplace_silu_and_mul"
    elif backend == "flagos_kerv":
        from KERVRuntimeOptimization.embodied_ops import kerv_silu_mul

        silu_and_mul = kerv_silu_mul
        implementation = "flagos_embodied_kerv_silu_mul"
    else:
        raise ValueError(f"unsupported SwiGLU backend: {backend}")

    def flagos_mlp_forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(value)
        up = self.up_proj(value)
        rows = gate.numel() // gate.shape[-1]
        if rows in allowed_rows:
            _record_hit(
                "silu_and_mul",
                gate,
                (gate.shape[-1],),
                implementation=implementation,
            )
            intermediate = silu_and_mul(gate, up)
        else:
            intermediate = self.act_fn(gate) * up
        return self.down_proj(intermediate)

    module.forward = MethodType(flagos_mlp_forward, module)


def _compatible_add_rms_norm_decoder(
    module: nn.Module,
    input_sizes: Sequence[int],
) -> bool:
    required = (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "mlp",
    )
    if not all(hasattr(module, name) for name in required):
        return False
    norm = module.post_attention_layernorm
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if weight.device.type != "cuda" or weight.is_meta or weight.ndim != 1:
        return False
    return not input_sizes or weight.numel() in input_sizes


def _patch_fused_add_rms_norm(
    module: nn.Module,
    fused_rows: Sequence[int],
    fused_operator,
    implementation: str,
) -> None:
    """Fuse attention residual-add with the following RMSNorm."""
    allowed_rows = frozenset(int(value) for value in fused_rows)
    attention_signature = inspect.signature(module.self_attn.forward)
    attention_parameters = attention_signature.parameters
    accepts_extra_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in attention_parameters.values()
    )

    def flagos_decoder_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        call_arguments = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "output_attentions": output_attentions,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "position_embeddings": position_embeddings,
        }
        if accepts_extra_kwargs:
            call_arguments.update(kwargs)
        else:
            call_arguments = {
                name: value
                for name, value in call_arguments.items()
                if name in attention_parameters
            }
        attention_outputs = self.self_attn(**call_arguments)
        hidden_states, self_attn_weights, present_key_value = attention_outputs

        rows = hidden_states.numel() // hidden_states.shape[-1]
        if rows in allowed_rows:
            norm = self.post_attention_layernorm
            weight = norm.weight
            epsilon = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
            _record_hit(
                "fused_add_rms_norm",
                hidden_states,
                (weight.numel(),),
                implementation=implementation,
            )
            if implementation == "flagos_native_inplace_add_rms_norm":
                torch.add(residual, hidden_states, out=hidden_states)
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
            else:
                hidden_states, residual = fused_operator(
                    hidden_states,
                    residual,
                    (weight.numel(),),
                    weight,
                    epsilon,
                )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    module.forward = MethodType(flagos_decoder_forward, module)


def _replace_group(
    module: nn.Module,
    names: Sequence[str],
    kind: str,
    fused_rows: Sequence[int],
) -> _LinearGroup:
    linears = [getattr(module, name) for name in names]
    group = _LinearGroup(linears, kind, fused_rows)
    for index, name in enumerate(names):
        setattr(module, name, FusedLinearSlice(group, index))
    return group


def fuse_transformer_linears(
    model: nn.Module,
    qkv_rows: Sequence[int] = (),
    gate_up_rows: Sequence[int] = (),
    qkv_input_sizes: Sequence[int] = (),
    gate_up_input_sizes: Sequence[int] = (),
    swiglu_rows: Sequence[int] = (),
    swiglu_input_sizes: Sequence[int] = (),
    swiglu_backend: str = "native_inplace",
    add_rms_norm_rows: Sequence[int] = (),
    add_rms_norm_input_sizes: Sequence[int] = (),
    add_rms_norm_backend: str = "native_inplace",
) -> dict[str, object]:
    """Replace standard sequential projections after final device placement.

    Only explicitly allowlisted flattened row counts use the packed GEMM. Other
    shapes execute the original per-projection calculation through weight views.
    This keeps dynamic decoding paths correct and avoids regressions on shapes
    for which the grouped cuBLAS call is slower.
    """
    qkv_rows = tuple(sorted(set(int(value) for value in qkv_rows)))
    gate_up_rows = tuple(sorted(set(int(value) for value in gate_up_rows)))
    qkv_input_sizes = tuple(sorted(set(int(value) for value in qkv_input_sizes)))
    gate_up_input_sizes = tuple(sorted(set(int(value) for value in gate_up_input_sizes)))
    swiglu_rows = tuple(sorted(set(int(value) for value in swiglu_rows)))
    swiglu_input_sizes = tuple(sorted(set(int(value) for value in swiglu_input_sizes)))
    add_rms_norm_rows = tuple(sorted(set(int(value) for value in add_rms_norm_rows)))
    add_rms_norm_input_sizes = tuple(sorted(set(int(value) for value in add_rms_norm_input_sizes)))
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if not isinstance(module, FusedLinearSlice)
        and (
            all(hasattr(module, item) for item in ("q_proj", "k_proj", "v_proj"))
            or all(hasattr(module, item) for item in ("gate_proj", "up_proj"))
        )
    ]
    groups: list[_LinearGroup] = []
    qkv_modules: list[str] = []
    gate_up_modules: list[str] = []
    swiglu_modules: list[str] = []
    add_rms_norm_modules: list[str] = []
    for name, module in candidates:
        if qkv_rows and all(hasattr(module, item) for item in ("q_proj", "k_proj", "v_proj")):
            values = [module.q_proj, module.k_proj, module.v_proj]
            if _compatible_linears(values) and (
                not qkv_input_sizes or values[0].in_features in qkv_input_sizes
            ):
                groups.append(
                    _replace_group(
                        module,
                        ("q_proj", "k_proj", "v_proj"),
                        "qkv_linear",
                        qkv_rows,
                    )
                )
                qkv_modules.append(name)
        if gate_up_rows and all(hasattr(module, item) for item in ("gate_proj", "up_proj")):
            values = [module.gate_proj, module.up_proj]
            if _compatible_linears(values) and (
                not gate_up_input_sizes or values[0].in_features in gate_up_input_sizes
            ):
                groups.append(
                    _replace_group(
                        module,
                        ("gate_proj", "up_proj"),
                        "gate_up_linear",
                        gate_up_rows,
                    )
                )
                gate_up_modules.append(name)
        if swiglu_rows and _compatible_swiglu_module(module, swiglu_input_sizes):
            _patch_swiglu(module, swiglu_rows, swiglu_backend)
            swiglu_modules.append(name)

    if add_rms_norm_rows:
        if add_rms_norm_backend == "adaptive":
            from KERVRuntimeOptimization.adaptive_rms_norm import (
                add_rms_norm_inference as fused_add_rms_norm,
            )

            add_rms_norm_implementation = "flagos_adaptive_add_rms_norm"
        elif add_rms_norm_backend == "flag_gems":
            import flag_gems

            fused_add_rms_norm = flag_gems.fused_add_rms_norm
            add_rms_norm_implementation = "flag_gems_fused_add_rms_norm"
        elif add_rms_norm_backend == "native_inplace":
            fused_add_rms_norm = None
            add_rms_norm_implementation = "flagos_native_inplace_add_rms_norm"
        elif add_rms_norm_backend == "flagos_kerv":
            from KERVRuntimeOptimization.embodied_ops import kerv_add_rms_norm

            def fused_add_rms_norm(
                attention_output: torch.Tensor,
                residual_input: torch.Tensor,
                _shape,
                norm_weight: torch.Tensor,
                norm_eps: float,
            ):
                # The decoder needs the post-add residual for the second
                # residual connection.  ``kerv_add_rms_norm`` intentionally
                # returns only the normalized tensor and updates its first
                # argument in place, so preserve the combined residual before
                # invoking it.  This keeps the fusion numerically equivalent
                # to ``residual + attention; rms_norm(residual)``.
                combined_residual = attention_output + residual_input
                normalized = kerv_add_rms_norm(
                    attention_output,
                    residual_input,
                    norm_weight,
                    norm_eps,
                )
                return normalized, combined_residual

            add_rms_norm_implementation = "flagos_embodied_kerv_add_rms_norm"
        else:
            raise ValueError(f"unsupported Add-RMSNorm backend: {add_rms_norm_backend}")

        for name, module in model.named_modules():
            if _compatible_add_rms_norm_decoder(module, add_rms_norm_input_sizes):
                _patch_fused_add_rms_norm(
                    module,
                    add_rms_norm_rows,
                    fused_add_rms_norm,
                    add_rms_norm_implementation,
                )
                add_rms_norm_modules.append(name)

    return {
        "qkv_group_count": len(qkv_modules),
        "gate_up_group_count": len(gate_up_modules),
        "total_group_count": len(groups),
        "swiglu_module_count": len(swiglu_modules),
        "add_rms_norm_module_count": len(add_rms_norm_modules),
        "packed_weight_bytes": sum(group.packed_bytes for group in groups),
        "qkv_fused_rows": list(qkv_rows),
        "gate_up_fused_rows": list(gate_up_rows),
        "qkv_input_sizes": list(qkv_input_sizes),
        "gate_up_input_sizes": list(gate_up_input_sizes),
        "swiglu_fused_rows": list(swiglu_rows),
        "swiglu_input_sizes": list(swiglu_input_sizes),
        "swiglu_backend": str(swiglu_backend),
        "add_rms_norm_fused_rows": list(add_rms_norm_rows),
        "add_rms_norm_input_sizes": list(add_rms_norm_input_sizes),
        "add_rms_norm_backend": add_rms_norm_backend,
        "qkv_modules": qkv_modules,
        "gate_up_modules": gate_up_modules,
        "swiglu_modules": swiglu_modules,
        "add_rms_norm_modules": add_rms_norm_modules,
    }


def enable_linear_fusion(
    model: nn.Module,
    enabled: bool,
    qkv_rows: str | None,
    gate_up_rows: str | None,
    qkv_input_sizes: str | None,
    gate_up_input_sizes: str | None,
    swiglu_rows: str | None,
    swiglu_input_sizes: str | None,
    swiglu_backend: str,
    add_rms_norm_rows: str | None,
    add_rms_norm_input_sizes: str | None,
    add_rms_norm_backend: str,
    record: bool,
    record_once: bool,
    log_path: str | None,
    manifest_path: str | None,
) -> dict[str, Any]:
    """Install shape-routed inference fusion and persist an auditable manifest."""
    parsed_qkv_rows = _parse_rows(qkv_rows)
    parsed_gate_up_rows = _parse_rows(gate_up_rows)
    parsed_qkv_input_sizes = _parse_rows(qkv_input_sizes)
    parsed_gate_up_input_sizes = _parse_rows(gate_up_input_sizes)
    parsed_swiglu_rows = _parse_rows(swiglu_rows)
    parsed_swiglu_input_sizes = _parse_rows(swiglu_input_sizes)
    parsed_add_rms_norm_rows = _parse_rows(add_rms_norm_rows)
    parsed_add_rms_norm_input_sizes = _parse_rows(add_rms_norm_input_sizes)
    manifest: dict[str, Any] = {
        "enabled": bool(enabled),
        "implementation": "flagos_grouped_cublas_linear",
        "model_class": type(model).__name__,
        "torch_version": torch.__version__,
        "qkv_fused_rows": list(parsed_qkv_rows),
        "gate_up_fused_rows": list(parsed_gate_up_rows),
        "qkv_input_sizes": list(parsed_qkv_input_sizes),
        "gate_up_input_sizes": list(parsed_gate_up_input_sizes),
        "swiglu_fused_rows": list(parsed_swiglu_rows),
        "swiglu_input_sizes": list(parsed_swiglu_input_sizes),
        "swiglu_backend": str(swiglu_backend),
        "add_rms_norm_fused_rows": list(parsed_add_rms_norm_rows),
        "add_rms_norm_input_sizes": list(parsed_add_rms_norm_input_sizes),
        "add_rms_norm_backend": str(add_rms_norm_backend),
        "native_shape_fallback": True,
        "inference_only": True,
    }
    if torch.cuda.is_available():
        manifest["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())

    if enabled:
        if (
            not parsed_qkv_rows
            and not parsed_gate_up_rows
            and not parsed_swiglu_rows
            and not parsed_add_rms_norm_rows
        ):
            raise ValueError("transformer fusion is enabled but all row allowlists are empty")
        if record and not log_path:
            raise ValueError("linear fusion recording requires a log path")
        configure_linear_fusion_recording(log_path, record, record_once)
        installed = fuse_transformer_linears(
            model,
            qkv_rows=parsed_qkv_rows,
            gate_up_rows=parsed_gate_up_rows,
            qkv_input_sizes=parsed_qkv_input_sizes,
            gate_up_input_sizes=parsed_gate_up_input_sizes,
            swiglu_rows=parsed_swiglu_rows,
            swiglu_input_sizes=parsed_swiglu_input_sizes,
            swiglu_backend=str(swiglu_backend),
            add_rms_norm_rows=parsed_add_rms_norm_rows,
            add_rms_norm_input_sizes=parsed_add_rms_norm_input_sizes,
            add_rms_norm_backend=str(add_rms_norm_backend),
        )
        if (
            installed["total_group_count"] == 0
            and installed["swiglu_module_count"] == 0
            and installed["add_rms_norm_module_count"] == 0
        ):
            raise RuntimeError(
                "transformer fusion found no compatible CUDA projection or SwiGLU modules"
            )
        manifest.update(installed)
    else:
        configure_linear_fusion_recording(None, False, record_once)
        manifest.update(
            {
                "qkv_group_count": 0,
                "gate_up_group_count": 0,
                "total_group_count": 0,
                "swiglu_module_count": 0,
                "add_rms_norm_module_count": 0,
                "packed_weight_bytes": 0,
                "qkv_modules": [],
                "gate_up_modules": [],
                "swiglu_modules": [],
                "add_rms_norm_modules": [],
            }
        )

    if manifest_path:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "[FlagOS/LinearFusion] "
        f"{'enabled' if enabled else 'disabled'}; "
        f"qkv_rows={list(parsed_qkv_rows)}; gate_up_rows={list(parsed_gate_up_rows)}; "
        f"swiglu_rows={list(parsed_swiglu_rows)}; groups={manifest['total_group_count']}; "
        f"swiglu_backend={swiglu_backend}; "
        f"swiglu_modules={manifest['swiglu_module_count']}; "
        f"add_rms_norm_rows={list(parsed_add_rms_norm_rows)}; "
        f"add_rms_norm_modules={manifest['add_rms_norm_module_count']}",
        flush=True,
    )
    return manifest
