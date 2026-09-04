"""Inject full BAGEL tensors into a TP=1 Megatron-FSDP wrapped model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .expected_state import ExpectedTarget
from .materialize import iter_materialized_targets


@dataclass(frozen=True)
class InjectionReport:
    copied: tuple[str, ...]
    regenerated: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


def _unwrap_for_parameters(model: Any) -> Any:
    """Remove common Float16/DDP wrappers while retaining DTensor parameters."""

    current = model
    visited = set()
    while hasattr(current, "module") and id(current) not in visited:
        visited.add(id(current))
        candidate = current.module
        if not hasattr(candidate, "named_parameters"):
            break
        current = candidate
    return current


def _copy_full_tensor_to_parameter(name: str, parameter: Any, full_tensor: torch.Tensor) -> None:
    """Copy a TP-local full tensor into a regular parameter or FSDP DTensor."""

    with torch.no_grad():
        if hasattr(parameter, "_local_tensor") and hasattr(parameter, "megatron_fsdp_slice"):
            original = getattr(parameter, "orig_param", None)
            original_shape = tuple(original.shape) if original is not None else tuple(parameter.shape)
            if tuple(full_tensor.shape) != original_shape:
                raise ValueError(
                    f"full tensor shape mismatch for {name}: "
                    f"target pre-FSDP shape={original_shape}, source={tuple(full_tensor.shape)}"
                )
            flat_source = full_tensor.reshape(-1)
            local_source = flat_source[parameter.megatron_fsdp_slice]
            local_target = parameter._local_tensor.reshape(-1)
            if local_source.numel() != local_target.numel():
                raise ValueError(
                    f"FSDP local slice mismatch for {name}: "
                    f"slice elements={local_source.numel()}, local parameter elements={local_target.numel()}"
                )
            local_target.copy_(local_source.to(device=local_target.device, dtype=local_target.dtype))
            return

        if tuple(parameter.shape) != tuple(full_tensor.shape):
            raise ValueError(
                f"parameter shape mismatch for {name}: "
                f"target={tuple(parameter.shape)}, source={tuple(full_tensor.shape)}"
            )
        parameter.copy_(full_tensor.to(device=parameter.device, dtype=parameter.dtype))


def inject_materialized_targets(
    model: Any,
    targets: Iterable[tuple[ExpectedTarget, torch.Tensor | None]],
) -> InjectionReport:
    """Inject already-materialized tensors into a wrapped or unwrapped model."""

    parameter_owner = _unwrap_for_parameters(model)
    parameters: Mapping[str, Any] = dict(parameter_owner.named_parameters())
    copied = []
    regenerated = []
    missing = []
    for expected, tensor in targets:
        if expected.regenerated:
            regenerated.append(expected.target)
            continue
        parameter = parameters.get(expected.target)
        if parameter is None:
            missing.append(expected.target)
            continue
        if tensor is None:
            raise ValueError(f"persistent target unexpectedly materialized as None: {expected.target}")
        _copy_full_tensor_to_parameter(expected.target, parameter, tensor)
        copied.append(expected.target)
    return InjectionReport(tuple(copied), tuple(regenerated), tuple(missing))


def inject_checkpoint(model: Any, checkpoint: Path) -> InjectionReport:
    """Stream and inject the released BAGEL checkpoint."""

    report = inject_materialized_targets(model, iter_materialized_targets(checkpoint))
    if not report.ok:
        preview = ", ".join(report.missing[:20])
        raise KeyError(f"{len(report.missing)} target parameters are absent from model: {preview}")
    return report


def install_optimized_model_weights(model: Any) -> None:
    """Finalize Megatron-FSDP model buffers after local DTensor copies."""

    candidates = [model]
    current = model
    visited = set()
    while hasattr(current, "module") and id(current) not in visited:
        visited.add(id(current))
        current = current.module
        candidates.append(current)
    for candidate in candidates:
        method = getattr(candidate, "install_optimized_model_weights", None)
        if method is not None:
            method()
            return
    raise TypeError("model is not a Megatron-FSDP wrapper with install_optimized_model_weights()")
