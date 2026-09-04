"""Compare an instantiated BAGEL model state dict with the conversion manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .expected_state import build_expected_targets


@dataclass(frozen=True)
class ShapeMismatch:
    key: str
    expected: tuple[int, ...]
    actual: tuple[int, ...]


@dataclass(frozen=True)
class ModelStateAudit:
    expected_count: int
    actual_count: int
    matched_count: int
    regenerated_count: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected and not self.shape_mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "ok": self.ok,
        }


def _normalize_key(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _is_ignorable_runtime_key(key: str) -> bool:
    return key.endswith("._extra_state")


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"state-dict value has no shape: {type(value).__name__}")
    return tuple(int(dim) for dim in shape)


def audit_model_state_dict(
    checkpoint: Path,
    state_dict: Mapping[str, Any],
) -> ModelStateAudit:
    """Audit full logical keys and shapes from a model/FSDP state dict."""

    expected_items = build_expected_targets(checkpoint)
    expected = {item.target: item for item in expected_items}
    actual = {}
    duplicate_normalized_keys = []
    for raw_key, value in state_dict.items():
        key = _normalize_key(raw_key)
        if _is_ignorable_runtime_key(key):
            continue
        if key in actual:
            duplicate_normalized_keys.append(key)
        actual[key] = value
    if duplicate_normalized_keys:
        raise ValueError(f"duplicate keys after removing wrapper prefixes: {duplicate_normalized_keys}")

    regenerated = {key for key, item in expected.items() if item.regenerated}
    persistent_expected = set(expected) - regenerated
    actual_keys = set(actual)
    missing = sorted(persistent_expected - actual_keys)
    unexpected = sorted(actual_keys - set(expected))
    mismatches = []
    for key in sorted(persistent_expected & actual_keys):
        expected_shape = expected[key].shape
        assert expected_shape is not None
        actual_shape = _shape(actual[key])
        if actual_shape != expected_shape:
            mismatches.append(ShapeMismatch(key, expected_shape, actual_shape))

    matched = len(persistent_expected & actual_keys) - len(mismatches)
    return ModelStateAudit(
        expected_count=len(expected),
        actual_count=len(actual),
        matched_count=matched,
        regenerated_count=len(regenerated),
        missing=tuple(missing),
        unexpected=tuple(unexpected),
        shape_mismatches=tuple(mismatches),
    )
