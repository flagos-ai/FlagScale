"""Build the expected full target state from BAGEL safetensors metadata."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from safetensors import safe_open

from .mapping import MappingKind, build_bagel_registry


@dataclass(frozen=True)
class ExpectedTarget:
    """Expected logical target tensor before TP/FSDP sharding."""

    target: str
    kind: str
    shape: tuple[int, ...] | None
    sources: tuple[str, ...]
    regenerated: bool = False


def _source_shapes(checkpoint: Path) -> Iterable[tuple[str, tuple[int, ...]]]:
    for filename in ("ema.safetensors", "ae.safetensors"):
        path = checkpoint / filename
        if not path.is_file():
            raise FileNotFoundError(f"required checkpoint file does not exist: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for source in handle.keys():
                yield source, tuple(handle.get_slice(source).get_shape())


def build_expected_targets(checkpoint: Path) -> list[ExpectedTarget]:
    """Return every expected full target tensor and its source membership."""

    registry = build_bagel_registry()
    grouped: dict[str, list[tuple[str, tuple[int, ...], MappingKind]]] = defaultdict(list)
    for source, shape in _source_shapes(checkpoint):
        source_mapping = registry.resolve(source)
        if source_mapping is None:
            raise KeyError(f"source registry has no mapping for {source}")
        if source_mapping.target is None:
            continue
        grouped[source_mapping.target].append((source, shape, source_mapping.kind))

    expected = []
    for target, members in sorted(grouped.items()):
        kinds = {member[2] for member in members}
        if len(kinds) != 1:
            raise ValueError(f"target {target} mixes mapping kinds: {kinds}")
        kind = next(iter(kinds))
        sources = tuple(member[0] for member in members)
        shapes = tuple(member[1] for member in members)
        if kind is MappingKind.REGENERATE:
            output_shape = None
        elif kind in (MappingKind.QKV, MappingKind.GATED_MLP):
            if len({shape[1:] for shape in shapes}) != 1:
                raise ValueError(f"cannot fuse incompatible shapes for {target}: {shapes}")
            output_shape = (sum(shape[0] for shape in shapes), *shapes[0][1:])
        else:
            if len(members) != 1:
                raise ValueError(f"direct target {target} has multiple sources: {sources}")
            output_shape = shapes[0]
        expected.append(
            ExpectedTarget(
                target=target,
                kind=kind.value,
                shape=output_shape,
                sources=sources,
                regenerated=kind is MappingKind.REGENERATE,
            )
        )
    return expected
