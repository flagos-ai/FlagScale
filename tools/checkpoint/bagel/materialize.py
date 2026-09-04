"""Stream released BAGEL tensors into full logical target tensors."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Iterator, Mapping, Protocol

import torch
from safetensors import safe_open

from .expected_state import ExpectedTarget, build_expected_targets
from .tensor_transforms import AttentionLayout, merge_gated_mlp, merge_qkv


LANGUAGE_ATTENTION_LAYOUT = AttentionLayout(
    hidden_size=3584,
    num_attention_heads=28,
    num_query_groups=4,
    kv_channels=128,
)
VISION_ATTENTION_LAYOUT = AttentionLayout(
    hidden_size=1152,
    num_attention_heads=16,
    num_query_groups=16,
    kv_channels=72,
)


class TensorReader(Protocol):
    def __getitem__(self, source: str) -> torch.Tensor: ...


class SafetensorsReader:
    """Read tensors from the two released files while keeping handles open."""

    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        self._stack = ExitStack()
        self._handles = []
        self._key_to_handle = {}

    def __enter__(self) -> "SafetensorsReader":
        for filename in ("ema.safetensors", "ae.safetensors"):
            path = self.checkpoint / filename
            if not path.is_file():
                raise FileNotFoundError(f"required checkpoint file does not exist: {path}")
            handle = self._stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            self._handles.append(handle)
            for key in handle.keys():
                if key in self._key_to_handle:
                    raise ValueError(f"duplicate source tensor key across checkpoint files: {key}")
                self._key_to_handle[key] = handle
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stack.close()
        self._handles.clear()
        self._key_to_handle.clear()

    def __getitem__(self, source: str) -> torch.Tensor:
        try:
            handle = self._key_to_handle[source]
        except KeyError as error:
            raise KeyError(f"source tensor does not exist: {source}") from error
        return handle.get_tensor(source)


def _projection_source(sources: tuple[str, ...], projection: str) -> str:
    markers = (f".{projection}_proj.", f".{projection}_proj_moe_gen.")
    matches = [source for source in sources if any(marker in source for marker in markers)]
    if len(matches) != 1:
        raise ValueError(f"expected one {projection} projection in {sources}, found {matches}")
    return matches[0]


def materialize_target(
    expected: ExpectedTarget,
    reader: TensorReader,
    *,
    language_layout: AttentionLayout = LANGUAGE_ATTENTION_LAYOUT,
    vision_layout: AttentionLayout = VISION_ATTENTION_LAYOUT,
) -> torch.Tensor | None:
    """Materialize one full target tensor, or return None for regeneration."""

    if expected.regenerated:
        return None
    if expected.kind == "direct":
        if len(expected.sources) != 1:
            raise ValueError(f"direct target {expected.target} has {len(expected.sources)} sources")
        tensor = reader[expected.sources[0]]
    elif expected.kind == "qkv":
        q = reader[_projection_source(expected.sources, "q")]
        k = reader[_projection_source(expected.sources, "k")]
        v = reader[_projection_source(expected.sources, "v")]
        layout = vision_layout if expected.target.startswith("vision_model.") else language_layout
        tensor = merge_qkv(q, k, v, layout)
    elif expected.kind == "gated_mlp":
        gate = reader[_projection_source(expected.sources, "gate")]
        up = reader[_projection_source(expected.sources, "up")]
        tensor = merge_gated_mlp(gate, up)
    else:
        raise ValueError(f"unsupported mapping kind for {expected.target}: {expected.kind}")

    if expected.shape is not None and tuple(tensor.shape) != expected.shape:
        raise ValueError(
            f"materialized shape mismatch for {expected.target}: "
            f"expected {expected.shape}, got {tuple(tensor.shape)}"
        )
    return tensor


def iter_materialized_targets(
    checkpoint: Path,
    *,
    language_layout: AttentionLayout = LANGUAGE_ATTENTION_LAYOUT,
    vision_layout: AttentionLayout = VISION_ATTENTION_LAYOUT,
) -> Iterator[tuple[ExpectedTarget, torch.Tensor | None]]:
    """Yield targets one at a time to bound conversion memory usage."""

    expected_targets = build_expected_targets(checkpoint)
    with SafetensorsReader(checkpoint) as reader:
        for expected in expected_targets:
            yield expected, materialize_target(
                expected,
                reader,
                language_layout=language_layout,
                vision_layout=vision_layout,
            )
