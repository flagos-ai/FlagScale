# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Inspect HF indexes/headers and MCore checkpoint metadata without constructing models."""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path, PurePosixPath, PureWindowsPath

from .. import MappingError
from ..utils import require_world_one


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors index key: {key}")
        result[key] = value
    return result


def read_safetensors_index(path):
    """Validate weight_map without opening shards or inferring shape/precision."""
    payload = json.loads(
        Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object
    )
    if not isinstance(payload, dict):
        raise ValueError("Safetensors index must be a JSON object")
    weights = payload.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Safetensors index requires a nonempty weight_map object")
    for name, shard in weights.items():
        if not isinstance(name, str) or not name or not isinstance(shard, str) or not shard:
            raise ValueError("Safetensors weight_map must map nonempty keys to shard filenames")
        posix = PurePosixPath(shard.replace("\\", "/"))
        windows = PureWindowsPath(shard)
        if (
            posix.is_absolute()
            or windows.drive
            or windows.root
            or ".." in posix.parts
            or posix.suffix != ".safetensors"
            or "\x00" in shard
        ):
            raise ValueError(f"Invalid relative safetensors shard path: {shard!r}")
    return payload


class HuggingFaceIOError(RuntimeError):
    pass


class SafetensorMetadataReader:
    """Read metadata and file sizes only; never read tensor payloads."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise HuggingFaceIOError(f"HF model directory does not exist: {self.model_dir}")
        index_path = self.model_dir / "model.safetensors.index.json"
        self.index_path = None
        self.index_metadata = None
        if index_path.is_file():
            index = read_safetensors_index(index_path)
            self.index_path = index_path
            self.index_metadata = index.get("metadata")
            self._weight_map = dict(index["weight_map"])
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if len(files) != 1:
                raise HuggingFaceIOError(
                    "Expected model.safetensors.index.json or exactly one safetensors file in "
                    f"{self.model_dir}"
                )
            if not files[0].resolve().is_relative_to(self.model_dir):
                raise HuggingFaceIOError(f"Shard escapes HF directory: {files[0].name}")
            header, _ = self._read_header(files[0])
            self._weight_map = {key: files[0].name for key in header}
        for filename in set(self._weight_map.values()):
            if not (self.model_dir / filename).resolve().is_relative_to(self.model_dir):
                raise HuggingFaceIOError(f"Shard escapes HF directory: {filename}")
        missing_files = sorted(
            filename
            for filename in set(self._weight_map.values())
            if not (self.model_dir / filename).is_file()
        )
        if missing_files:
            raise HuggingFaceIOError(f"Missing safetensors shards: {missing_files[:20]}")
        self._metadata = {}
        self._data_starts = {}
        for filename in sorted(set(self._weight_map.values())):
            header, start = self._read_header(self.model_dir / filename)
            self._data_starts[filename] = start
            for name, entry in header.items():
                if self._weight_map.get(name) != filename:
                    raise HuggingFaceIOError(f"Unindexed or duplicate tensor: {name}")
                self._metadata[name] = entry
        missing = self.keys - self._metadata.keys()
        if missing:
            raise HuggingFaceIOError(f"Index keys absent from shards: {sorted(missing)[:20]}")

    @staticmethod
    def _read_header(path: Path) -> tuple[dict, int]:
        # Preserve storage dtype names even when installed tensor libraries do
        # not implement them. Numerical interpretation belongs to precision.py.
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise HuggingFaceIOError(f"Duplicate safetensors header key: {key}")
                result[key] = value
            return result

        with path.open("rb") as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise HuggingFaceIOError(f"Truncated safetensors header: {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            file_size = path.stat().st_size
            if length > 100_000_000 or length + 8 > file_size:
                raise HuggingFaceIOError(f"Invalid safetensors header length: {path}")
            header = json.loads(stream.read(length), object_pairs_hook=unique_object)
        if not isinstance(header, dict):
            raise HuggingFaceIOError(f"Safetensors header must be an object: {path}")
        header.pop("__metadata__", None)
        data_size = file_size - 8 - length
        spans = []
        byte_sizes = {
            "BF16": 2,
            "F16": 2,
            "F32": 4,
            "I32": 4,
            "I64": 8,
            "I8": 1,
            "U8": 1,
            "F8_E4M3": 1,
            "F8_E4M3FN": 1,
            "F8_E8M0": 1,
            "F8_E8M0FNU": 1,
        }
        for name, entry in header.items():
            if not isinstance(entry, dict):
                raise HuggingFaceIOError(f"Invalid tensor metadata: {name}")
            offsets, shape, dtype = (
                entry.get(field) for field in ("data_offsets", "shape", "dtype")
            )
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int for n in offsets)
            ):
                raise HuggingFaceIOError(f"Invalid data offsets for {name}")
            begin, end = offsets
            if not (0 <= begin <= end <= data_size):
                raise HuggingFaceIOError(f"Invalid data offsets for {name}")
            if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
                raise HuggingFaceIOError(f"Invalid shape for {name}")
            if not isinstance(dtype, str) or not dtype:
                raise HuggingFaceIOError(f"Invalid storage dtype for {name}")
            size = byte_sizes.get(dtype)
            if size is not None and end - begin != size * math.prod(shape):
                raise HuggingFaceIOError(f"Invalid tensor byte count: {name}")
            if end > begin:
                spans.append((begin, end, name))
        cursor = 0
        for begin, end, name in sorted(spans):
            if begin != cursor:
                raise HuggingFaceIOError(f"Overlapping or incomplete safetensors storage: {name}")
            cursor = end
        if cursor != data_size:
            raise HuggingFaceIOError(f"Unindexed safetensors storage: {path}")
        return header, 8 + length

    @property
    def keys(self) -> set[str]:
        return set(self._weight_map)

    @property
    def weight_map(self) -> dict[str, str]:
        return dict(self._weight_map)

    def tensor_dtypes(self) -> dict[str, str]:
        return {name: entry["dtype"].upper() for name, entry in self._metadata.items()}

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(entry["shape"]) for name, entry in self._metadata.items()}

    def tensor_storage(self, name: str) -> tuple[str, tuple[int, ...], int]:
        entry = self._metadata[name]
        begin, end = entry["data_offsets"]
        return entry["dtype"], tuple(entry["shape"]), end - begin


def resolve_iteration_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser().resolve()
    if (path / ".metadata").is_file():
        _validate_checkpoint_dir(path)
        return path
    tracker = path / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text(encoding="utf-8").strip()
        if value == "release":
            candidate = path / "release"
        else:
            candidate = path / f"iter_{int(value):07d}"
        if (candidate / ".metadata").is_file():
            _validate_checkpoint_dir(candidate)
            return candidate
        raise MappingError(f"Tracker points to an incomplete checkpoint: {candidate}")
    candidates = sorted(
        item
        for item in path.glob("iter_*")
        if (item / ".metadata").is_file() and (item / "metadata.json").is_file()
    )
    if candidates:
        _validate_checkpoint_dir(candidates[-1])
        return candidates[-1]
    raise MappingError(f"No MCore torch_dist checkpoint found under {path}")


def _validate_checkpoint_dir(path: Path) -> None:
    marker = path / "metadata.json"
    if not marker.is_file():
        raise MappingError(f"Missing MCore completion marker: {marker}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("sharded_backend") != "torch_dist":
        raise MappingError(f"Expected MCore torch_dist backend in {marker}")


def is_extra_state(name: str) -> bool:
    return re.search(r"(?:^|\.)_extra_state\d*(?:/|$)", name) is not None


def _plain(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(item) for key, item in value.items()}
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device"):
        return str(value)
    if (
        callable(value)
        and getattr(value, "__module__", None)
        and getattr(value, "__qualname__", None)
    ):
        return {"callable": value.__module__ + "." + value.__qualname__}
    raise TypeError("Runtime-only configuration value")


def json_args(args):
    """Keep all serializable saved defaults; report runtime objects explicitly."""
    values, omitted = {}, []
    for name, value in args.items():
        try:
            values[name] = _plain(value)
        except (TypeError, ValueError):
            omitted.append(name)
    return values, sorted(omitted)


def read_mcore_artifacts(path):
    require_world_one()
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    directory = resolve_iteration_dir(path)
    # Our model-only outputs have standalone common.pt; avoid importing the
    # training runtime just to read them. Embedded common state uses MCore's reader.
    if (directory / "common.pt").is_file():
        import torch

        common = torch.load(directory / "common.pt", map_location="cpu", weights_only=False)
    else:
        from megatron.core.dist_checkpointing.serialization import load_common_state_dict

        common = load_common_state_dict(str(directory))
    if not isinstance(common, dict) or "args" not in common:
        raise MappingError("Existing MCore checkpoint must contain common state args")
    saved = common["args"]
    raw_args = dict(saved) if isinstance(saved, dict) else dict(vars(saved))
    values, omitted = json_args(raw_args)
    content = common.get("content_metadata") or {}
    if not isinstance(content, dict):
        raise MappingError("MCore content_metadata must be a mapping")
    if (
        content.get("singleton_local_shards", False)
        or content.get("non_homogeneous_layers") is False
    ):
        raise MappingError("Only non-homogeneous, non-singleton DCP keys are supported")
    metadata = FileSystemReader(directory).read_metadata()
    ignored = (
        "optimizer",
        "rng_state",
        "rerun_state_machine",
        "opt_param_scheduler",
        "common_state",
        "num_floating_point_operations_so_far",
    )
    schema = {
        name: {"shape": list(entry.size), "dtype": str(entry.properties.dtype)}
        for name, entry in sorted(metadata.state_dict_metadata.items())
        if isinstance(entry, TensorStorageMetadata)
        and not is_extra_state(name)
        and not any(
            name == prefix or name.startswith(prefix + ".") or name.startswith(prefix + "/")
            for prefix in ignored
        )
    }
    if not schema:
        raise MappingError("No MCore model tensor metadata was found")
    return {
        "directory": str(directory),
        "args": values,
        "omitted_runtime_args": omitted,
        "model_tensors": schema,
        "iteration": common.get("iteration"),
    }
