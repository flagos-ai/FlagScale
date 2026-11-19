# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Minimal safetensors reader/writer for HF checkpoints."""

from __future__ import annotations

import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from ..utils import parse_size
from .metadata import HuggingFaceIOError, SafetensorMetadataReader


class SafetensorReader(SafetensorMetadataReader):
    def get_bytes(self, name: str) -> torch.Tensor:
        """Read one-byte storage (FP8, packed FP4 or E8M0), without a dtype cast."""
        entry = self._metadata[name]
        begin, end = entry["data_offsets"]
        if end - begin != math.prod(entry["shape"]):
            raise HuggingFaceIOError(f"Expected byte storage for {name}")
        filename = self._weight_map[name]
        with (self.model_dir / filename).open("rb") as stream:
            stream.seek(self._data_starts[filename] + begin)
            payload = bytearray(stream.read(end - begin))
        if len(payload) != end - begin:
            raise HuggingFaceIOError(f"Truncated tensor: {name}")
        return torch.frombuffer(payload, dtype=torch.uint8).reshape(entry["shape"])

    def get_tensor(self, name: str) -> torch.Tensor:
        if name not in self._weight_map:
            raise HuggingFaceIOError(f"HF tensor not found: {name}")
        entry = self._metadata[name]
        dense = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
            "I32": torch.int32,
            "I64": torch.int64,
            "I8": torch.int8,
            "U8": torch.uint8,
        }
        if entry["dtype"] in dense:
            # Read dense values directly; quantized payloads/scales use get_bytes.
            dtype = dense[entry["dtype"]]
            begin, end = entry["data_offsets"]
            sizes = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "I8": 1, "U8": 1}
            if end - begin != math.prod(entry["shape"]) * sizes[entry["dtype"]]:
                raise HuggingFaceIOError(f"Invalid tensor byte count: {name}")
            if begin == end:
                return torch.empty(entry["shape"], dtype=dtype)
            filename = self._weight_map[name]
            with (self.model_dir / filename).open("rb") as stream:
                stream.seek(self._data_starts[filename] + begin)
                payload = bytearray(stream.read(end - begin))
            if len(payload) != end - begin:
                raise HuggingFaceIOError(f"Truncated tensor: {name}")
            return torch.frombuffer(payload, dtype=dtype).reshape(entry["shape"])
        raise HuggingFaceIOError(f"Expected a dense tensor, got {entry['dtype']}: {name}")


def copy_hf_assets(reference_dir: str | Path, output_dir: str | Path) -> None:
    source = Path(reference_dir).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if (
            item.name == "conversion_report.json"
            or item.name.startswith(".")
            or item.is_dir()
            or item.suffix in {".safetensors", ".bin", ".pt", ".pth"}
            or item.name.endswith(".index.json")
        ):
            continue
        destination = target / item.name
        shutil.copy2(item, destination)


@dataclass(frozen=True)
class RawEncoded:
    """A payload already in its final safetensors byte layout.

    Used for storage safetensors has no torch dtype for (E8M0 scales) or
    that must not round-trip through a dense cast (FP8 E4M3 payloads).
    `tensor` is a uint8 tensor holding the exact bytes to write and
    `dtype_code` is the safetensors header code describing those bytes.
    """

    tensor: torch.Tensor
    dtype_code: str


class ShardedSafetensorWriter:
    def __init__(self, output_dir: str | Path, max_shard_size: int | str = "10GB"):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_shard_size = (
            parse_size(max_shard_size) if isinstance(max_shard_size, str) else max_shard_size
        )
        if self.max_shard_size <= 0:
            raise HuggingFaceIOError("max_shard_size must be positive")
        self._current: dict[str, torch.Tensor | RawEncoded] = {}
        self._current_size = 0
        self._temporary_files: list[Path] = []
        self._temporary_weight_map: dict[str, Path] = {}
        self._total_size = 0

    @staticmethod
    def _tensor_size(tensor: torch.Tensor | RawEncoded) -> int:
        if isinstance(tensor, RawEncoded):
            return tensor.tensor.numel()
        return tensor.numel() * tensor.element_size()

    def add(self, name: str, tensor: torch.Tensor | RawEncoded) -> None:
        if name in self._temporary_weight_map or name in self._current:
            raise HuggingFaceIOError(f"Duplicate HF output tensor: {name}")
        # Own storage: expert gate/up and grouped projections can share a source
        # allocation, which safetensors.save_file otherwise rejects as aliases.
        if not isinstance(tensor, RawEncoded):
            tensor = tensor.detach().to("cpu").contiguous().clone()
        tensor_size = self._tensor_size(tensor)
        if self._current and self._current_size + tensor_size > self.max_shard_size:
            self._flush()
        self._current[name] = tensor
        self._current_size += tensor_size
        self._total_size += tensor_size

    def _flush(self) -> None:
        if not self._current:
            return
        temporary_path = (
            self.output_dir / f"model-part-{len(self._temporary_files) + 1:05d}.safetensors"
        )
        if any(isinstance(tensor, RawEncoded) for tensor in self._current.values()):
            self._write_raw_shard(temporary_path)
        else:
            try:
                from safetensors.torch import save_file
            except ImportError as error:
                raise HuggingFaceIOError("safetensors is required for HF conversion") from error
            save_file(self._current, temporary_path, metadata={"format": "pt"})
        self._temporary_files.append(temporary_path)
        for name in self._current:
            self._temporary_weight_map[name] = temporary_path
        self._current = {}
        self._current_size = 0

    def _write_raw_shard(self, path: Path) -> None:
        # Safetensors stores an eight-byte header length, padded JSON, then
        # contiguous payloads. No native E8M0/FP4 tensor or kernel is needed.
        codes = {
            torch.bfloat16: "BF16",
            torch.float16: "F16",
            torch.float32: "F32",
            torch.int32: "I32",
            torch.int64: "I64",
            torch.int8: "I8",
            torch.uint8: "U8",
        }
        header, offset = {"__metadata__": {"format": "pt"}}, 0
        for name, tensor in self._current.items():
            if isinstance(tensor, RawEncoded):
                if tensor.tensor.dtype != torch.uint8:
                    raise HuggingFaceIOError(
                        f"RawEncoded payload must be uint8 bytes: {name}: {tensor.tensor.dtype}"
                    )
                code, shape, size = (
                    tensor.dtype_code,
                    tuple(tensor.tensor.shape),
                    tensor.tensor.numel(),
                )
            else:
                if tensor.dtype not in codes:
                    raise HuggingFaceIOError(
                        f"Unsupported dense dtype in raw shard: {name}: {tensor.dtype}"
                    )
                code, shape, size = (
                    codes[tensor.dtype],
                    tuple(tensor.shape),
                    self._tensor_size(tensor),
                )
            header[name] = {
                "dtype": code,
                "shape": list(shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size
        encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encoded += b" " * (-len(encoded) % 8)
        with path.open("xb") as stream:
            stream.write(struct.pack("<Q", len(encoded)))
            stream.write(encoded)
            for name, tensor in self._current.items():
                if isinstance(tensor, RawEncoded):
                    data = tensor.tensor.reshape(-1)
                    for start in range(0, data.numel(), 8 * 1024**2):
                        stream.write(data[start : start + 8 * 1024**2].numpy().tobytes())
                else:
                    data = tensor.reshape(-1).view(torch.uint8)
                    for start in range(0, data.numel(), 8 * 1024**2):
                        stream.write(data[start : start + 8 * 1024**2].numpy().tobytes())

    def finish(self) -> Path:
        self._flush()
        if not self._temporary_files:
            raise HuggingFaceIOError("No tensors were written")
        count = len(self._temporary_files)
        renamed: dict[Path, Path] = {}
        for index, temporary_path in enumerate(self._temporary_files, start=1):
            final_path = self.output_dir / f"model-{index:05d}-of-{count:05d}.safetensors"
            temporary_path.replace(final_path)
            renamed[temporary_path] = final_path
        weight_map = {
            name: renamed[temporary_path].name
            for name, temporary_path in self._temporary_weight_map.items()
        }
        index_path = self.output_dir / "model.safetensors.index.json"
        payload = {
            "metadata": {"total_size": self._total_size},
            "weight_map": weight_map,
        }
        index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return index_path
