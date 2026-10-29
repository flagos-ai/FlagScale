# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Serial checkpoint conversion using DCP storage plans, without a resident GPTModel.

One global save plan is committed once. Write items are packed into file plans
with a target payload size, then written serially to bound per-file buffering.
The reader intersects logical regions with actual DCP chunk offsets, independent
of the TP/PP/EP rank numbers that originally wrote them.
"""

from __future__ import annotations

import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import DEFAULT_SUFFIX, _StoragePrefix, _write_item
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    ReadItem,
    SavePlan,
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

from .. import MappingError
from ..utils import parse_size, require_world_one, validate_regions
from .metadata import is_extra_state, resolve_iteration_dir
from .precision import DENSE_DTYPES, HEADER_DTYPES

MCORE_OUTPUT_OVERRIDES = dict(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=1,
    context_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    world_size=1,
    data_parallel_size=1,
    rank=0,
    local_rank=0,
    sequence_parallel=False,
    no_save_optim=True,
    no_save_rng=True,
    iteration=0,
    ckpt_format="torch_dist",
    use_dist_ckpt=True,
    use_distributed_optimizer=False,
    fp8=None,
    fp4=None,
    fp8_param_gather=False,
    fp4_param_gather=False,
    fp8_param=False,
    fp4_param=False,
)


def _groups(profile):
    groups = defaultdict(list)
    for piece in profile.pieces:
        groups[piece.key].append(piece)
    for key, pieces in groups.items():
        shape = pieces[0].global_shape
        if any(piece.global_shape != shape for piece in pieces):
            raise MappingError(f"Conflicting mapping shapes for {key}")
        validate_regions(key, shape, [(p.offset, p.shape) for p in pieces])
    return groups


def _storage_dtype(pieces, source, policy):
    types = {source.specs[p.hf].dtype for p in pieces}
    integers = {"I32": torch.int32, "I64": torch.int64}
    if types.intersection(integers):
        if len(types) != 1:
            raise MappingError("Cannot fuse different integer/floating storage dtypes")
        return integers[next(iter(types))]
    dtypes = {
        HEADER_DTYPES[code]
        if code in HEADER_DTYPES and policy.preserve_source_dtypes
        else torch.float32
        if code == "F32"
        else DENSE_DTYPES[policy.dtype]
        for code in types
    }
    if len(dtypes) != 1:
        raise MappingError(
            f"Cannot fuse different source dtypes into one DCP tensor without casting: "
            f"{pieces[0].key}: {sorted(types)}; choose an explicit --dtype if appropriate"
        )
    return next(iter(dtypes))


def _checkpoint_args(model_args, policy):
    if model_args is None:
        raise MappingError("MCore output requires explicit model args")
    values = dict(model_args)
    values.update(MCORE_OUTPUT_OVERRIDES)
    values.update(
        bf16=policy.dtype == "bf16",
        fp16=policy.dtype == "fp16",
        params_dtype=DENSE_DTYPES[policy.dtype],
    )
    return SimpleNamespace(**values)


def _piece_tensor(tensor, piece):
    if piece.source_offset is not None:
        for axis, (offset, size) in enumerate(zip(piece.source_offset, piece.source_shape)):
            tensor = tensor.narrow(axis, offset, size)
    if piece.axes is not None:
        tensor = tensor.permute(piece.axes)
    return tensor.reshape(piece.shape)


class _LazySaveData:
    def __init__(self, source, lookup, dtypes):
        self.source, self.lookup, self.dtypes = source, lookup, dtypes
        self.resolved = set()
        counts = Counter(p.hf for p in lookup.values())
        destinations = defaultdict(set)
        for piece in lookup.values():
            destinations[piece.hf].add(piece.key)
        # QKV fragments reuse each source matrix within one global target.
        # Release on its final fragment; never cache across unrelated targets.
        self._remaining = {
            name: count
            for name, count in counts.items()
            if count > 1 and len(destinations[name]) == 1
        }
        self._source_cache = {}

    def _source_tensor(self, name):
        if name not in self._remaining:
            return self.source.get_tensor(name)
        if name not in self._source_cache:
            self._source_cache[name] = self.source.get_tensor(name)
        tensor = self._source_cache[name]
        self._remaining[name] -= 1
        if self._remaining[name] == 0:
            del self._remaining[name], self._source_cache[name]
        return tensor

    def resolve_data(self, item):
        self.resolved.add(item.index)
        if item.type == WriteItemType.BYTE_IO:
            payload = io.BytesIO()
            # MCore's strategy serializes a list of ShardedObject.data values.
            # None is a supported dense TE extra-state placeholder (no history).
            torch.save([None], payload)
            payload.seek(0)
            return payload
        piece = self.lookup[item.index]
        tensor = self._source_tensor(piece.hf)
        tensor = _piece_tensor(tensor, piece).to(device="cpu", dtype=self.dtypes[piece.key])
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise MappingError(f"Non-finite or overflowing tensor: {piece.hf}")
        # Isolate the storage of mHC slices and dequantized views for torch.save.
        return tensor.contiguous().clone()


def _file_plans(items, max_shard_size):
    """Pack physical chunks, not global tensor sizes, without loading weights."""
    plans, bucket, size = [], [], 0
    for item in items:
        if item.type == WriteItemType.BYTE_IO:
            # Conservative estimate for the serialized [None] extra-state blob.
            item_size = 4096
        else:
            data = item.tensor_data
            item_size = math.prod(data.chunk.sizes) * data.properties.dtype.itemsize
        if bucket and size + item_size > max_shard_size:
            plans.append(SavePlan(bucket))
            bucket, size = [], 0
        bucket.append(item)
        size += item_size
        # A single oversized chunk stays intact in its own file. Serialization
        # overhead is additional; the limit is a payload target, not a hard cap.
        if size >= max_shard_size:
            plans.append(SavePlan(bucket))
            bucket, size = [], 0
    if bucket:
        plans.append(SavePlan(bucket))
    return plans


def _write_bucket(writer, plan, resolver):
    """Write CPU tensors serially without staging the complete model."""
    storage_plan: _StoragePrefix = plan.storage_data
    file_name = f"{storage_plan.prefix}0{DEFAULT_SUFFIX}"
    path = writer.fs.concat_path(writer.path, file_name)
    tensor_items = [item for item in plan.items if item.type != WriteItemType.BYTE_IO]
    bytes_items = [item for item in plan.items if item.type == WriteItemType.BYTE_IO]
    write_results = []
    with writer.fs.create_stream(path, "wb") as stream:
        for item in bytes_items:
            data = resolver.resolve_data(item)
            write_results.append(
                _write_item(
                    writer.transforms, stream, data, item, file_name, writer.serialization_format
                )
            )
        for item in tensor_items:
            data = resolver.resolve_data(item)
            if data.device.type != "cpu":
                data = data.to("cpu")
            write_results.append(
                _write_item(
                    writer.transforms, stream, data, item, file_name, writer.serialization_format
                )
            )
        os.fsync(stream.fileno())
    return write_results


def save_streaming_checkpoint(
    source, profile, policy, config, output_root, *, model_args, max_shard_size="4GB"
):
    require_world_one()
    max_shard_size = (
        parse_size(max_shard_size) if isinstance(max_shard_size, str) else max_shard_size
    )
    if (
        not isinstance(max_shard_size, int)
        or isinstance(max_shard_size, bool)
        or max_shard_size <= 0
    ):
        raise MappingError("max_shard_size must be a positive byte count")
    saved_args = _checkpoint_args(model_args, policy)
    root = Path(output_root)
    directory = root / "iter_0000000"
    directory.mkdir(parents=True, exist_ok=False)
    groups = _groups(profile)
    dtypes = {key: _storage_dtype(pieces, source, policy) for key, pieces in groups.items()}
    storage_metadata, items, lookup = {}, [], {}
    for key, pieces in groups.items():
        properties = TensorProperties(dtype=dtypes[key])
        chunks = []
        for i, piece in enumerate(pieces):
            chunk = ChunkStorageMetadata(torch.Size(piece.offset), torch.Size(piece.shape))
            chunks.append(chunk)
            index = MetadataIndex(key, piece.offset, i)
            items.append(
                WriteItem(
                    index,
                    WriteItemType.SHARD,
                    tensor_data=TensorWriteData(chunk, properties, torch.Size(piece.global_shape)),
                )
            )
            lookup[index] = piece
        storage_metadata[key] = TensorStorageMetadata(
            properties, torch.Size(pieces[0].global_shape), chunks
        )
    extra_keys = sorted(set(profile.extra_keys))
    for key in extra_keys:
        storage_metadata[key] = BytesStorageMetadata()
        items.append(WriteItem(MetadataIndex(key), WriteItemType.BYTE_IO))
    # No high-level save wrapper: neither a real distributed group nor an
    # async worker/pinned copy of the whole state dictionary is needed.
    writer = FileSystemWriter(
        directory,
        single_file_per_rank=True,
        thread_count=1,
        per_thread_copy_ahead=0,
        sync_files=True,
        overwrite=False,
    )
    writer.set_up_storage_writer(is_coordinator=True)
    plans = [writer.prepare_local_plan(plan) for plan in _file_plans(items, max_shard_size)]
    # Allocate every file prefix in one call. These are storage buckets, not
    # distributed ranks; separately planning each bucket would reuse __0_0.distcp.
    plans = writer.prepare_global_plan(plans)
    resolver = _LazySaveData(source, lookup, dtypes)
    results = []
    for plan in plans:
        written = _write_bucket(writer, plan, resolver)
        expected_file = {item.index for item in plan.items}
        if len(written) != len(expected_file) or {r.index for r in written} != expected_file:
            raise MappingError("DCP writer did not persist the complete file plan")
        results.extend(written)
    expected = {item.index for item in items}
    if (
        resolver.resolved != expected
        or len(results) != len(expected)
        or {r.index for r in results} != expected
    ):
        raise MappingError("DCP writer did not resolve and persist the complete global plan")
    storage_files = {r.storage_data.relative_path for r in results}
    if len(storage_files) != len(plans):
        raise MappingError("DCP file plans did not produce distinct storage files")
    source.validate_consumed()
    # Only after all tensor/scale and region checks pass is the global metadata
    # finalized. No repeated save() calls overwrite the same checkpoint.
    writer.finish(Metadata(storage_metadata), [results])
    common = {
        "args": saved_args,
        "checkpoint_version": 3.0,
        "iteration": 0,
        "content_metadata": {
            "non_homogeneous_layers": True,
            "singleton_local_shards": False,
            "checkpoint_content": "model_only",
            "model_config_source": "hf_config",
            "conversion": policy.report(),
            "profile_revision": profile.revision,
        },
    }
    torch.save(common, directory / "common.pt")
    marker = {
        "sharded_backend": "torch_dist",
        "sharded_backend_version": 1,
        "common_backend": "torch",
        "common_backend_version": 1,
    }
    (directory / "metadata.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    (root / "latest_checkpointed_iteration.txt").write_text("0", encoding="utf-8")
    return {
        "iteration_dir": str(directory),
        "stored_tensor_count": len(groups),
        "stored_chunk_count": len(lookup),
        "extra_state_count": len(extra_keys),
        "storage_file_count": len(storage_files),
        "max_shard_size_bytes": max_shard_size,
        "largest_storage_file_bytes": max(
            ((directory / name).stat().st_size for name in storage_files), default=0
        ),
    }


class _ReadDestination:
    def __init__(self, tensor):
        self.tensor = tensor

    def resolve_tensor(self, item):
        result = self.tensor
        for axis, (offset, size) in enumerate(zip(item.dest_offsets, item.lengths)):
            result = result.narrow(axis, offset, size)
        return result

    def commit_tensor(self, item, tensor):
        pass  # FileSystemReader has already copied into the destination view.


class StreamingCheckpointReader:
    def __init__(self, checkpoint_path, profile, *, ignore=()):
        require_world_one()
        self.directory = resolve_iteration_dir(checkpoint_path)
        self.reader = FileSystemReader(self.directory)
        self.metadata = self.reader.read_metadata()
        self.reader.set_up_storage_reader(self.metadata, is_coordinator=True)
        self.profile, self.used = profile, set()
        self.ignore = set(ignore)
        self.by_hf = defaultdict(list)
        self.groups = _groups(profile)
        for piece in profile.pieces:
            self.by_hf[piece.hf].append(piece)
        if (self.directory / "preserved_quantization.json").is_file():
            raise MappingError(
                "This checkpoint uses removed preserved-payload adapters; convert from the original HF source"
            )
        self._validate_metadata()
        self.tensor_dtypes()  # Reject implicit promotion when joining MCore pieces.

    def tensor_dtypes(self):
        dtypes = {}
        for name, pieces in self.by_hf.items():
            types = {
                self.metadata.state_dict_metadata[piece.key].properties.dtype for piece in pieces
            }
            if len(types) != 1:
                raise MappingError(
                    f"Inconsistent source DCP dtypes for {name}: {sorted(map(str, types))}"
                )
            dtypes[name] = next(iter(types))
        return dtypes

    def _validate_metadata(self):
        ignored = (
            "optimizer",
            "rng_state",
            "rerun_state_machine",
            "opt_param_scheduler",
            "common_state",
            "num_floating_point_operations_so_far",
        )

        def _is_ignored(key: str) -> bool:
            if any(
                key == prefix or key.startswith(prefix + ".") or key.startswith(prefix + "/")
                for prefix in ignored
            ):
                return True
            # Megatron training saves persist trainer rerun-state bytes as
            # rerun_state_machine_state/shard_<r>_<world> — trainer metadata,
            # not model tensors. Accept it like the other ignored prefixes.
            return key.startswith("rerun_state_machine_state/")

        unknown = []
        for key, entry in self.metadata.state_dict_metadata.items():
            if key not in self.groups:
                if key in self.ignore:
                    continue
                if not is_extra_state(key) and not _is_ignored(key):
                    unknown.append(key)
                continue
            if not isinstance(entry, TensorStorageMetadata):
                raise MappingError(f"Expected DCP tensor: {key}")
        if unknown or self.groups.keys() - self.metadata.state_dict_metadata.keys():
            raise MappingError(
                f"DCP schema differs from mapping: unknown={unknown[:20]}, "
                f"missing={sorted(self.groups.keys() - self.metadata.state_dict_metadata.keys())[:20]}"
            )
        file_sizes = {}
        for key, pieces in self.groups.items():
            entry = self.metadata.state_dict_metadata[key]
            expected, actual = pieces[0].global_shape, tuple(entry.size)
            if actual != self.profile.physical_shapes.get(key, expected):
                raise MappingError(f"DCP global shape mismatch: {key}: {actual} != {expected}")
            dtype = entry.properties.dtype
            if dtype not in (
                torch.float32,
                torch.float16,
                torch.bfloat16,
                torch.int32,
                torch.int64,
            ):
                raise MappingError(
                    f"Unsupported raw DCP precision {dtype}: {key}; "
                    "this adapter requires standard dense model tensors"
                )
            validate_regions(
                key, actual, [(tuple(c.offsets), tuple(c.sizes)) for c in entry.chunks]
            )
            for i, chunk in enumerate(entry.chunks):
                index = MetadataIndex(key, chunk.offsets, i)
                info = self.metadata.storage_data.get(index)
                if info is None:
                    raise MappingError(f"Missing DCP storage entry: {index}")
                path = (self.directory / info.relative_path).resolve()
                if path not in file_sizes:
                    if not path.is_relative_to(self.directory.resolve()) or not path.is_file():
                        raise MappingError(f"Missing or external DCP shard: {info.relative_path}")
                    file_sizes[path] = path.stat().st_size
                if (
                    info.offset < 0
                    or info.length <= 0
                    or info.offset + info.length > file_sizes[path]
                ):
                    raise MappingError(f"Invalid DCP byte range: {index}")

    def _read_piece(self, piece):
        entry = self.metadata.state_dict_metadata[piece.key]
        tensor = torch.empty(piece.shape, dtype=entry.properties.dtype, device="cpu")
        requests, covered = [], 0
        for i, chunk in enumerate(entry.chunks):
            start = tuple(max(a, b) for a, b in zip(piece.offset, chunk.offsets))
            end = tuple(
                min(a + n, b + m)
                for a, n, b, m in zip(piece.offset, piece.shape, chunk.offsets, chunk.sizes)
            )
            lengths = tuple(b - a for a, b in zip(start, end))
            if any(n <= 0 for n in lengths):
                continue
            covered += math.prod(lengths)
            requests.append(
                ReadItem(
                    type=LoadItemType.TENSOR,
                    dest_index=MetadataIndex(piece.hf),
                    dest_offsets=torch.Size(a - b for a, b in zip(start, piece.offset)),
                    storage_index=MetadataIndex(piece.key, chunk.offsets, i),
                    storage_offsets=torch.Size(a - b for a, b in zip(start, chunk.offsets)),
                    lengths=torch.Size(lengths),
                )
            )
        if covered != math.prod(piece.shape):
            raise MappingError(f"Incomplete read coverage for {piece.hf}")
        plan = self.reader.prepare_local_plan(LoadPlan(requests))
        plan = self.reader.prepare_global_plan([plan])[0]
        self.reader.read_data(plan, _ReadDestination(tensor)).wait()
        return tensor

    def tensors(self):
        dtypes = self.tensor_dtypes()
        for name, pieces in self.by_hf.items():
            first = self._read_piece(pieces[0])
            shape = self.profile.shapes[name]
            tensor = torch.empty(shape, dtype=dtypes[name], device="cpu")
            for index, piece in enumerate(pieces):
                value = first if index == 0 else self._read_piece(piece)
                if not isinstance(value, torch.Tensor):
                    raise MappingError(f"Mixed raw/dense fragments for {name}")
                fragment = piece.source_shape or shape
                if piece.axes is not None:
                    value = value.reshape([fragment[i] for i in piece.axes])
                    value = value.permute([piece.axes.index(i) for i in range(len(piece.axes))])
                else:
                    value = value.reshape(fragment)
                target = tensor
                offset = piece.source_offset or (0,) * len(shape)
                for axis, (start, size) in enumerate(zip(offset, fragment)):
                    target = target.narrow(axis, start, size)
                target.copy_(value)
            self.used.add(name)
            yield name, tensor
            del tensor, first
        if self.used != set(self.profile.shapes):
            raise MappingError("Incomplete HF export coverage")
