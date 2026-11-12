# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Compile declarative layouts into tensor nodes and global checkpoint regions."""

from __future__ import annotations

import fnmatch
from dataclasses import asdict, dataclass, field

import torch

from .. import MappingError
from ..schema.config import set_field
from ..schema.mapping import expand_rules
from ..storage.precision import DENSE_DTYPES, HEADER_DTYPES, TensorSpec
from ..utils import validate_regions
from .operators import transform


@dataclass(frozen=True)
class Piece:
    hf: str
    key: str
    shape: tuple[int, ...]
    offset: tuple[int, ...]
    global_shape: tuple[int, ...]
    source_offset: tuple[int, ...] | None = None
    source_shape: tuple[int, ...] | None = None
    axes: tuple[int, ...] | None = None


@dataclass
class Profile:
    pieces: list[Piece]
    shapes: dict[str, tuple[int, ...]]

    name: str = "mapping"
    revision: str = "hf-mcore-v1"
    extra_keys: tuple[str, ...] = ()
    physical_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def report(self):
        return {
            "profile": self.name,
            "revision": self.revision,
            "physical_contract": {"non_homogeneous_layers": True, "singleton_local_shards": False},
            "pieces": [asdict(piece) for piece in self.pieces],
        }


STORAGE_DTYPES = {**HEADER_DTYPES, "I32": torch.int32, "I64": torch.int64}
DTYPE_CODES = {str(value): key for key, value in STORAGE_DTYPES.items()}


def converted_dtype(code, requested):
    if code not in STORAGE_DTYPES:
        raise MappingError(
            f"Table mappings currently require dense storage, got {code}; "
            "decode the source encoding before applying tensor layouts"
        )
    if requested == "auto" or code in ("F32", "I32", "I64"):
        return code
    return DTYPE_CODES[str(DENSE_DTYPES[requested])]


def compile_mapping(mapping, specs, configs, *, forward, dtype="auto"):
    source_side, target_side = ("hf", "mcore") if forward else ("mcore", "hf")
    patterns = mapping.get("ignore", {}).get(source_side, [])
    if not isinstance(patterns, list) or any(not isinstance(p, str) for p in patterns):
        raise MappingError("ignore.hf / ignore.mcore must be lists of glob patterns")
    zero_checks = [
        name
        for name in specs
        if any(
            fnmatch.fnmatchcase(name, p)
            for p in mapping.get("zero_checks", {}).get(source_side, [])
        )
    ]
    ignored = {name for name in specs if any(fnmatch.fnmatchcase(name, p) for p in patterns)} | set(
        zero_checks
    )
    available = specs.keys() - ignored
    nodes, outputs, used, shape_fields = [], {}, set(), {}
    for row in expand_rules(mapping, available, configs, forward):
        names, targets = row[source_side], row[target_side]
        if row.get("generate"):
            spec = row["generate"]
            if spec.get("side", "hf") != target_side:
                continue
            if names:
                raise MappingError("Generated tensors must not declare source names")
            result = [
                torch.empty(spec["shape"], device="meta", dtype=STORAGE_DTYPES[spec["dtype"]])
                for _ in targets
            ]
            codes = []
        else:
            result = None
        missing = set(names) - available
        if missing:
            raise MappingError(f"Rule {row['id']} has missing inputs: {sorted(missing)}")
        if result is None:
            codes = [converted_dtype(specs[name].dtype, dtype) for name in names]
            inputs = [
                torch.empty(specs[name].shape, device="meta", dtype=STORAGE_DTYPES[code])
                for name, code in zip(names, codes)
            ]
            result = transform(inputs, row.get("op", "identity"), row["params"], forward)
        if row.get("export_slice") and not forward:
            if row.get("op", "identity") != "identity":
                raise MappingError("export_slice currently applies to identity rules")
            axis, size = row["export_slice"]["axis"], row["export_slice"]["size"]
            if type(size) is not int or size <= 0 or size > result[0].shape[axis]:
                raise MappingError("Invalid export_slice size")
            result = [result[0].narrow(axis, 0, size)]
        if len(result) != len(targets):
            raise MappingError(
                f"Rule {row['id']} produces {len(result)} tensors for {len(targets)} targets"
            )
        extras = row.get("extra_state", [])
        if not isinstance(extras, list) or any(not isinstance(x, str) for x in extras):
            raise MappingError("extra_state must be a list of MCore storage keys")
        node = {
            "id": row["id"],
            "inputs": names,
            "outputs": targets,
            "op": row.get("op", "identity"),
            "params": row["params"],
            "forward": forward,
            "input_dtypes": codes,
            "extra_state": extras,
            "hf": row["hf"],
            "mcore": row["mcore"],
            "hf_shapes": [list(specs[n].shape) for n in names]
            if forward
            else [list(t.shape) for t in result],
            "mcore_shapes": [list(t.shape) for t in result]
            if forward
            else [list(specs[n].shape) for n in names],
            "export_slice": row.get("export_slice"),
            "generate": row.get("generate"),
            "shape_config": row.get("shape_config", {}),
        }
        for config_field, axis in node["shape_config"].items():
            value = node["hf_shapes"][0][axis]
            if config_field in shape_fields and shape_fields[config_field] != value:
                raise MappingError(
                    f"Conflicting tensor dimensions for {config_field}: {shape_fields[config_field]} vs {value}"
                )
            shape_fields[config_field] = value
            set_field(configs, config_field, value)
        for name, tensor in zip(targets, result):
            if name in outputs:
                raise MappingError(f"Multiple rules write target {name}")
            if not tensor.shape or any(n <= 0 for n in tensor.shape):
                raise MappingError(
                    f"Target {name} needs a nonempty shape; reshape scalars explicitly"
                )
            outputs[name] = TensorSpec(name, tuple(tensor.shape), DTYPE_CODES[str(tensor.dtype)])
        nodes.append(node)
        used.update(names)
    if available - used:
        raise MappingError(
            f"Unmapped source tensors (use explicit ignore rules): {sorted(available - used)[:20]}"
        )
    if not outputs:
        raise MappingError("Mapping produced no tensors")
    return {
        "nodes": nodes,
        "outputs": outputs,
        "ignored": sorted(ignored),
        "zero_checks": zero_checks,
    }


def plan_report(plan):
    return {**plan, "outputs": {name: asdict(spec) for name, spec in plan["outputs"].items()}}


def compile_regions(plan, name="mapping"):
    pieces, shapes, physical, extras = [], {}, {}, []
    for node in plan["nodes"]:
        if node.get("generate"):
            continue
        hf, mc = node["hf"], node["mcore"]
        hs, ms = node["hf_shapes"], node["mcore_shapes"]
        shapes.update(zip(hf, map(tuple, hs)))
        extras.extend(node["extra_state"])
        op, params = node["op"], node["params"]
        if node.get("export_slice") and not node["forward"]:
            physical[mc[0]] = tuple(ms[0])
            ms = hs  # Identity only: logical DCP coverage excludes explicitly cropped rows.

        def add(i, j, shape=None, offset=None, source_offset=None, source_shape=None, axes=None):
            shape = tuple(ms[j] if shape is None else shape)
            fragment = tuple(hs[i] if source_shape is None else source_shape)
            if source_offset is not None and not any(source_offset) and fragment == tuple(hs[i]):
                source_offset = None
            pieces.append(
                Piece(
                    hf[i],
                    mc[j],
                    shape,
                    tuple(offset or [0] * len(shape)),
                    tuple(ms[j]),
                    source_offset=tuple(source_offset) if source_offset is not None else None,
                    source_shape=fragment,
                    axes=tuple(axes) if axes is not None else None,
                )
            )

        if op in ("identity", "reshape", "permute"):
            add(0, 0, axes=params.get("dims") if op == "permute" else None)
        elif op in ("concat", "swiglu"):
            axis = params.get("axis", 0) % len(ms[0])
            start = 0
            for i, shape in enumerate(hs):
                offset = [0] * len(shape)
                offset[axis] = start
                add(i, 0, shape, offset)
                start += shape[axis]
        elif op == "split":
            axis = params.get("axis", 0) % len(hs[0])
            start = 0
            for j, shape in enumerate(ms):
                offset = [0] * len(shape)
                offset[axis] = start
                add(0, j, source_offset=offset, source_shape=shape)
                start += shape[axis]
        elif op == "stack":
            axis = params.get("axis", 0) % len(ms[0])
            for i, shape in enumerate(hs):
                shape = list(shape)
                shape.insert(axis, 1)
                offset = [0] * len(shape)
                offset[axis] = i
                add(i, 0, shape, offset)
        elif op == "expert_swiglu":
            for i, shape in enumerate(hs):
                add(i, 0, (1, *shape), (i // 2, (i % 2) * shape[0], *([0] * (len(shape) - 1))))
        elif op == "packed_qkv":
            heads = params["heads"]
            dim = hs[0][0] // (3 * heads)
            tail = tuple(hs[0][1:])
            zeros = (0,) * len(tail)
            for head in range(heads):
                for projection in range(3):
                    add(
                        0,
                        0,
                        (dim, *tail),
                        ((head * 3 + projection) * dim, *zeros),
                        ((projection * heads + head) * dim, *zeros),
                        (dim, *tail),
                    )
        elif op == "qkv":
            groups = params["groups"]
            if params.get("output_gate", False):
                heads_per_group = params["heads"] // groups
                dim = hs[1][0] // groups
                tail = tuple(hs[0][1:])
                zeros = (0,) * len(tail)
                for group in range(groups):
                    base = group * (2 * heads_per_group + 2) * dim
                    for head in range(heads_per_group):
                        for gate in range(2):
                            src = ((group * heads_per_group + head) * 2 + gate) * dim
                            dst = base + (gate * heads_per_group + head) * dim
                            add(0, 0, (dim, *tail), (dst, *zeros), (src, *zeros), (dim, *tail))
                    for i in (1, 2):
                        dst = base + (2 * heads_per_group + i - 1) * dim
                        add(i, 0, (dim, *tail), (dst, *zeros), (group * dim, *zeros), (dim, *tail))
                continue
            order = params.get("order", ["q", "k", "v"])
            sizes = [shape[0] // groups for shape in hs]
            for group in range(groups):
                target_start = group * sum(sizes)
                for label in order:
                    i = ["q", "k", "v"].index(label)
                    shape = (sizes[i], *hs[i][1:])
                    add(
                        i,
                        0,
                        shape,
                        (target_start, *([0] * (len(shape) - 1))),
                        (group * sizes[i], *([0] * (len(shape) - 1))),
                        shape,
                    )
                    target_start += sizes[i]
        else:
            raise MappingError(f"No region compiler for {op}")
    profile = Profile(
        pieces, shapes, name, extra_keys=tuple(sorted(set(extras))), physical_shapes=physical
    )
    # Reverse writes must cover each HF output once, including arbitrary split axes.
    if plan["nodes"] and not plan["nodes"][0]["forward"]:
        for key, shape in shapes.items():
            validate_regions(
                key,
                shape,
                [
                    (p.source_offset or (0,) * len(shape), p.source_shape)
                    for p in pieces
                    if p.hf == key
                ],
            )
    return profile
