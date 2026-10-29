# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Shared naming, size, region, process and output-transaction helpers; standard library only."""

from __future__ import annotations

import math
import os
import re
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from string import Formatter

from . import MappingError


def parse_size(value: str) -> int:
    units = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
    normalized = value.strip().upper()
    for unit in sorted(units, key=len, reverse=True):
        if normalized.endswith(unit):
            number = normalized[: -len(unit)].strip()
            return int(float(number) * units[unit])
    return int(normalized)


def numeric_pattern(template):
    pattern, seen = "", set()
    for literal, field, spec, conversion in Formatter().parse(template):
        pattern += re.escape(literal)
        if field is not None:
            if not field.isidentifier() or spec or conversion:
                raise MappingError(f"Only plain numeric placeholders are supported: {template}")
            pattern += f"(?P={field})" if field in seen else f"(?P<{field}>[0-9]+)"
            seen.add(field)
    return re.compile("^" + pattern + "$"), seen


def normalize_name(name, naming):
    name = name.removeprefix(naming.get("strip_prefix", ""))
    aliases = naming.get("segments", {})
    name = ".".join(aliases.get(part, part) for part in name.split("."))
    for before, after in naming.get("replace", {}).items():
        name = name.replace(before, after)
    return name


def validate_regions(key, shape, regions):
    """Check bounds, overlaps and complete volume without allocating a bitmap."""
    if not shape or any(n <= 0 for n in shape):
        raise MappingError(f"Invalid global tensor shape for {key}: {shape}")
    boxes = []
    for offset, size in regions:
        if len(offset) != len(shape) or len(size) != len(shape):
            raise MappingError(f"Invalid region dimensionality: {key}")
        if any(o < 0 or n <= 0 or o + n > total for o, n, total in zip(offset, size, shape)):
            raise MappingError(f"Out-of-bounds region: {key}: {offset}/{size} in {shape}")
        boxes.append((tuple(offset), tuple(o + n for o, n in zip(offset, size))))
    boxes.sort()
    active = []
    for start, end in boxes:
        active = [(a, b) for a, b in active if b[0] > start[0]]
        for other_start, other_end in active:
            if all(max(a, c) < min(b, d) for a, b, c, d in zip(start, end, other_start, other_end)):
                raise MappingError(f"Overlapping checkpoint regions: {key}: {start}, {other_start}")
        active.append((start, end))
    if sum(math.prod(b - a for a, b in zip(start, end)) for start, end in boxes) != math.prod(
        shape
    ):
        raise MappingError(f"Incomplete checkpoint region coverage: {key}")


def require_world_one(environ=None, distributed=None):
    env = os.environ if environ is None else environ
    sizes = (
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "SLURM_NTASKS",
    )
    ranks = ("RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID")
    for name, expected in [(key, 1) for key in sizes] + [(key, 0) for key in ranks]:
        if name in env:
            try:
                value = int(env[name])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid launcher setting {name}={env[name]!r}") from error
            if value != expected:
                raise ValueError(
                    f"Checkpoint conversion requires real world_size=1/rank=0; {name}={value}"
                )
    if distributed is None:
        torch = sys.modules.get("torch")
        distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_initialized():
        if distributed.get_world_size() != 1 or distributed.get_rank() != 0:
            raise ValueError(
                "Checkpoint conversion refuses an initialized multi-rank process group"
            )


@contextmanager
def staged_output(path, overwrite, inputs):
    output = Path(path).expanduser().resolve()
    for item in inputs:
        source = Path(item).expanduser().resolve()
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise MappingError(f"Output overlaps an input: {output} / {source}")
    if output.exists() and (not output.is_dir() or (any(output.iterdir()) and not overwrite)):
        raise MappingError(f"Output is not an empty directory: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.incomplete-{uuid.uuid4().hex}")
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        yield stage
        if output.exists():
            output.rename(backup)
        try:
            stage.rename(output)
        except OSError:
            if backup.exists():
                backup.rename(output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
