#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Validate planned unit targets before the legacy torchrun launcher."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def targets(root: Path, selection: dict, device: str) -> list[str]:
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if selection.get("source_sha") != actual or not re.fullmatch(
        r"[0-9a-f]{64}", selection.get("plan_hash", "")
    ):
        raise ValueError("Unit selection has invalid source/plan identity")
    paths = selection["targets"][device]
    if not isinstance(paths, list) or len(paths) != len(set(paths)):
        raise ValueError("Unit targets must be a unique list")
    if selection["mode"] == "full":
        if paths:
            raise ValueError("Full selection cannot override legacy unit targets")
        return []
    if selection["mode"] != "affected" or not paths:
        raise ValueError("Affected selection must contain tests")
    for path in paths:
        if not isinstance(path, str) or not re.fullmatch(
            r"tests/unit_tests/[A-Za-z0-9_./-]+\.py", path
        ):
            raise ValueError(f"Unsafe test target: {path}")
        resolved = (root / path).resolve()
        if (
            not resolved.is_relative_to((root / "tests/unit_tests").resolve())
            or not resolved.is_file()
        ):
            raise ValueError(f"Missing or escaping test target: {path}")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    print(" ".join(targets(Path.cwd(), json.loads(args.selection_file.read_text()), args.device)))
