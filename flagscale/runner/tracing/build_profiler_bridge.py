# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Build the optional bridge using the official Profiler v5 SDK headers."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def build_bridge(*, profiler_include: str, output: str, cxx: str = "g++") -> Path:
    if os.name != "posix":
        raise RuntimeError("The Profiler bridge requires Linux")
    compiler = shutil.which(cxx)
    if compiler is None:
        raise RuntimeError(f"C++ compiler not found: {cxx}")
    include = Path(profiler_include).expanduser().resolve()
    for name in ("profiler.h", "profiler_v5.h", "common.h"):
        if not (include / name).is_file():
            raise ValueError(f"Official Profiler v5 SDK header missing: {include / name}")
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-fvisibility=hidden",
            "-shared",
            "-Wall",
            "-Wextra",
            str(Path(__file__).parent / "native/profiler_bridge.cpp"),
            "-I",
            str(include),
            "-o",
            str(destination),
            "-ldl",
            "-pthread",
        ],
        check=True,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiler-include", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    print(build_bridge(profiler_include=args.profiler_include, output=args.output, cxx=args.cxx))


if __name__ == "__main__":
    main()
