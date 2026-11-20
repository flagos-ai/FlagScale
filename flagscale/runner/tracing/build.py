# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build the Linux NCCL/RCCL ``LD_PRELOAD`` probe."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _rccl_include_dirs(
    rccl_home: str | os.PathLike[str] | None, rocm_home: str | os.PathLike[str] | None
) -> list[Path]:
    roots = [
        rccl_home or os.getenv("RCCL_HOME"),
        rocm_home,
        os.getenv("ROCM_HOME"),
        os.getenv("ROCM_PATH"),
        os.getenv("HIP_PATH"),
        "/opt/rocm",
        "/opt/dtk",
    ]
    includes = []
    for root in roots:
        if root:
            base = Path(root).expanduser()
            for directory in (base / "include/rccl", base / "include", base / "hip/include"):
                if directory.is_dir() and directory not in includes:
                    includes.append(directory)
    return includes


def build_probe(
    *,
    output: str | os.PathLike[str] | None = None,
    cxx: str = "g++",
    nccl_home: str | os.PathLike[str] | None = None,
    cuda_home: str | os.PathLike[str] | None = None,
    backend: str = "auto",
    rccl_home: str | os.PathLike[str] | None = None,
    rocm_home: str | os.PathLike[str] | None = None,
) -> Path:
    """Compile the probe and return the generated shared-library path."""
    if os.name != "posix":
        raise RuntimeError("The NCCL preload probe can only be built on a POSIX/Linux host")

    compiler = shutil.which(cxx)
    if compiler is None:
        raise RuntimeError(f"C++ compiler not found: {cxx}")

    native_dir = Path(__file__).resolve().parent / "native"
    source = native_dir / "nccl_probe.cpp"
    output_path = (
        Path(output).expanduser().resolve() if output else native_dir / "libflagscale_nccl_probe.so"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend not in {"auto", "nccl", "rccl"}:
        raise ValueError(f"Unsupported communication backend: {backend}")
    nccl_root = nccl_home or os.getenv("NCCL_HOME")
    cuda_root = cuda_home or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    rccl_includes = _rccl_include_dirs(rccl_home, rocm_home)
    if backend == "auto":
        # Explicit roots win over discovery on hosts with both SDKs installed.
        if rccl_home or rocm_home:
            backend = "rccl"
        elif nccl_root or cuda_root:
            backend = "nccl"
        else:
            backend = (
                "rccl" if any((path / "rccl.h").is_file() for path in rccl_includes) else "nccl"
            )
    include_dirs: list[Path] = []
    if nccl_root:
        include_dirs.append(Path(nccl_root).expanduser() / "include")
    if cuda_root:
        include_dirs.append(Path(cuda_root).expanduser() / "include")
    if backend == "rccl":
        if not any((path / "rccl.h").is_file() for path in rccl_includes):
            raise RuntimeError("RCCL headers not found; set --rccl-home or --rocm-home")
        include_dirs = rccl_includes

    cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wall",
        "-Wextra",
        "-shared",
        str(source),
        "-o",
        str(output_path),
        "-ldl",
        "-pthread",
    ]
    if backend == "rccl":
        cmd.extend(["-DFLAGSCALE_TRACE_USE_RCCL=1", "-D__HIP_PLATFORM_AMD__=1"])
    for include_dir in include_dirs:
        cmd.extend(["-I", str(include_dir)])

    subprocess.run(cmd, check=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--cxx", default=os.getenv("CXX", "g++"))
    parser.add_argument("--nccl-home")
    parser.add_argument("--cuda-home")
    parser.add_argument("--backend", choices=("auto", "nccl", "rccl"), default="auto")
    parser.add_argument("--rccl-home")
    parser.add_argument("--rocm-home")
    args = parser.parse_args()
    output = build_probe(
        output=args.output,
        cxx=args.cxx,
        nccl_home=args.nccl_home,
        cuda_home=args.cuda_home,
        backend=args.backend,
        rccl_home=args.rccl_home,
        rocm_home=args.rocm_home,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
