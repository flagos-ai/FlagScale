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

"""Build the Linux NCCL ``LD_PRELOAD`` probe."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def build_probe(
    *,
    output: str | os.PathLike[str] | None = None,
    cxx: str = "g++",
    nccl_home: str | os.PathLike[str] | None = None,
    cuda_home: str | os.PathLike[str] | None = None,
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

    include_dirs: list[Path] = []
    nccl_root = nccl_home or os.getenv("NCCL_HOME")
    cuda_root = cuda_home or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
    if nccl_root:
        include_dirs.append(Path(nccl_root).expanduser() / "include")
    if cuda_root:
        include_dirs.append(Path(cuda_root).expanduser() / "include")

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
    args = parser.parse_args()
    output = build_probe(
        output=args.output,
        cxx=args.cxx,
        nccl_home=args.nccl_home,
        cuda_home=args.cuda_home,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
