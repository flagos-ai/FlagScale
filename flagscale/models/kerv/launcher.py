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

"""Validated process launcher shared by the KERV train and inference entrypoints.

KERV builds on the OpenVLA training stack and the SpecVLA drafter recipe.  The
upstream projects intentionally remain separate dependencies; this module maps
a composed FlagScale configuration to their public command-line interfaces.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_SUPPORTED_LAUNCHERS = {"python", "torchrun", "deepspeed"}


def load_kerv_config(config_path: str | Path) -> DictConfig:
    """Load a composed FlagScale config and return its KERV task section."""

    config = OmegaConf.load(config_path)
    if "kerv" in config:
        kerv = config.kerv
    elif config.get("train") and "kerv" in config.train:
        kerv = config.train.kerv
    elif config.get("inference") and "kerv" in config.inference:
        kerv = config.inference.kerv
    else:
        raise ValueError("KERV configuration must contain kerv, train.kerv, or inference.kerv")
    for required in ("stage", "source_root", "entrypoint", "launcher"):
        if not kerv.get(required):
            raise ValueError(f"kerv.{required} is required")
    return kerv


def _resolve_entrypoint(source_root: str | Path, entrypoint: str | Path) -> tuple[Path, Path]:
    root = Path(source_root).expanduser().resolve()
    script = Path(entrypoint)
    if script.is_absolute():
        resolved = script.resolve()
    else:
        resolved = (root / script).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("kerv.entrypoint must be located under kerv.source_root")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"KERV entrypoint does not exist: {resolved}. "
            "Set KERV_SOURCE_ROOT to a KERV/OpenVLA checkout."
        )
    return root, resolved


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return str(value)


def _append_arguments(command: list[str], arguments: Mapping[str, Any] | None) -> None:
    for name, value in (arguments or {}).items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            raise TypeError(f"nested command argument is not supported: {name}")
        command.extend((f"--{name}", _render_value(value)))


def build_kerv_command(kerv: Mapping[str, Any]) -> tuple[list[str], Path, dict[str, str]]:
    """Build a deterministic KERV command without starting a subprocess."""

    root, script = _resolve_entrypoint(kerv["source_root"], kerv["entrypoint"])
    launcher = str(kerv["launcher"]).lower()
    if launcher not in _SUPPORTED_LAUNCHERS:
        raise ValueError(f"unsupported KERV launcher: {launcher}")

    python_executable = str(kerv.get("python_executable") or sys.executable)
    distributed = kerv.get("distributed") or {}
    if launcher == "python":
        command = [python_executable, str(script)]
    elif launcher == "torchrun":
        command = [
            python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            str(distributed.get("nnodes", 1)),
            "--nproc-per-node",
            str(distributed.get("nproc_per_node", 1)),
        ]
        if distributed.get("master_port") is not None:
            command.extend(("--master-port", str(distributed.master_port)))
        command.append(str(script))
    else:
        command = [str(kerv.get("deepspeed_executable") or "deepspeed")]
        if distributed.get("master_port") is not None:
            command.extend(("--master_port", str(distributed.master_port)))
        if distributed.get("include"):
            command.extend(("--include", str(distributed.include)))
        command.append(str(script))

    _append_arguments(command, kerv.get("arguments"))
    environment = os.environ.copy()
    environment.update(
        {str(key): _render_value(value) for key, value in (kerv.get("env") or {}).items()}
    )
    configured_python_paths = [
        str(Path(value).expanduser().resolve()) for value in kerv.get("python_paths", [])
    ]
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(root), *configured_python_paths, existing_pythonpath) if value
    )
    work_dir = Path(kerv.get("work_dir") or root).expanduser().resolve()
    return command, work_dir, environment


def launch_kerv_stage(kerv: Mapping[str, Any], *, dry_run: bool = False) -> list[str]:
    """Launch one configured KERV stage and return its argument vector."""

    command, work_dir, environment = build_kerv_command(kerv)
    print(f"[FlagScale/KERV] stage={kerv['stage']} command={shlex.join(command)}", flush=True)
    if dry_run:
        return command
    work_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=work_dir, env=environment, check=True)
    return command
