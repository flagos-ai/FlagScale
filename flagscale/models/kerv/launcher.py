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

"""Compatibility helpers for the native KERV entrypoint adapter.

FlagScale owns worker creation. KERV stages therefore run inside the current
worker instead of starting a nested Python, torchrun, or DeepSpeed process.
The command builder is retained for configuration validation and diagnostics;
``launch_kerv_stage`` delegates to the same in-process adapter used by the
public train and inference commands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flagscale.kerv_runtime import (
    KERVEntrypointError,
    build_inprocess_argv,
    load_kerv_task_config,
    run_kerv_entrypoint,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omegaconf import DictConfig


def load_kerv_config(config_path: str | Path) -> DictConfig:
    """Backward-compatible alias for :func:`load_kerv_task_config`."""

    return load_kerv_task_config(config_path)


def _resolve_entrypoint(kerv: Mapping[str, Any]) -> tuple[Path, Path]:
    root = Path(str(kerv.get("source_root") or "")).expanduser().resolve()
    if not root.is_dir():
        raise KERVEntrypointError(f"KERV source checkout does not exist: {root}")
    target = str(kerv.get("entrypoint") or "").partition(":")[0]
    script = Path(target).expanduser()
    script = script.resolve() if script.is_absolute() else (root / script).resolve()
    if not script.is_relative_to(root):
        raise KERVEntrypointError("kerv.entrypoint must be located under kerv.source_root")
    if not script.is_file():
        raise KERVEntrypointError(f"KERV entrypoint does not exist: {script}")
    return root, script


def build_kerv_command(kerv: Mapping[str, Any]) -> tuple[list[str], Path, dict[str, str]]:
    """Build a diagnostic argv for the in-process KERV stage.

    The returned vector resembles a direct Python invocation, but it is never
    executed as a child process. Distributed workers are created by the
    FlagScale runner before the stage entrypoint is called.
    """

    launcher = str(kerv.get("launcher") or "python").lower()
    if launcher != "python":
        raise KERVEntrypointError(
            f"nested launcher='{launcher}' is unsupported; configure launcher=python"
        )
    root, script = _resolve_entrypoint(kerv)
    command = [sys.executable, str(script), *build_inprocess_argv(kerv)]
    work_dir = Path(str(kerv.get("work_dir") or root)).expanduser().resolve()
    environment = os.environ.copy()
    python_paths = [str(root)] + [
        str(Path(str(value)).expanduser().resolve()) for value in kerv.get("python_paths", [])
    ]
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (*python_paths, environment.get("PYTHONPATH", "")) if value
    )
    environment.update({str(key): str(value) for key, value in (kerv.get("env") or {}).items()})
    return command, work_dir, environment


def launch_kerv_stage(kerv: Mapping[str, Any], *, dry_run: bool = False) -> Any:
    """Execute one KERV stage in the current FlagScale worker."""

    return run_kerv_entrypoint(kerv, dry_run=dry_run)


__all__ = ["build_kerv_command", "launch_kerv_stage", "load_kerv_config"]
