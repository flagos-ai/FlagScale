from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

HIPPROF_RUNNER_KEYS = ("hipprof_bin_path", "hipprof_output_dir")
HIPPROF_WRAPPER_PATH = "tools/profiling/hipprof_python_wrapper.sh"
LAUNCHER_PROFILING_RUNNER_KEYS = (
    "nsys_bin_path",
    "nsys_rep_file_path",
    *HIPPROF_RUNNER_KEYS,
)


def remove_launcher_profiling_args(runner_args: dict) -> None:
    for key in LAUNCHER_PROFILING_RUNNER_KEYS:
        runner_args.pop(key, None)


def configure_hipprof_env(
    runner_config: Mapping,
    model_config: Mapping,
    current_env: Mapping | None = None,
) -> dict:
    env = dict(current_env or {})
    hipprof_bin_path = runner_config.get("hipprof_bin_path", None)
    hipprof_output_dir = runner_config.get("hipprof_output_dir", None)
    enabled = bool(model_config.get("use_hipprof_profiler", False))
    requested = enabled or hipprof_bin_path is not None or hipprof_output_dir is not None

    if not requested:
        return env
    if not enabled:
        raise ValueError(
            "hipprof runner configuration requires train.model.use_hipprof_profiler: true"
        )
    if not model_config.get("profile", False):
        raise ValueError("hipprof step profiling requires train.model.profile: true")
    if not hipprof_bin_path or not hipprof_output_dir:
        raise ValueError(
            "hipprof profiling requires both experiment.runner.hipprof_bin_path "
            "and experiment.runner.hipprof_output_dir"
        )
    if runner_config.get("nsys_bin_path", None) or runner_config.get("nsys_rep_file_path", None):
        raise ValueError("nsys and hipprof launcher profiling cannot be enabled together")

    hipprof_executable = str(hipprof_bin_path)
    if os.path.basename(hipprof_executable) != "hipprof":
        hipprof_executable = os.path.join(hipprof_executable, "hipprof")

    previous_python = env.get("PYTHON_EXEC")
    if previous_python and not env.get("HIPPROF_REAL_PYTHON"):
        env["HIPPROF_REAL_PYTHON"] = previous_python
    env["PYTHON_EXEC"] = HIPPROF_WRAPPER_PATH
    env["HIPPROF_BIN_PATH"] = hipprof_executable
    env["HIPPROF_OUTPUT_DIR"] = str(hipprof_output_dir)
    return env
