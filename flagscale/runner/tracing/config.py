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

"""Configuration and shell integration for the CPU-side NCCL probe."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_PROBE_LIBRARY_NAME = "libflagscale_nccl_probe.so"


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tracing.{name} must be a positive number, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"tracing.{name} must be greater than zero, got {parsed}")
    return parsed


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"tracing.{name} must be a boolean, got {value!r}")


def _non_negative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tracing.{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"tracing.{name} must be a non-negative integer")
    return parsed


@dataclass(frozen=True)
class TraceLaunchConfig:
    """Resolved launch-time settings for one trace run."""

    enabled: bool
    run_id: str = ""
    trace_dir: str = ""
    probe_library: str = ""
    heartbeat_dir: str = ""
    heartbeat_timeout_s: float = 30.0
    collective_timeout_s: float = 60.0
    delayed_enter_threshold_s: float = 30.0
    p2p_timeout_s: float = 60.0
    p2p_match_window_s: float = 30.0
    missing_exit_timeout_s: float = 60.0
    failure_grace_period_s: float = 60.0
    scan_interval_s: float = 1.0
    monitor_nice: int = 10
    hardware_health_enabled: bool = False
    hardware_health_stale_after_s: float = 180.0
    inspector_enabled: bool = False
    inspector_plugin_library: str = ""
    inspector_dump_dir: str = ""
    inspector_dump_interval_us: int = 500_000
    inspector_min_size_bytes: int = 8192
    inspector_require_kernel_timing: bool = True
    inspector_correlation_window_s: float = 5.0

    @property
    def completion_file(self) -> str:
        return os.path.join(self.trace_dir, "training.exit_code")

    @property
    def monitor_pid_file(self) -> str:
        return os.path.join(self.trace_dir, "analyzer.pid")

    @property
    def report_file(self) -> str:
        return os.path.join(self.trace_dir, "findings.jsonl")

    @property
    def monitor_log_file(self) -> str:
        return os.path.join(self.trace_dir, "analyzer.log")

    def shell_setup_lines(self, node_rank: int) -> list[str]:
        """Return shell lines that enable tracing without touching the train entrypoint."""
        if not self.enabled:
            return []

        qdir = shlex.quote(self.trace_dir)
        qlib = shlex.quote(self.probe_library)
        lines = [
            "# FlagScale CPU-side NCCL tracing",
            f"mkdir -p {qdir}",
            f"if [ ! -r {qlib} ]; then",
            (
                "  echo "
                + shlex.quote(f"FlagScale NCCL probe not found or unreadable: {self.probe_library}")
                + " >&2"
            ),
            "  exit 1",
            "fi",
            "export FLAGSCALE_TRACE_ENABLE=1",
            f"export FLAGSCALE_TRACE_RUN_ID={shlex.quote(self.run_id)}",
            f"export FLAGSCALE_TRACE_DIR={qdir}",
            f'export LD_PRELOAD={qlib}"${{LD_PRELOAD:+:$LD_PRELOAD}}"',
        ]
        if self.inspector_enabled:
            inspector_library = shlex.quote(self.inspector_plugin_library)
            inspector_dir = shlex.quote(self.inspector_dump_dir)
            lines.extend(
                [
                    f"if [ ! -r {inspector_library} ]; then",
                    (
                        "  echo "
                        + shlex.quote(
                            "NCCL Inspector plugin not found or unreadable: "
                            f"{self.inspector_plugin_library}"
                        )
                        + " >&2"
                    ),
                    "  exit 1",
                    "fi",
                    f"mkdir -p {inspector_dir}",
                    f"export NCCL_PROFILER_PLUGIN={inspector_library}",
                    "export NCCL_INSPECTOR_ENABLE=1",
                    "export NCCL_INSPECTOR_DUMP_VERBOSE=1",
                    f"export NCCL_INSPECTOR_DUMP_DIR={inspector_dir}",
                    (
                        "export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS="
                        f"{self.inspector_dump_interval_us}"
                    ),
                    (f"export NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES={self.inspector_min_size_bytes}"),
                    (
                        "export NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING="
                        f"{int(self.inspector_require_kernel_timing)}"
                    ),
                ]
            )

        # With a shared trace directory, a single analyzer on node zero sees all ranks.
        if node_rank == 0:
            monitor_cmd = [
                "python",
                "-m",
                "flagscale.runner.tracing.monitor",
                "--trace-dir",
                self.trace_dir,
                "--run-id",
                self.run_id,
                "--collective-timeout",
                f"{self.collective_timeout_s:g}",
                "--delayed-enter-threshold",
                f"{self.delayed_enter_threshold_s:g}",
                "--p2p-timeout",
                f"{self.p2p_timeout_s:g}",
                "--p2p-match-window",
                f"{self.p2p_match_window_s:g}",
                "--missing-exit-timeout",
                f"{self.missing_exit_timeout_s:g}",
                "--failure-grace-period",
                f"{self.failure_grace_period_s:g}",
                "--scan-interval",
                f"{self.scan_interval_s:g}",
                "--completion-file",
                self.completion_file,
                "--report-file",
                self.report_file,
                "--nice",
                str(self.monitor_nice),
            ]
            if self.heartbeat_dir:
                monitor_cmd.extend(
                    [
                        "--heartbeat-dir",
                        self.heartbeat_dir,
                        "--heartbeat-timeout",
                        f"{self.heartbeat_timeout_s:g}",
                        "--hardware-health-stale-after",
                        f"{self.hardware_health_stale_after_s:g}",
                    ]
                )
                if self.hardware_health_enabled:
                    monitor_cmd.append("--hardware-health")
            if self.inspector_enabled:
                monitor_cmd.extend(
                    [
                        "--inspector-dir",
                        self.inspector_dump_dir,
                        "--inspector-correlation-window",
                        f"{self.inspector_correlation_window_s:g}",
                    ]
                )
            lines.extend(
                [
                    f"rm -f {shlex.quote(self.completion_file)}",
                    (
                        f"nohup {shlex.join(monitor_cmd)} "
                        f">> {shlex.quote(self.monitor_log_file)} 2>&1 &"
                    ),
                    f"echo $! > {shlex.quote(self.monitor_pid_file)}",
                ]
            )
        return lines

    def command_body(self, node_rank: int) -> str:
        """Wrap the training command so the analyzer can distinguish normal completion."""
        exit_actions = self.command_exit_actions(node_rank)
        if not exit_actions:
            return "$cmd; sync"
        # Dollar signs are escaped because this string is embedded in the outer shell's
        # double-quoted `bash -c` argument. The child shell must expand them.
        return "$cmd; rc=\\$?; " + "; ".join(exit_actions) + "; sync; exit \\$rc"

    def command_exit_actions(self, node_rank: int) -> list[str]:
        """Return analyzer completion actions for a shared diagnostic wrapper."""
        if not self.enabled or node_rank != 0:
            return []
        completion = shlex.quote(self.completion_file)
        return [f"printf '%s\\n' \\$rc > {completion}"]

    def stop_shell_lines(self, node_rank: int) -> list[str]:
        if not self.enabled or node_rank != 0:
            return []
        pid_file = shlex.quote(self.monitor_pid_file)
        return [
            f"if [ -f {pid_file} ]; then",
            f'    kill "$(cat {pid_file})" 2>/dev/null || true',
            "fi",
        ]


def _default_probe_path() -> str:
    return str(Path(__file__).resolve().parent / "native" / _PROBE_LIBRARY_NAME)


def prepare_trace_launch_config(
    config: DictConfig, run_id: str, heartbeat_config: Any | None = None
) -> TraceLaunchConfig:
    """Resolve ``experiment.runner.tracing`` without leaking it to torchrun.

    Tracing is opt-in. When disabled this function is intentionally a no-op so existing
    training launches retain their previous command and environment.
    """

    runner = config.experiment.runner
    raw = runner.get("tracing", None)
    if raw is None:
        return TraceLaunchConfig(enabled=False)

    raw_dict = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else raw
    if not isinstance(raw_dict, dict):
        raise ValueError("experiment.runner.tracing must be a mapping")

    enabled = _bool(raw_dict.get("enabled", False), "enabled")
    if not enabled:
        return TraceLaunchConfig(enabled=False)

    legacy_heartbeat_keys = {
        "heartbeat_interval_s",
        "heartbeat_timeout_s",
        "initial_heartbeat_timeout_s",
    }.intersection(raw_dict)
    if legacy_heartbeat_keys:
        moved = ", ".join(sorted(legacy_heartbeat_keys))
        raise ValueError(
            f"tracing heartbeat settings ({moved}) moved to experiment.runner.heartbeat"
        )

    if config.experiment.runner.get("no_shared_fs", False):
        raise ValueError(
            "CPU NCCL tracing currently requires a shared filesystem so node 0 can analyze "
            "events from every rank; no_shared_fs=true is not supported yet"
        )

    logging = config.train.system.logging
    trace_dir = raw_dict.get("log_dir") or os.path.join(logging.log_dir, "tracing", run_id)
    trace_dir = os.path.abspath(os.path.expanduser(str(trace_dir)))

    probe_library = raw_dict.get("probe_library") or os.getenv(
        "FLAGSCALE_NCCL_PROBE_LIBRARY", _default_probe_path()
    )
    probe_library = os.path.abspath(os.path.expanduser(str(probe_library)))

    try:
        monitor_nice = int(raw_dict.get("monitor_nice", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("tracing.monitor_nice must be an integer") from exc
    if not -20 <= monitor_nice <= 19:
        raise ValueError("tracing.monitor_nice must be between -20 and 19")

    collective_timeout_s = _positive_float(
        raw_dict.get("collective_timeout_s", 60.0), "collective_timeout_s"
    )
    p2p_timeout_s = _positive_float(
        raw_dict.get("p2p_timeout_s", collective_timeout_s), "p2p_timeout_s"
    )
    p2p_match_window_s = _positive_float(
        raw_dict.get("p2p_match_window_s", min(30.0, p2p_timeout_s)),
        "p2p_match_window_s",
    )
    missing_exit_timeout_s = _positive_float(
        raw_dict.get("missing_exit_timeout_s", collective_timeout_s),
        "missing_exit_timeout_s",
    )
    failure_grace_period_s = _positive_float(
        raw_dict.get(
            "failure_grace_period_s",
            max(
                collective_timeout_s,
                p2p_timeout_s,
                missing_exit_timeout_s,
            ),
        ),
        "failure_grace_period_s",
    )
    inspector = raw_dict.get("inspector", {})
    if not isinstance(inspector, dict):
        raise ValueError("tracing.inspector must be a mapping")
    inspector_enabled = _bool(inspector.get("enabled", False), "inspector.enabled")
    inspector_plugin_library = str(
        inspector.get("plugin_library") or os.getenv("NCCL_PROFILER_PLUGIN") or ""
    )
    if inspector_enabled and not inspector_plugin_library:
        raise ValueError("tracing.inspector.plugin_library is required when Inspector is enabled")
    if inspector_plugin_library:
        inspector_plugin_library = os.path.abspath(os.path.expanduser(inspector_plugin_library))
    inspector_dump_dir = os.path.abspath(
        os.path.expanduser(str(inspector.get("dump_dir") or os.path.join(trace_dir, "inspector")))
    )
    inspector_dump_interval_us = _non_negative_int(
        inspector.get("dump_interval_us", 500_000),
        "inspector.dump_interval_us",
    )
    inspector_min_size_bytes = _non_negative_int(
        inspector.get("min_size_bytes", 8192),
        "inspector.min_size_bytes",
    )
    inspector_require_kernel_timing = _bool(
        inspector.get("require_kernel_timing", True),
        "inspector.require_kernel_timing",
    )
    inspector_correlation_window_s = _positive_float(
        inspector.get("correlation_window_s", 5.0),
        "inspector.correlation_window_s",
    )

    resolved = TraceLaunchConfig(
        enabled=True,
        run_id=str(run_id),
        trace_dir=trace_dir,
        probe_library=probe_library,
        heartbeat_dir=(
            heartbeat_config.heartbeat_dir
            if heartbeat_config is not None and heartbeat_config.enabled
            else ""
        ),
        heartbeat_timeout_s=(
            heartbeat_config.process_timeout_s
            if heartbeat_config is not None and heartbeat_config.enabled
            else 30.0
        ),
        collective_timeout_s=collective_timeout_s,
        delayed_enter_threshold_s=_positive_float(
            raw_dict.get("delayed_enter_threshold_s", 30.0),
            "delayed_enter_threshold_s",
        ),
        p2p_timeout_s=p2p_timeout_s,
        p2p_match_window_s=p2p_match_window_s,
        missing_exit_timeout_s=missing_exit_timeout_s,
        failure_grace_period_s=failure_grace_period_s,
        scan_interval_s=_positive_float(raw_dict.get("scan_interval_s", 1.0), "scan_interval_s"),
        monitor_nice=monitor_nice,
        hardware_health_enabled=bool(
            heartbeat_config is not None
            and heartbeat_config.enabled
            and heartbeat_config.hardware_health_enabled
        ),
        hardware_health_stale_after_s=(
            heartbeat_config.hardware_health_stale_after_s
            if heartbeat_config is not None and heartbeat_config.enabled
            else 180.0
        ),
        inspector_enabled=inspector_enabled,
        inspector_plugin_library=inspector_plugin_library,
        inspector_dump_dir=inspector_dump_dir,
        inspector_dump_interval_us=inspector_dump_interval_us,
        inspector_min_size_bytes=inspector_min_size_bytes,
        inspector_require_kernel_timing=inspector_require_kernel_timing,
        inspector_correlation_window_s=inspector_correlation_window_s,
    )

    if resolved.p2p_match_window_s > resolved.p2p_timeout_s:
        raise ValueError("tracing.p2p_match_window_s must not exceed p2p_timeout_s")
    minimum_failure_grace = max(
        resolved.collective_timeout_s,
        resolved.p2p_timeout_s,
        resolved.missing_exit_timeout_s,
    )
    if resolved.failure_grace_period_s < minimum_failure_grace:
        raise ValueError(
            "tracing.failure_grace_period_s must be at least the largest enabled detection timeout"
        )

    return resolved
