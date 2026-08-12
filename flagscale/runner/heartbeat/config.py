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

"""Launch integration for training-progress-driven GPU heartbeat monitoring."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"heartbeat.{name} must be a positive number, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"heartbeat.{name} must be greater than zero, got {parsed}")
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
    raise ValueError(f"heartbeat.{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class HeartbeatLaunchConfig:
    """Resolved settings for one GPU-progress heartbeat run."""

    enabled: bool
    run_id: str = ""
    heartbeat_dir: str = ""
    publish_interval_s: float = 5.0
    initial_process_timeout_s: float = 30.0
    process_timeout_s: float = 30.0
    initial_progress_timeout_s: float = 600.0
    progress_timeout_s: float = 300.0
    checkpoint_timeout_s: float = 1800.0
    failure_grace_period_s: float = 30.0
    scan_interval_s: float = 1.0
    monitor_nice: int = 10
    expected_world_size: int = 0
    expected_node_count: int = 0
    hardware_health_enabled: bool = False
    hardware_health_interval_s: float = 60.0
    hardware_health_command_timeout_s: float = 10.0
    hardware_health_stale_after_s: float = 180.0

    @property
    def completion_file(self) -> str:
        return os.path.join(self.heartbeat_dir, "training.exit_code")

    @property
    def monitor_pid_file(self) -> str:
        return os.path.join(self.heartbeat_dir, "monitor.pid")

    @property
    def report_file(self) -> str:
        return os.path.join(self.heartbeat_dir, "findings.jsonl")

    @property
    def status_file(self) -> str:
        return os.path.join(self.heartbeat_dir, "status.json")

    @property
    def monitor_log_file(self) -> str:
        return os.path.join(self.heartbeat_dir, "monitor.log")

    def hardware_health_file(self, node_rank: int) -> str:
        return os.path.join(self.heartbeat_dir, f"gpu_health_node_{node_rank}.json")

    def hardware_health_pid_file(self, node_rank: int) -> str:
        return os.path.join(self.heartbeat_dir, f"gpu_health_node_{node_rank}.pid")

    def hardware_health_log_file(self, node_rank: int) -> str:
        return os.path.join(self.heartbeat_dir, f"gpu_health_node_{node_rank}.log")

    def training_command_body(self, node_rank: int) -> str:
        """Run training and notify the node-zero heartbeat monitor on exit."""
        if not self.enabled:
            return "$cmd; sync"
        exit_actions: list[str] = []
        if node_rank == 0:
            completion_file = shlex.quote(self.completion_file)
            exit_actions.append(f"printf '%s\\n' \\$rc > {completion_file}")
        if self.hardware_health_enabled:
            health_pid_file = shlex.quote(self.hardware_health_pid_file(node_rank))
            exit_actions.append(
                f"if [ -f {health_pid_file} ]; then "
                f'kill \\"\\$(cat {health_pid_file})\\" 2>/dev/null || true; fi'
            )
        if not exit_actions:
            return "$cmd; sync"
        return "$cmd; rc=\\$?; " + "; ".join(exit_actions) + "; sync; exit \\$rc"

    def shell_setup_lines(self, node_rank: int) -> list[str]:
        if not self.enabled:
            return []

        qdir = shlex.quote(self.heartbeat_dir)
        lines = [
            "# FlagScale GPU progress heartbeat",
            f"mkdir -p {qdir}",
            "export FLAGSCALE_GPU_HEARTBEAT_ENABLE=1",
            f"export FLAGSCALE_GPU_HEARTBEAT_RUN_ID={shlex.quote(self.run_id)}",
            f"export FLAGSCALE_GPU_HEARTBEAT_DIR={qdir}",
            (f"export FLAGSCALE_GPU_HEARTBEAT_INTERVAL_SEC={self.publish_interval_s:g}"),
        ]
        if self.hardware_health_enabled:
            health_cmd = [
                "python",
                "-m",
                "flagscale.runner.heartbeat.gpu_health",
                "--output-file",
                self.hardware_health_file(node_rank),
                "--run-id",
                self.run_id,
                "--node-rank",
                str(node_rank),
                "--interval",
                f"{self.hardware_health_interval_s:g}",
                "--command-timeout",
                f"{self.hardware_health_command_timeout_s:g}",
                "--nice",
                str(self.monitor_nice),
            ]
            lines.extend(
                [
                    "# FlagScale low-frequency GPU hardware health",
                    f"rm -f {shlex.quote(self.hardware_health_file(node_rank))}",
                    f"nohup {shlex.join(health_cmd)} "
                    f">> {shlex.quote(self.hardware_health_log_file(node_rank))} 2>&1 &",
                    f"echo $! > {shlex.quote(self.hardware_health_pid_file(node_rank))}",
                ]
            )
        if node_rank == 0:
            monitor_cmd = [
                "python",
                "-m",
                "flagscale.runner.heartbeat.monitor",
                "--heartbeat-dir",
                self.heartbeat_dir,
                "--run-id",
                self.run_id,
                "--expected-world-size",
                str(self.expected_world_size),
                "--initial-process-timeout",
                f"{self.initial_process_timeout_s:g}",
                "--process-timeout",
                f"{self.process_timeout_s:g}",
                "--initial-progress-timeout",
                f"{self.initial_progress_timeout_s:g}",
                "--progress-timeout",
                f"{self.progress_timeout_s:g}",
                "--checkpoint-timeout",
                f"{self.checkpoint_timeout_s:g}",
                "--failure-grace-period",
                f"{self.failure_grace_period_s:g}",
                "--scan-interval",
                f"{self.scan_interval_s:g}",
                "--completion-file",
                self.completion_file,
                "--report-file",
                self.report_file,
                "--status-file",
                self.status_file,
                "--nice",
                str(self.monitor_nice),
            ]
            if self.hardware_health_enabled:
                monitor_cmd.extend(
                    [
                        "--hardware-health-enabled",
                        "--expected-node-count",
                        str(self.expected_node_count),
                        "--hardware-health-stale-after",
                        f"{self.hardware_health_stale_after_s:g}",
                    ]
                )
            lines.extend(
                [
                    f"rm -f {shlex.quote(self.completion_file)}",
                    f"nohup {shlex.join(monitor_cmd)} "
                    f">> {shlex.quote(self.monitor_log_file)} 2>&1 &",
                    f"echo $! > {shlex.quote(self.monitor_pid_file)}",
                ]
            )
        return lines

    def stop_shell_lines(self, node_rank: int) -> list[str]:
        if not self.enabled:
            return []
        lines: list[str] = []
        if self.hardware_health_enabled:
            health_pid_file = shlex.quote(self.hardware_health_pid_file(node_rank))
            lines.extend(
                [
                    f"if [ -f {health_pid_file} ]; then",
                    f'    kill "$(cat {health_pid_file})" 2>/dev/null || true',
                    "fi",
                ]
            )
        if node_rank == 0:
            pid_file = shlex.quote(self.monitor_pid_file)
            lines.extend(
                [
                    f"if [ -f {pid_file} ]; then",
                    f'    kill "$(cat {pid_file})" 2>/dev/null || true',
                    "fi",
                ]
            )
        return lines


def _infer_world_size(runner: Any) -> int:
    try:
        nnodes = int(runner.get("nnodes", 1))
        nproc_per_node = int(runner.get("nproc_per_node", 0))
    except (TypeError, ValueError):
        return 0
    return nnodes * nproc_per_node if nnodes > 0 and nproc_per_node > 0 else 0


def _infer_node_count(runner: Any) -> int:
    try:
        nnodes = int(runner.get("nnodes", 1))
    except (TypeError, ValueError):
        return 0
    return nnodes if nnodes > 0 else 0


def prepare_heartbeat_launch_config(config: DictConfig, run_id: str) -> HeartbeatLaunchConfig:
    """Resolve ``experiment.runner.heartbeat`` without leaking it to torchrun."""

    raw = config.experiment.runner.get("heartbeat", None)
    if raw is None:
        return HeartbeatLaunchConfig(enabled=False)
    raw_dict = OmegaConf.to_container(raw, resolve=True) if isinstance(raw, DictConfig) else raw
    if not isinstance(raw_dict, dict):
        raise ValueError("experiment.runner.heartbeat must be a mapping")
    if not _bool(raw_dict.get("enabled", False), "enabled"):
        return HeartbeatLaunchConfig(enabled=False)
    runner_type = str(config.experiment.runner.get("type", "ssh")).lower()
    if runner_type == "cloud":
        raise NotImplementedError("GPU progress heartbeat is not supported by CloudTrainRunner")
    if config.experiment.runner.get("no_shared_fs", False):
        raise ValueError(
            "GPU progress heartbeat currently requires a shared filesystem for cross-node "
            "monitoring; no_shared_fs=true is not supported yet"
        )
    task = config.experiment.get("task", {})
    backend = task.get("backend", "megatron") if hasattr(task, "get") else "megatron"
    if str(backend).lower() != "megatron":
        raise ValueError(
            "GPU progress heartbeat currently supports the Megatron backend because it relies "
            "on the common Megatron training-step hooks"
        )
    logging = config.train.system.logging
    heartbeat_dir = raw_dict.get("log_dir") or os.path.join(
        logging.log_dir, "heartbeat", str(run_id)
    )
    heartbeat_dir = os.path.abspath(os.path.expanduser(str(heartbeat_dir)))

    publish_interval_s = _positive_float(
        raw_dict.get("publish_interval_s", raw_dict.get("interval_s", 5.0)),
        "publish_interval_s",
    )
    initial_process_timeout_s = _positive_float(
        raw_dict.get("initial_process_timeout_s", 30.0), "initial_process_timeout_s"
    )
    process_timeout_s = _positive_float(
        raw_dict.get("process_timeout_s", 30.0), "process_timeout_s"
    )
    initial_progress_timeout_s = _positive_float(
        raw_dict.get("initial_progress_timeout_s", 600.0),
        "initial_progress_timeout_s",
    )
    progress_timeout_s = _positive_float(
        raw_dict.get("progress_timeout_s", 300.0), "progress_timeout_s"
    )
    checkpoint_timeout_s = _positive_float(
        raw_dict.get("checkpoint_timeout_s", 1800.0), "checkpoint_timeout_s"
    )
    failure_grace_period_s = _positive_float(
        raw_dict.get(
            "failure_grace_period_s",
            max(initial_process_timeout_s, process_timeout_s),
        ),
        "failure_grace_period_s",
    )

    for name, timeout in (
        ("initial_process_timeout_s", initial_process_timeout_s),
        ("process_timeout_s", process_timeout_s),
        ("initial_progress_timeout_s", initial_progress_timeout_s),
        ("progress_timeout_s", progress_timeout_s),
        ("checkpoint_timeout_s", checkpoint_timeout_s),
    ):
        if timeout <= publish_interval_s:
            raise ValueError(f"heartbeat.{name} must be greater than publish_interval_s")
    if checkpoint_timeout_s < progress_timeout_s:
        raise ValueError(
            "heartbeat.checkpoint_timeout_s must be greater than or equal to progress_timeout_s"
        )
    if failure_grace_period_s < max(initial_process_timeout_s, process_timeout_s):
        raise ValueError(
            "heartbeat.failure_grace_period_s must cover both process heartbeat timeouts"
        )

    try:
        monitor_nice = int(raw_dict.get("monitor_nice", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("heartbeat.monitor_nice must be an integer") from exc
    if not -20 <= monitor_nice <= 19:
        raise ValueError("heartbeat.monitor_nice must be between -20 and 19")
    try:
        expected_world_size = int(
            raw_dict.get("expected_world_size", _infer_world_size(config.experiment.runner))
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("heartbeat.expected_world_size must be a non-negative integer") from exc
    if expected_world_size < 0:
        raise ValueError("heartbeat.expected_world_size must be a non-negative integer")
    expected_node_count = _infer_node_count(config.experiment.runner)

    hardware_health = raw_dict.get("hardware_health", {})
    if not isinstance(hardware_health, dict):
        raise ValueError("heartbeat.hardware_health must be a mapping")
    hardware_health_enabled = _bool(
        hardware_health.get("enabled", False), "hardware_health.enabled"
    )
    hardware_health_interval_s = _positive_float(
        hardware_health.get("interval_s", 60.0), "hardware_health.interval_s"
    )
    hardware_health_command_timeout_s = _positive_float(
        hardware_health.get("command_timeout_s", 10.0),
        "hardware_health.command_timeout_s",
    )
    hardware_health_stale_after_s = _positive_float(
        hardware_health.get("stale_after_s", 3 * hardware_health_interval_s),
        "hardware_health.stale_after_s",
    )
    if hardware_health_command_timeout_s >= hardware_health_interval_s:
        raise ValueError("heartbeat.hardware_health.command_timeout_s must be less than interval_s")
    if hardware_health_stale_after_s <= hardware_health_interval_s:
        raise ValueError("heartbeat.hardware_health.stale_after_s must be greater than interval_s")

    return HeartbeatLaunchConfig(
        enabled=True,
        run_id=str(run_id),
        heartbeat_dir=heartbeat_dir,
        publish_interval_s=publish_interval_s,
        initial_process_timeout_s=initial_process_timeout_s,
        process_timeout_s=process_timeout_s,
        initial_progress_timeout_s=initial_progress_timeout_s,
        progress_timeout_s=progress_timeout_s,
        checkpoint_timeout_s=checkpoint_timeout_s,
        failure_grace_period_s=failure_grace_period_s,
        scan_interval_s=_positive_float(raw_dict.get("scan_interval_s", 1.0), "scan_interval_s"),
        monitor_nice=monitor_nice,
        expected_world_size=expected_world_size,
        expected_node_count=expected_node_count,
        hardware_health_enabled=hardware_health_enabled,
        hardware_health_interval_s=hardware_health_interval_s,
        hardware_health_command_timeout_s=hardware_health_command_timeout_s,
        hardware_health_stale_after_s=hardware_health_stale_after_s,
    )
