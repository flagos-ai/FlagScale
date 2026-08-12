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

"""Low-frequency, node-local NVIDIA GPU hardware health collector."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Any

logger = logging.getLogger("flagscale.heartbeat.gpu_health")

QUERY_FIELDS = (
    "index",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks_event_reasons.hw_thermal_slowdown",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.uncorrected.volatile.total",
    "retired_pages.pending",
    "remapped_rows.pending",
    "remapped_rows.failure",
    "gpu_recovery_action",
)

_MISSING_VALUES = {"", "n/a", "[n/a]", "not supported", "not found", "unknown"}


def _missing(value: str) -> bool:
    return value.strip().lower() in _MISSING_VALUES


def _integer(value: str) -> int | None:
    if _missing(value):
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def _number(value: str) -> float | None:
    if _missing(value):
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def _yes(value: str) -> bool | None:
    if _missing(value):
        return None
    normalized = value.strip().lower()
    if normalized in {"yes", "active", "true", "1"}:
        return True
    if normalized in {"no", "not active", "false", "0", "none"}:
        return False
    return None


def _recovery_required(value: str) -> bool:
    return value.strip().lower() not in _MISSING_VALUES | {"none"}


def parse_nvidia_smi_csv(
    output: str, corrected_ecc_baseline: dict[str, int] | None = None
) -> list[dict[str, Any]]:
    """Parse and classify one ``nvidia-smi --query-gpu`` response."""

    baseline = corrected_ecc_baseline if corrected_ecc_baseline is not None else {}
    gpus: list[dict[str, Any]] = []
    for values in csv.reader(output.splitlines(), skipinitialspace=True):
        if not values:
            continue
        if len(values) != len(QUERY_FIELDS):
            raise ValueError(
                f"nvidia-smi returned {len(values)} columns; expected {len(QUERY_FIELDS)}"
            )
        row = dict(zip(QUERY_FIELDS, (value.strip() for value in values), strict=True))
        uuid = row["uuid"]
        corrected_ecc = _integer(row["ecc.errors.corrected.volatile.total"])
        uncorrected_ecc = _integer(row["ecc.errors.uncorrected.volatile.total"])
        previous_corrected = baseline.setdefault(uuid, corrected_ecc or 0)
        corrected_delta = (
            max(0, corrected_ecc - previous_corrected) if corrected_ecc is not None else None
        )

        issues: list[dict[str, str]] = []
        if uncorrected_ecc is not None and uncorrected_ecc > 0:
            issues.append(
                {
                    "severity": "unhealthy",
                    "reason": "volatile_uncorrected_ecc_present",
                }
            )
        if _yes(row["remapped_rows.failure"]) is True:
            issues.append({"severity": "unhealthy", "reason": "row_remap_failure"})
        if _recovery_required(row["gpu_recovery_action"]):
            issues.append(
                {
                    "severity": "unhealthy",
                    "reason": "gpu_recovery_action_required",
                }
            )
        if corrected_delta is not None and corrected_delta > 0:
            issues.append(
                {
                    "severity": "warning",
                    "reason": "new_volatile_corrected_ecc",
                }
            )
        if _yes(row["retired_pages.pending"]) is True:
            issues.append({"severity": "warning", "reason": "retired_pages_pending"})
        if _yes(row["remapped_rows.pending"]) is True:
            issues.append({"severity": "warning", "reason": "row_remap_pending"})
        if _yes(row["clocks_event_reasons.hw_thermal_slowdown"]) is True:
            issues.append({"severity": "warning", "reason": "hardware_thermal_slowdown"})

        status = "healthy"
        if any(issue["severity"] == "unhealthy" for issue in issues):
            status = "unhealthy"
        elif issues:
            status = "warning"

        gpus.append(
            {
                "index": _integer(row["index"]),
                "uuid": uuid,
                "pci_bus_id": row["pci.bus_id"],
                "driver_version": row["driver_version"],
                "status": status,
                "issues": issues,
                "temperature_c": _number(row["temperature.gpu"]),
                "utilization_percent": _number(row["utilization.gpu"]),
                "memory_used_mib": _number(row["memory.used"]),
                "memory_total_mib": _number(row["memory.total"]),
                "power_draw_w": _number(row["power.draw"]),
                "power_limit_w": _number(row["power.limit"]),
                "hardware_thermal_slowdown": _yes(row["clocks_event_reasons.hw_thermal_slowdown"]),
                "volatile_corrected_ecc": corrected_ecc,
                "volatile_corrected_ecc_delta": corrected_delta,
                "volatile_uncorrected_ecc": uncorrected_ecc,
                "retired_pages_pending": _yes(row["retired_pages.pending"]),
                "row_remap_pending": _yes(row["remapped_rows.pending"]),
                "row_remap_failure": _yes(row["remapped_rows.failure"]),
                "gpu_recovery_action": (
                    row["gpu_recovery_action"]
                    if _recovery_required(row["gpu_recovery_action"])
                    else None
                ),
            }
        )
    return gpus


def _aggregate_status(gpus: list[dict[str, Any]]) -> str:
    statuses = {str(gpu.get("status")) for gpu in gpus}
    if "unhealthy" in statuses:
        return "unhealthy"
    if "warning" in statuses:
        return "warning"
    return "healthy" if gpus else "unavailable"


class NvidiaSmiHealthSampler:
    def __init__(self, command_timeout_s: float) -> None:
        self.command_timeout_s = command_timeout_s
        self.corrected_ecc_baseline: dict[str, int] = {}

    def sample(self, run_id: str, node_rank: int) -> dict[str, Any]:
        collected_at_unix_ns = time.time_ns()
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_s,
            )
            gpus = parse_nvidia_smi_csv(result.stdout, self.corrected_ecc_baseline)
            return {
                "schema_version": 1,
                "component": "gpu_hardware_health",
                "run_id": run_id,
                "node_rank": node_rank,
                "hostname": socket.gethostname(),
                "collector_pid": os.getpid(),
                "collected_at_unix_ns": collected_at_unix_ns,
                "source": "nvidia-smi_nvml",
                "status": _aggregate_status(gpus),
                "gpus": gpus,
            }
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            logger.warning("GPU hardware health collection failed: %s", exc)
            return {
                "schema_version": 1,
                "component": "gpu_hardware_health",
                "run_id": run_id,
                "node_rank": node_rank,
                "hostname": socket.gethostname(),
                "collector_pid": os.getpid(),
                "collected_at_unix_ns": collected_at_unix_ns,
                "source": "nvidia-smi_nvml",
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "gpus": [],
            }


def _write_snapshot(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_collector(args: argparse.Namespace) -> int:
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.nice:
        try:
            os.nice(args.nice)
        except (AttributeError, OSError):
            logger.debug("Could not adjust GPU health collector niceness", exc_info=True)

    stopping = Event()

    def request_stop(_signum, _frame) -> None:
        stopping.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    sampler = NvidiaSmiHealthSampler(args.command_timeout)
    while not stopping.is_set():
        _write_snapshot(output_file, sampler.sample(args.run_id, args.node_rank))
        stopping.wait(args.interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--command-timeout", type=float, default=10.0)
    parser.add_argument("--nice", type=int, default=10)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    if args.interval <= 0 or args.command_timeout <= 0:
        raise SystemExit("--interval and --command-timeout must be greater than zero")
    return run_collector(args)


if __name__ == "__main__":
    raise SystemExit(main())
