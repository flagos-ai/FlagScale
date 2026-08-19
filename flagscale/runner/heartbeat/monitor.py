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

"""Low-priority monitor for per-rank process and GPU-progress heartbeats."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .health_reader import HardwareHealthIndex, HardwareHealthReader

logger = logging.getLogger("flagscale.heartbeat")


class JsonlTailer:
    def __init__(self, heartbeat_dir: Path) -> None:
        self.heartbeat_dir = heartbeat_dir
        self._offsets: dict[Path, int] = {}
        self._remainders: dict[Path, str] = {}

    def poll(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for path in sorted(self.heartbeat_dir.glob("rank_*_pid_*.heartbeat.jsonl")):
            offset = self._offsets.get(path, 0)
            try:
                size = path.stat().st_size
                if size < offset:
                    offset = 0
                    self._remainders.pop(path, None)
                with path.open("r", encoding="utf-8", errors="replace") as file_obj:
                    file_obj.seek(offset)
                    chunk = file_obj.read()
                    self._offsets[path] = file_obj.tell()
            except OSError as exc:
                logger.debug("Could not read %s: %s", path, exc)
                continue
            if not chunk:
                continue
            text = self._remainders.pop(path, "") + chunk
            for line in text.splitlines(keepends=True):
                if not line.endswith(("\n", "\r")):
                    self._remainders[path] = line
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Ignored malformed heartbeat line in %s", path)
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events


@dataclass
class RankState:
    start: dict[str, Any]
    start_seen_s: float
    heartbeat: dict[str, Any] | None = None
    heartbeat_seen_s: float | None = None
    progress: dict[str, Any] | None = None
    progress_seen_s: float | None = None
    progress_seq: int = 0
    ended: bool = False


class HeartbeatAnalyzer:
    def __init__(
        self,
        run_id: str,
        initial_process_timeout_s: float,
        process_timeout_s: float,
        initial_progress_timeout_s: float,
        progress_timeout_s: float,
        checkpoint_timeout_s: float,
        expected_world_size: int = 0,
        monitor_started_s: float = 0.0,
    ) -> None:
        self.run_id = run_id
        self.initial_process_timeout_s = initial_process_timeout_s
        self.process_timeout_s = process_timeout_s
        self.initial_progress_timeout_s = initial_progress_timeout_s
        self.progress_timeout_s = progress_timeout_s
        self.checkpoint_timeout_s = checkpoint_timeout_s
        self.expected_world_size = expected_world_size
        self.monitor_started_s = monitor_started_s
        self.ranks: dict[int, RankState] = {}
        # In-memory deduplication for active timeout episodes. Persisted findings
        # remain in findings.jsonl after their keys are cleared on recovery.
        self._reported: set[tuple[int, int, str]] = set()

    def ingest(self, event: dict[str, Any], observed_s: float) -> None:
        if event.get("run_id") != self.run_id:
            return
        rank = _as_int(event.get("rank"), -1)
        pid = _as_int(event.get("pid"), -1)
        if rank < 0 or pid < 0:
            return
        state = self.ranks.get(rank)
        if state is None or _as_int(state.start.get("pid"), -2) != pid:
            self._reported = {key for key in self._reported if key[0] != rank}
            state = RankState(start=event, start_seen_s=observed_s)
            self.ranks[rank] = state

        event_type = event.get("event")
        if event_type == "process_start":
            state.start = event
            state.start_seen_s = observed_s
            state.ended = False
        elif event_type == "heartbeat":
            state.heartbeat = event
            state.heartbeat_seen_s = observed_s
            self._reported.discard((rank, pid, "initial_process_heartbeat"))
            self._reported.discard((rank, pid, "subsequent_process_heartbeat"))
        elif event_type == "process_end":
            state.heartbeat = event
            state.heartbeat_seen_s = observed_s
            state.ended = True

        progress_seq = _as_int(event.get("progress_seq"), 0)
        if progress_seq > state.progress_seq:
            state.progress_seq = progress_seq
            state.progress = event
            state.progress_seen_s = observed_s
            self._reported.discard((rank, pid, "initial_gpu_progress"))
            self._reported.discard((rank, pid, "subsequent_gpu_progress"))

    def _process_age(self, state: RankState, now_s: float) -> tuple[float, str]:
        if state.heartbeat_seen_s is None:
            return now_s - state.start_seen_s, "initial_process_heartbeat"
        return now_s - state.heartbeat_seen_s, "subsequent_process_heartbeat"

    def _progress_age(self, state: RankState, now_s: float) -> tuple[float, str, float]:
        source = state.heartbeat or state.progress or state.start
        phase = str(source.get("phase") or "setup")
        if state.progress_seen_s is None:
            return now_s - state.start_seen_s, phase, self.initial_progress_timeout_s
        timeout_s = (
            self.checkpoint_timeout_s if phase == "checkpointing" else self.progress_timeout_s
        )
        return now_s - state.progress_seen_s, phase, timeout_s

    def scan(
        self,
        now_s: float,
        now_unix_ns: int,
        hardware_health: HardwareHealthIndex | None = None,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for rank in range(self.expected_world_size):
            if rank in self.ranks:
                continue
            age_s = now_s - self.monitor_started_s
            key = (rank, -1, "initial_process_heartbeat")
            if age_s <= self.initial_process_timeout_s or key in self._reported:
                continue
            self._reported.add(key)
            findings.append(
                {
                    "finding_type": "rank_process_heartbeat_timeout",
                    "run_id": self.run_id,
                    "detected_at_unix_ns": now_unix_ns,
                    "rank": rank,
                    "pid": None,
                    "hostname": None,
                    "timeout_type": "initial_process_heartbeat",
                    "heartbeat_age_s": max(0.0, age_s),
                    "reason": "rank_process_heartbeat_never_observed",
                    "confidence": "observed",
                }
            )

        for rank, state in sorted(self.ranks.items()):
            if state.ended:
                continue
            pid = _as_int(state.start.get("pid"), -1)
            process_age_s, process_timeout_type = self._process_age(state, now_s)
            process_timeout_s = (
                self.initial_process_timeout_s
                if state.heartbeat_seen_s is None
                else self.process_timeout_s
            )
            process_alive = process_age_s <= process_timeout_s
            process_key = (rank, pid, process_timeout_type)
            if not process_alive and process_key not in self._reported:
                self._reported.add(process_key)
                findings.append(
                    {
                        "finding_type": "rank_process_heartbeat_timeout",
                        "run_id": self.run_id,
                        "detected_at_unix_ns": now_unix_ns,
                        "rank": rank,
                        "pid": pid,
                        "hostname": state.start.get("hostname"),
                        "timeout_type": process_timeout_type,
                        "heartbeat_age_s": max(0.0, process_age_s),
                        "reason": "rank_process_or_heartbeat_thread_unresponsive",
                        "confidence": "suspected",
                    }
                )

            # A stale process already explains absent progress. Report a GPU-progress
            # timeout only when the CPU-side liveness publisher is still responsive.
            if not process_alive:
                continue
            progress_age_s, phase, progress_timeout_s = self._progress_age(state, now_s)
            progress_timeout_type = (
                "initial_gpu_progress"
                if state.progress_seen_s is None
                else "subsequent_gpu_progress"
            )
            progress_key = (rank, pid, progress_timeout_type)
            if progress_age_s <= progress_timeout_s or progress_key in self._reported:
                continue
            self._reported.add(progress_key)
            source = state.heartbeat or state.progress or state.start
            hardware = (
                hardware_health.for_rank(source)
                if hardware_health is not None
                else {"gpu_device_health": "not_collected"}
            )
            findings.append(
                {
                    "finding_type": "gpu_progress_timeout",
                    "run_id": self.run_id,
                    "detected_at_unix_ns": now_unix_ns,
                    "rank": rank,
                    "local_rank": source.get("local_rank"),
                    "pid": pid,
                    "hostname": source.get("hostname"),
                    "phase": phase,
                    "iteration": source.get("iteration"),
                    "progress_seq": state.progress_seq,
                    "timeout_type": progress_timeout_type,
                    "progress_age_s": max(0.0, progress_age_s),
                    "process_liveness": "alive",
                    "gpu_progress": "stalled",
                    **hardware,
                    "reason": "rank_alive_but_gpu_progress_not_advancing",
                    "confidence": "suspected",
                }
            )
        return findings

    def snapshot(
        self,
        now_s: float,
        now_unix_ns: int,
        hardware_health: HardwareHealthIndex | None = None,
    ) -> dict[str, Any]:
        ranks: list[dict[str, Any]] = []
        for rank in range(self.expected_world_size):
            if rank not in self.ranks:
                ranks.append(
                    {
                        "rank": rank,
                        "process_liveness": "not_observed",
                        "gpu_progress": "not_observed",
                        "gpu_device_health": (
                            hardware_health.overall_status
                            if hardware_health is not None
                            else "not_collected"
                        ),
                        "process_heartbeat_age_s": max(0.0, now_s - self.monitor_started_s),
                        "progress_age_s": None,
                    }
                )

        for rank, state in sorted(self.ranks.items()):
            process_age_s, _ = self._process_age(state, now_s)
            process_timeout_s = (
                self.initial_process_timeout_s
                if state.heartbeat_seen_s is None
                else self.process_timeout_s
            )
            process_alive = process_age_s <= process_timeout_s
            progress_age_s, phase, progress_timeout_s = self._progress_age(state, now_s)
            if state.progress_seen_s is None:
                gpu_progress = (
                    "starting" if progress_age_s <= self.initial_progress_timeout_s else "stalled"
                )
            else:
                gpu_progress = "progressing" if progress_age_s <= progress_timeout_s else "stalled"
            source = state.heartbeat or state.progress or state.start
            hardware = (
                hardware_health.for_rank(source)
                if hardware_health is not None
                else {"gpu_device_health": "not_collected"}
            )
            ranks.append(
                {
                    "rank": rank,
                    "local_rank": source.get("local_rank"),
                    "pid": source.get("pid"),
                    "hostname": source.get("hostname"),
                    "world_size": source.get("world_size"),
                    "cuda_visible_devices": source.get("cuda_visible_devices"),
                    "assigned_gpu": source.get("assigned_gpu"),
                    "phase": phase,
                    "iteration": source.get("iteration"),
                    "progress_seq": state.progress_seq,
                    "process_liveness": (
                        "exited" if state.ended else ("alive" if process_alive else "stale")
                    ),
                    "process_heartbeat_age_s": max(0.0, process_age_s),
                    "gpu_progress": (
                        "completed"
                        if state.ended
                        else (gpu_progress if process_alive else "unknown")
                    ),
                    "progress_age_s": max(0.0, progress_age_s),
                    **hardware,
                }
            )
        ranks.sort(key=lambda item: int(item["rank"]))
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "updated_at_unix_ns": now_unix_ns,
            "health_scope": "gpu_training_progress_and_optional_hardware",
            "gpu_device_health": (
                hardware_health.overall_status if hardware_health is not None else "not_collected"
            ),
            "hardware_health": (
                hardware_health.summary()
                if hardware_health is not None
                else {"enabled": False, "status": "not_collected", "nodes": []}
            ),
            "ranks": ranks,
        }


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _completion_exit_code(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _append_findings(file_obj: TextIO, findings: list[dict[str, Any]]) -> None:
    for finding in findings:
        payload = json.dumps(finding, sort_keys=True, separators=(",", ":"))
        file_obj.write(payload + "\n")
        logger.warning("GPU heartbeat finding: %s", payload)
    if findings:
        file_obj.flush()
        os.fsync(file_obj.fileno())


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_monitor(args: argparse.Namespace) -> int:
    heartbeat_dir = Path(args.heartbeat_dir)
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    completion_file = Path(args.completion_file) if args.completion_file else None
    report_file = Path(args.report_file)
    status_file = Path(args.status_file)
    if args.nice:
        try:
            os.nice(args.nice)
        except (AttributeError, OSError):
            logger.debug("Could not adjust heartbeat monitor niceness", exc_info=True)

    monitor_started_s = time.monotonic()
    monitor_started_unix_ns = time.time_ns()
    analyzer = HeartbeatAnalyzer(
        args.run_id,
        args.initial_process_timeout,
        args.process_timeout,
        args.initial_progress_timeout,
        args.progress_timeout,
        args.checkpoint_timeout,
        expected_world_size=args.expected_world_size,
        monitor_started_s=monitor_started_s,
    )
    tailer = JsonlTailer(heartbeat_dir)
    hardware_reader = HardwareHealthReader(
        heartbeat_dir,
        args.run_id,
        args.hardware_health_enabled,
        args.hardware_health_stale_after,
        expected_node_count=args.expected_node_count,
        monitor_started_unix_ns=monitor_started_unix_ns,
    )
    stopping = False
    failed_seen_s: float | None = None

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    with report_file.open("a", encoding="utf-8") as output:
        while not stopping:
            now_s = time.monotonic()
            for event in tailer.poll():
                analyzer.ingest(event, now_s)
            now_unix_ns = time.time_ns()
            hardware_health, hardware_findings = hardware_reader.poll(now_unix_ns)
            findings = hardware_findings + analyzer.scan(now_s, now_unix_ns, hardware_health)
            _append_findings(output, findings)
            _write_status(status_file, analyzer.snapshot(now_s, now_unix_ns, hardware_health))

            exit_code = _completion_exit_code(completion_file)
            if exit_code == 0:
                break
            if exit_code is not None and failed_seen_s is None:
                failed_seen_s = now_s
            if failed_seen_s is not None and now_s - failed_seen_s >= args.failure_grace_period:
                break
            time.sleep(args.scan_interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartbeat-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-world-size", type=int, default=0)
    parser.add_argument("--expected-node-count", type=int, default=0)
    parser.add_argument("--initial-process-timeout", type=float, default=30.0)
    parser.add_argument("--process-timeout", type=float, default=30.0)
    parser.add_argument("--initial-progress-timeout", type=float, default=600.0)
    parser.add_argument("--progress-timeout", type=float, default=300.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=1800.0)
    parser.add_argument("--failure-grace-period", type=float, default=30.0)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--completion-file")
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--hardware-health-enabled", action="store_true")
    parser.add_argument("--hardware-health-stale-after", type=float, default=180.0)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()
    if args.expected_world_size < 0:
        raise SystemExit("--expected-world-size must be non-negative")
    if args.expected_node_count < 0:
        raise SystemExit("--expected-node-count must be non-negative")
    for name in (
        "initial_process_timeout",
        "process_timeout",
        "initial_progress_timeout",
        "progress_timeout",
        "checkpoint_timeout",
        "failure_grace_period",
        "scan_interval",
        "hardware_health_stale_after",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero")
    return run_monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
