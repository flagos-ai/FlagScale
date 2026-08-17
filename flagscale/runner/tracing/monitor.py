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

"""Low-priority CPU process that tails NCCL probe JSONL files."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path
from typing import TextIO

from .analyzer import Finding, TraceAnalyzer
from .inspector import InspectorReader
from flagscale.runner.heartbeat.health_reader import HardwareHealthReader

logger = logging.getLogger("flagscale.tracing")


class JsonlTailer:
    """Incrementally read complete JSON lines from per-process trace files."""

    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir
        self._offsets: dict[Path, int] = {}
        self._remainders: dict[Path, str] = {}

    def poll(self) -> list[dict]:
        events: list[dict] = []
        for path in sorted(self.trace_dir.glob("rank_*_pid_*.jsonl")):
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
            lines = text.splitlines(keepends=True)
            for line in lines:
                if not line.endswith(("\n", "\r")):
                    self._remainders[path] = line
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Ignored malformed trace line in %s", path)
                    continue
                if isinstance(event, dict):
                    events.append(event)
        return events


def _append_findings(file_obj: TextIO, findings: list[Finding]) -> None:
    for finding in findings:
        payload = json.dumps(finding.to_dict(), sort_keys=True, separators=(",", ":"))
        file_obj.write(payload + "\n")
        logger.warning("NCCL trace finding: %s", payload)
    if findings:
        file_obj.flush()
        os.fsync(file_obj.fileno())


def _completion_exit_code(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def run_monitor(args: argparse.Namespace) -> int:
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    report_file = Path(args.report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    completion_file = Path(args.completion_file) if args.completion_file else None

    if args.nice:
        try:
            os.nice(args.nice)
        except (AttributeError, OSError):
            logger.debug("Could not adjust analyzer niceness", exc_info=True)

    analyzer = TraceAnalyzer(
        run_id=args.run_id,
        heartbeat_timeout_s=args.heartbeat_timeout,
        collective_timeout_s=args.collective_timeout,
        delayed_enter_threshold_s=args.delayed_enter_threshold,
        checkpoint_timeout_s=args.checkpoint_timeout,
        p2p_timeout_s=args.p2p_timeout,
        p2p_match_window_s=args.p2p_match_window,
        missing_exit_timeout_s=args.missing_exit_timeout,
    )
    tailers = [JsonlTailer(trace_dir)]
    if args.heartbeat_dir:
        tailers.append(JsonlTailer(Path(args.heartbeat_dir)))
    hardware_reader = HardwareHealthReader(
        Path(args.heartbeat_dir) if args.heartbeat_dir else trace_dir,
        args.run_id,
        args.hardware_health,
        args.hardware_health_stale_after,
    )
    inspector_reader = InspectorReader(
        Path(args.inspector_dir) if args.inspector_dir else trace_dir / "inspector",
        enabled=bool(args.inspector_dir),
        correlation_window_s=args.inspector_correlation_window,
    )
    stopping = False
    failed_completion_seen_monotonic_s: float | None = None

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    with report_file.open("a", encoding="utf-8") as output:
        while not stopping:
            observed_monotonic_s = time.monotonic()
            observed_unix_ns = time.time_ns()
            for tailer in tailers:
                for event in tailer.poll():
                    analyzer.ingest(
                        event,
                        observed_monotonic_s=observed_monotonic_s,
                        observed_unix_ns=observed_unix_ns,
                    )

            scan_unix_ns = time.time_ns()
            hardware_health, _hardware_findings = hardware_reader.poll(scan_unix_ns)
            inspector = inspector_reader.poll()
            findings = analyzer.scan(
                now_monotonic_s=time.monotonic(),
                now_unix_ns=scan_unix_ns,
                hardware_health=hardware_health,
                inspector=inspector,
            )
            _append_findings(output, findings)

            exit_code = _completion_exit_code(completion_file)
            if exit_code is not None and exit_code != 0:
                if failed_completion_seen_monotonic_s is None:
                    failed_completion_seen_monotonic_s = time.monotonic()
                    logger.info(
                        "Training exited with code %s; keeping analyzer alive for %.1fs",
                        exit_code,
                        args.failure_grace_period,
                    )

            failure_grace_elapsed = (
                failed_completion_seen_monotonic_s is not None
                and time.monotonic() - failed_completion_seen_monotonic_s
                >= args.failure_grace_period
            )
            if exit_code == 0 or failure_grace_elapsed:
                # One last poll prevents losing lines appended immediately before the
                # training shell wrote its completion marker. A failed run stays alive
                # for a bounded grace period so timeout-based crash evidence can mature.
                for tailer in tailers:
                    for event in tailer.poll():
                        analyzer.ingest(
                            event,
                            observed_monotonic_s=time.monotonic(),
                            observed_unix_ns=time.time_ns(),
                        )
                final_unix_ns = time.time_ns()
                final_hardware_health, _hardware_findings = hardware_reader.poll(final_unix_ns)
                _append_findings(
                    output,
                    analyzer.scan(
                        now_monotonic_s=time.monotonic(),
                        now_unix_ns=final_unix_ns,
                        hardware_health=final_hardware_health,
                        inspector=inspector_reader.poll(),
                    ),
                )
                break
            time.sleep(args.scan_interval)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--heartbeat-dir")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--heartbeat-timeout", type=float, default=30.0)
    parser.add_argument("--collective-timeout", type=float, default=60.0)
    parser.add_argument("--delayed-enter-threshold", type=float, default=30.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=1800.0)
    parser.add_argument("--p2p-timeout", type=float, default=60.0)
    parser.add_argument("--p2p-match-window", type=float, default=30.0)
    parser.add_argument("--missing-exit-timeout", type=float, default=60.0)
    parser.add_argument("--failure-grace-period", type=float, default=60.0)
    parser.add_argument("--scan-interval", type=float, default=1.0)
    parser.add_argument("--completion-file")
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--nice", type=int, default=10)
    parser.add_argument("--hardware-health", action="store_true")
    parser.add_argument("--hardware-health-stale-after", type=float, default=180.0)
    parser.add_argument("--inspector-dir")
    parser.add_argument("--inspector-correlation-window", type=float, default=5.0)
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    for name in (
        "heartbeat_timeout",
        "collective_timeout",
        "delayed_enter_threshold",
        "checkpoint_timeout",
        "p2p_timeout",
        "p2p_match_window",
        "missing_exit_timeout",
        "failure_grace_period",
        "scan_interval",
        "hardware_health_stale_after",
        "inspector_correlation_window",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be greater than zero")
    if args.p2p_match_window > args.p2p_timeout:
        raise SystemExit("--p2p-match-window must not exceed --p2p-timeout")
    return run_monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
