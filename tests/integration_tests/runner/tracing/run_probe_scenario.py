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

"""Run one real NCCL probe scenario and validate its analyzer output."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

EXPECTED_FINDING = {"not_enter": "collective_missing_enter"}


def _stop_process_group(process: subprocess.Popen, grace_s: float = 3.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=grace_s)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def run_scenario(scenario: str, timeout_s: float) -> int:
    repo = Path(__file__).resolve().parents[4]
    probe = repo / "flagscale/runner/tracing/native/libflagscale_nccl_probe.so"
    workload = Path(__file__).with_name("nccl_scenarios.py")
    if not probe.is_file():
        raise RuntimeError(f"probe has not been built: {probe}")

    trace_dir = Path(tempfile.mkdtemp(prefix=f"flagscale-{scenario}-"))
    heartbeat_dir = trace_dir / "heartbeat"
    heartbeat_dir.mkdir()
    run_id = f"integration-{scenario}-{uuid.uuid4().hex[:8]}"
    completion = trace_dir / "training.exit_code"
    findings_path = trace_dir / "findings.jsonl"
    workload_log = trace_dir / "workload.log"
    monitor_log = trace_dir / "monitor.log"

    monitor_command = [
        sys.executable,
        "-m",
        "flagscale.runner.tracing.monitor",
        "--trace-dir",
        str(trace_dir),
        "--run-id",
        run_id,
        "--collective-timeout",
        "1",
        "--delayed-enter-threshold",
        "1",
        "--failure-grace-period",
        "2",
        "--scan-interval",
        "0.1",
        "--completion-file",
        str(completion),
        "--report-file",
        str(findings_path),
        "--nice",
        "10",
    ]
    workload_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(workload),
        "--scenario",
        scenario,
    ]
    workload_env = os.environ.copy()
    workload_env.update(
        {
            "PYTHONPATH": str(repo)
            + (os.pathsep + workload_env["PYTHONPATH"] if workload_env.get("PYTHONPATH") else ""),
            "FLAGSCALE_TRACE_ENABLE": "1",
            "FLAGSCALE_TRACE_RUN_ID": run_id,
            "FLAGSCALE_TRACE_DIR": str(trace_dir),
            "LD_PRELOAD": str(probe)
            + (f":{workload_env['LD_PRELOAD']}" if workload_env.get("LD_PRELOAD") else ""),
        }
    )
    if scenario == "not_enter":
        monitor_command.extend(["--heartbeat-dir", str(heartbeat_dir), "--heartbeat-timeout", "1"])
        workload_env.update(
            {
                "FLAGSCALE_GPU_HEARTBEAT_ENABLE": "1",
                "FLAGSCALE_GPU_HEARTBEAT_RUN_ID": run_id,
                "FLAGSCALE_GPU_HEARTBEAT_DIR": str(heartbeat_dir),
                "FLAGSCALE_GPU_HEARTBEAT_INTERVAL_SEC": "0.25",
            }
        )

    with (
        monitor_log.open("w", encoding="utf-8") as monitor_output,
        workload_log.open("w", encoding="utf-8") as workload_output,
    ):
        monitor = subprocess.Popen(
            monitor_command,
            cwd=repo,
            stdout=monitor_output,
            stderr=subprocess.STDOUT,
        )
        workload_process = subprocess.Popen(
            workload_command,
            cwd=repo,
            env=workload_env,
            stdout=workload_output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            return_code = workload_process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_group(workload_process)
            return_code = 124

        completion.write_text(f"{return_code}\n", encoding="utf-8")
        try:
            monitor.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            monitor.terminate()
            monitor.wait(timeout=3.0)

    findings = _read_jsonl(findings_path)
    finding_types = [str(finding.get("hang_type")) for finding in findings]
    raw_files = sorted(trace_dir.glob("rank_*_pid_*.jsonl"))

    result = {
        "scenario": scenario,
        "workload_return_code": return_code,
        "timed_out": timed_out,
        "finding_types": finding_types,
        "raw_trace_files": len(raw_files),
        "trace_dir": str(trace_dir),
    }
    print(json.dumps(result, sort_keys=True))

    if scenario == "sanity":
        return 0 if return_code == 0 and not findings else 1

    if scenario == "subprocess":
        # Only the two torchrun workers should own trace files. The helper spawned
        # by each worker performs no NCCL work and must not create another file.
        return 0 if return_code == 0 and not findings and len(raw_files) == 2 else 1

    expected = EXPECTED_FINDING[scenario]
    if expected not in finding_types:
        print(workload_log.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
        print(monitor_log.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        required=True,
        choices=("sanity", "subprocess", *EXPECTED_FINDING),
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    return run_scenario(args.scenario, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
