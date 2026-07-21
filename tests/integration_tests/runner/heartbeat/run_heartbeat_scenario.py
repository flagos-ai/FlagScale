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

"""Run the standalone GPU-progress heartbeat under a real two-worker torchrun."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("normal", "stall"), default="normal")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[4]
    output_dir = Path(tempfile.mkdtemp(prefix="flagscale-gpu-heartbeat-"))
    run_id = f"integration-heartbeat-{uuid.uuid4().hex[:8]}"
    completion = output_dir / "training.exit_code"
    findings_path = output_dir / "findings.jsonl"
    status_path = output_dir / "status.json"
    monitor_log = output_dir / "monitor.log"

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(repo)
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "FLAGSCALE_GPU_HEARTBEAT_ENABLE": "1",
            "FLAGSCALE_GPU_HEARTBEAT_RUN_ID": run_id,
            "FLAGSCALE_GPU_HEARTBEAT_DIR": str(output_dir),
            "FLAGSCALE_GPU_HEARTBEAT_INTERVAL_SEC": "0.1",
        }
    )
    workload_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(Path(__file__).with_name("heartbeat_workload.py")),
        "--scenario",
        args.scenario,
    ]
    monitor_command = [
        sys.executable,
        "-m",
        "flagscale.runner.heartbeat.monitor",
        "--heartbeat-dir",
        str(output_dir),
        "--run-id",
        run_id,
        "--expected-world-size",
        "2",
        "--initial-process-timeout",
        "5",
        "--process-timeout",
        "2",
        "--initial-progress-timeout",
        "2",
        "--progress-timeout",
        "1.2",
        "--checkpoint-timeout",
        "2",
        "--failure-grace-period",
        "0.6",
        "--scan-interval",
        "0.05",
        "--completion-file",
        str(completion),
        "--report-file",
        str(findings_path),
        "--status-file",
        str(status_path),
        "--nice",
        "10",
    ]

    with monitor_log.open("w", encoding="utf-8") as monitor_output:
        monitor = subprocess.Popen(
            monitor_command,
            cwd=repo,
            stdout=monitor_output,
            stderr=subprocess.STDOUT,
        )
        result = subprocess.run(
            workload_command,
            cwd=repo,
            env=env,
            timeout=10,
            check=False,
        )
        completion.write_text(f"{result.returncode}\n", encoding="utf-8")
        monitor.wait(timeout=5)

    raw_files = sorted(output_dir.glob("rank_*_pid_*.heartbeat.jsonl"))
    progress_max = []
    for path in raw_files:
        records = _read_jsonl(path)
        progress_max.append(max(int(record.get("progress_seq", 0)) for record in records))
    findings = _read_jsonl(findings_path)
    finding_types = [finding.get("finding_type") for finding in findings]
    result_payload = {
        "scenario": args.scenario,
        "return_code": result.returncode,
        "raw_files": len(raw_files),
        "progress_max": progress_max,
        "finding_types": finding_types,
        "output_dir": str(output_dir),
    }
    print(json.dumps(result_payload, sort_keys=True))

    base_ok = result.returncode == 0 and len(raw_files) == 2
    if args.scenario == "normal":
        return 0 if base_ok and min(progress_max) == 25 and not findings else 1
    stalled_ranks = [
        finding.get("rank")
        for finding in findings
        if finding.get("finding_type") == "gpu_progress_timeout"
    ]
    return 0 if base_ok and stalled_ranks == [1] else 1


if __name__ == "__main__":
    raise SystemExit(main())
