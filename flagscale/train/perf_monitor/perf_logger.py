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

"""File logger for performance monitor output."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path


class PerfMonitorLogger:
    """Rank-0 logger that writes human-readable and json summaries."""

    def __init__(
        self,
        log_dir="logs/perf_monitor",
        log_level=logging.INFO,
        enable_console=False,
        max_log_files=10,
        log_format="both",
    ):
        self.rank = 0
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                self.rank = dist.get_rank()
        except ImportError:
            pass

        self.enabled = self.rank == 0
        self.json_data = []
        if not self.enabled:
            return

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_log_files = max_log_files
        self.log_format = log_format
        self.session_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

        self.metrics_file = self.log_dir / f"perf_metrics_{self.session_timestamp}.log"
        self.summary_file = self.log_dir / f"perf_summary_{self.session_timestamp}.json"
        self.realtime_file = self.log_dir / "perf_realtime.log"

        self.logger = logging.getLogger(f"perf_monitor_{self.session_timestamp}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []

        if self.log_format in ("text", "both"):
            file_handler = logging.FileHandler(self.metrics_file)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(file_handler)

            if enable_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(console_handler)

        self._write_header()

    def _write_header(self):
        if not self.enabled or self.log_format not in ("text", "both"):
            return

        header = "=" * 96 + "\n"
        header += (
            f"Performance Monitor Session Started: "
            f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        header += "=" * 96 + "\n"
        header += (
            f"{'Timestamp':<20} {'Step':<8} {'TFLOPS/GPU':<12} {'TFLOPS':<10} "
            f"{'Samples/s':<12} {'Tokens/s':<12} {'Time(ms)':<10} {'Memory(GB)':<10}\n"
        )
        header += "-" * 96
        self.logger.info(header)
        self.realtime_file.write_text(f"{header}\n")

    def log_metrics(self, iteration, metrics_dict):
        if not self.enabled:
            return

        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        if self.log_format in ("text", "both"):
            log_line = (
                f"{timestamp:<20} {iteration:<8} "
                f"{metrics_dict.get('TFLOPS_per_GPU', 0.0):<12.2f} "
                f"{metrics_dict.get('TFLOPS_total', 0.0):<10.2f} "
                f"{metrics_dict.get('samples_per_sec', 0.0):<12.1f} "
                f"{metrics_dict.get('tokens_per_sec', 0.0):<12.0f} "
                f"{metrics_dict.get('step_time_ms', 0.0):<10.1f} "
                f"{metrics_dict.get('memory_GB', 0.0):<10.2f}"
            )
            self.logger.info(log_line)
            with self.realtime_file.open("a") as file_obj:
                file_obj.write(f"{log_line}\n")

        if self.log_format in ("json", "both"):
            self.json_data.append(
                {
                    "iteration": iteration,
                    "timestamp": timestamp,
                    **metrics_dict,
                }
            )

    def log_breakdown(self, iteration, breakdown):
        if not self.enabled or self.log_format not in ("text", "both"):
            return

        lines = [f"Estimated FLOPS Breakdown (Iteration {iteration}):"]
        for key, value in breakdown.items():
            lines.append(f"  {key}: {value / 1e12:.2f} TFLOPS")
        self.logger.info("\n".join(lines))

    def save_summary(self, final_stats=None, total_iterations=None):
        if not self.enabled:
            return

        summary = {
            "session_info": {
                "start_time": self.session_timestamp,
                "end_time": datetime.now().astimezone().isoformat(),
                "total_iterations": (
                    total_iterations if total_iterations is not None else len(self.json_data)
                ),
            },
            "final_statistics": final_stats or {},
            "iteration_logs": self.json_data,
        }
        if self.log_format in ("json", "both"):
            with self.summary_file.open("w") as file_obj:
                json.dump(summary, file_obj, indent=2)
        self._cleanup_old_logs()

    def _cleanup_old_logs(self):
        if not self.enabled or self.max_log_files <= 0:
            return

        log_files = sorted(self.log_dir.glob("perf_metrics_*.log"))
        if len(log_files) <= self.max_log_files:
            return

        for old_file in log_files[: -self.max_log_files]:
            try:
                old_file.unlink()
                summary_file = self.log_dir / (
                    f"perf_summary_{old_file.stem.replace('perf_metrics_', '')}.json"
                )
                if summary_file.exists():
                    summary_file.unlink()
            except OSError:
                continue
