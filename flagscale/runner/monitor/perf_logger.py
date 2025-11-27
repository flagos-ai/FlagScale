"""
Performance Monitor File Logger
This module provides a dedicated logging system for performance metrics,
supporting both text and JSON output formats.
"""

import json
import logging
import os

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PerfMonitorLogger:
    """
    Dedicated file logger for performance monitoring.
    This logger handles:
    - Writing metrics to text log files
    - Saving JSON summaries
    - Managing log rotation
    - Ensuring only rank 0 writes in distributed settings
    """

    def __init__(
        self,
        log_dir: str = "logs/perf_monitor",
        log_level: int = logging.INFO,
        enable_console: bool = False,
        max_log_files: int = 10,
    ):
        """
        Initialize the performance monitor logger.
        Args:
            log_dir: Directory for log files
            log_level: Logging level
            enable_console: Whether to also output to console
            max_log_files: Maximum number of log files to keep
        """
        # Check if we're in distributed training
        self.rank = 0
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                self.rank = dist.get_rank()
        except ImportError:
            pass

        self.enabled = self.rank == 0  # Only rank 0 writes logs

        if not self.enabled:
            return

        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup timestamp for this session (use local timezone)
        self.session_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

        # Setup log files
        self.metrics_file = self.log_dir / f"perf_metrics_{self.session_timestamp}.log"
        self.summary_file = self.log_dir / f"perf_summary_{self.session_timestamp}.json"
        self.realtime_file = self.log_dir / "perf_realtime.log"

        # Setup logger
        self.logger = logging.getLogger(f"perf_monitor_{self.session_timestamp}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear any existing handlers

        # File handler for metrics
        file_handler = logging.FileHandler(self.metrics_file)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler if requested
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(file_formatter)
            self.logger.addHandler(console_handler)

        # JSON data cache for summary
        self.json_data = []
        self.max_log_files = max_log_files

        # Write header
        self._write_header()

    def _write_header(self):
        """Write header information to log file."""
        if not self.enabled:
            return

        header = "=" * 80 + "\n"
        header += f"Performance Monitor Session Started: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "=" * 80 + "\n"
        header += f"{'Timestamp':<20} {'Step':<8} {'TFLOPS/GPU':<12} {'TFLOPS':<10} "
        header += f"{'Samples/s':<12} {'Tokens/s':<12} {'Time(ms)':<10} {'Memory(GB)':<10}\n"
        header += "-" * 80

        self.logger.info(header)

        # Also write to realtime file
        with open(self.realtime_file, 'w') as f:
            f.write(header + "\n")

    def log_metrics(self, iteration: int, metrics_dict: Dict[str, float]):
        """
        Log performance metrics to file.
        Args:
            iteration: Current training iteration
            metrics_dict: Dictionary of metric names to values
        """
        if not self.enabled:
            return

        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

        # Format metrics for text log
        tflops_gpu = metrics_dict.get('TFLOPS_per_GPU', 0.0)
        tflops_total = metrics_dict.get('TFLOPS_total', 0.0)
        samples_sec = metrics_dict.get('samples_per_sec', 0.0)
        tokens_sec = metrics_dict.get('tokens_per_sec', 0.0)
        step_time = metrics_dict.get('step_time_ms', 0.0)
        memory_gb = metrics_dict.get('memory_GB', 0.0)

        # Create formatted log line
        log_line = f"{timestamp:<20} {iteration:<8} {tflops_gpu:<12.2f} {tflops_total:<10.2f} "
        log_line += (
            f"{samples_sec:<12.1f} {tokens_sec:<12.0f} {step_time:<10.1f} {memory_gb:<10.2f}"
        )

        # Write to main log file
        self.logger.info(log_line)

        # Write to realtime file (append mode)
        with open(self.realtime_file, 'a') as f:
            f.write(log_line + "\n")

        # Store for JSON summary
        json_entry = {"iteration": iteration, "timestamp": timestamp, **metrics_dict}
        self.json_data.append(json_entry)

    def log_performance_breakdown(self, iteration: int, breakdown: Dict[str, float]):
        """
        Log detailed performance breakdown.
        Args:
            iteration: Current iteration
            breakdown: Dictionary with timing breakdowns
        """
        if not self.enabled:
            return

        log_msg = f"\n  Performance Breakdown (Iteration {iteration}):\n"
        for component, time_ms in breakdown.items():
            log_msg += f"    {component:<20}: {time_ms:>8.2f} ms\n"

        self.logger.info(log_msg)

    def save_summary(self, final_stats: Optional[Dict[str, Any]] = None):
        """
        Save final summary to JSON file.
        Args:
            final_stats: Optional final statistics to include
        """
        if not self.enabled:
            return

        # Calculate statistics if not provided
        if final_stats is None and self.json_data:
            final_stats = self._calculate_statistics()

        summary = {
            "session_info": {
                "start_time": self.session_timestamp,
                "end_time": datetime.now().astimezone().isoformat(),
                "total_iterations": len(self.json_data),
            },
            "final_statistics": final_stats or {},
            "iteration_logs": self.json_data,
        }

        # Write JSON summary
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Log completion
        footer = "\n" + "=" * 80 + "\n"
        footer += f"Performance Monitor Session Completed: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"Summary saved to: {self.summary_file}\n"
        footer += "=" * 80

        self.logger.info(footer)

        # Cleanup old log files if needed
        self._cleanup_old_logs()

    def _calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistics from logged data."""
        if not self.json_data:
            return {}

        # Extract metrics
        tflops = [d.get('TFLOPS_per_GPU', 0) for d in self.json_data]
        throughput = [d.get('tokens_per_sec', 0) for d in self.json_data]
        step_times = [d.get('step_time_ms', 0) for d in self.json_data]
        memory = [d.get('memory_GB', 0) for d in self.json_data if d.get('memory_GB')]

        stats = {
            'avg_tflops_per_gpu': sum(tflops) / len(tflops) if tflops else 0,
            'max_tflops_per_gpu': max(tflops) if tflops else 0,
            'min_tflops_per_gpu': min(tflops) if tflops else 0,
            'avg_throughput_tokens': sum(throughput) / len(throughput) if throughput else 0,
            'avg_step_time_ms': sum(step_times) / len(step_times) if step_times else 0,
            'min_step_time_ms': min(step_times) if step_times else 0,
            'max_step_time_ms': max(step_times) if step_times else 0,
        }

        if memory:
            stats['peak_memory_gb'] = max(memory)
            stats['avg_memory_gb'] = sum(memory) / len(memory)

        return stats

    def _cleanup_old_logs(self):
        """Remove old log files if exceeding max_log_files limit."""
        if not self.enabled or self.max_log_files <= 0:
            return

        # Find all metrics log files
        log_files = sorted(self.log_dir.glob("perf_metrics_*.log"))

        # Remove oldest files if exceeding limit
        if len(log_files) > self.max_log_files:
            for old_file in log_files[: -self.max_log_files]:
                try:
                    old_file.unlink()
                    # Also remove corresponding JSON file
                    json_file = old_file.with_suffix('.json')
                    json_file = (
                        self.log_dir
                        / f"perf_summary_{old_file.stem.replace('perf_metrics_', '')}.json"
                    )
                    if json_file.exists():
                        json_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove old log file {old_file}: {e}")

    def close(self):
        """Close the logger and save final summary."""
        if not self.enabled:
            return

        self.save_summary()

        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
