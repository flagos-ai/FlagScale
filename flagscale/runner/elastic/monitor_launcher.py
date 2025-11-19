#!/usr/bi/env python3
"""
An independent monitoring service launcher
for starting the monitoring service during the training process, including log collection and diagnostic report generation
"""

import argparse
import os
import subprocess
import sys
import time

from pathlib import Path

from omegaconf import OmegaConf

from flagscale.runner.elastic.monitor_service import MonitorService
from flagscale.runner.runner_base import JobStatus
from flagscale.runner.utils import logger


class MonitorRunner:
    """
    The dummy Runner class used for monitoring services
    Provide status query function
    """

    def __init__(self, config, pid_file_path):
        self.resources = None
        self.config = config
        self.pid_file_path = pid_file_path

    def _query_status(self):
        """Query the status of the training process"""
        if os.path.exists(self.pid_file_path):
            try:
                with open(self.pid_file_path, "r") as f:
                    pid = int(f.read().strip())
                result = subprocess.run(["ps", "-p", str(pid)], capture_output=True)
                if result.returncode == 0:
                    return JobStatus.RUNNING
                else:
                    return JobStatus.COMPLETED_OR_IDLE
            except Exception:
                return JobStatus.COMPLETED_OR_IDLE
        else:
            return JobStatus.COMPLETED_OR_IDLE


def main():
    """Monitor the main function of the service"""
    parser = argparse.ArgumentParser(description="Start the FlagScale training monitoring service")
    parser.add_argument("--log-dir", required=True, help="Log directory path")
    parser.add_argument(
        "--pid-file", required=True, help="The path of the PID file for the training process"
    )
    parser.add_argument("--host", required=True, help="Hostname or IP of this node")
    parser.add_argument("--node-rank", type=int, required=True, help="Node rank of this node")
    parser.add_argument(
        "--no-shared-fs", action="store_true", help="Whether it is in non-shared file system mode"
    )
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval (seconds)")
    parser.add_argument(
        "--enable-log-collection", action="store_true", help="Enable log collection"
    )
    parser.add_argument(
        "--enable-diagnostic", action="store_true", help="Enable diagnostic reports"
    )

    args = parser.parse_args()

    # Wait for the training process to start by checking for the PID file
    logger.info(f"Waiting for the training process to start by checking for {args.pid_file}...")
    wait_start_time = time.time()
    while not os.path.exists(args.pid_file):
        if time.time() - wait_start_time > 60:  # 1-minute timeout
            logger.error(f"Timeout waiting for PID file {args.pid_file}. Exiting.")
            sys.exit(1)
        time.sleep(1)
    logger.info("Training process PID file found.")

    # Create configuration
    config = OmegaConf.create(
        {
            "train": {"system": {"logging": {"log_dir": args.log_dir}}},
            "experiment": {
                "runner": {"no_shared_fs": args.no_shared_fs, "ssh_port": args.ssh_port}
            },
        }
    )

    # Create dummy runners and monitoring services
    runner = MonitorRunner(config, args.pid_file)
    monitor = MonitorService(
        config, runner, interval=args.interval, host=args.host, node_rank=args.node_rank
    )

    # Start the monitoring service
    monitor.start_monitoring(
        enable_log_collection=args.enable_log_collection, enable_diagnostic=args.enable_diagnostic
    )

    logger.info(f"The monitoring service has been started. Interval: {args.interval} seconds")
    logger.info(f"PID file: {args.pid_file}")
    logger.info(f"Log directory: {args.log_dir}")

    # Keep the monitoring running until the training is over
    try:
        while True:
            status = runner._query_status()
            if status == JobStatus.COMPLETED_OR_IDLE:
                logger.info("The training has been completed. Stop the monitoring service")
                break
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("The monitoring service was interrupted by the user")
    except Exception as e:
        logger.error(f"An error occurred in the monitoring service: {e}")
    finally:
        monitor.stop()
        logger.info("The monitoring service has been stopped")


if __name__ == "__main__":
    main()
