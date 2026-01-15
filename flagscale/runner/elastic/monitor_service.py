import os
import signal
import subprocess
import sys
import threading
import time

from flagscale.runner.elastic.diagnostic import generate_diagnostic_report
from flagscale.runner.elastic.log_collector import collect_logs
from flagscale.runner.utils import JobStatus, get_remote_file_mtime, logger


class MonitorService:
    """
    An independent monitoring service for background monitoring of training task status, log collection, and diagnostic report generation.
    """

    def __init__(self, config, runner_instance, interval=10, host=None, node_rank=None):
        """
        Initializing service

        Args:
            config: Configuration object
            runner_instance: runner instance
            interval: interval time
            host: Hostname or IP of this node (for single-node mode)
            node_rank: Node rank of this node (for single-node mode)
        """
        self.config = config
        self.runner = runner_instance
        self.interval = interval
        self.is_running = False
        self.monitor_thread = None
        self.log_collection_enabled = True
        self.diagnostic_enabled = True
        self.hang_detection_timeout = config.experiment.runner.get(
            "hang_detection_timeout", 1800
        )  # 30 minutes in seconds
        self.last_log_check_times = {}  # Track last modification time for each log file
        self.last_job_status = None  # Track previous job status for kill detection
        self.process_start_time = time.time()  # Track when monitoring started
        # Single-node monitoring mode (each node monitors itself)
        self.single_node_mode = host is not None and node_rank is not None
        self.monitored_host = host
        self.monitored_node_rank = node_rank
        self.monitor_log_dir = os.path.join(config.train.system.logging.log_dir, "monitor")
        os.makedirs(self.monitor_log_dir, exist_ok=True)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, stopping monitor service...")
        self.stop()
        sys.exit(0)

    def start_monitoring(self):
        """
        Start monitoring service (non-blocking)

        """
        if self.is_running:
            logger.warning("Monitor service is already running")
            return

        self.is_running = True

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Monitor service started with interval={self.interval}s")
        logger.info(f"Monitor logs will be saved to: {self.monitor_log_dir}")

    def stop(self):
        """
        Stop monitoring service
        """
        if not self.is_running:
            return

        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        logger.info("Monitor service stopped")

    def _monitor_loop(self):
        logger.info("Starting monitoring loop...")

        time.sleep(self.interval)

        try:
            while self.is_running:
                start_time = time.time()

                try:
                    job_status = self._get_job_status()
                    logger.info(f"Job Status: {job_status.name}")

                    # Check for manual kill detection
                    self._check_for_manual_kill(job_status)

                    self._log_status(job_status)

                    if job_status == JobStatus.COMPLETED_OR_IDLE:
                        logger.info("Job completed, stopping monitoring")
                        break

                    if self.log_collection_enabled:
                        self._collect_logs()

                    if self.diagnostic_enabled:
                        self._generate_diagnostics()

                    if self.diagnostic_enabled:
                        self._check_and_report_hang()

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)

                if self.is_running:
                    time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Monitor loop crashed: {e}")
        finally:
            logger.info("Monitor loop ended")
            self.is_running = False

    def _get_job_status(self) -> JobStatus:
        return self.runner._query_status()

    def _check_for_manual_kill(self, current_status: JobStatus):
        """
        Check if the process was manually killed and write diagnostic entry

        Args:
            current_status: Current job status
        """
        try:
            # If this is the first check, just record the status
            if self.last_job_status is None:
                self.last_job_status = current_status
                return

            # Check if process went from RUNNING to COMPLETED_OR_IDLE suddenly
            if (
                self.last_job_status == JobStatus.RUNNING
                and current_status == JobStatus.COMPLETED_OR_IDLE
            ):
                # Check if it happened too quickly (likely manual kill)
                running_time = time.time() - self.process_start_time
                if running_time < 300:  # Less than 5 minutes, likely manual kill
                    logger.warning("Detected potential manual kill - process terminated quickly")
                    self._write_manual_kill_diagnostic()
                else:
                    # Try to detect if it was a clean shutdown vs manual kill
                    if self._detect_abnormal_termination():
                        logger.warning("Detected abnormal process termination - likely manual kill")
                        self._write_manual_kill_diagnostic()

            # Update last status
            self.last_job_status = current_status

        except Exception as e:
            logger.error(f"Error in manual kill detection: {e}")

    def _detect_abnormal_termination(self) -> bool:
        """
        Detect if the termination was abnormal (not a clean shutdown)
        Returns True if abnormal termination is detected
        """
        try:
            # Check if PID files still exist but processes are gone
            if not hasattr(self.runner, "resources") or self.runner.resources is None:
                # Local mode
                return self._check_pid_file_anomaly("localhost", 0)
            else:
                # Multi-node mode
                for node_rank, (host, _) in enumerate(self.runner.resources.items()):
                    if self._check_pid_file_anomaly(host, node_rank):
                        return True
            return False
        except Exception as e:
            logger.error(f"Error detecting abnormal termination: {e}")
            return False

    def _check_pid_file_anomaly(self, host: str, node_rank: int) -> bool:
        """
        Check if PID file exists but process is gone (indicating manual kill)
        """
        try:
            logging_config = self.config.train.system.logging
            pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

            if os.path.exists(pid_file):
                # PID file exists, check if process is still running
                with open(pid_file, "r") as f:
                    pid = int(f.read().strip())

                try:
                    result = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True)
                    if result.returncode != 0:
                        # PID file exists but process is gone - likely manual kill
                        return True
                except Exception:
                    return True

            return False
        except Exception as e:
            logger.error(f"Error checking PID file anomaly for {host}:{node_rank}: {e}")
            return False

    def _write_manual_kill_diagnostic(self):
        """
        Write manual kill detection entry to diagnostic files
        """
        try:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # Write monitor-detected kill entry (won't be re-detected by diagnostic.py)
            kill_entry = f"[{current_time}] MonitorDetected: MANUAL KILL DETECTED - Process terminated unexpectedly, likely killed manually"

            if self.single_node_mode:
                # Single-node monitoring mode
                self._write_diagnostic_entry(
                    self.monitored_host, self.monitored_node_rank, kill_entry
                )
            elif not hasattr(self.runner, "resources") or self.runner.resources is None:
                # Local mode (backward compatibility)
                self._write_diagnostic_entry("localhost", 0, kill_entry)
            else:
                # Multi-node mode (centralized monitoring)
                for node_rank, (host, _) in enumerate(self.runner.resources.items()):
                    self._write_diagnostic_entry(host, node_rank, kill_entry)

            logger.warning("⚠️ MANUAL KILL DETECTED - Diagnostic entry written to files")

        except Exception as e:
            logger.error(f"Failed to write manual kill diagnostic: {e}")

    def _write_diagnostic_entry(self, host: str, node_rank: int, entry: str):
        """
        Write a diagnostic entry directly to the diagnostic file
        """
        try:
            diagnostic_file = os.path.join(
                self.monitor_log_dir, f"host_{node_rank}_{host}_diagnostic.txt"
            )

            # Ensure diagnostic file exists with header if it doesn't
            if not os.path.exists(diagnostic_file):
                os.makedirs(os.path.dirname(diagnostic_file), exist_ok=True)
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                header_content = f"Diagnostic Report for {host} (node {node_rank})\n"
                header_content += f"Generated at {current_time}\n"
                header_content += "Analysis:\n"

                with open(diagnostic_file, "w", encoding="utf-8") as f:
                    f.write(header_content)

            # Append the entry
            with open(diagnostic_file, "a", encoding="utf-8") as f:
                f.write(f"{entry}\n")

            logger.debug(f"Diagnostic entry written for {host} (node {node_rank}): {entry}")

        except Exception as e:
            logger.error(f"Failed to write diagnostic entry for {host}:{node_rank}: {e}")

    def _log_status(self, status: JobStatus):
        status_log_file = os.path.join(self.monitor_log_dir, "status.log")

        try:
            with open(status_log_file, "a", encoding="utf-8") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                f.write(f"[{timestamp}] Status: {status.name}\n")
        except Exception as e:
            logger.error(f"Failed to write status log: {e}")

    def _collect_logs(self):
        if self.single_node_mode:
            # Single-node monitoring mode - each node monitors only itself
            self._collect_logs_for_host(self.monitored_host, self.monitored_node_rank)
        elif not hasattr(self.runner, "resources") or self.runner.resources is None:
            # Local mode (backward compatibility)
            self._collect_logs_for_host("localhost", 0)
        else:
            # Multi-node mode (centralized monitoring)
            for node_rank, (host, _) in enumerate(self.runner.resources.items()):
                self._collect_logs_for_host(host, node_rank)

    def _collect_logs_for_host(self, host: str, node_rank: int):
        try:
            log_file = collect_logs(
                self.config, host, node_rank, self.monitor_log_dir, dryrun=False
            )

            if log_file:
                logger.debug(f"Collected logs for {host} (node {node_rank}): {log_file}")
        except Exception as e:
            logger.error(f"Failed to collect logs for {host} (node {node_rank}): {e}")

    def _generate_diagnostics(self):
        """Generate diagnostc report"""
        if self.single_node_mode:
            # Single-node monitoring mode - each node monitors only itself
            self._generate_diagnostic_for_host(self.monitored_host, self.monitored_node_rank)
        elif not hasattr(self.runner, "resources") or self.runner.resources is None:
            self._generate_diagnostic_for_host("localhost", 0)
        else:
            # Multi-nodes (centralized monitoring)
            for node_rank, (host, _) in enumerate(self.runner.resources.items()):
                self._generate_diagnostic_for_host(host, node_rank)

    def _generate_diagnostic_for_host(self, host: str, node_rank: int):
        try:
            log_file_path = None
            current_log_file = os.path.join(
                self.monitor_log_dir, f"host_{node_rank}_{host}_current.log"
            )
            if os.path.exists(current_log_file):
                log_file_path = current_log_file
            else:
                no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
                if no_shared_fs:
                    logger.warning(
                        "no_shared_fs path has NOT been fully tested and may behave incorrectly!"
                    )
                    src_log_file = os.path.join(
                        self.config.train.system.logging.log_dir, "host.output"
                    )
                else:
                    src_log_file = os.path.join(
                        self.config.train.system.logging.log_dir, f"host_{node_rank}_{host}.output"
                    )

                if os.path.exists(src_log_file):
                    log_file_path = src_log_file
                    logger.debug(f"Using source log file for diagnostic: {src_log_file}")

            if log_file_path and os.path.exists(log_file_path):
                diagnostic_file = generate_diagnostic_report(
                    self.config, host, node_rank, log_file_path, return_content=False
                )
                if diagnostic_file:
                    logger.debug(
                        f"Generated diagnostic for {host} (node {node_rank}): {diagnostic_file}"
                    )
            else:
                logger.debug(
                    f"No log file available for diagnostic generation: {host} (node {node_rank})"
                )
        except Exception as e:
            logger.error(f"Failed to generate diagnostic for {host} (node {node_rank}): {e}")

    def _check_log_hang(self, host: str, node_rank: int) -> bool:
        """
        Check if log file has not been updated for too long (hang detection)

        Args:
            host (str): Hostname
            node_rank (int): Node rank

        Returns:
            bool: True if log appears to be hanging, False otherwise
        """
        try:
            # Determine log file path
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                log_file = os.path.join(self.config.train.system.logging.log_dir, "host.output")
            else:
                log_file = os.path.join(
                    self.config.train.system.logging.log_dir, f"host_{node_rank}_{host}.output"
                )

            # Get current modification time
            if no_shared_fs:
                logger.warning(
                    "no_shared_fs path has NOT been fully tested and may behave incorrectly!"
                )
                # For remote files, get mtime via SSH
                ssh_port = self.config.experiment.runner.get("ssh_port", 22)
                current_mtime = get_remote_file_mtime(host, log_file, ssh_port)
                if current_mtime == -1:
                    # Remote file doesn't exist or SSH failed
                    return False
            else:
                # For local/shared files, check existence and get mtime
                if not os.path.exists(log_file):
                    return False
                current_mtime = os.path.getmtime(log_file)
            current_time = time.time()

            # Check if log file hasn't been updated for too long
            time_since_update = current_time - current_mtime

            if time_since_update > self.hang_detection_timeout:
                logger.warning(
                    f"Log file {log_file} has not been updated for {time_since_update:.0f} seconds"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking log hang for {host} (node {node_rank}): {e}")
            return False

    def _generate_hang_diagnostic(self, host: str, node_rank: int):
        """
        Generate hang diagnostic entry when log file is not updating

        Args:
            host (str): Hostname
            node_rank (int): Node rank
        """
        try:
            # Create a temporary diagnostic content for hang detection
            log_dir = self.monitor_log_dir
            diagnostic_file = os.path.join(log_dir, f"host_{node_rank}_{host}_diagnostic.txt")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Determine log file name for reference
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                logger.warning(
                    "no_shared_fs path has NOT been fully tested and may behave incorrectly!"
                )
                log_filename = "host.output"
            else:
                log_filename = f"host_{node_rank}_{host}.output"

            hang_entry = f"[{current_time}] HangError: Process appears to be hanging - log file not updated for over {self.hang_detection_timeout // 60} minutes. Check {log_filename}"

            # Ensure diagnostic file exists with header if it doesn't
            if not os.path.exists(diagnostic_file):
                os.makedirs(os.path.dirname(diagnostic_file), exist_ok=True)
                header_content = f"Diagnostic Report for {host} (node {node_rank})\n"
                header_content += f"Generated at {current_time}\n"
                header_content += "Analysis:\n"

                with open(diagnostic_file, "w", encoding="utf-8") as f:
                    f.write(header_content)

            # Append hang detection entry
            with open(diagnostic_file, "a", encoding="utf-8") as f:
                f.write(f"{hang_entry}\n")

            logger.info(
                f"Added hang detection entry to diagnostic report for {host} (node {node_rank})"
            )

        except Exception as e:
            logger.error(f"Failed to generate hang diagnostic for {host} (node {node_rank}): {e}")

    def _check_and_report_hang(self):
        """Check for hanging processes and report them"""
        if self.single_node_mode:
            # Single-node monitoring mode
            if self._check_log_hang(self.monitored_host, self.monitored_node_rank):
                self._generate_hang_diagnostic(self.monitored_host, self.monitored_node_rank)
        elif not hasattr(self.runner, "resources") or self.runner.resources is None:
            # Local mode (backward compatibility)
            if self._check_log_hang("localhost", 0):
                self._generate_hang_diagnostic("localhost", 0)
        else:
            # Multi-node mode (centralized monitoring)
            for node_rank, (host, _) in enumerate(self.runner.resources.items()):
                if self._check_log_hang(host, node_rank):
                    self._generate_hang_diagnostic(host, node_rank)
