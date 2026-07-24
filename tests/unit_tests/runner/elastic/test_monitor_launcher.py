import subprocess
from unittest.mock import MagicMock, patch

import pytest

from flagscale.runner.elastic import monitor_launcher
from flagscale.runner.elastic.monitor_launcher import MonitorRunner
from flagscale.runner.utils import JobStatus


def test_monitor_runner_query_status_returns_completed_when_pid_file_missing(tmp_path):
    runner = MonitorRunner(MagicMock(), str(tmp_path / "missing.pid"))

    assert runner._query_status() == JobStatus.COMPLETED_OR_IDLE


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0, JobStatus.RUNNING), (1, JobStatus.COMPLETED_OR_IDLE)],
)
def test_monitor_runner_query_status_reads_pid_and_checks_process(tmp_path, returncode, expected):
    pid_file = tmp_path / "train.pid"
    pid_file.write_text("12345")
    runner = MonitorRunner(MagicMock(), str(pid_file))

    with patch(
        "flagscale.runner.elastic.monitor_launcher.subprocess.run",
        return_value=subprocess.CompletedProcess(["ps"], returncode),
    ) as run:
        assert runner._query_status() == expected

    run.assert_called_once_with(["ps", "-p", "12345"], capture_output=True)


def test_monitor_runner_query_status_handles_invalid_pid(tmp_path):
    pid_file = tmp_path / "train.pid"
    pid_file.write_text("not-a-pid")
    runner = MonitorRunner(MagicMock(), str(pid_file))

    assert runner._query_status() == JobStatus.COMPLETED_OR_IDLE


def test_main_waits_for_pid_timeout_and_exits(tmp_path):
    argv = [
        "monitor_launcher.py",
        "--log-dir",
        str(tmp_path / "logs"),
        "--pid-file",
        str(tmp_path / "train.pid"),
        "--host",
        "localhost",
        "--node-rank",
        "0",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "flagscale.runner.elastic.monitor_launcher.os.path.exists",
            return_value=False,
        ),
        patch("flagscale.runner.elastic.monitor_launcher.time.time", side_effect=[0, 61]),
        patch("flagscale.runner.elastic.monitor_launcher.time.sleep") as sleep,
        patch("flagscale.runner.elastic.monitor_launcher.logger") as logger,
        pytest.raises(SystemExit) as exc,
    ):
        monitor_launcher.main()

    assert exc.value.code == 1
    sleep.assert_not_called()
    logger.error.assert_called_once()


def test_main_starts_monitor_and_stops_after_training_completed(tmp_path):
    pid_file = tmp_path / "train.pid"
    pid_file.write_text("12345")
    monitor = MagicMock()
    argv = [
        "monitor_launcher.py",
        "--log-dir",
        str(tmp_path / "logs"),
        "--pid-file",
        str(pid_file),
        "--host",
        "worker0",
        "--node-rank",
        "3",
        "--no-shared-fs",
        "--ssh-port",
        "2222",
        "--interval",
        "7",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "flagscale.runner.elastic.monitor_launcher.MonitorService",
            return_value=monitor,
        ) as service_cls,
        patch.object(
            monitor_launcher.MonitorRunner,
            "_query_status",
            side_effect=[JobStatus.RUNNING, JobStatus.COMPLETED_OR_IDLE],
        ),
        patch("flagscale.runner.elastic.monitor_launcher.time.sleep") as sleep,
    ):
        monitor_launcher.main()

    monitor.start_monitoring.assert_called_once()
    monitor.stop.assert_called_once()
    sleep.assert_called_once_with(10)
    _, kwargs = service_cls.call_args
    assert kwargs == {"interval": 7, "host": "worker0", "node_rank": 3}
    config = service_cls.call_args.args[0]
    assert config.train.system.logging.log_dir == str(tmp_path / "logs")
    assert config.experiment.runner.no_shared_fs is True
    assert config.experiment.runner.ssh_port == 2222


def test_main_stops_monitor_on_keyboard_interrupt(tmp_path):
    pid_file = tmp_path / "train.pid"
    pid_file.write_text("12345")
    monitor = MagicMock()
    argv = [
        "monitor_launcher.py",
        "--log-dir",
        str(tmp_path / "logs"),
        "--pid-file",
        str(pid_file),
        "--host",
        "localhost",
        "--node-rank",
        "0",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "flagscale.runner.elastic.monitor_launcher.MonitorService",
            return_value=monitor,
        ),
        patch.object(
            monitor_launcher.MonitorRunner,
            "_query_status",
            side_effect=KeyboardInterrupt,
        ),
        patch("flagscale.runner.elastic.monitor_launcher.logger") as logger,
    ):
        monitor_launcher.main()

    monitor.stop.assert_called_once()
    logger.info.assert_any_call("The monitoring service was interrupted by the user")


def test_main_logs_unexpected_monitoring_error_and_stops(tmp_path):
    pid_file = tmp_path / "train.pid"
    pid_file.write_text("12345")
    monitor = MagicMock()
    argv = [
        "monitor_launcher.py",
        "--log-dir",
        str(tmp_path / "logs"),
        "--pid-file",
        str(pid_file),
        "--host",
        "localhost",
        "--node-rank",
        "0",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "flagscale.runner.elastic.monitor_launcher.MonitorService",
            return_value=monitor,
        ),
        patch.object(
            monitor_launcher.MonitorRunner,
            "_query_status",
            side_effect=RuntimeError("query failed"),
        ),
        patch("flagscale.runner.elastic.monitor_launcher.logger") as logger,
    ):
        monitor_launcher.main()

    monitor.stop.assert_called_once()
    logger.error.assert_called_once_with(
        "An error occurred in the monitoring service: query failed"
    )
