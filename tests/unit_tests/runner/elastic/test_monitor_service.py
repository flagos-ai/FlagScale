import os
import subprocess

import pytest
from omegaconf import OmegaConf

from flagscale.runner.elastic.monitor_service import MonitorService
from flagscale.runner.utils import JobStatus


class FakeRunner:
    def __init__(self, resources=None, statuses=None):
        self.resources = resources
        self.statuses = list(statuses or [JobStatus.RUNNING])

    def _query_status(self):
        return self.statuses.pop(0) if self.statuses else JobStatus.COMPLETED_OR_IDLE


def make_monitor_config(tmp_path, no_shared_fs=False, timeout=60):
    return OmegaConf.create(
        {
            "experiment": {
                "runner": {
                    "no_shared_fs": no_shared_fs,
                    "ssh_port": 2222,
                    "hang_detection_timeout": timeout,
                }
            },
            "train": {
                "system": {
                    "logging": {
                        "log_dir": str(tmp_path / "logs"),
                        "pids_dir": str(tmp_path / "logs" / "pids"),
                    }
                }
            },
        }
    )


def test_start_monitoring_is_idempotent_and_stop_joins_thread(tmp_path, mocker):
    config = make_monitor_config(tmp_path)
    runner = FakeRunner()
    service = MonitorService(config, runner, interval=1)

    class FakeThread:
        def __init__(self, target=None, daemon=False):
            nonlocal target_ref, daemon_ref
            target_ref = target
            daemon_ref = daemon
            self.started = False
            self.joined = False

        def start(self):
            self.started = True

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.joined = timeout

    target_ref = None
    daemon_ref = None
    thread_cls = mocker.patch(
        "flagscale.runner.elastic.monitor_service.threading.Thread", FakeThread
    )
    warning = mocker.patch("flagscale.runner.elastic.monitor_service.logger.warning")

    service.start_monitoring()
    first_thread = service.monitor_thread
    service.start_monitoring()
    service.stop()

    assert thread_cls is FakeThread
    assert target_ref == service._monitor_loop
    assert daemon_ref is True
    assert first_thread.started is True
    assert first_thread.joined == 5
    warning.assert_called_once_with("Monitor service is already running")
    assert service.is_running is False


def test_log_status_writes_status_file(tmp_path):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)

    service._log_status(JobStatus.RUNNING)

    status_log = os.path.join(service.monitor_log_dir, "status.log")
    assert os.path.exists(status_log)
    assert "Status: RUNNING" in open(status_log, encoding="utf-8").read()


def test_check_for_manual_kill_writes_diagnostic_on_fast_termination(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    service.last_job_status = JobStatus.RUNNING
    service.process_start_time = 100
    mocker.patch("flagscale.runner.elastic.monitor_service.time.time", return_value=120)
    write = mocker.patch.object(service, "_write_manual_kill_diagnostic")

    service._check_for_manual_kill(JobStatus.COMPLETED_OR_IDLE)

    write.assert_called_once()
    assert service.last_job_status == JobStatus.COMPLETED_OR_IDLE


def test_check_pid_file_anomaly_detects_dead_process(tmp_path, mocker):
    config = make_monitor_config(tmp_path)
    service = MonitorService(config, FakeRunner(), interval=1)
    pid_dir = tmp_path / "logs" / "pids"
    pid_dir.mkdir(parents=True)
    (pid_dir / "host_0_localhost.pid").write_text("12345")
    mocker.patch(
        "flagscale.runner.elastic.monitor_service.subprocess.run",
        return_value=subprocess.CompletedProcess(["ps"], 1),
    )

    assert service._check_pid_file_anomaly("localhost", 0) is True


def test_collect_logs_and_diagnostics_route_to_each_resource(tmp_path, mocker):
    resources = {"worker0": {}, "worker1": {}}
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(resources), interval=1)
    collect = mocker.patch.object(service, "_collect_logs_for_host")
    diagnostic = mocker.patch.object(service, "_generate_diagnostic_for_host")

    service._collect_logs()
    service._generate_diagnostics()

    assert collect.call_args_list[0].args == ("worker0", 0)
    assert collect.call_args_list[1].args == ("worker1", 1)
    assert diagnostic.call_args_list[0].args == ("worker0", 0)
    assert diagnostic.call_args_list[1].args == ("worker1", 1)


def test_check_log_hang_detects_stale_local_log(tmp_path, mocker):
    config = make_monitor_config(tmp_path, timeout=30)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "host_0_localhost.output"
    log_file.write_text("old log")
    service = MonitorService(config, FakeRunner(), interval=1)
    mocker.patch("flagscale.runner.elastic.monitor_service.os.path.getmtime", return_value=100)
    mocker.patch("flagscale.runner.elastic.monitor_service.time.time", return_value=131)

    assert service._check_log_hang("localhost", 0) is True


def test_check_log_hang_no_shared_fs_uses_remote_mtime(tmp_path, mocker):
    config = make_monitor_config(tmp_path, no_shared_fs=True, timeout=30)
    service = MonitorService(config, FakeRunner(), interval=1)
    remote_mtime = mocker.patch(
        "flagscale.runner.elastic.monitor_service.get_remote_file_mtime",
        return_value=100,
    )
    mocker.patch("flagscale.runner.elastic.monitor_service.time.time", return_value=120)

    assert service._check_log_hang("worker0", 0) is False
    remote_mtime.assert_called_once_with(
        "worker0",
        os.path.join(config.train.system.logging.log_dir, "host.output"),
        2222,
    )


def test_check_and_report_hang_generates_diagnostic_for_hanging_nodes(tmp_path, mocker):
    service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner({"worker0": {}, "worker1": {}}),
        interval=1,
    )
    check = mocker.patch.object(service, "_check_log_hang", side_effect=[True, False])
    generate = mocker.patch.object(service, "_generate_hang_diagnostic")

    service._check_and_report_hang()

    assert check.call_args_list[0].args == ("worker0", 0)
    assert check.call_args_list[1].args == ("worker1", 1)
    generate.assert_called_once_with("worker0", 0)


def test_monitor_loop_runs_collection_diagnostic_and_stops_on_completed(tmp_path, mocker):
    service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner(statuses=[JobStatus.RUNNING, JobStatus.COMPLETED_OR_IDLE]),
        interval=1,
    )
    service.is_running = True
    log_status = mocker.patch.object(service, "_log_status")
    collect = mocker.patch.object(service, "_collect_logs")
    diagnostic = mocker.patch.object(service, "_generate_diagnostics")
    hang = mocker.patch.object(service, "_check_and_report_hang")
    mocker.patch("flagscale.runner.elastic.monitor_service.time.sleep")
    mocker.patch("flagscale.runner.elastic.monitor_service.time.time", return_value=0)

    service._monitor_loop()

    assert [call.args[0] for call in log_status.call_args_list] == [
        JobStatus.RUNNING,
        JobStatus.COMPLETED_OR_IDLE,
    ]
    collect.assert_called_once()
    diagnostic.assert_called_once()
    hang.assert_called_once()
    assert service.is_running is False


def test_signal_handler_stops_service_and_exits(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    stop = mocker.patch.object(service, "stop")

    with pytest.raises(SystemExit) as exc:
        service._signal_handler(15, None)

    assert exc.value.code == 0
    stop.assert_called_once()


def test_detect_abnormal_termination_local_and_multi_node(tmp_path, mocker):
    local_service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    local_check = mocker.patch.object(local_service, "_check_pid_file_anomaly", return_value=True)

    assert local_service._detect_abnormal_termination() is True
    local_check.assert_called_once_with("localhost", 0)

    multi_service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner({"worker0": {}, "worker1": {}}),
        interval=1,
    )
    multi_check = mocker.patch.object(
        multi_service, "_check_pid_file_anomaly", side_effect=[False, True]
    )

    assert multi_service._detect_abnormal_termination() is True
    assert [call.args for call in multi_check.call_args_list] == [
        ("worker0", 0),
        ("worker1", 1),
    ]


def test_detect_abnormal_termination_handles_exception(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    mocker.patch.object(service, "_check_pid_file_anomaly", side_effect=RuntimeError("bad"))
    logger = mocker.patch("flagscale.runner.elastic.monitor_service.logger")

    assert service._detect_abnormal_termination() is False
    logger.error.assert_called_once()


def test_check_pid_file_anomaly_alive_missing_and_invalid_pid(tmp_path, mocker):
    config = make_monitor_config(tmp_path)
    service = MonitorService(config, FakeRunner(), interval=1)
    pid_dir = tmp_path / "logs" / "pids"
    pid_dir.mkdir(parents=True)

    assert service._check_pid_file_anomaly("localhost", 0) is False

    pid_file = pid_dir / "host_0_localhost.pid"
    pid_file.write_text("12345")
    mocker.patch(
        "flagscale.runner.elastic.monitor_service.subprocess.run",
        return_value=subprocess.CompletedProcess(["ps"], 0),
    )
    assert service._check_pid_file_anomaly("localhost", 0) is False

    pid_file.write_text("not-a-pid")
    assert service._check_pid_file_anomaly("localhost", 0) is False


def test_check_pid_file_anomaly_treats_ps_exception_as_abnormal(tmp_path, mocker):
    config = make_monitor_config(tmp_path)
    service = MonitorService(config, FakeRunner(), interval=1)
    pid_dir = tmp_path / "logs" / "pids"
    pid_dir.mkdir(parents=True)
    (pid_dir / "host_0_localhost.pid").write_text("12345")
    mocker.patch(
        "flagscale.runner.elastic.monitor_service.subprocess.run",
        side_effect=RuntimeError("ps failed"),
    )

    assert service._check_pid_file_anomaly("localhost", 0) is True


def test_write_manual_kill_diagnostic_routes_single_local_and_multi(tmp_path, mocker):
    single_service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner(),
        interval=1,
        host="worker0",
        node_rank=2,
    )
    single_write = mocker.patch.object(single_service, "_write_diagnostic_entry")
    single_service._write_manual_kill_diagnostic()
    assert single_write.call_args.args[:2] == ("worker0", 2)
    assert "MANUAL KILL DETECTED" in single_write.call_args.args[2]

    local_service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    local_write = mocker.patch.object(local_service, "_write_diagnostic_entry")
    local_service._write_manual_kill_diagnostic()
    assert local_write.call_args.args[:2] == ("localhost", 0)

    multi_service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner({"worker0": {}, "worker1": {}}),
        interval=1,
    )
    multi_write = mocker.patch.object(multi_service, "_write_diagnostic_entry")
    multi_service._write_manual_kill_diagnostic()
    assert [call.args[:2] for call in multi_write.call_args_list] == [
        ("worker0", 0),
        ("worker1", 1),
    ]


def test_write_diagnostic_entry_creates_header_and_appends(tmp_path):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)

    service._write_diagnostic_entry("localhost", 0, "manual kill")
    service._write_diagnostic_entry("localhost", 0, "second entry")

    diagnostic_file = os.path.join(service.monitor_log_dir, "host_0_localhost_diagnostic.txt")
    content = open(diagnostic_file, encoding="utf-8").read()
    assert "Diagnostic Report for localhost (node 0)" in content
    assert "manual kill" in content
    assert "second entry" in content


def test_write_diagnostic_entry_logs_write_error(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    logger = mocker.patch("flagscale.runner.elastic.monitor_service.logger")
    mocker.patch("builtins.open", side_effect=PermissionError("readonly"))

    service._write_diagnostic_entry("localhost", 0, "entry")

    logger.error.assert_called_once()


def test_collect_logs_for_host_logs_success_and_exception(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    logger = mocker.patch("flagscale.runner.elastic.monitor_service.logger")
    collect = mocker.patch(
        "flagscale.runner.elastic.monitor_service.collect_logs",
        return_value="current.log",
    )

    service._collect_logs_for_host("localhost", 0)

    collect.assert_called_once_with(
        service.config, "localhost", 0, service.monitor_log_dir, dryrun=False
    )
    logger.debug.assert_called_once()

    collect.side_effect = RuntimeError("collect failed")
    service._collect_logs_for_host("localhost", 0)
    logger.error.assert_called_once()


def test_generate_diagnostic_for_host_uses_current_log(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    current_log = os.path.join(service.monitor_log_dir, "host_0_localhost_current.log")
    with open(current_log, "w", encoding="utf-8") as f:
        f.write("cuda error")
    generate = mocker.patch(
        "flagscale.runner.elastic.monitor_service.generate_diagnostic_report",
        return_value="diagnostic.txt",
    )

    service._generate_diagnostic_for_host("localhost", 0)

    generate.assert_called_once_with(
        service.config, "localhost", 0, current_log, return_content=False
    )


def test_generate_diagnostic_for_host_uses_source_log_and_no_shared_fs(tmp_path, mocker):
    config = make_monitor_config(tmp_path, no_shared_fs=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    source_log = log_dir / "host.output"
    source_log.write_text("fatal error")
    service = MonitorService(config, FakeRunner(), interval=1)
    generate = mocker.patch(
        "flagscale.runner.elastic.monitor_service.generate_diagnostic_report",
        return_value="diagnostic.txt",
    )

    service._generate_diagnostic_for_host("worker0", 0)

    generate.assert_called_once_with(
        service.config, "worker0", 0, str(source_log), return_content=False
    )


def test_generate_diagnostic_for_host_logs_no_file_and_exception(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    logger = mocker.patch("flagscale.runner.elastic.monitor_service.logger")

    service._generate_diagnostic_for_host("localhost", 0)
    logger.debug.assert_called()

    current_log = os.path.join(service.monitor_log_dir, "host_0_localhost_current.log")
    with open(current_log, "w", encoding="utf-8") as f:
        f.write("cuda error")
    mocker.patch(
        "flagscale.runner.elastic.monitor_service.generate_diagnostic_report",
        side_effect=RuntimeError("diagnostic failed"),
    )
    service._generate_diagnostic_for_host("localhost", 0)
    logger.error.assert_called_once()


def test_generate_hang_diagnostic_creates_file_for_local_and_no_shared_fs(tmp_path):
    service = MonitorService(make_monitor_config(tmp_path, timeout=120), FakeRunner(), interval=1)
    service._generate_hang_diagnostic("localhost", 0)

    diagnostic_file = os.path.join(service.monitor_log_dir, "host_0_localhost_diagnostic.txt")
    content = open(diagnostic_file, encoding="utf-8").read()
    assert "HangError" in content
    assert "host_0_localhost.output" in content
    assert "2 minutes" in content

    no_shared_service = MonitorService(
        make_monitor_config(tmp_path, no_shared_fs=True, timeout=60),
        FakeRunner(),
        interval=1,
    )
    no_shared_service._generate_hang_diagnostic("worker0", 1)
    no_shared_file = os.path.join(
        no_shared_service.monitor_log_dir, "host_1_worker0_diagnostic.txt"
    )
    assert "host.output" in open(no_shared_file, encoding="utf-8").read()


def test_generate_hang_diagnostic_logs_error_on_write_failure(tmp_path, mocker):
    service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    logger = mocker.patch("flagscale.runner.elastic.monitor_service.logger")
    mocker.patch("builtins.open", side_effect=PermissionError("readonly"))

    service._generate_hang_diagnostic("localhost", 0)

    logger.error.assert_called_once()


def test_check_and_report_hang_single_and_local_modes(tmp_path, mocker):
    single_service = MonitorService(
        make_monitor_config(tmp_path),
        FakeRunner(),
        interval=1,
        host="worker0",
        node_rank=1,
    )
    single_check = mocker.patch.object(single_service, "_check_log_hang", return_value=True)
    single_generate = mocker.patch.object(single_service, "_generate_hang_diagnostic")
    single_service._check_and_report_hang()
    single_check.assert_called_once_with("worker0", 1)
    single_generate.assert_called_once_with("worker0", 1)

    local_service = MonitorService(make_monitor_config(tmp_path), FakeRunner(), interval=1)
    local_check = mocker.patch.object(local_service, "_check_log_hang", return_value=True)
    local_generate = mocker.patch.object(local_service, "_generate_hang_diagnostic")
    local_service._check_and_report_hang()
    local_check.assert_called_once_with("localhost", 0)
    local_generate.assert_called_once_with("localhost", 0)
