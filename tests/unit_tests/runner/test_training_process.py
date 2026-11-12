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

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flagscale.runner import training_process as lifecycle


def _wait(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("Test process did not reach the requested state")


def _start(tmp_path, program):
    state = tmp_path / "training.json"
    completed = tmp_path / "exit-code"
    command = shlex.join(
        [
            sys.executable,
            lifecycle.__file__,
            "run",
            "--state-file",
            str(state),
            "--",
            sys.executable,
            "-c",
            program,
        ]
    )
    wrapper = subprocess.Popen(
        ["bash", "-c", f"{command}; echo $? > {shlex.quote(str(completed))}"],
        start_new_session=True,
    )
    _wait(
        lambda: state.exists()
        and json.loads(state.read_text()).get("status") in {"running", "exited"}
    )
    return wrapper, state, completed


@pytest.mark.parametrize("ignore_term", [False, True])
def test_bounded_stop_reaps_detached_workers_without_harming_other_jobs(tmp_path, ignore_term):
    ready = tmp_path / "ready"
    worker_ready = tmp_path / "worker-ready"
    worker = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(worker_ready)!r},'w').close(); time.sleep(120)"
    )
    program = (
        "import signal,subprocess,sys,time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + f"subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session=True); "
        + f"open({str(ready)!r},'w').close(); time.sleep(120)"
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(120)"], start_new_session=True
    )
    wrapper, state, completed = _start(tmp_path, program)
    try:
        _wait(ready.exists)
        _wait(worker_ready.exists)
        code = lifecycle.stop(state, term_timeout_s=0.4, kill_timeout_s=2)
        wrapper.wait(timeout=3)
        result = json.loads(Path(str(state) + ".stop.json").read_text())
        assert code == 0
        assert result["confirmed_stopped"]
        assert result["remaining"] == []
        assert result["status"] == "forced"  # Detached worker deliberately ignores TERM.
        assert result["elapsed_s"] < 6
        assert unrelated.poll() is None
        assert completed.read_text().strip() in {"137", "143"}
        assert lifecycle.stop(state, 0.2, 1) == 0  # Repeated stop is harmless.
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)
        if wrapper.poll() is None:
            lifecycle.stop(state, 0.1, 1)
            wrapper.wait(timeout=3)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_normal_and_abnormal_exit_codes_are_preserved(tmp_path, exit_code):
    wrapper, state, completed = _start(tmp_path, f"raise SystemExit({exit_code})")
    wrapper.wait(timeout=5)
    assert completed.read_text().strip() == str(exit_code)
    assert json.loads(state.read_text())["training_exit_code"] == exit_code
    assert lifecycle.stop(state, 0.2, 1) == 0


def test_missing_ownership_fails_closed(tmp_path):
    assert lifecycle.stop(tmp_path / "missing.json", 0.1, 0.1) == 1


def test_runner_environment_assignments_reach_the_training_process(tmp_path):
    state = tmp_path / "environment.json"
    program = (
        "import os; "
        "assert os.environ['FLAGSCALE_TEST_VALUE'] == 'value with spaces=ok'; "
        f"assert os.environ[{lifecycle._TOKEN_ENV!r}] != 'untrusted'"
    )
    result = subprocess.run(
        [
            sys.executable,
            lifecycle.__file__,
            "run",
            "--state-file",
            str(state),
            "--",
            "FLAGSCALE_TEST_VALUE=value with spaces=ok",
            f"{lifecycle._TOKEN_ENV}=untrusted",
            sys.executable,
            "-c",
            program,
        ],
        timeout=10,
    )
    assert result.returncode == 0
    assert json.loads(state.read_text())["training_exit_code"] == 0


@pytest.mark.parametrize("command", [["/nonexistent/flagscale-training"], ["ONLY_ENV=value"]])
def test_startup_failure_records_a_terminal_state(tmp_path, command):
    state = tmp_path / "startup.json"
    result = subprocess.run(
        [sys.executable, lifecycle.__file__, "run", "--state-file", str(state), "--", *command],
        timeout=10,
    )
    assert result.returncode == 127
    actual = json.loads(state.read_text())
    assert actual["status"] == "exited"
    assert actual["training_exit_code"] == 127
    assert actual["startup_error"]


def test_reused_pid_identity_is_never_signalled(tmp_path):
    state = tmp_path / "stale.json"
    identity = lifecycle._identity(os.getpid())
    identity["start_ticks"] = "0"
    lifecycle._write(
        state,
        dict(
            version=1,
            hostname=os.uname().nodename,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            token="0" * 32,
            supervisor=identity,
            wrapper=identity,
            status="exited",
        ),
    )
    assert lifecycle.stop(state, 0.1, 0.1) == 0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_wait_time_is_rejected(value):
    with pytest.raises(ValueError):
        lifecycle._timeouts({"stop_timeout_s": value})
