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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from flagscale.runner.launcher.launcher_ssh import SshLauncher, _get_runner_cmd_train


def _launcher():
    launcher = SshLauncher.__new__(SshLauncher)
    launcher.task_type = "train"
    launcher.resources = None
    launcher.backend = SimpleNamespace(
        generate_stop_script=Mock(return_value="/tmp/task logs/stop.sh"),
        active_run_id_file="/tmp/task/pids/diagnostics.active_run_id",
        diagnostics_run_id="run-1",
    )
    launcher.config = OmegaConf.create(
        dict(
            experiment=dict(
                runner=dict(
                    ssh_port=22, no_shared_fs=False, stop_timeout_s=30, stop_kill_timeout_s=5
                )
            ),
            train=dict(
                system=dict(logging=dict(scripts_dir="/tmp/task logs", details_dir="/tmp/details"))
            ),
        )
    )
    return launcher


@pytest.mark.parametrize("return_code", [0, 1])
def test_ssh_stop_waits_and_propagates_failure(return_code):
    launcher = _launcher()
    with patch("flagscale.runner.launcher.launcher_ssh.run_ssh_command") as ssh:
        ssh.return_value = SimpleNamespace(
            returncode=return_code, stdout="result", stderr="failure"
        )
        if return_code:
            with pytest.raises(RuntimeError, match="Training stop failed"):
                launcher._stop_each("worker", 0)
        else:
            launcher._stop_each("worker", 0)
        assert ssh.call_args.kwargs == dict(background=False, query=True)
        assert "'/tmp/task logs/stop.sh'" in ssh.call_args.args[1]


def test_active_state_is_not_cleared_on_stop_failure():
    launcher = _launcher()
    launcher._stop_each = Mock(side_effect=RuntimeError("still alive"))
    launcher._clear_stopped_run = Mock()
    with pytest.raises(RuntimeError):
        launcher.stop()
    launcher._clear_stopped_run.assert_not_called()


def test_active_state_is_cleared_only_after_confirmed_stop():
    launcher = _launcher()
    calls = []
    launcher._stop_each = lambda *_: calls.append("confirmed exit")
    launcher._clear_stopped_run = lambda: calls.append("clear active")
    launcher.stop()
    assert calls == ["confirmed exit", "clear active"]


def test_stop_wait_settings_do_not_leak_to_torchrun():
    command = _get_runner_cmd_train("localhost", "127.0.0.1", 29500, 1, 0, 1, _launcher().config)
    assert "--stop_timeout_s" not in command
    assert "--stop_kill_timeout_s" not in command
