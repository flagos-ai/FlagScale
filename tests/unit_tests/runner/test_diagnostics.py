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

from flagscale.runner.diagnostics import (
    active_run_id_cleanup_action,
    active_run_id_cleanup_lines,
    active_run_id_file,
    active_run_id_setup_lines,
    diagnostic_command_body,
    read_active_run_id,
    stop_process_shell_lines,
    wait_for_file_removal_shell_lines,
)


class _DiagnosticConfig:
    enabled = True

    def command_exit_actions(self, node_rank: int) -> list[str]:
        return [f"record-node-{node_rank}"]


def test_active_run_id_path_and_reading(tmp_path):
    path = active_run_id_file(str(tmp_path))
    assert path == str(tmp_path / "diagnostics.active_run_id")

    assert read_active_run_id(path) is None
    (tmp_path / "diagnostics.active_run_id").write_text("run-123\n", encoding="utf-8")
    assert read_active_run_id(path) == "run-123"

    (tmp_path / "diagnostics.active_run_id").write_text("../other-run\n", encoding="utf-8")
    assert read_active_run_id(path) is None


def test_active_run_id_setup_is_atomic_and_node_zero_only(tmp_path):
    path = active_run_id_file(str(tmp_path))
    lines = active_run_id_setup_lines(path, "run-123", 0)

    assert active_run_id_setup_lines(path, "run-123", 1) == []
    assert any("$$.tmp" in line for line in lines)
    assert any("printf" in line and "run-123" in line for line in lines)
    assert any(line.startswith("mv -f") for line in lines)


def test_command_body_preserves_exit_actions_and_clears_matching_active_run(tmp_path):
    path = active_run_id_file(str(tmp_path))
    command = diagnostic_command_body(
        0,
        _DiagnosticConfig(),
        active_run_id_path=path,
        active_run_id="run-123",
    )

    assert "record-node-0" in command
    assert "$(cat" in command
    assert "run-123" in command
    assert "rm -f" in command
    assert "diagnostics.active_run_id" in command
    assert command.endswith("sync; exit \\$rc")


def test_cleanup_only_removes_the_matching_run(tmp_path):
    path = active_run_id_file(str(tmp_path))
    action = active_run_id_cleanup_action(path, "run-123")
    lines = active_run_id_cleanup_lines(path, "run-123")

    assert "$(cat" in action
    assert "run-123" in action
    assert "rm -f" in action
    assert "$(cat" in "\n".join(lines)
    assert "run-123" in "\n".join(lines)


def test_stop_process_validates_pid_waits_and_forces_termination(tmp_path):
    lines = stop_process_shell_lines(
        str(tmp_path / "monitor.pid"),
        "flagscale.runner.heartbeat.monitor",
        timeout_s=1,
        poll_interval_s=0.1,
    )
    script = "\n".join(lines)

    assert '[[ "$diagnostics_pid" =~ ^[0-9]+$ ]]' in script
    assert "flagscale.runner.heartbeat.monitor" in script
    assert "diagnostics_wait<10" in script
    assert "kill -KILL" in script
    assert "rm -f" in script


def test_wait_for_file_removal_is_bounded(tmp_path):
    lines = wait_for_file_removal_shell_lines(
        str(tmp_path / "monitor.pid"),
        timeout_s=1,
        poll_interval_s=0.1,
    )
    script = "\n".join(lines)

    assert "diagnostics_wait<10" in script
    assert "[ ! -f" in script
    assert "sleep 0.1" in script
