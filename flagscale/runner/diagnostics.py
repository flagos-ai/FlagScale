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

"""Shared lifecycle helpers for opt-in runner diagnostics."""

from __future__ import annotations

import math
import os
import re
import shlex
from typing import Protocol

ACTIVE_RUN_ID_FILENAME = "diagnostics.active_run_id"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class DiagnosticLaunchConfig(Protocol):
    enabled: bool

    def command_exit_actions(self, node_rank: int) -> list[str]: ...


def active_run_id_file(pids_dir: str) -> str:
    """Return the stable file used to find the currently active diagnostic run."""

    return os.path.join(str(pids_dir), ACTIVE_RUN_ID_FILENAME)


def read_active_run_id(path: str) -> str | None:
    """Read and validate the active diagnostic run id."""

    try:
        with open(path, encoding="utf-8") as file_obj:
            run_id = file_obj.read().strip()
    except OSError:
        return None

    if not _RUN_ID_PATTERN.fullmatch(run_id):
        return None
    return run_id


def active_run_id_setup_lines(path: str, run_id: str, node_rank: int) -> list[str]:
    """Atomically publish one run id from node zero when its run script starts."""

    if node_rank != 0:
        return []
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid diagnostics run id: {run_id!r}")

    qpath = shlex.quote(path)
    qdir = shlex.quote(os.path.dirname(path))
    qrun_id = shlex.quote(run_id)
    return [
        f"mkdir -p {qdir}",
        f"diagnostics_run_id_tmp={qpath}.$$.tmp",
        f"printf '%s\\n' {qrun_id} > \"$diagnostics_run_id_tmp\"",
        f'mv -f "$diagnostics_run_id_tmp" {qpath}',
    ]


def active_run_id_cleanup_action(path: str, run_id: str) -> str:
    """Remove the active marker only when it still belongs to this run."""

    qpath = shlex.quote(path)
    qrun_id = shlex.quote(run_id)
    return f'if [ -f {qpath} ] && [ \\"\\$(cat {qpath})\\" = {qrun_id} ]; then rm -f {qpath}; fi'


def active_run_id_cleanup_lines(path: str, run_id: str) -> list[str]:
    """Return direct shell lines that clear matching active state during stop."""

    qpath = shlex.quote(path)
    qrun_id = shlex.quote(run_id)
    return [
        f'if [ -f {qpath} ] && [ "$(cat {qpath})" = {qrun_id} ]; then',
        f"    rm -f {qpath}",
        "fi",
    ]


def diagnostic_command_body(
    node_rank: int,
    *configs: DiagnosticLaunchConfig,
    active_run_id_path: str = "",
    active_run_id: str = "",
) -> str:
    """Write monitor completion markers and clear active state on normal exit."""

    actions = [
        action
        for config in configs
        if config.enabled
        for action in config.command_exit_actions(node_rank)
    ]
    if node_rank == 0 and active_run_id_path and active_run_id:
        actions.append(active_run_id_cleanup_action(active_run_id_path, active_run_id))
    if not actions:
        return "$cmd; sync"
    return f"$cmd; rc=\\$?; {'; '.join(dict.fromkeys(actions))}; sync; exit \\$rc"


def stop_process_shell_lines(
    pid_file: str,
    process_marker: str,
    *,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.1,
) -> list[str]:
    """Safely terminate one recorded diagnostic process with a bounded wait."""

    attempts = max(1, math.ceil(timeout_s / poll_interval_s))
    qpid_file = shlex.quote(pid_file)
    qmarker = shlex.quote(process_marker)
    return [
        f"if [ -f {qpid_file} ]; then",
        f'    diagnostics_pid="$(cat {qpid_file})"',
        '    if [[ "$diagnostics_pid" =~ ^[0-9]+$ ]] && '
        f'ps -p "$diagnostics_pid" -o args= 2>/dev/null | grep -F -- {qmarker} >/dev/null; then',
        '        kill "$diagnostics_pid" 2>/dev/null || true',
        f"        for ((diagnostics_wait=0; diagnostics_wait<{attempts}; diagnostics_wait++)); do",
        '            kill -0 "$diagnostics_pid" 2>/dev/null || break',
        f"            sleep {poll_interval_s:g}",
        "        done",
        '        if kill -0 "$diagnostics_pid" 2>/dev/null; then',
        '            kill -KILL "$diagnostics_pid" 2>/dev/null || true',
        "        fi",
        "    fi",
        f"    rm -f {qpid_file}",
        "fi",
    ]


def wait_for_file_removal_shell_lines(
    path: str,
    *,
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.1,
) -> list[str]:
    """Wait for node zero to finish stopping a shared diagnostic process."""

    attempts = max(1, math.ceil(timeout_s / poll_interval_s))
    qpath = shlex.quote(path)
    return [
        f"for ((diagnostics_wait=0; diagnostics_wait<{attempts}; diagnostics_wait++)); do",
        f"    [ ! -f {qpath} ] && break",
        f"    sleep {poll_interval_s:g}",
        "done",
    ]
