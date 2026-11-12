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

"""Linux, run-scoped training ownership and bounded termination.

Workers can create their own sessions, so a process-group kill is not enough.
Only children inheriting a fresh launch token are eligible for termination.
PID start times and pidfds guard against stale PID files and PID reuse.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

_TOKEN_ENV = "FLAGSCALE_TRAINING_PROCESS_TOKEN"
_SCRIPT_PATH = str(Path(__file__).resolve())


def _identity(pid):
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"pid": int(pid), "start_ticks": fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def _alive(identity):
    return bool(identity and _identity(identity["pid"]) == identity)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text())


def _owned(token):
    marker = f"{_TOKEN_ENV}={token}".encode()
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            identity = _identity(int(entry.name))
            if identity and marker in (entry / "environ").read_bytes().split(b"\0"):
                result.append(identity)
        except OSError:
            continue
    return result


def _signal(identity, sig):
    if not _alive(identity):
        return
    # Some Conda Python/glibc builds omit pidfd wrappers despite a recent kernel.
    # These numbers match Linux x86_64 unistd_64.h; fail closed elsewhere.
    native = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
    if not native and platform.machine() != "x86_64":
        raise RuntimeError("This Python/platform needs native pidfd support for safe stopping")
    libc = ctypes.CDLL(None, use_errno=True) if not native else None

    def syscall(number, *arguments):
        value = libc.syscall(ctypes.c_long(number), *arguments)
        if value < 0:
            code = ctypes.get_errno()
            if code == errno.ESRCH:
                raise ProcessLookupError(code, os.strerror(code))
            raise OSError(code, os.strerror(code))
        return value

    try:
        descriptor = os.pidfd_open(identity["pid"]) if native else syscall(434, identity["pid"], 0)
    except ProcessLookupError:
        return
    try:
        if _alive(identity):
            if native:
                signal.pidfd_send_signal(descriptor, sig)
            else:
                syscall(424, descriptor, int(sig), ctypes.c_void_p(), 0)
    except ProcessLookupError:
        pass
    finally:
        os.close(descriptor)


def run(state_file, command):
    if not command:
        raise ValueError("Missing training command")
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    launch_lock = open(str(state_file) + ".lock", "a")
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if Path(state_file).exists():
        previous = _read(state_file)
        if _alive(previous.get("supervisor")) or _owned(previous["token"]):
            raise RuntimeError("A training launch still owns this state file")
    token = uuid.uuid4().hex
    Path(str(state_file) + ".stop_requested").unlink(missing_ok=True)
    state = {
        "version": 1,
        "token": token,
        "hostname": os.uname().nodename,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "supervisor": _identity(os.getpid()),
        "wrapper": _identity(os.getppid()),
        "status": "starting",
    }
    _write(state_file, state)
    # Runner commands start with shell-style NAME=value environment assignments.
    # They are passed as argv by the outer diagnostic shell, not as an executable.
    env = dict(os.environ)
    command = list(command)
    while command and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", command[0]):
        name, value = command.pop(0).split("=", 1)
        env[name] = value
    env[_TOKEN_ENV] = token
    try:
        if not command:
            raise FileNotFoundError(
                errno.ENOENT, "Missing executable after environment assignments"
            )
        child = subprocess.Popen(command, env=env, start_new_session=True)
    except OSError as error:
        exit_code = 127 if error.errno == errno.ENOENT else 126
        state.update(status="exited", training_exit_code=exit_code, startup_error=str(error))
        _write(state_file, state)
        launch_lock.close()
        return exit_code
    state.update(launcher=_identity(child.pid), status="running")
    _write(state_file, state)

    def forward_stop(_signum, _frame):
        for identity in _owned(token):
            _signal(identity, signal.SIGTERM)

    signal.signal(signal.SIGTERM, forward_stop)
    signal.signal(signal.SIGINT, forward_stop)
    return_code = child.wait()
    exit_code = return_code if return_code >= 0 else 128 - return_code
    state.update(status="exited", training_exit_code=exit_code)
    _write(state_file, state)
    launch_lock.close()
    return exit_code


def stop(state_file, term_timeout_s=30.0, kill_timeout_s=5.0):
    """Wait for the wrapper too: it must finish recording diagnostic exit codes."""
    started = time.monotonic()
    result_file = str(state_file) + ".stop.json"
    if not Path(state_file).exists():
        # A legacy PID alone cannot prove ownership. Never use a global pkill.
        result = {"status": "ownership_missing", "confirmed_stopped": False}
        _write(result_file, result)
        return 1
    state = _read(state_file)
    if (
        state.get("version") != 1
        or state.get("hostname") != os.uname().nodename
        or state.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        raise ValueError("Training state does not belong to this host/schema")
    token = state["token"]
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("Invalid training ownership token")
    Path(str(state_file) + ".stop_requested").touch()
    wrapper = state.get("wrapper")
    supervisor = state.get("supervisor")

    def remaining():
        identities = _owned(token)
        for identity in (supervisor, wrapper):
            if _alive(identity) and identity not in identities:
                identities.append(identity)
        return identities

    if Path(result_file).exists():
        previous = _read(result_file)
        if previous.get("token") == token and previous.get("confirmed_stopped") and not remaining():
            print(json.dumps(previous), flush=True)
            return 0

    targets = _owned(token)
    for identity in targets:
        _signal(identity, signal.SIGTERM)
    deadline = time.monotonic() + term_timeout_s
    while remaining() and time.monotonic() < deadline:
        time.sleep(0.1)
    forced = remaining()
    if forced:
        # Kill workers/launcher first; keep supervisor/wrapper to reap and record rc.
        for identity in _owned(token):
            _signal(identity, signal.SIGKILL)
        deadline = time.monotonic() + kill_timeout_s
        while remaining() and time.monotonic() < deadline:
            time.sleep(0.1)
        for identity in remaining():
            _signal(identity, signal.SIGKILL)
        deadline = time.monotonic() + kill_timeout_s
        while remaining() and time.monotonic() < deadline:
            time.sleep(0.1)
    survivors = remaining()
    final_state = _read(state_file)
    result = {
        "status": "stop_failed" if survivors else ("forced" if forced else "terminated"),
        "confirmed_stopped": not survivors,
        "token": token,
        "elapsed_s": round(time.monotonic() - started, 3),
        "training_exit_code": final_state.get("training_exit_code"),
        "term_targets": targets,
        "forced_targets": forced,
        "remaining": survivors,
    }
    _write(result_file, result)
    print(json.dumps(result), flush=True)
    return int(bool(survivors))


def _timeouts(runner):
    values = []
    for key, default in (("stop_timeout_s", 30.0), ("stop_kill_timeout_s", 5.0)):
        value = float(runner.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"runner.{key} must be finite and greater than zero")
        values.append(value)
    return values


def tracked_command_body(command_body, pid_file):
    # Retain the existing outer Bash diagnostic wrapper and its exact exit actions.
    command = shlex.join([sys.executable, _SCRIPT_PATH, "run", "--state-file"])
    command += " " + shlex.quote(str(pid_file) + ".training.json") + " -- $cmd"
    if command_body == "$cmd; sync":
        return f"{command}; rc=\\$?; sync; exit \\$rc"
    return command_body.replace("$cmd", command, 1)


def stop_shell_lines(pid_file, runner):
    term, kill = _timeouts(runner)
    command = shlex.join(
        [
            sys.executable,
            _SCRIPT_PATH,
            "stop",
            "--state-file",
            str(pid_file) + ".training.json",
            "--term-timeout",
            str(term),
            "--kill-timeout",
            str(kill),
        ]
    )
    return [
        command,
        "training_stop_rc=$?",
        '[ "$training_stop_rc" -eq 0 ] || exit "$training_stop_rc"',
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "stop"))
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--term-timeout", type=float, default=30)
    parser.add_argument("--kill-timeout", type=float, default=5)
    args, command = parser.parse_known_args()
    if command[:1] == ["--"]:
        command = command[1:]
    if args.action == "run":
        return run(args.state_file, command)
    if command:
        parser.error("Unexpected stop arguments")
    _timeouts({"stop_timeout_s": args.term_timeout, "stop_kill_timeout_s": args.kill_timeout})
    return stop(args.state_file, args.term_timeout, args.kill_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
