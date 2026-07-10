import os
import re
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "tools/profiling/hipprof_python_wrapper.sh"


class HipprofWrapperTests(unittest.TestCase):
    def _write_executable(self, path, body):
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
        path.chmod(0o755)

    def test_unselected_rank_bypasses_hipprof(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            log = temp / "events.log"
            fake_python = temp / "python"
            fake_hipprof = temp / "hipprof"
            self._write_executable(
                fake_python,
                'printf "python_args=%s\\n" "$*" >> "$HIPPROF_TEST_LOG"\n',
            )
            self._write_executable(
                fake_hipprof,
                'printf "hipprof_called\\n" >> "$HIPPROF_TEST_LOG"\nexit 99\n',
            )
            env = os.environ.copy()
            env.update(
                {
                    "RANK": "3",
                    "LOCAL_RANK": "1",
                    "HIPPROF_REAL_PYTHON": str(fake_python),
                    "HIPPROF_BIN_PATH": str(fake_hipprof),
                    "HIPPROF_TEST_LOG": str(log),
                }
            )

            result = subprocess.run(
                [
                    str(WRAPPER),
                    "-u",
                    "train.py",
                    "--profile-ranks",
                    "2",
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            events = log.read_text()
            self.assertIn("python_args=-u train.py --profile-ranks 2", events)
            self.assertNotIn("hipprof_called", events)

    def test_selected_rank_launches_trace_off_with_matching_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            log = temp / "events.log"
            output = temp / "output"
            fake_python = temp / "python"
            fake_hipprof = temp / "hipprof"
            self._write_executable(
                fake_python,
                (
                    'printf "python_session=%s\\n" "$HIPPROF_SESSION_ID" '
                    '>> "$HIPPROF_TEST_LOG"\n'
                    'printf "python_args=%s\\n" "$*" >> "$HIPPROF_TEST_LOG"\n'
                ),
            )
            self._write_executable(
                fake_hipprof,
                (
                    'printf "hipprof_args=%s\\n" "$*" >> "$HIPPROF_TEST_LOG"\n'
                    'while [[ "$#" -gt 0 ]]; do\n'
                    '  if [[ "$1" == "$HIPPROF_REAL_PYTHON" ]]; then\n'
                    "    shift\n"
                    '    exec "$HIPPROF_REAL_PYTHON" "$@"\n'
                    "  fi\n"
                    "  shift\n"
                    "done\n"
                    "exit 3\n"
                ),
            )
            env = os.environ.copy()
            env.update(
                {
                    "RANK": "2",
                    "LOCAL_RANK": "0",
                    "HOSTNAME": "testhost",
                    "HIPPROF_REAL_PYTHON": str(fake_python),
                    "HIPPROF_BIN_PATH": str(fake_hipprof),
                    "HIPPROF_OUTPUT_DIR": str(output),
                    "HIPPROF_TRACE": "HIP,HSA",
                    "HIPPROF_GROUP_STREAM": "1",
                    "HIPPROF_SEGMENT_SIZE": "6000",
                    "HIPPROF_SESSION_ID": "stale-shared-session",
                    "HIPPROF_TEST_LOG": str(log),
                }
            )

            result = subprocess.run(
                [
                    str(WRAPPER),
                    "-u",
                    "train.py",
                    "--profile-ranks",
                    "2",
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            events = log.read_text()
            launch = next(
                line for line in events.splitlines() if line.startswith("hipprof_args=")
            )
            session_match = re.search(r"--session ([^ ]+)", launch)
            self.assertIsNotNone(session_match)
            session = session_match.group(1)
            self.assertNotEqual(session, "stale-shared-session")
            self.assertIn("_rank2_", session)
            self.assertIn(f"python_session={session}", events)
            for argument in (
                "--trace-off",
                "--hip-trace",
                "--hsa-trace",
                "--group-stream",
                "--segment-size 6000",
                "-d ",
                "-o ",
            ):
                self.assertIn(argument, launch)
            self.assertNotIn("--rccl-trace", launch)

    def test_wrapper_exec_replaces_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            output = temp / "output"
            pid_file = temp / "training.pid"
            fake_python = temp / "python"
            fake_hipprof = temp / "hipprof"
            self._write_executable(
                fake_python,
                'printf "%s\\n" "$$" > "$HIPPROF_PID_FILE"\nexec sleep 30\n',
            )
            self._write_executable(
                fake_hipprof,
                (
                    'while [[ "$#" -gt 0 ]]; do\n'
                    '  if [[ "$1" == "$HIPPROF_REAL_PYTHON" ]]; then\n'
                    "    shift\n"
                    '    exec "$HIPPROF_REAL_PYTHON" "$@"\n'
                    "  fi\n"
                    "  shift\n"
                    "done\n"
                    "exit 3\n"
                ),
            )
            env = os.environ.copy()
            env.update(
                {
                    "RANK": "0",
                    "LOCAL_RANK": "0",
                    "HIPPROF_REAL_PYTHON": str(fake_python),
                    "HIPPROF_BIN_PATH": str(fake_hipprof),
                    "HIPPROF_OUTPUT_DIR": str(output),
                    "HIPPROF_PID_FILE": str(pid_file),
                }
            )
            process = subprocess.Popen(
                [str(WRAPPER), "train.py", "--profile-ranks", "0"],
                env=env,
            )
            training_pid = None
            try:
                deadline = time.time() + 5
                while time.time() < deadline and not pid_file.exists():
                    time.sleep(0.05)
                self.assertTrue(pid_file.exists(), "fake training did not start")
                training_pid = int(pid_file.read_text().strip())
                self.assertEqual(training_pid, process.pid)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                if training_pid is not None and training_pid != process.pid:
                    try:
                        os.kill(training_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
