import ast
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from flagscale.train.megatron.hipprof import hipprof_session_control


class HipprofSessionControlTests(unittest.TestCase):
    @patch("flagscale.train.megatron.hipprof.subprocess.run")
    def test_start_uses_session_client_from_wrapper_environment(self, run):
        with patch.dict(
            os.environ,
            {
                "HIPPROF_SESSION_ID": "session-7",
                "HIPPROF_BIN_PATH": "/opt/dtk/bin/hipprof",
            },
            clear=True,
        ):
            hipprof_session_control("start")

        run.assert_called_once_with(
            ["/opt/dtk/bin/hipprof", "--session-client", "session-7", "--start"],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_missing_session_points_to_yaml(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "experiment.runner"),
        ):
            hipprof_session_control("stop")

    def test_invalid_action_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            hipprof_session_control("flush")

    @patch("flagscale.train.megatron.hipprof.subprocess.run")
    def test_control_failure_includes_output(self, run):
        run.side_effect = subprocess.CalledProcessError(
            9,
            ["hipprof"],
            output="out text",
            stderr="err text",
        )
        with (
            patch.dict(
                os.environ,
                {"HIPPROF_SESSION_ID": "session-9"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "out text") as error,
        ):
            hipprof_session_control("start")
        self.assertIn("err text", str(error.exception))


class HipprofConfigOwnershipTests(unittest.TestCase):
    def test_runtime_connection_fields_are_not_model_arguments(self):
        root = Path(__file__).resolve().parents[3]
        source = root / "flagscale/train/megatron/training/config/common_config.py"
        tree = ast.parse(source.read_text())
        profiling = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ProfilingConfig"
        )
        fields = {
            node.target.id
            for node in profiling.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

        self.assertIn("use_hipprof_profiler", fields)
        self.assertNotIn("hipprof_bin_path", fields)
        self.assertNotIn("hipprof_session_id", fields)


if __name__ == "__main__":
    unittest.main()
