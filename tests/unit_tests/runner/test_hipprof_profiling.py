import unittest

from flagscale.runner.profiling import (
    configure_hipprof_env,
    remove_launcher_profiling_args,
)


class HipprofEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.runner = {
            "hipprof_bin_path": "/opt/dtk/bin",
            "hipprof_output_dir": "/tmp/hipprof",
        }
        self.model = {
            "profile": True,
            "use_hipprof_profiler": True,
        }

    def test_injects_wrapper_and_preserves_custom_python(self):
        result = configure_hipprof_env(
            self.runner,
            self.model,
            {"PYTHON_EXEC": "/venv/bin/python"},
        )

        self.assertEqual(
            result["PYTHON_EXEC"],
            "tools/profiling/hipprof_python_wrapper.sh",
        )
        self.assertEqual(result["HIPPROF_REAL_PYTHON"], "/venv/bin/python")
        self.assertEqual(result["HIPPROF_BIN_PATH"], "/opt/dtk/bin/hipprof")
        self.assertEqual(result["HIPPROF_OUTPUT_DIR"], "/tmp/hipprof")

    def test_accepts_full_executable_path(self):
        self.runner["hipprof_bin_path"] = "/opt/dtk/bin/hipprof"

        result = configure_hipprof_env(self.runner, self.model, {})

        self.assertEqual(result["HIPPROF_BIN_PATH"], "/opt/dtk/bin/hipprof")

    def test_disabled_configuration_is_unchanged(self):
        result = configure_hipprof_env({}, {}, {"TOKEN": "value"})

        self.assertEqual(result, {"TOKEN": "value"})

    def test_launcher_only_profiling_arguments_are_removed(self):
        runner_args = {
            "nsys_bin_path": "/opt/nsight/bin/nsys",
            "nsys_rep_file_path": "/tmp/nsys",
            "hipprof_bin_path": "/opt/dtk/bin/hipprof",
            "hipprof_output_dir": "/tmp/hipprof",
            "nproc_per_node": 8,
        }

        remove_launcher_profiling_args(runner_args)

        self.assertEqual(runner_args, {"nproc_per_node": 8})

    def test_rejects_invalid_configurations(self):
        cases = [
            ({"hipprof_bin_path": "/opt/dtk/bin"}, self.model, "requires both"),
            (
                self.runner,
                {"profile": True, "use_hipprof_profiler": False},
                "use_hipprof_profiler: true",
            ),
            (
                self.runner,
                {"profile": False, "use_hipprof_profiler": True},
                "profile: true",
            ),
            (
                {
                    **self.runner,
                    "nsys_bin_path": "/opt/nsight/bin/nsys",
                    "nsys_rep_file_path": "/tmp/nsys",
                },
                self.model,
                "cannot be enabled together",
            ),
        ]
        for runner, model, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    configure_hipprof_env(runner, model, {})


if __name__ == "__main__":
    unittest.main()
