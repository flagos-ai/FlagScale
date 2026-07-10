from omegaconf import OmegaConf

from flagscale.runner.launcher.launcher_ssh import _get_runner_cmd_train


def test_get_runner_cmd_train_strips_perf_monitor_runner_keys():
    config = OmegaConf.create(
        {
            "experiment": {
                "runner": {
                    "backend": "torchrun",
                    "nnodes": 1,
                    "nproc_per_node": 8,
                    "rdzv_backend": "static",
                    "enable_perf_monitor": True,
                    "perf_log_interval": 5,
                    "perf_log_dir": "/tmp/perf_monitor",
                    "perf_console_output": True,
                }
            },
            "train": {
                "system": {
                    "logging": {
                        "details_dir": "/tmp/details",
                    }
                }
            },
        }
    )

    cmd = _get_runner_cmd_train("localhost", "127.0.0.1", 29500, 1, 0, 8, config)

    assert cmd[0] == "torchrun"
    assert "--enable_perf_monitor" not in cmd
    assert "--perf_log_interval" not in cmd
    assert "--perf_log_dir" not in cmd
    assert "--perf_console_output" not in cmd
    assert "--log_dir" in cmd
    assert "--rdzv_endpoint" in cmd


def test_get_runner_cmd_train_strips_hipprof_runner_keys():
    config = OmegaConf.create(
        {
            "experiment": {
                "runner": {
                    "backend": "torchrun",
                    "nnodes": 1,
                    "nproc_per_node": 8,
                    "rdzv_backend": "static",
                    "hipprof_bin_path": "/opt/dtk/bin/hipprof",
                    "hipprof_output_dir": "/tmp/hipprof",
                }
            },
            "train": {
                "system": {
                    "logging": {
                        "details_dir": "/tmp/details",
                    }
                },
                "model": {
                    "profile": True,
                    "use_hipprof_profiler": True,
                },
            },
        }
    )

    cmd = _get_runner_cmd_train("localhost", "127.0.0.1", 29500, 1, 0, 8, config)

    assert "--hipprof_bin_path" not in cmd
    assert "--hipprof_output_dir" not in cmd
