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
