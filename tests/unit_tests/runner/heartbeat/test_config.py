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

import os

import pytest
from omegaconf import OmegaConf

from flagscale.runner.heartbeat.config import prepare_heartbeat_launch_config


def _config(tmp_path, heartbeat):
    return OmegaConf.create(
        {
            "experiment": {"runner": {"heartbeat": heartbeat}},
            "train": {"system": {"logging": {"log_dir": str(tmp_path / "logs")}}},
        }
    )


def test_disabled_heartbeat_is_a_noop(tmp_path):
    resolved = prepare_heartbeat_launch_config(_config(tmp_path, {"enabled": False}), "run")
    assert resolved.enabled is False
    assert resolved.shell_setup_lines(0) == []


def test_cloud_runner_rejects_enabled_heartbeat(tmp_path):
    config = _config(tmp_path, {"enabled": True})
    config.experiment.runner.type = "cloud"

    with pytest.raises(NotImplementedError, match="CloudTrainRunner"):
        prepare_heartbeat_launch_config(config, "run")


def test_enabled_gpu_progress_heartbeat_has_no_preload_or_nccl_dependency(tmp_path):
    resolved = prepare_heartbeat_launch_config(
        _config(
            tmp_path,
            {
                "enabled": True,
                "publish_interval_s": 2,
                "initial_process_timeout_s": 10,
                "process_timeout_s": 10,
                "initial_progress_timeout_s": 60,
                "progress_timeout_s": 30,
                "checkpoint_timeout_s": 120,
                "failure_grace_period_s": 10,
            },
        ),
        "run-123",
    )
    shell = "\n".join(resolved.shell_setup_lines(0))

    assert resolved.heartbeat_dir == os.path.join(str(tmp_path / "logs"), "heartbeat", "run-123")
    assert "FLAGSCALE_GPU_HEARTBEAT_ENABLE=1" in shell
    assert "flagscale.runner.heartbeat.monitor" in shell
    assert "LD_PRELOAD" not in shell
    assert "libflagscale" not in shell
    assert "nccl" not in shell.lower()

    command_body = resolved.training_command_body(0)
    assert command_body.startswith("$cmd; rc=\\$?")
    assert resolved.completion_file in command_body
    assert resolved.training_command_body(1) == "$cmd; sync"


def test_optional_hardware_health_starts_one_node_local_cpu_collector(tmp_path):
    config = _config(
        tmp_path,
        {
            "enabled": True,
            "hardware_health": {
                "enabled": True,
                "interval_s": 60,
                "command_timeout_s": 10,
                "stale_after_s": 180,
            },
        },
    )
    config.experiment.runner.nnodes = 2
    resolved = prepare_heartbeat_launch_config(config, "run")
    node_zero = "\n".join(resolved.shell_setup_lines(0))
    node_one = "\n".join(resolved.shell_setup_lines(1))

    assert "flagscale.runner.heartbeat.gpu_health" in node_zero
    assert "gpu_health_node_0.json" in node_zero
    assert "--hardware-health-enabled" in node_zero
    assert "--expected-node-count 2" in node_zero
    assert resolved.expected_node_count == 2
    assert "flagscale.runner.heartbeat.gpu_health" in node_one
    assert "gpu_health_node_1.json" in node_one
    assert "flagscale.runner.heartbeat.monitor" not in node_one
    assert "CUDA" not in node_zero
    assert "gpu_health_node_0.pid" in resolved.training_command_body(0)
    assert "gpu_health_node_1.pid" in resolved.training_command_body(1)
    assert '\\"\\$(cat ' in resolved.training_command_body(0)


def test_hardware_health_command_timeout_must_be_less_than_interval(tmp_path):
    config = _config(
        tmp_path,
        {
            "enabled": True,
            "hardware_health": {
                "enabled": True,
                "interval_s": 10,
                "command_timeout_s": 10,
            },
        },
    )
    with pytest.raises(ValueError, match="command_timeout_s"):
        prepare_heartbeat_launch_config(config, "run")


def test_process_timeout_must_exceed_publish_interval(tmp_path):
    config = _config(
        tmp_path,
        {
            "enabled": True,
            "publish_interval_s": 5,
            "process_timeout_s": 5,
        },
    )
    with pytest.raises(ValueError, match="greater than publish_interval_s"):
        prepare_heartbeat_launch_config(config, "run")


def test_checkpoint_timeout_covers_regular_progress_timeout(tmp_path):
    config = _config(
        tmp_path,
        {
            "enabled": True,
            "progress_timeout_s": 60,
            "checkpoint_timeout_s": 30,
        },
    )
    with pytest.raises(ValueError, match="checkpoint_timeout_s"):
        prepare_heartbeat_launch_config(config, "run")


def test_no_shared_filesystem_is_rejected(tmp_path):
    config = _config(tmp_path, {"enabled": True})
    config.experiment.runner.no_shared_fs = True
    with pytest.raises(ValueError, match="shared filesystem"):
        prepare_heartbeat_launch_config(config, "run")


def test_backend_without_training_progress_hook_is_rejected(tmp_path):
    config = _config(tmp_path, {"enabled": True})
    config.experiment.task = {"backend": "native"}
    with pytest.raises(ValueError, match="Megatron backend"):
        prepare_heartbeat_launch_config(config, "run")
