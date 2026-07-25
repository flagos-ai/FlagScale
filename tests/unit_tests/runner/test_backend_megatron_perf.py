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
import sys
import tempfile
import types
from unittest.mock import patch

from omegaconf import OmegaConf

hydra_module = types.ModuleType("hydra")
hydra_core_module = types.ModuleType("hydra.core")
hydra_config_module = types.ModuleType("hydra.core.hydra_config")


class _HydraConfig:
    @staticmethod
    def get():
        raise RuntimeError("HydraConfig.get() is not expected in this test")


hydra_config_module.HydraConfig = _HydraConfig
sys.modules.setdefault("hydra", hydra_module)
sys.modules.setdefault("hydra.core", hydra_core_module)
sys.modules.setdefault("hydra.core.hydra_config", hydra_config_module)

from flagscale.runner.backend.backend_megatron import MegatronBackend


def _make_config():
    return OmegaConf.create(
        {
            "experiment": {
                "exp_dir": "/tmp/test_exp",
                "task": {
                    "type": "train",
                    "backend": "megatron",
                    "entrypoint": "flagscale/train/megatron/train_gpt.py",
                },
                "runner": {
                    "hostfile": None,
                    "enable_perf_monitor": True,
                    "perf_log_interval": 5,
                },
                "envs": {},
            },
            "train": {
                "system": {
                    "checkpoint": {
                        "save": "/tmp/test_exp/checkpoints",
                        "load": "/tmp/test_exp/checkpoints",
                    },
                    "logging": {
                        "log_dir": "/tmp/test_exp/logs",
                        "scripts_dir": "/tmp/test_exp/logs/scripts",
                        "pids_dir": "/tmp/test_exp/logs/pids",
                        "details_dir": "/tmp/test_exp/logs/details",
                        "tensorboard_dir": "/tmp/test_exp/tensorboard",
                        "wandb_save_dir": "/tmp/test_exp/wandb",
                        "straggler_dir": "/tmp/test_exp/logs/straggler",
                    },
                    "straggler_log_dir": "/tmp/test_exp/logs/straggler",
                },
                "model": {},
                "data": {},
            },
        }
    )


def test_backend_prepare_copies_perf_monitor_config_from_runner():
    config = _make_config()
    with (
        patch("flagscale.runner.backend.backend_megatron._get_args_megatron", return_value=[]),
        patch("flagscale.runner.backend.backend_megatron._update_config_train"),
        patch("flagscale.runner.backend.backend_megatron.parse_hostfile", return_value=None),
        patch("flagscale.runner.backend.backend_megatron.logger"),
    ):
        backend = MegatronBackend(config)

    assert backend.config.train.system.enable_perf_monitor is True
    assert backend.config.train.system.perf_log_interval == 5
    assert backend.config.train.system.perf_log_dir == "/tmp/test_exp/logs/perf_monitor"


def test_generate_run_script_creates_perf_log_dir():
    config = _make_config()
    with (
        patch("flagscale.runner.backend.backend_megatron._get_args_megatron", return_value=[]),
        patch("flagscale.runner.backend.backend_megatron._update_config_train"),
        patch("flagscale.runner.backend.backend_megatron.parse_hostfile", return_value=None),
        patch("flagscale.runner.backend.backend_megatron.logger"),
    ):
        backend = MegatronBackend(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        config.train.system.logging.scripts_dir = os.path.join(tmpdir, "scripts")
        config.train.system.logging.log_dir = os.path.join(tmpdir, "logs")
        config.train.system.logging.pids_dir = os.path.join(tmpdir, "pids")
        config.train.system.perf_log_dir = os.path.join(tmpdir, "logs", "perf_monitor")

        with (
            patch("os.path.exists", return_value=True),
            patch("flagscale.runner.backend.backend_megatron.get_pkg_dir", return_value=tmpdir),
        ):
            script_path = backend.generate_run_script(
                config, "localhost", 0, "python train.py", background=True
            )

        with open(script_path, "r") as file_obj:
            content = file_obj.read()

        assert f"mkdir -p {config.train.system.perf_log_dir}" in content
