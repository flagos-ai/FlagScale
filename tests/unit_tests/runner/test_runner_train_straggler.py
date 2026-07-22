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
import types

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

from flagscale.runner.runner_train import _update_config_train


def test_update_config_train_sets_default_straggler_dirs(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {
                "exp_dir": str(tmp_path / "exp"),
                "runner": {},
                "task": {"backend": "megatron"},
            },
            "train": {
                "system": {
                    "checkpoint": {},
                    "logging": {},
                },
                "model": {},
                "data": {},
            },
        }
    )

    _update_config_train(config)

    assert config.train.system.logging.straggler_dir == os.path.join(
        config.train.system.logging.log_dir, "straggler"
    )
    assert config.train.system.straggler_log_dir == config.train.system.logging.straggler_dir
