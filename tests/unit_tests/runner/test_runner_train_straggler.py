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
