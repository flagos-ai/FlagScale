import unittest

from omegaconf import OmegaConf
from pydantic import ValidationError

from flagscale.train.train_config import (
    CheckpointConfig,
    DataConfig,
    FreezeConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    SystemConfig,
    TrainConfig,
)


class TestFreezeConfig(unittest.TestCase):
    """Test FreezeConfig"""

    def test_defaults_are_none(self):
        config = FreezeConfig()
        self.assertIsNone(config.freeze_patterns)
        self.assertIsNone(config.keep_patterns)

    def test_custom_patterns(self):
        config = FreezeConfig(freeze_patterns=["backbone.*"], keep_patterns=["backbone.layer4.*"])
        self.assertEqual(config.freeze_patterns, ["backbone.*"])
        self.assertEqual(config.keep_patterns, ["backbone.layer4.*"])


class TestOptimizerConfig(unittest.TestCase):
    """Test OptimizerConfig validation and defaults"""

    def test_default_values(self):
        config = OptimizerConfig()
        self.assertEqual(config.name, "AdamW")
        self.assertIsNone(config.lr)
        self.assertIsNone(config.betas)
        self.assertIsNone(config.eps)
        self.assertIsNone(config.weight_decay)
        self.assertIsNone(config.param_groups)

    def test_custom_values(self):
        config = OptimizerConfig(
            name="SGD", lr=1e-4, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.001
        )
        self.assertEqual(config.name, "SGD")
        self.assertEqual(config.lr, 1e-4)
        self.assertEqual(config.betas, (0.8, 0.9))
        self.assertEqual(config.eps, 1e-6)
        self.assertEqual(config.weight_decay, 0.001)

    def test_betas_list_converted_to_tuple(self):
        config = OptimizerConfig(betas=[0.9, 0.95])
        self.assertIsInstance(config.betas, tuple)
        self.assertEqual(config.betas, (0.9, 0.95))

    def test_get_optimizer_kwargs_excludes_none(self):
        config = OptimizerConfig(lr=1e-4, weight_decay=0.01)
        kwargs = config.get_optimizer_kwargs()
        self.assertEqual(kwargs, {"lr": 1e-4, "weight_decay": 0.01})
        self.assertNotIn("betas", kwargs)
        self.assertNotIn("eps", kwargs)

    def test_get_optimizer_kwargs_all_set(self):
        config = OptimizerConfig(lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
        kwargs = config.get_optimizer_kwargs()
        self.assertEqual(len(kwargs), 4)

    def test_param_groups(self):
        config = OptimizerConfig(
            lr=1e-4,
            param_groups={"encoder": {"lr": 1e-5}, "decoder": {"lr": 1e-3}},
        )
        self.assertEqual(len(config.param_groups), 2)
        self.assertEqual(config.param_groups["encoder"]["lr"], 1e-5)

    def test_type_validation(self):
        with self.assertRaises(ValidationError):
            OptimizerConfig(lr="invalid")

        with self.assertRaises(ValidationError):
            OptimizerConfig(betas=[0.9, 0.95, 0.99])


class TestSchedulerConfig(unittest.TestCase):
    """Test SchedulerConfig validation and defaults"""

    def test_default_values(self):
        config = SchedulerConfig()
        self.assertIsNone(config.name)
        self.assertEqual(config.warmup_steps, 1000)
        self.assertIsNone(config.scheduler_kwargs)
        # Legacy fields
        self.assertEqual(config.decay_steps, 30000)
        self.assertEqual(config.decay_lr, 2.5e-6)

    def test_transformers_scheduler(self):
        config = SchedulerConfig(
            name="cosine",
            warmup_steps=500,
            scheduler_kwargs={"min_lr": 1e-6},
        )
        self.assertEqual(config.name, "cosine")
        self.assertEqual(config.scheduler_kwargs["min_lr"], 1e-6)

    def test_legacy_fields_still_work(self):
        config = SchedulerConfig(warmup_steps=500, decay_steps=10000, decay_lr=1e-6)
        self.assertEqual(config.warmup_steps, 500)
        self.assertEqual(config.decay_steps, 10000)
        self.assertEqual(config.decay_lr, 1e-6)


class TestCheckpointConfig(unittest.TestCase):
    """Test CheckpointConfig validation"""

    def test_default_values(self):
        config = CheckpointConfig(output_directory="/tmp/ckpt")
        self.assertEqual(config.save_checkpoint, True)
        self.assertEqual(config.save_freq, 1000)
        self.assertEqual(config.output_directory, "/tmp/ckpt")

    def test_custom_values(self):
        config = CheckpointConfig(
            save_checkpoint=False, save_freq=500, output_directory="/custom/path"
        )
        self.assertEqual(config.save_checkpoint, False)
        self.assertEqual(config.save_freq, 500)
        self.assertEqual(config.output_directory, "/custom/path")


class TestSystemConfig(unittest.TestCase):
    """Test hierarchical SystemConfig with subconfigs"""

    def test_hierarchical_structure(self):
        config = SystemConfig(
            batch_size=8,
            checkpoint=CheckpointConfig(output_directory="/tmp"),
        )

        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.checkpoint.output_directory, "/tmp")

    def test_from_dict(self):
        config_dict = {
            "batch_size": 16,
            "train_steps": 5000,
            "checkpoint": {"output_directory": "/output", "save_freq": 100},
        }
        config = SystemConfig(**config_dict)

        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.train_steps, 5000)
        self.assertEqual(config.checkpoint.save_freq, 100)

    def test_raw_dictconfig_passthrough(self):
        raw = OmegaConf.create({"custom_field": 42})
        config = SystemConfig(
            checkpoint=CheckpointConfig(output_directory="/tmp"),
            raw=raw,
        )
        self.assertEqual(config.custom_field, 42)

    def test_raw_missing_attr_raises(self):
        config = SystemConfig(
            checkpoint=CheckpointConfig(output_directory="/tmp"),
        )
        with self.assertRaises(AttributeError):
            _ = config.nonexistent_field


class TestDataConfig(unittest.TestCase):
    """Test DataConfig with rename_map"""

    def test_basic_config(self):
        config = DataConfig(data_path="/path/to/data", tolerance_s=0.001, use_imagenet_stats=False)
        self.assertEqual(config.data_path, "/path/to/data")
        self.assertEqual(config.tolerance_s, 0.001)
        self.assertEqual(config.use_imagenet_stats, False)
        self.assertIsNone(config.rename_map)

    def test_rename_map_as_dict(self):
        rename_map = {
            "observation.images.cam_high": "observation.images.base_0_rgb",
            "observation.images.cam_left": "observation.images.left_0_rgb",
        }
        config = DataConfig(data_path="/data", rename_map=rename_map)
        self.assertEqual(config.rename_map, rename_map)
        self.assertEqual(len(config.rename_map), 2)

    def test_raw_dictconfig_passthrough(self):
        raw = OmegaConf.create({"vla_data": {"image_features": ["img1"]}})
        config = DataConfig(data_path="/data", raw=raw)
        self.assertEqual(config.vla_data.image_features, ["img1"])


class TestModelConfig(unittest.TestCase):
    """Test flexible ModelConfig that accepts extra fields"""

    def test_required_fields(self):
        config = ModelConfig(model_name="pi0", checkpoint_dir="/path/to/checkpoint")
        self.assertEqual(config.model_name, "pi0")
        self.assertEqual(config.checkpoint_dir, "/path/to/checkpoint")

    def test_extra_fields_allowed(self):
        config = ModelConfig(
            model_name="pi0",
            checkpoint_dir="/path/to/checkpoint",
            tokenizer_path="/path/to/tokenizer",
            tokenizer_max_length=48,
            action_steps=50,
        )

        model_dict = config.get_model_config_dict()
        self.assertEqual(model_dict["tokenizer_path"], "/path/to/tokenizer")
        self.assertEqual(model_dict["tokenizer_max_length"], 48)
        self.assertEqual(model_dict["action_steps"], 50)

    def test_qwen_gr00t_model_name(self):
        config = ModelConfig(model_name="pi0", checkpoint_dir="/path")
        self.assertEqual(config.model_name, "pi0")

    def test_invalid_model_name(self):
        with self.assertRaisesRegex(ValidationError, "Invalid model_name"):
            ModelConfig(model_name="invalid_model", checkpoint_dir="/path")

    def test_freeze_config(self):
        config = ModelConfig(
            model_name="pi0",
            checkpoint_dir="/path",
            freeze=FreezeConfig(freeze_patterns=["vlm.*"]),
        )
        self.assertIsNotNone(config.freeze)
        self.assertEqual(config.freeze.freeze_patterns, ["vlm.*"])

    def test_get_model_config_dict_excludes_train_fields(self):
        config = ModelConfig(
            model_name="pi0",
            checkpoint_dir="/checkpoint",
            freeze=FreezeConfig(),
            tokenizer_path="/tokenizer",
            action_steps=50,
        )

        model_dict = config.get_model_config_dict()

        self.assertNotIn("model_name", model_dict)
        self.assertNotIn("checkpoint_dir", model_dict)
        self.assertNotIn("freeze", model_dict)
        self.assertNotIn("optimizer", model_dict)

        self.assertIn("tokenizer_path", model_dict)
        self.assertIn("action_steps", model_dict)


class TestTrainConfig(unittest.TestCase):
    """Test top-level TrainConfig integration"""

    def test_full_config_creation(self):
        config_dict = {
            "system": {
                "batch_size": 4,
                "train_steps": 10000,
                "checkpoint": {"output_directory": "/tmp/ckpt"},
            },
            "model": {
                "model_name": "pi0",
                "checkpoint_dir": "/model",
                "tokenizer_path": "/tokenizer",
                "action_steps": 50,
                "optimizer": {"lr": 1e-4, "scheduler": {"warmup_steps": 500}},
            },
            "data": {"data_path": "/data", "use_imagenet_stats": True},
        }

        config = TrainConfig(**config_dict)

        self.assertEqual(config.system.batch_size, 4)
        self.assertEqual(config.system.train_steps, 10000)
        self.assertEqual(config.system.checkpoint.output_directory, "/tmp/ckpt")

        self.assertEqual(config.model.model_name, "pi0")
        self.assertEqual(config.model.checkpoint_dir, "/model")
        self.assertEqual(config.model.optimizer.lr, 1e-4)
        self.assertEqual(config.model.optimizer.scheduler.warmup_steps, 500)

        self.assertEqual(config.data.data_path, "/data")
        self.assertEqual(config.data.use_imagenet_stats, True)

    def test_from_hydra_config(self):
        hydra_dict = {
            "train": {
                "system": {
                    "batch_size": 8,
                    "checkpoint": {"output_directory": "/out"},
                },
                "model": {
                    "model_name": "pi0.5",
                    "checkpoint_dir": "/ckpt",
                    "optimizer": {"lr": 2e-5, "scheduler": {}},
                },
                "data": {"data_path": "/dataset"},
            }
        }

        hydra_config = OmegaConf.create(hydra_dict)
        config = TrainConfig.from_hydra_config(hydra_config)

        self.assertEqual(config.system.batch_size, 8)
        self.assertEqual(config.model.optimizer.lr, 2e-5)
        self.assertEqual(config.model.model_name, "pi0.5")
        self.assertEqual(config.data.data_path, "/dataset")

    def test_from_hydra_preserves_raw(self):
        hydra_dict = {
            "train": {
                "system": {
                    "checkpoint": {"output_directory": "/out"},
                    "custom_sys_field": "hello",
                },
                "model": {"model_name": "pi0", "checkpoint_dir": "/ckpt"},
                "data": {"data_path": "/dataset", "vla_data": {"key": "val"}},
            }
        }
        hydra_config = OmegaConf.create(hydra_dict)
        config = TrainConfig.from_hydra_config(hydra_config)

        self.assertEqual(config.system.custom_sys_field, "hello")
        self.assertEqual(config.data.vla_data.key, "val")

    def test_type_validation_error(self):
        config_dict = {
            "system": {
                "batch_size": "invalid",
                "checkpoint": {"output_directory": "/tmp"},
            },
            "model": {"model_name": "pi0", "checkpoint_dir": "/model"},
            "data": {"data_path": "/data"},
        }

        with self.assertRaises(ValidationError):
            TrainConfig(**config_dict)

    def test_missing_required_field(self):
        config_dict = {
            "system": {
                "checkpoint": {"output_directory": "/tmp"},
            },
            "model": {
                # Missing required model_name
                "checkpoint_dir": "/model",
            },
            "data": {"data_path": "/data"},
        }

        with self.assertRaises(ValidationError):
            TrainConfig(**config_dict)


class TestConfigSerialization(unittest.TestCase):
    """Test config serialization and deserialization"""

    def test_dict_roundtrip(self):
        config = TrainConfig(
            system=SystemConfig(
                batch_size=16,
                checkpoint=CheckpointConfig(output_directory="/tmp"),
            ),
            model=ModelConfig(
                model_name="pi0",
                checkpoint_dir="/model",
                action_steps=50,
                optimizer=OptimizerConfig(),
            ),
            data=DataConfig(data_path="/data"),
        )

        config_dict = config.model_dump()
        config_restored = TrainConfig(**config_dict)

        self.assertEqual(config_restored.system.batch_size, config.system.batch_size)
        self.assertEqual(config_restored.model.model_name, config.model.model_name)
        self.assertEqual(config_restored.data.data_path, config.data.data_path)
