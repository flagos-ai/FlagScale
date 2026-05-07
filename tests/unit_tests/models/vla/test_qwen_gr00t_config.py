"""Unit tests for QwenGr00tConfig and PreTrainedConfig."""

import json
import tempfile
from pathlib import Path

import pytest

from flagscale.models.configs.types import FeatureType, NormalizationMode, PolicyFeature
from flagscale.models.utils.constants import ACTION, OBS_STATE
from flagscale.models.vla.action_model.gr00t_action_header import GR00TActionHeadConfig
from flagscale.models.vla.pretrained_config import CONFIG_NAME, PreTrainedConfig
from flagscale.models.vla.qwen_gr00t import QwenGr00tConfig
from flagscale.models.vla.vlm.qwenvl_backbone import QwenVLConfig


def _make_features():
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        "observation.images.image_0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return input_features, output_features


def _make_config(**overrides):
    input_features, output_features = _make_features()
    defaults = dict(
        input_features=input_features,
        output_features=output_features,
        vlm=QwenVLConfig(
            type="qwen3-vl",
            base_vlm="/models/Qwen3-VL-4B",
            load_pretrained=True,
            attn_implementation="flash_attention_2",
        ),
        action_model=GR00TActionHeadConfig(
            type="gr00t_action_head",
            action_model_type="DiT-B",
            hidden_size=1024,
            action_dim=7,
            state_dim=7,
            future_action_window_size=7,
            diffusion_model_cfg={
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "num_layers": 16,
            },
        ),
        prompt_template="Your task is {instruction}.",
    )
    defaults.update(overrides)
    return QwenGr00tConfig(**defaults)


class TestPreTrainedConfigFeatureAccessors:
    def test_robot_state_feature(self):
        cfg = _make_config()
        assert cfg.robot_state_feature is not None
        assert cfg.robot_state_feature.type is FeatureType.STATE
        assert cfg.robot_state_feature.shape == (7,)

    def test_image_features(self):
        cfg = _make_config()
        imgs = cfg.image_features
        assert len(imgs) == 1
        assert "observation.images.image_0" in imgs

    def test_action_feature(self):
        cfg = _make_config()
        assert cfg.action_feature is not None
        assert cfg.action_feature.type is FeatureType.ACTION
        assert cfg.action_feature.shape == (7,)

    def test_empty_features(self):
        cfg = _make_config(input_features=None, output_features=None)
        assert cfg.robot_state_feature is None
        assert cfg.image_features == {}
        assert cfg.action_feature is None


class TestQwenGr00tConfigRoundTrip:
    def test_save_and_load(self):
        cfg = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg._save_pretrained(Path(tmpdir))

            config_path = Path(tmpdir) / CONFIG_NAME
            assert config_path.exists()

            loaded = QwenGr00tConfig.from_pretrained(tmpdir)

        assert loaded.vlm.type == cfg.vlm.type
        assert loaded.vlm.base_vlm == cfg.vlm.base_vlm
        assert loaded.vlm.load_pretrained == cfg.vlm.load_pretrained
        assert loaded.vlm.attn_implementation == cfg.vlm.attn_implementation

        assert loaded.action_model.type == cfg.action_model.type
        assert loaded.action_model.action_model_type == cfg.action_model.action_model_type
        assert loaded.action_model.hidden_size == cfg.action_model.hidden_size
        assert loaded.action_model.action_dim == cfg.action_model.action_dim
        assert loaded.action_model.diffusion_model_cfg == cfg.action_model.diffusion_model_cfg

        assert loaded.prompt_template == cfg.prompt_template
        assert loaded.observation_delta_indices == cfg.observation_delta_indices
        assert loaded.action_delta_indices == cfg.action_delta_indices

    def test_feature_round_trip(self):
        cfg = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg._save_pretrained(Path(tmpdir))
            loaded = QwenGr00tConfig.from_pretrained(tmpdir)

        assert loaded.robot_state_feature == cfg.robot_state_feature
        assert loaded.action_feature == cfg.action_feature
        assert set(loaded.image_features.keys()) == set(cfg.image_features.keys())

    def test_json_is_valid(self):
        cfg = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg._save_pretrained(Path(tmpdir))
            with open(Path(tmpdir) / CONFIG_NAME) as f:
                data = json.load(f)

        assert data["type"] == "QwenGr00t"
        assert "vlm" in data
        assert "action_model" in data
        assert data["vlm"]["type"] == "qwen3-vl"
        assert data["action_model"]["action_dim"] == 7
        assert data["input_features"][OBS_STATE]["type"] == "STATE"

    def test_none_features_round_trip(self):
        cfg = _make_config(input_features=None, output_features=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg._save_pretrained(Path(tmpdir))
            loaded = QwenGr00tConfig.from_pretrained(tmpdir)
        assert loaded.input_features is None
        assert loaded.output_features is None


class TestQwenGr00tConfigValidation:
    def test_validate_features_passes(self):
        cfg = _make_config()
        cfg.validate_features()

    def test_validate_features_missing_action(self):
        input_features, _ = _make_features()
        cfg = _make_config(output_features={})
        with pytest.raises(ValueError, match="output_features must be set"):
            cfg.validate_features()

    def test_validate_features_wrong_type(self):
        input_features, _ = _make_features()
        cfg = _make_config(
            output_features={ACTION: PolicyFeature(type=FeatureType.STATE, shape=(7,))}
        )
        with pytest.raises(ValueError, match="ACTION"):
            cfg.validate_features()


class TestQwenGr00tConfigFromTrainConfig:
    def test_from_omegaconf(self):
        from omegaconf import OmegaConf

        raw = OmegaConf.create(
            {
                "model": {
                    "model_name": "qwen_gr00t",
                    "vlm": {
                        "type": "qwen3-vl",
                        "base_vlm": "/models/Qwen3-VL",
                        "load_pretrained": True,
                        "attn_implementation": "flash_attention_2",
                    },
                    "action_model": {
                        "type": "gr00t_action_head",
                        "action_model_type": "DiT-B",
                        "hidden_size": 1024,
                        "action_dim": 7,
                        "state_dim": 7,
                        "future_action_window_size": 7,
                        "action_horizon": 8,
                        "use_state": False,
                        "repeated_diffusion_steps": 4,
                        "add_pos_embed": True,
                        "max_seq_len": 1024,
                        "noise_beta_alpha": 1.5,
                        "noise_beta_beta": 1.0,
                        "noise_s": 0.999,
                        "num_timestep_buckets": 1000,
                        "num_inference_timesteps": 4,
                        "num_target_vision_tokens": 32,
                        "diffusion_model_cfg": {
                            "cross_attention_dim": 2048,
                            "dropout": 0.2,
                        },
                    },
                    "prompt_template": "Your task is {instruction}.",
                },
                "data": {
                    "data_path": "/data",
                    "tolerance_s": 0.0001,
                },
            }
        )

        # Minimal mock of TrainConfig structure (OmegaConf namespaces)
        class _FakeTrainConfig:
            model = raw.model
            data = raw.data

        cfg = QwenGr00tConfig.from_train_config(_FakeTrainConfig())

        assert cfg.vlm.type == "qwen3-vl"
        assert cfg.vlm.base_vlm == "/models/Qwen3-VL"
        assert cfg.vlm.attn_implementation == "flash_attention_2"
        assert cfg.action_model.action_dim == 7
        assert cfg.action_model.diffusion_model_cfg["cross_attention_dim"] == 2048
        assert cfg.prompt_template == "Your task is {instruction}."
        assert cfg.observation_delta_indices == [0]
        assert cfg.action_delta_indices == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_from_omegaconf_with_normalization_mapping(self):
        from omegaconf import OmegaConf

        raw = OmegaConf.create(
            {
                "model": {
                    "model_name": "qwen_gr00t",
                    "vlm": {
                        "type": "qwen3-vl",
                        "base_vlm": "/models/Qwen3-VL",
                    },
                    "action_model": {
                        "type": "gr00t_action_head",
                        "action_model_type": "DiT-B",
                        "hidden_size": 1024,
                        "action_dim": 7,
                        "state_dim": 7,
                        "future_action_window_size": 7,
                        "action_horizon": 8,
                        "num_inference_timesteps": 4,
                        "num_target_vision_tokens": 32,
                        "diffusion_model_cfg": {"cross_attention_dim": 2048},
                    },
                    "normalization_mapping": {
                        "VISUAL": "IDENTITY",
                        "STATE": "MEAN_STD",
                        "ACTION": "MEAN_STD",
                    },
                },
                "data": {
                    "data_path": "/data",
                    "tolerance_s": 0.0001,
                },
            }
        )

        class _FakeTrainConfig:
            model = raw.model
            data = raw.data

        cfg = QwenGr00tConfig.from_train_config(_FakeTrainConfig())
        assert cfg.normalization_mapping["STATE"] is NormalizationMode.MEAN_STD
        assert cfg.normalization_mapping["ACTION"] is NormalizationMode.MEAN_STD
        assert cfg.normalization_mapping["VISUAL"] is NormalizationMode.IDENTITY

    def test_from_omegaconf_default_normalization_mapping(self):
        from omegaconf import OmegaConf

        raw = OmegaConf.create(
            {
                "model": {
                    "model_name": "qwen_gr00t",
                    "vlm": {
                        "type": "qwen3-vl",
                        "base_vlm": "/models/Qwen3-VL",
                    },
                    "action_model": {
                        "type": "gr00t_action_head",
                        "action_model_type": "DiT-B",
                        "hidden_size": 1024,
                        "action_dim": 7,
                        "state_dim": 7,
                        "future_action_window_size": 7,
                        "action_horizon": 8,
                        "num_inference_timesteps": 4,
                        "num_target_vision_tokens": 32,
                        "diffusion_model_cfg": {"cross_attention_dim": 2048},
                    },
                },
                "data": {"data_path": "/data", "tolerance_s": 0.0001},
            }
        )

        class _FakeTrainConfig:
            model = raw.model
            data = raw.data

        cfg = QwenGr00tConfig.from_train_config(_FakeTrainConfig())
        assert cfg.normalization_mapping["STATE"] is NormalizationMode.MIN_MAX
        assert cfg.normalization_mapping["ACTION"] is NormalizationMode.MIN_MAX
        assert cfg.normalization_mapping["VISUAL"] is NormalizationMode.IDENTITY

    def test_from_omegaconf_round_trip(self):
        from omegaconf import OmegaConf

        raw = OmegaConf.create(
            {
                "model": {
                    "model_name": "qwen_gr00t",
                    "vlm": {
                        "type": "qwen3-vl",
                        "base_vlm": "/models/Qwen3-VL",
                    },
                    "action_model": {
                        "type": "gr00t_action_head",
                        "action_model_type": "DiT-B",
                        "hidden_size": 1024,
                        "action_dim": 7,
                        "state_dim": 7,
                        "future_action_window_size": 7,
                        "action_horizon": 8,
                        "num_inference_timesteps": 4,
                        "num_target_vision_tokens": 32,
                        "diffusion_model_cfg": {
                            "cross_attention_dim": 2048,
                        },
                    },
                },
                "data": {"data_path": "/data", "tolerance_s": 0.0001},
            }
        )

        class _FakeTrainConfig:
            model = raw.model
            data = raw.data

        cfg = QwenGr00tConfig.from_train_config(_FakeTrainConfig())

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg._save_pretrained(Path(tmpdir))
            loaded = QwenGr00tConfig.from_pretrained(tmpdir)

        assert loaded.vlm.type == cfg.vlm.type
        assert loaded.vlm.base_vlm == cfg.vlm.base_vlm
        assert loaded.action_model.hidden_size == cfg.action_model.hidden_size
        assert loaded.action_feature == cfg.action_feature


class TestGR00TActionHeadConfigFromOmegaconf:
    def test_missing_diffusion_model_cfg_raises(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "type": "gr00t_action_head",
                "action_model_type": "DiT-B",
                "hidden_size": 1024,
                "action_dim": 7,
            }
        )
        with pytest.raises(ValueError, match="diffusion_model_cfg is required"):
            GR00TActionHeadConfig.from_omegaconf(cfg)


class TestDeltaIndices:
    def test_delta_indices_computed_from_future_action_window_size(self):
        cfg = _make_config()
        assert cfg.observation_delta_indices == [0]
        assert cfg.action_delta_indices == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_delta_indices_with_different_window_size(self):
        cfg = _make_config(
            action_model=GR00TActionHeadConfig(
                future_action_window_size=3,
                diffusion_model_cfg={"cross_attention_dim": 2048},
            ),
        )
        assert cfg.action_delta_indices == [0, 1, 2, 3]

    def test_delta_indices_always_present(self):
        input_features, output_features = _make_features()
        cfg = QwenGr00tConfig(
            input_features=input_features,
            output_features=output_features,
        )
        assert cfg.observation_delta_indices == [0]
        assert cfg.action_delta_indices == [0, 1, 2, 3, 4, 5, 6, 7]


class TestPolymorphicDispatch:
    def test_registry_contains_qwen_gr00t(self):
        assert "QwenGr00t" in PreTrainedConfig._registry
        assert PreTrainedConfig._registry["QwenGr00t"] is QwenGr00tConfig

    def test_base_class_from_pretrained_dispatches(self):
        cfg = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg._save_pretrained(Path(tmpdir))
            loaded = PreTrainedConfig.from_pretrained(tmpdir)

        assert type(loaded) is QwenGr00tConfig
        assert loaded.vlm.base_vlm == cfg.vlm.base_vlm
        assert loaded.action_model.action_dim == cfg.action_model.action_dim

    def test_concrete_class_from_pretrained_works(self):
        cfg = _make_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg._save_pretrained(Path(tmpdir))
            loaded = QwenGr00tConfig.from_pretrained(tmpdir)

        assert type(loaded) is QwenGr00tConfig
        assert loaded.vlm.base_vlm == cfg.vlm.base_vlm

    def test_missing_type_field_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / CONFIG_NAME
            config_file.write_text(json.dumps({"input_features": {}, "output_features": {}}))
            with pytest.raises(ValueError, match="missing the 'type' field"):
                PreTrainedConfig.from_pretrained(tmpdir)

    def test_unknown_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / CONFIG_NAME
            config_file.write_text(json.dumps({"type": "nonexistent_policy"}))
            with pytest.raises(ValueError, match="Unknown config type"):
                PreTrainedConfig.from_pretrained(tmpdir)

    def test_type_name_attribute(self):
        assert QwenGr00tConfig._type_name == "QwenGr00t"

    def test_base_class_from_train_config_dispatches(self):
        from omegaconf import OmegaConf

        raw = OmegaConf.create(
            {
                "model": {
                    "model_name": "qwen_gr00t",
                    "vlm": {
                        "type": "qwen3-vl",
                        "base_vlm": "/models/Qwen3-VL",
                    },
                    "action_model": {
                        "type": "gr00t_action_head",
                        "action_model_type": "DiT-B",
                        "hidden_size": 1024,
                        "action_dim": 7,
                        "state_dim": 7,
                        "future_action_window_size": 7,
                        "action_horizon": 8,
                        "num_inference_timesteps": 4,
                        "num_target_vision_tokens": 32,
                        "diffusion_model_cfg": {"cross_attention_dim": 2048},
                    },
                },
                "data": {"data_path": "/data", "tolerance_s": 0.0001},
            }
        )

        class _FakeTrainConfig:
            model = raw.model
            data = raw.data

        cfg = PreTrainedConfig.from_train_config(_FakeTrainConfig())
        assert type(cfg) is QwenGr00tConfig
        assert cfg.vlm.base_vlm == "/models/Qwen3-VL"
        assert cfg.action_model.action_dim == 7

    def test_from_train_config_missing_model_name_raises(self):
        from omegaconf import OmegaConf

        raw = OmegaConf.create({"model": {}, "data": {}})

        class _FakeTrainConfig:
            model = raw.model
            data = raw.data

        with pytest.raises(ValueError, match="model_name is required"):
            PreTrainedConfig.from_train_config(_FakeTrainConfig())

    def test_from_train_config_unknown_model_name_raises(self):
        from omegaconf import OmegaConf

        raw = OmegaConf.create(
            {
                "model": {"model_name": "nonexistent_model"},
                "data": {},
            }
        )

        class _FakeTrainConfig:
            model = raw.model
            data = raw.data

        with pytest.raises(ValueError, match="No config registered"):
            PreTrainedConfig.from_train_config(_FakeTrainConfig())
