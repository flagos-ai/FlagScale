"""Regression tests for the Qwen3.5 release checkpoint contract."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

CONVERTER = Path(__file__).resolve().parents[3] / "tools/checkpoint/qwen35"
sys.path.insert(0, str(CONVERTER))
from qwen35.config import Config
from qwen35.io import save_megatron_release_checkpoint


@pytest.mark.parametrize(
    "torch_version,linear",
    [
        ("2.7.1+cu128", False),
        ("2.9.0", False),
        ("2.9.0+cu128", True),
        ("2.10.0", True),
        ("2.11.0", False),
    ],
)
def test_vision_layout_matches_runtime(torch_version, linear):
    model_yaml = CONVERTER.parents[2] / "examples/qwen35/conf/train/4b.yaml"
    with patch.object(torch, "__version__", torch_version):
        config = Config(model_yaml)
    assert config.use_linear_proj is linear


def test_vision_layout_can_target_another_runtime(tmp_path):
    model_yaml = CONVERTER.parents[2] / "examples/qwen35/conf/train/4b.yaml"
    raw = yaml.safe_load(model_yaml.read_text())
    raw["model"]["vision_patch_embed_linear"] = True
    target = tmp_path / "model.yaml"
    target.write_text(yaml.safe_dump(raw))
    with patch.object(torch, "__version__", "2.7.1"):
        assert Config(target).use_linear_proj


def test_release_preserves_modern_qkv_layout(tmp_path):
    model_yaml = CONVERTER.parents[2] / "examples/qwen35/conf/train/4b.yaml"
    config = Config(model_yaml)
    config.tp = 1
    config.pp = 1
    weights = {"language_model.decoder.layers.0.test": torch.arange(6)}
    save_megatron_release_checkpoint({(0, 0): weights}, tmp_path, config)
    checkpoint = torch.load(tmp_path / "release/mp_rank_00/model_optim_rng.pt", weights_only=True)
    assert checkpoint["checkpoint_version"] == 3.0
    assert checkpoint["iteration"] == 0
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "release\n"
    torch.testing.assert_close(
        checkpoint["model"][next(iter(weights))], next(iter(weights.values()))
    )


def test_extra_states_only_for_te_modules():
    from qwen35.converter import DenseConverter

    state = {
        "vision_model.decoder.final_layernorm.weight": torch.ones(4),
        "vision_model.decoder.layers.0.self_attention.linear_qkv.weight": torch.ones(4, 4),
    }
    result = DenseConverter._add_extra_states(state)
    assert "vision_model.decoder.final_layernorm._extra_state" not in result
    assert "vision_model.decoder.layers.0.self_attention.linear_qkv._extra_state" in result


@pytest.mark.parametrize("module", ["enorm", "hnorm", "eh_proj", "final_layernorm"])
def test_mtp_extra_states_belong_to_modules(module):
    from qwen35.converter import DenseConverter

    base = f"language_model.mtp.layers.0.{module}"
    state = {f"{base}.weight": torch.ones(4, 8)}
    result = DenseConverter._add_extra_states(state)
    assert "language_model.mtp.layers._extra_state" not in result
    assert f"{base}._extra_state" in result
