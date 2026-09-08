# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flagscale.models.kerv import build_kerv_command, load_kerv_config

CONFIG_DIR = Path(__file__).resolve().parents[4] / "examples" / "kerv" / "conf"


def _compose(name: str):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name)


def test_all_public_kerv_configs_compose(monkeypatch):
    monkeypatch.setenv("KERV_SOURCE_ROOT", "/tmp/KERV")
    monkeypatch.setenv("OPENVLA_SOURCE_ROOT", "/tmp/openvla")
    for stage in ("verifier_lora", "verifier_full", "drafter"):
        monkeypatch.setenv("KERV_TRAIN_STAGE", stage)
        config = _compose("train")
        OmegaConf.resolve(config)
        assert config.train.kerv.stage == stage
        assert config.train.system is not None
        assert config.train.kerv.launcher == "python"
        assert config.experiment.task.entrypoint == "flagscale/train/train_kerv.py"

    for name in ("generate_draft_data", "inference"):
        config = _compose(name)
        OmegaConf.resolve(config)
        task_config = config.inference if name == "inference" else config.train
        assert task_config.kerv.stage
        assert config.experiment.task.entrypoint
        if name != "inference":
            assert task_config.system is not None
            assert task_config.kerv.launcher == "python"


def test_inference_uses_safe_profile(monkeypatch):
    monkeypatch.setenv("KERV_SOURCE_ROOT", "/tmp/KERV")
    config = _compose("inference")
    OmegaConf.resolve(config)
    arguments = config.inference.kerv.arguments
    assert arguments.use_spec is True
    assert arguments.use_kalman_fallback is True
    assert arguments.use_kalman_tree is True
    assert arguments.use_flagos_embodied_ops is True
    assert "kerv_static_tree_attention" in arguments.flagos_embodied_ops_include
    assert arguments.flagos_tree_operatorized is False
    assert arguments.flagos_persistent_tree_capacity >= max(
        int(value) for value in arguments.flagos_tree_node_buckets.split(",")
    )
    assert arguments.flagos_qkv_fusion_rows
    assert arguments.flagos_gate_up_fusion_rows
    assert arguments.flagos_swiglu_fusion_rows
    assert arguments.flagos_linear_fusion_record is False
    assert arguments.flagos_rope_fusion_record is False


def test_all_public_configs_build_commands_without_weights(tmp_path, monkeypatch):
    kerv_root = tmp_path / "KERV"
    openvla_root = tmp_path / "openvla"
    training_entrypoints = {
        "verifier_lora": (openvla_root, "vla-scripts/finetune.py"),
        "verifier_full": (openvla_root, "vla-scripts/train.py"),
        "drafter": (kerv_root, "training/train_drafter.py"),
    }
    other_entrypoints = {
        "generate_draft_data": (kerv_root, "training/generate_drafter_data.py"),
        "inference": (
            kerv_root,
            "openvla/experiments/robot/libero/run_kerv_libero.py",
        ),
    }
    for source_root, relative in (*training_entrypoints.values(), *other_entrypoints.values()):
        script = source_root / relative
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("print('smoke')\n", encoding="utf-8")

    monkeypatch.setenv("KERV_SOURCE_ROOT", str(kerv_root))
    monkeypatch.setenv("OPENVLA_SOURCE_ROOT", str(openvla_root))
    for stage, (source_root, relative) in training_entrypoints.items():
        monkeypatch.setenv("KERV_TRAIN_STAGE", stage)
        config = _compose("train")
        OmegaConf.resolve(config)
        config_path = tmp_path / f"train_{stage}.yaml"
        OmegaConf.save(config, config_path)
        kerv = load_kerv_config(config_path)
        command, _, _ = build_kerv_command(kerv)
        assert command[1] == str(source_root / relative)

    for name, (source_root, relative) in other_entrypoints.items():
        config = _compose(name)
        OmegaConf.resolve(config)
        config_path = tmp_path / f"{name}.yaml"
        OmegaConf.save(config, config_path)
        kerv = load_kerv_config(config_path)
        command, _, _ = build_kerv_command(kerv)
        assert command[1] == str(source_root / relative)
