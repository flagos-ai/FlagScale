# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from flagscale.models.kerv import build_kerv_command, load_kerv_config


def _make_script(root: Path, relative: str = "training/stage.py") -> Path:
    script = root / relative
    script.parent.mkdir(parents=True)
    script.write_text("print('smoke')\n", encoding="utf-8")
    return script


def test_load_kerv_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "kerv:\n"
        "  stage: inference\n"
        f"  source_root: {tmp_path}\n"
        "  entrypoint: run.py\n"
        "  launcher: python\n",
        encoding="utf-8",
    )
    config = load_kerv_config(config_path)
    assert config.stage == "inference"


def test_load_nested_native_train_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "train:\n"
        "  system: {}\n"
        "  kerv:\n"
        "    stage: drafter\n"
        f"    source_root: {tmp_path}\n"
        "    entrypoint: training/run.py\n"
        "    launcher: python\n",
        encoding="utf-8",
    )
    config = load_kerv_config(config_path)
    assert config.stage == "drafter"


def test_python_command_and_argument_rendering(tmp_path):
    script = _make_script(tmp_path)
    config = OmegaConf.create(
        {
            "stage": "draft_data",
            "source_root": str(tmp_path),
            "entrypoint": str(script.relative_to(tmp_path)),
            "launcher": "python",
            "arguments": {"enabled": True, "buckets": [224, 240], "skip": None},
        }
    )
    command, work_dir, environment = build_kerv_command(config)
    assert command[1] == str(script)
    assert command[-4:] == ["--enabled", "True", "--buckets", "224,240"]
    assert work_dir == tmp_path
    assert environment["PYTHONPATH"].split(":")[0] == str(tmp_path)


def test_workdir_and_python_paths_are_absolute(tmp_path, monkeypatch):
    source_root = tmp_path / "KERV"
    _make_script(source_root)
    runtime_root = source_root / "runtime_opt"
    runtime_root.mkdir()
    monkeypatch.chdir(tmp_path)
    config = OmegaConf.create(
        {
            "stage": "inference",
            "source_root": "KERV",
            "entrypoint": "training/stage.py",
            "launcher": "python",
            "work_dir": ".",
            "python_paths": ["KERV/runtime_opt"],
        }
    )
    _, work_dir, environment = build_kerv_command(config)
    assert work_dir == tmp_path
    assert environment["PYTHONPATH"].split(":")[:2] == [
        str(source_root),
        str(runtime_root),
    ]


def test_torchrun_command(tmp_path):
    script = _make_script(tmp_path)
    config = OmegaConf.create(
        {
            "stage": "verifier_lora",
            "source_root": str(tmp_path),
            "entrypoint": str(script.relative_to(tmp_path)),
            "launcher": "torchrun",
            "distributed": {"nnodes": 1, "nproc_per_node": 4, "master_port": 23456},
        }
    )
    command, _, _ = build_kerv_command(config)
    assert "torch.distributed.run" in command
    assert command[command.index("--nproc-per-node") + 1] == "4"
    assert command[command.index("--master-port") + 1] == "23456"


def test_deepspeed_command(tmp_path):
    script = _make_script(tmp_path)
    config = OmegaConf.create(
        {
            "stage": "drafter",
            "source_root": str(tmp_path),
            "entrypoint": str(script.relative_to(tmp_path)),
            "launcher": "deepspeed",
            "distributed": {"master_port": 23333, "include": "localhost:0,1"},
        }
    )
    command, _, _ = build_kerv_command(config)
    assert command[:5] == [
        "deepspeed",
        "--master_port",
        "23333",
        "--include",
        "localhost:0,1",
    ]


def test_entrypoint_cannot_escape_source_root(tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")
    config = {
        "stage": "inference",
        "source_root": str(tmp_path),
        "entrypoint": str(outside),
        "launcher": "python",
    }
    with pytest.raises(ValueError, match="under kerv.source_root"):
        build_kerv_command(config)


def test_missing_entrypoint_has_actionable_error(tmp_path):
    config = {
        "stage": "inference",
        "source_root": str(tmp_path),
        "entrypoint": "missing.py",
        "launcher": "python",
    }
    with pytest.raises(FileNotFoundError, match="KERV_SOURCE_ROOT"):
        build_kerv_command(config)
