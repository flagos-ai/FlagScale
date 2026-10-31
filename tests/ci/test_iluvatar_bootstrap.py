# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CPU checks for image bootstrap isolation and shared candidate validation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts/ci"))
from generate_matrix import load_platforms


def test_iluvatar_uses_validated_image_through_shared_ci():
    config = yaml.safe_load((ROOT / ".github/configs/iluvatar.yml").read_text())
    assert config["ci_train_image"] == (
        "harbor.baai.ac.cn/flagos-dev/flagscale-train:manual-20261030-lluvatar-bi-gpu8-dev"
    )
    assert "ci_image" not in config  # SDK images cannot be a training fallback.
    assert "ci_enabled" not in config
    assert "iluvatar" in load_platforms(ROOT / ".github/configs")
    assert config["runner_labels"] == ["flagscale-lluvatar-bi-gpu8"]
    build = config["image_build"]
    assert build["enabled"] is True
    assert set(build["tasks"]) == {"train"}
    assert build["runtime_smoke_nproc"] == "2"
    assert build["runtime_device_count"] == "8"
    task = build["tasks"]["train"]
    for path in (task["dockerfile"], build["validation_script"]):
        assert (ROOT / path).is_file()
    assert task["base_image"] == (
        "harbor.baai.ac.cn/flagos-base/flagos-base-iluvatar-corex4.5.0:2.3.0"
    )
    assert not list((ROOT / ".github/workflows").glob("*iluvatar*"))


@pytest.mark.parametrize("phase,override", [("false", ""), ("false", "typer"), ("true", "")])
def test_train_installer_respects_common_phase_controls(phase, override):
    # Exercise skipped task + explicit pip override paths as well as full task.
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "FLAGSCALE_PKG_MGR": "pip",
        "FLAGSCALE_INSTALL_TASK": phase,
        "FLAGSCALE_PIP_DEPS": override,
        "FLAGSCALE_ONLY_PIP": "true",
        "FLAGSCALE_SRC_DEPS": "",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "tools/install/iluvatar/install_train.sh"), "--debug"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "git clone" not in result.stderr
    if override:
        assert "install --root-user-action=ignore typer" in result.stderr
    elif phase == "false":
        assert "pip install" not in result.stderr
    else:
        assert "requirements/iluvatar/train.txt" in result.stderr


def test_iluvatar_changes_keep_all_shared_platform_plans_executable():
    from flagcicd_plan import build_plan

    plan = build_plan(
        ROOT,
        profile="pr",
        changed=[
            ".github/configs/iluvatar.yml",
            "docker/iluvatar/Dockerfile.train",
            "tests/functional_tests/train/qwen3/conf/0_6b_iluvatar.yaml",
            "tests/functional_tests/train/qwen3/conf/train/0_6b_iluvatar.yaml",
            "tests/ci/test_iluvatar_bootstrap.py",
        ],
        labels=[],
    )
    assert set(plan["platforms"]) == {
        "ascend",
        "cuda",
        "enflame",
        "hygon",
        "iluvatar",
        "kunlunxin",
        "metax",
        "musa",
        "ppu",
    }
    assert "iluvatar" in plan["slices"]
