# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CPU contract checks for the Iluvatar platform configuration and installers."""

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts/ci"))
from generate_matrix import load_platforms

CONFIG = ROOT / ".github/configs/iluvatar.yml"
CONSTRAINTS = ROOT / "requirements/iluvatar/constraints.txt"


def _image(value):
    return isinstance(value, str) and re.fullmatch(r"harbor\.baai\.ac\.cn/[^\s:@]+[:@]\S+", value)


def test_iluvatar_uses_validated_image_through_shared_ci():
    config = yaml.safe_load(CONFIG.read_text())
    assert _image(config["ci_train_image"])
    assert "ci_image" not in config  # SDK images cannot be a training fallback.
    assert "iluvatar" in load_platforms(ROOT / ".github/configs")
    labels = config["runner_labels"]
    assert labels and all(isinstance(label, str) and label for label in labels)
    build = config["image_build"]
    assert build["enabled"] is True
    assert set(build["tasks"]) == {"train"}
    assert int(build["runtime_device_count"]) >= int(build["runtime_smoke_nproc"]) >= 2
    task = build["tasks"]["train"]
    assert _image(task["base_image"])
    for path in (task["dockerfile"], build["validation_script"]):
        assert (ROOT / path).is_file()
    assert not list((ROOT / ".github/workflows").glob("*iluvatar*"))


def test_corex_runtime_versions_have_a_single_source():
    pins = dict(
        line.split("==", 1)
        for line in CONSTRAINTS.read_text().splitlines()
        if line and not line.startswith("#")
    )
    torch = pins["torch"]
    assert "+corex." in torch
    for path in (
        "requirements/iluvatar/base.txt",
        "requirements/iluvatar/train.txt",
        "tools/install/iluvatar/install_train.sh",
        "tools/install/iluvatar/validate_image_build.sh",
    ):
        assert torch not in (ROOT / path).read_text(), path


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


def test_iluvatar_owned_changes_select_only_iluvatar():
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
            "tests/functional_tests/benchmark/qwen3/conf/iluvatar0_6b_tp1_pp1-reference.yaml",
            "tests/functional_tests/benchmark/qwen3/conf/train/iluvatar0_6b_tp1_pp1-reference.yaml",
            "tests/functional_tests/benchmark/qwen3/gold_values/iluvatar0_6b_tp1_pp1-reference.json",
        ],
        labels=[],
    )
    assert plan["platforms"] == ["iluvatar"]
    assert "iluvatar" in plan["slices"]


def test_iluvatar_benchmark_uses_shared_plan_and_measured_thresholds():
    from flagcicd_plan import build_plan, outputs

    plan = build_plan(
        ROOT, profile="requested", changed=None, labels=[], requested_platforms=["iluvatar"]
    )
    cases = json.loads(outputs(plan, "iluvatar")["benchmark_test_matrix"])
    assert len(cases) == 1
    case = cases[0]
    assert (case["device"], case["task"], case["model"]) == ("iluvatar", "benchmark", "qwen3")
    root = ROOT / "tests/functional_tests/benchmark/qwen3"
    config = yaml.safe_load((root / "conf" / (case["case"] + ".yaml")).read_text())
    assert config["experiment"]["runner"]["nproc_per_node"] == 1
    train = yaml.safe_load((root / "conf/train" / (case["case"] + ".yaml")).read_text())
    assert train["system"]["logging"]["log_throughput"] is True
    assert train["model"]["train_iters"] >= 20
    assert train["model"]["te_fl_prefer"] == "reference"
    metrics = json.loads((root / "gold_values" / (case["case"] + ".json")).read_text())["iluvatar"][
        "iluvatar"
    ]
    for key, direction in (
        ("elapsed time per iteration (ms):", "upper_bound"),
        ("throughput per GPU (TFLOP/s/GPU):", "lower_bound"),
    ):
        assert len(metrics[key]["values"]) >= 20
        assert all(math.isfinite(value) and value > 0 for value in metrics[key]["values"])
        assert metrics[key]["threshold"]["type"] == direction
        assert metrics[key]["threshold"]["tolerance"] == 0.1
    assert (
        metrics["throughput per GPU (TFLOP/s/GPU):"]["threshold"]["aggregation"] == "harmonic_mean"
    )
