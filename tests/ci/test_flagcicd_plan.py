# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Platform/case routing, explicit impact and configured regression equivalence."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_flagcicd_contract import edit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts/ci"))
import flagcicd_plan as planner
from generate_matrix import changed_files, documentation


def plan(changed=None, profile="pr", labels=None, root=ROOT, **kwargs):
    return planner.build_plan(root, profile=profile, changed=changed, labels=labels or [], **kwargs)


@pytest.mark.parametrize("profile", ["branch", "nightly"])
def test_full_plan_matches_every_enabled_configured_case(profile):
    result = plan(profile=profile)
    assert len(result["platforms"]) == 8
    for platform, s in result["slices"].items():
        config = yaml.safe_load(
            (ROOT / f"tests/test_utils/config/platforms/{platform}.yaml").read_text()
        )
        expected = {
            (d, t, m, c)
            for d in config["device_types"]
            for t in planner.ACTIVE_TASKS
            for m, cases in config[d]["tests"]["functional"].get(t, {}).items()
            for c in cases
        }
        assert {
            tuple(c[k] for k in ("device", "task", "model", "case")) for c in s["cases"]
        } == expected
        assert s["unit_mode"] == "full"
    assert "mode" not in result and "capability_readiness" not in result
    assert all(
        "exclusive" not in c and "level" not in c
        for s in result["slices"].values()
        for c in s["cases"]
    )


def test_registered_unit_group_is_used_without_feature_flag():
    result = plan(["tests/unit_tests/checkpoint/test_qwen35_converter.py"])
    assert result["selected_groups"] == ["checkpoint"]
    for s in result["slices"].values():
        assert s["unit_mode"] == "affected" and not s["cases"]
        assert all(
            v == ["tests/unit_tests/checkpoint/test_qwen35_converter.py"]
            for v in s["unit_targets"].values()
        )


def test_ordinary_labels_do_not_change_execution_selection():
    files = ["tests/unit_tests/checkpoint/test_qwen35_converter.py"]
    assert (
        plan(files)["plan_hash"]
        == plan(files, labels=["bug", "documentation", "urgent"])["plan_hash"]
    )


def test_platform_labels_only_add_platforms_and_request_full_supplement():
    p = plan(["docker/cuda/Dockerfile.train"], labels=["platform:ascend"])
    assert p["platforms"] == ["ascend", "cuda"]
    assert all(s["unit_mode"] == "full" for s in p["slices"].values())
    assert len(plan(["flagscale/run.py"], labels=["platform:ascend"])["platforms"]) == 8


@pytest.mark.parametrize(
    "changed",
    [
        None,
        [],
        ["new/unknown.py"],
        [".github/workflows/ci.yml"],
        ["tests/conftest.py"],
        ["flagscale/runner/runner_train.py"],
    ],
)
def test_shared_unknown_and_missing_diff_keep_configured_regression(changed):
    p = plan(changed)
    assert len(p["platforms"]) == 8
    assert all(s["unit_mode"] == "full" and s["cases"] for s in p["slices"].values())


def test_backend_source_includes_cross_group_unit_and_training_launch_chain():
    p = plan(["flagscale/runner/backend/backend_megatron.py"])
    assert p["direct_groups"] == ["megatron-train"]
    assert {"megatron-train", "runner", "train-utils"} <= set(p["selected_groups"])
    assert all(s["unit_mode"] == "affected" and s["cases"] for s in p["slices"].values())


def test_transformation_inference_and_models_are_not_skipped_with_disabled_e2e():
    p = plan(["flagscale/inference/runtime_context.py"])
    assert {"transformations", "kerv"} <= set(p["selected_groups"])
    assert p["gaps"]
    assert all(
        any("/transformations/" in f for f in targets)
        for s in p["slices"].values()
        for targets in s["unit_targets"].values()
    )


def test_multiple_changes_union_groups_and_cases():
    p = plan(
        [
            "tools/checkpoint/qwen35/qwen35/io.py",
            "flagscale/serve/processor/image_resize_processor.py",
        ]
    )
    assert {"checkpoint", "serve-processor"} <= set(p["selected_groups"])
    assert not any(s["cases"] for s in p["slices"].values())


def test_case_configuration_reverse_index_chooses_platform_and_case():
    p = plan(["tests/functional_tests/train/qwen3/conf/train/0_6b_ascend.yaml"])
    assert p["platforms"] == ["ascend"]
    assert [c["case"] for c in p["slices"]["ascend"]["cases"]] == ["0_6b_ascend"]
    p = plan(["tests/functional_tests/train/qwen3/conf/train/data.yaml"])
    assert {"ascend", "enflame", "metax", "musa", "ppu"} <= set(p["platforms"])


def test_shared_case_used_by_two_platforms_is_not_routed_by_filename(policy_repo):
    p = policy_repo / "tests/test_utils/config/platforms/hygon.yaml"
    config = yaml.safe_load(p.read_text())
    config["bw1000"]["tests"]["functional"]["train"]["qwen3"].append("0_6b_ascend")
    p.write_text(yaml.safe_dump(config))
    result = plan(["tests/functional_tests/train/qwen3/conf/0_6b_ascend.yaml"], root=policy_repo)
    assert result["platforms"] == ["ascend", "hygon"]


def test_deleted_case_or_unit_keeps_conservative_execution(policy_repo):
    path = "tests/unit_tests/checkpoint/test_qwen35_converter.py"
    (policy_repo / path).unlink()
    p = plan([path], root=policy_repo)
    assert all(s["unit_mode"] == "full" for s in p["slices"].values())


def test_git_rename_diff_preserves_old_and_new_paths(policy_repo):
    old = "tests/unit_tests/checkpoint/test_qwen35_converter.py"
    new = "tests/unit_tests/runner/test_moved_converter.py"
    subprocess.run(["git", "mv", old, new], cwd=policy_repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "move test",
        ],
        cwd=policy_repo,
        check=True,
    )
    result = changed_files("HEAD^", "HEAD", root=policy_repo)
    assert set(result) == {old, new}
    p = plan(result, root=policy_repo)
    assert {"checkpoint", "runner"} <= set(p["selected_groups"])


def test_unknown_test_cannot_hide_behind_shared_fallback(policy_repo):
    new = policy_repo / "tests/unit_tests/new_suite/test_new.py"
    new.parent.mkdir()
    new.write_text("def test_new(): pass\n")
    with pytest.raises(ValueError, match="Unowned test"):
        plan([".github/workflows/ci.yml", str(new.relative_to(policy_repo))], root=policy_repo)


def test_project_configuration_changes_behavior(policy_repo):
    edit(policy_repo, lambda p: p["profiles"]["pr"].update(selection="configured"))
    p = plan(["tests/unit_tests/checkpoint/test_qwen35_converter.py"], root=policy_repo)
    assert all(s["unit_mode"] == "full" and s["cases"] for s in p["slices"].values())


def test_requested_profile_is_configured_scope_for_exact_platforms():
    p = plan(profile="requested", requested_platforms=["kunlunxin"])
    assert p["platforms"] == ["kunlunxin"]
    assert p["slices"]["kunlunxin"]["unit_mode"] == "full"
    assert len(p["slices"]["kunlunxin"]["cases"]) == 2
    with pytest.raises(ValueError):
        plan(profile="requested", requested_platforms=["absent"])


def test_exclusions_still_apply_and_no_global_core_group():
    p = plan(["tests/unit_tests/models/kerv/test_configs.py"])
    targets = p["slices"]["ascend"]["unit_targets"]["unit-910c"]
    assert "tests/unit_tests/models/kerv/test_configs.py" in targets
    assert "tests/unit_tests/models/kerv/test_ops.py" not in targets
    assert "unit-core" not in p["selected_groups"]


def test_plan_identity_and_configuration_drift_rejected(policy_repo):
    p = plan(profile="nightly", root=policy_repo)
    planner.verify_plan(policy_repo, p)
    changed = copy.deepcopy(p)
    changed["source_sha"] = "a" * 40
    with pytest.raises(ValueError, match="hash"):
        planner.verify_plan(policy_repo, changed)
    changed["plan_hash"] = planner.fingerprint(
        {k: v for k, v in changed.items() if k != "plan_hash"}
    )
    with pytest.raises(ValueError, match="source SHA"):
        planner.verify_plan(policy_repo, changed)
    file = policy_repo / p["slices"]["ascend"]["cases"][0]["config"]
    file.write_text(file.read_text() + "\n# drift\n")
    with pytest.raises(ValueError, match="configuration"):
        planner.verify_plan(policy_repo, p)


def test_documentation_skips_hardware_but_examples_are_runtime_inputs():
    assert plan(["README.md"])["platforms"] == []
    assert not documentation("examples/kerv/conf/train.yaml")
    assert not documentation("examples/runtime.py")
    assert plan(["examples/kerv/conf/runtime.yaml"])["platforms"]


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/test.py", "-x", "a/../x", "a b.py"])
def test_unsafe_paths_fail_before_commands(path):
    with pytest.raises(ValueError, match="path"):
        plan([path])


def test_unconfigured_new_functional_case_cannot_hide_behind_full(policy_repo):
    path = policy_repo / "tests/functional_tests/train/new_model/conf/new.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("experiment:\n  task:\n    type: train\n    backend: native\n")
    with pytest.raises(ValueError, match="Unconfigured functional"):
        plan([".github/workflows/ci.yml", str(path.relative_to(policy_repo))], root=policy_repo)


def test_new_test_excluded_on_every_platform_requires_execution_adapter(policy_repo):
    file = policy_repo / "tests/unit_tests/runner/test_new_launcher.py"
    file.write_text("def test_new(): pass\n")
    for name in planner.load_platforms(policy_repo / ".github/configs"):
        path = policy_repo / f"tests/test_utils/config/platforms/{name}.yaml"
        config = yaml.safe_load(path.read_text())
        for device in config["device_types"]:
            config[device]["tests"]["unit"].setdefault("exclude", []).append(
                str(file.relative_to(policy_repo))
            )
        path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="excluded on every platform"):
        plan([".github/workflows/ci.yml"], root=policy_repo)


def test_selection_is_deterministic_for_diff_order():
    files = ["tools/checkpoint/qwen35/qwen35/io.py", "flagscale/models/kerv/config.py"]
    assert plan(files)["plan_hash"] == plan(list(reversed(files)))["plan_hash"]


def test_standalone_platform_cli_uses_configured_case_consumers():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / ".github/scripts/ci/generate_matrix.py"),
            "--event-name",
            "pull_request",
            "--changed-files",
            "tests/functional_tests/train/qwen3/conf/0_6b_ascend.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.splitlines()[0].removeprefix("platforms=")) == ["ascend"]
