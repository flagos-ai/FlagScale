# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise real pytest evidence and planned-target validation without devices."""

import copy
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"
sys.path.insert(0, str(SCRIPTS))
import check_results as gate
import flagcicd_plan as planner
import flagcicd_unit as unit


def selection():
    plan = planner.build_plan(
        ROOT,
        profile="pr",
        labels=[],
        changed=["tests/unit_tests/checkpoint/test_qwen35_converter.py"],
    )
    return {
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "mode": "affected",
        "targets": plan["slices"]["ascend"]["unit_targets"],
    }


def test_unit_target_identity_and_full_compatibility():
    data = selection()
    assert unit.targets(ROOT, data, "unit-910c") == [
        "tests/unit_tests/checkpoint/test_qwen35_converter.py"
    ]
    data.update(mode="full", targets={"unit-910c": []})
    assert unit.targets(ROOT, data, "unit-910c") == []


@pytest.mark.parametrize("mutation", ["empty", "stale", "escape", "option", "missing", "duplicate"])
def test_invalid_selection_fails(mutation):
    data = selection()
    path = data["targets"]["unit-910c"][0]
    if mutation == "stale":
        data["source_sha"] = "0" * 40
    else:
        data["targets"]["unit-910c"] = {
            "empty": [],
            "escape": ["tests/unit_tests/../../pyproject.toml.py"],
            "option": ["-k expression"],
            "missing": ["tests/unit_tests/absent.py"],
            "duplicate": [path, path],
        }[mutation]
    with pytest.raises(ValueError):
        unit.targets(ROOT, data, "unit-910c")


@pytest.mark.parametrize(
    "body,passed",
    [
        ("def test_ok(): assert True", True),
        ("import pytest\ndef test_skip(): pytest.skip('no device')", False),
        ("def test_bad(): assert False", False),
        ("# no tests", False),
        ("raise RuntimeError('collection failed')", False),
    ],
)
def test_real_pytest_plugin_never_passes_empty_or_all_skipped_runs(tmp_path, body, passed):
    (tmp_path / "test_sample.py").write_text(body)
    report = tmp_path / "reports"
    env = {
        **os.environ,
        "PYTHONPATH": str(SCRIPTS),
        "FLAGSCALE_CI_JUNIT_DIR": str(report),
        "LOCAL_RANK": "3",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "flagcicd_pytest", "-p", "no:cacheprovider", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) == passed, result.stdout + result.stderr
    tree = ET.parse(report / "junit-rank-3.xml")
    assert tree.getroot().tag == "testsuites"


def test_matrix_slices_and_gate_reject_missing_planned_train():
    plan = planner.build_plan(ROOT, profile="nightly", labels=[], changed=None)
    needs = {
        name: {"result": "success"}
        for name in (
            "checkout_and_config",
            "cli_validation",
            "unit_tests",
            "functional_tests_train",
            "functional_tests_benchmark",
        )
    }
    needs.update(
        {
            name: {"result": "skipped"}
            for name in (
                "functional_tests_hetero_train",
                "functional_tests_inference",
                "functional_tests_serve",
            )
        }
    )
    assert not planner.check_platform(plan, "ascend", needs)
    for state in ("skipped", "failure", "cancelled"):
        broken = copy.deepcopy(needs)
        broken["functional_tests_train"]["result"] = state
        assert "functional_tests_train" in planner.check_platform(plan, "ascend", broken)


def test_optional_workflow_inputs_preserve_legacy_callers_and_secrets():
    def workflow(name):
        return yaml.load(
            (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
        )

    pipeline = workflow("_platform_pipeline")
    common = workflow("all_tests_common")
    unit_workflow = workflow("unit_tests_common")
    assert pipeline["on"]["workflow_call"]["inputs"]["test_plan"]["default"] == ""
    assert common["on"]["workflow_call"]["inputs"]["test_plan"]["default"] == ""
    assert unit_workflow["on"]["workflow_call"]["inputs"]["unit_selection"]["default"] == ""
    assert "test_plan" not in pipeline["jobs"]["prepare"]["with"]
    assert pipeline["jobs"]["test"]["with"]["test_plan"] == "${{ inputs.test_plan }}"
    assert common["jobs"]["functional_tests_inference"]["if"].startswith("false")
    assert common["jobs"]["functional_tests_serve"]["if"].startswith("false")
    assert "functional_tests_rl" not in common["jobs"]


def test_pr_and_nightly_share_execution_but_not_trigger_or_cancellation():
    def workflow(name):
        return yaml.load(
            (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
        )

    ci, nightly = workflow("ci"), workflow("nightly")
    assert "schedule" not in ci["on"]
    assert set(nightly["on"]) == {"schedule", "workflow_dispatch"}
    assert nightly["on"]["schedule"] == [{"cron": "0 18 * * *"}]
    assert nightly["concurrency"]["cancel-in-progress"] == "false"
    assert ci["concurrency"]["group"] != nightly["concurrency"]["group"]
    for entry in (ci, nightly):
        pipeline = entry["jobs"]["platform-pipeline"]
        assert pipeline["uses"] == "./.github/workflows/_platform_pipeline.yml"
        assert pipeline["with"]["test_plan"] == "${{ needs.generate-matrix.outputs.plan }}"
        assert "source_sha" not in pipeline["with"]  # do not disable trusted main cache saves
        compute = next(
            step for step in entry["jobs"]["generate-matrix"]["steps"] if step.get("id") == "matrix"
        )
        assert "flagcicd_plan.py plan" in compute["run"]
    nightly_compute = next(
        step for step in nightly["jobs"]["generate-matrix"]["steps"] if step.get("id") == "matrix"
    )
    assert nightly_compute["env"]["PROFILE"] == "nightly"
    assert "ROLLOUT_MODE" not in nightly_compute["env"]
    assert nightly["jobs"]["ci-complete"]["name"] == "Nightly Complete"
    assert "Nightly" in workflow("notify-feishu")["on"]["workflow_run"]["workflows"]
    assert "Nightly" in workflow("ci-infra-retry")["on"]["workflow_run"]["workflows"]


def test_global_gate_rejects_missing_stale_or_modified_plan():
    plan = planner.build_plan(ROOT, profile="nightly", labels=[], changed=None)
    assert not gate.plan_failures(plan, plan["platforms"], plan["source_sha"])
    assert gate.plan_failures({}, [], plan["source_sha"])
    assert gate.plan_failures(plan, [], plan["source_sha"])
    assert gate.plan_failures(plan, plan["platforms"], "f" * 40)
    changed = copy.deepcopy(plan)
    changed["slices"]["ascend"]["cases"] = []
    assert gate.plan_failures(changed, plan["platforms"], plan["source_sha"])


@pytest.mark.parametrize("platform,device", [("cuda", "a100"), ("enflame", "s60")])
@pytest.mark.parametrize("selection_mode", [None, "full", "affected"])
def test_real_shell_runner_retains_topology_and_exclusions(
    tmp_path, platform, device, selection_mode
):
    planned = selection_mode is not None
    root = tmp_path / "repo"
    runner_dir = root / "tests/test_utils/runners"
    runner_dir.mkdir(parents=True)
    scripts = root / ".github/scripts"
    (scripts / "ci").mkdir(parents=True)
    for filename in ("run_unit_tests.sh", "helpers.py"):
        shutil.copy(ROOT / "tests/test_utils/runners" / filename, runner_dir / filename)
    for helper in ("flagcicd_unit.py", "flagcicd_evidence.py"):
        shutil.copy(SCRIPTS / helper, scripts / "ci" / helper)
    (scripts / "set_env_common.sh").write_text("""
ci_resolve_python_bin() { export CI_PYTHON_BIN="$CHECK_PYTHON"; }
ci_configure_training_pythonpath() { export PYTHONPATH="$PROJECT_ROOT"; }
""")
    # Use the real configuration-Python wrapper. A stub here would miss the
    # vendor startup/stdout boundary that differs between CUDA and Enflame.
    (runner_dir / "utils.sh").write_text(
        (ROOT / "tests/test_utils/runners/utils.sh").read_text()
        + """
log_info() { :; }; log_error() { echo "$*" >&2; }; log_success() { :; }
validate_platform() { :; }; validate_device() { :; }; wait_for_gpu() { :; }
default_dist_backend() { echo gloo; }; default_torch_device_type() { echo cpu; }
"""
    )
    (runner_dir / "parse_config.py").write_text("""
import json
print(json.dumps({"include": "*", "exclude": ["tests/unit_tests/ignored.py"], "nproc_per_node": 2}))
""")
    target = root / "tests/unit_tests/test_sample.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_ok(): assert True\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=CI test",
            "-c",
            "user.email=ci@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    workspace = tmp_path / "workspace"
    (workspace / "megatron-lm-fl-install").mkdir(parents=True)
    binary = tmp_path / "bin"
    binary.mkdir()
    torchrun = binary / "torchrun"
    torchrun.write_text(
        f'#!{sys.executable}\nimport json, os, sys\nopen(os.environ["TRACE"], "w").write(json.dumps(sys.argv[1:]))\n'
    )
    torchrun.chmod(0o755)
    python_bin = sys.executable
    if platform == "enflame":
        # Model the image's .pth hook: ordinary Python prints vendor diagnostics
        # before a script runs, whereas -S prevents that startup side effect.
        python_wrapper = binary / "vendor-python"
        python_wrapper.write_text(
            f"#!{sys.executable}\nimport os, sys\n"
            'if "-S" not in sys.argv[1:]:\n'
            '    print("CUDA vendor backend skipped (CUDA build was disabled at build time)", flush=True)\n'
            f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
        )
        python_wrapper.chmod(0o755)
        python_bin = str(python_wrapper)
    env = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "CHECK_PYTHON": python_bin,
        "GITHUB_WORKSPACE": str(workspace),
        "TRACE": str(tmp_path / "trace.json"),
    }
    env.pop("COVERAGE_DIR", None)
    args = [
        "bash",
        str(runner_dir / "run_unit_tests.sh"),
        "--platform",
        platform,
        "--device",
        device,
    ]
    if planned:
        selection_file = tmp_path / "selection.json"
        selection_file.write_text(
            json.dumps(
                {
                    "source_sha": sha,
                    "plan_hash": "a" * 64,
                    "mode": selection_mode,
                    "targets": {
                        device: ["tests/unit_tests/test_sample.py"]
                        if selection_mode == "affected"
                        else []
                    },
                }
            )
        )
        args.extend(["--selection-file", str(selection_file)])
    result = subprocess.run(args, env=env, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = json.loads((tmp_path / "trace.json").read_text())
    assert argv[0] == "--nproc_per_node=2"
    assert "--ignore=tests/unit_tests/ignored.py" in argv
    assert ("flagcicd_pytest" in argv) == planned
    assert (
        "tests/unit_tests/test_sample.py" if selection_mode == "affected" else "tests/unit_tests/"
    ) in argv
    assert "CUDA" not in argv
    if planned:
        assert (root / f"test-results/unit/{device}/selection.json").is_file()


def test_all_planned_entrypoints_and_gates_consume_execution_evidence():
    def workflow(name):
        return yaml.load(
            (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
        )

    requested = workflow("ci-request")
    assert (
        requested["jobs"]["platform-pipeline"]["with"]["test_plan"]
        == "${{ needs.resolve.outputs.plan }}"
    )
    controller = next(s for s in requested["jobs"]["resolve"]["steps"] if s.get("id") == "matrix")
    assert (
        "--profile requested" in controller["run"]
        and '--root "$GITHUB_WORKSPACE/candidate"' in controller["run"]
    )
    assert "secrets" not in requested["jobs"]["platform-pipeline"]
    for name in ("ci", "nightly", "ci-request"):
        entry = workflow(name)
        gate_job = entry["jobs"]["requested-complete" if name == "ci-request" else "ci-complete"]
        assert gate_job["if"].startswith("always()")
        assert any(s.get("uses") == "actions/download-artifact@v4" for s in gate_job["steps"])
        assert any("check_results.py" in s.get("run", "") for s in gate_job["steps"])
        assert "CI_AFFECTED_TESTS_ENABLED" not in json.dumps(entry)
    common = workflow("all_tests_common")["jobs"]["all_tests_complete"]
    assert any("flagcicd_evidence.py verify" in s.get("run", "") for s in common["steps"])
    for task in ("train", "hetero_train", "benchmark"):
        steps = next(iter(workflow("functional_tests_" + task)["jobs"].values()))["steps"]
        assert any("flagcicd_evidence.py" in s.get("run", "") for s in steps)
        assert any(
            s.get("uses") == "actions/upload-artifact@v4"
            and s.get("with", {}).get("name", "").startswith("ci-evidence-")
            for s in steps
        )
