# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Real pytest/shell evidence and gates reject omitted, stale or wrong execution."""

import copy
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"
sys.path.insert(0, str(SCRIPTS))
import flagcicd_evidence as evidence
import flagcicd_plan as planner


def setup_runners(root, tmp_path):
    directory = root / "tests/test_utils/runners"
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        "run_unit_tests.sh",
        "run_tests.sh",
        "run_functional_tests.sh",
        "helpers.py",
        "parse_config.py",
        "utils.sh",
    ):
        shutil.copy(ROOT / "tests/test_utils/runners" / name, directory / name)
    with (directory / "utils.sh").open("a") as f:
        f.write("\nwait_for_gpu() { :; }\n")
    shutil.copytree(
        SCRIPTS,
        root / ".github/scripts/ci",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (
        root / ".github/scripts/set_env_common.sh"
    ).write_text(f"""ci_resolve_python_bin() {{ export CI_PYTHON_BIN={sys.executable}; }}
ci_configure_training_pythonpath() {{ export PYTHONPATH="$PROJECT_ROOT"; }}
ci_setup_isolated_hf_modules_cache() {{ :; }}
""")
    (root / "tests/conftest.py").write_text("""def pytest_addoption(parser):
    for name in ('path','task','model','case','platform','device'):
        parser.addoption('--'+name, default='')
""")
    binary = tmp_path / "bin"
    binary.mkdir()
    launch = binary / "torchrun"
    launch.write_text(f"""#!{sys.executable}
import os,subprocess,sys
n=int(sys.argv[1].split('=')[1])
for rank in range(n):
    result=subprocess.run([sys.executable,*sys.argv[2:]],env={{**os.environ,'LOCAL_RANK':str(rank),'WORLD_SIZE':str(n)}})
    if result.returncode: sys.exit(result.returncode)
""")
    launch.chmod(0o755)
    cli = binary / "flagscale"
    cli.write_text(
        f'#!{sys.executable}\nfrom pathlib import Path\nPath("cli-executed").write_text("ok")\n'
    )
    cli.chmod(0o755)
    (tmp_path / "megatron-lm-fl-install").mkdir()
    return {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "GITHUB_WORKSPACE": str(tmp_path),
        "PROJECT_ROOT": str(root),
        "GITHUB_RUN_ID": "77",
        "GITHUB_RUN_ATTEMPT": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def test_affected_plan_runs_real_shell_pytest_and_both_gates(policy_repo, tmp_path, monkeypatch):
    root = policy_repo
    target = root / "tests/unit_tests/checkpoint/test_qwen35_converter.py"
    target.write_text("def test_converter(): assert True\n")
    env = setup_runners(root, tmp_path)
    monkeypatch.setattr(
        planner, "select_platforms", lambda *a, **kw: (["ascend"], "fixture platform")
    )
    plan = planner.build_plan(
        root, profile="pr", changed=[str(target.relative_to(root))], labels=[]
    )
    selection = root / "selection.json"
    selection.write_text(planner.outputs(plan, "ascend")["unit_selection"])
    run = subprocess.run(
        [
            "bash",
            str(root / "tests/test_utils/runners/run_unit_tests.sh"),
            "--platform",
            "ascend",
            "--device",
            "unit-910c",
            "--selection-file",
            str(selection),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=40,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    artifacts = tmp_path / "downloaded"
    shutil.copytree(root / "test-results/evidence", artifacts)
    evidence.verify_results(plan, artifacts, platform="ascend", run_id="77", attempt=1)
    needs = {
        "lint": {"result": "success"},
        "generate-matrix": {"result": "success"},
        "platform-pipeline": {"result": "success"},
    }
    gate = subprocess.run(
        [sys.executable, str(SCRIPTS / "check_results.py")],
        env={
            **env,
            "PLAN_JSON": json.dumps(plan),
            "SOURCE_SHA": plan["source_sha"],
            "PLATFORMS": json.dumps(plan["platforms"]),
            "JOB_RESULTS": json.dumps(needs),
            "EVIDENCE_ROOT": str(artifacts),
        },
        capture_output=True,
        text=True,
    )
    assert gate.returncode == 0, gate.stdout + gate.stderr
    next(artifacts.rglob("junit-rank-1.xml")).unlink()
    with pytest.raises(ValueError, match="JUnit"):
        evidence.verify_results(plan, artifacts, run_id="77", attempt=1)


def test_functional_shared_runner_cli_and_numerical_comparator_produce_bound_evidence(
    policy_repo, tmp_path
):
    root = policy_repo
    env = setup_runners(root, tmp_path)
    checker = root / "tests/test_utils/runners/check_results.py"
    checker.write_text(
        'from pathlib import Path\ndef test_train_equal(): assert Path("cli-executed").read_text()=="ok"\n'
    )
    plan = planner.build_plan(
        root, profile="requested", changed=None, labels=[], requested_platforms=["ascend"]
    )
    case = next(c for c in plan["slices"]["ascend"]["cases"] if c["task"] == "train")
    spec = planner.functional_spec(plan, "ascend", case)
    spec_file = root / "functional.json"
    spec_file.write_text(json.dumps(spec))
    command = [
        sys.executable,
        str(SCRIPTS / "flagcicd_evidence.py"),
        "functional",
        "--spec-file",
        str(spec_file),
        "--",
        "bash",
        str(root / "tests/test_utils/runners/run_tests.sh"),
        "--platform",
        "ascend",
        "--device",
        case["device"],
        "--type",
        "functional",
        "--task",
        case["task"],
        "--model",
        case["model"],
        "--list",
        case["case"],
    ]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    path = next((root / "test-results/evidence").rglob("result.json"))
    evidence.verify_result(path, spec, run_id="77", attempt=1)
    wrong = copy.deepcopy(spec)
    wrong["case"] = "other_case"
    with pytest.raises(ValueError, match="identity"):
        evidence.verify_result(path, wrong, run_id="77", attempt=1)
    report = path.parent / "junit.xml"
    report.write_text(report.read_text().replace(f'value="{case["case"]}"', 'value="wrong-case"'))
    with pytest.raises(ValueError, match="comparator identity"):
        evidence.verify_result(path, spec, run_id="77", attempt=1)


@pytest.fixture
def proof(tmp_path):
    directory = tmp_path / "proof"
    directory.mkdir()
    spec = {
        "source_sha": "a" * 40,
        "plan_hash": "b" * 64,
        "kind": "unit",
        "platform": "ascend",
        "id": "unit:d",
        "device": "d",
        "mode": "affected",
        "targets": {"d": ["tests/unit_tests/test_a.py"]},
        "expected_files": ["tests/unit_tests/test_a.py"],
        "allowed_files": ["tests/unit_tests/test_a.py"],
        "command_targets": ["tests/unit_tests/test_a.py"],
        "ranks": 2,
        "deselect": [],
    }
    for rank in range(2):
        (directory / f"junit-rank-{rank}.xml").write_text(
            '<testsuites><testsuite tests="1"><testcase name="test_a"/></testsuite></testsuites>'
        )
        (directory / f"collection-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": 2,
                    "args": spec["command_targets"],
                    "files": spec["expected_files"],
                    "collection_skips": [],
                    "nodeids": ["tests/unit_tests/test_a.py::test_a"],
                    "collected_nodeids": ["tests/unit_tests/test_a.py::test_a"],
                    "deselected_nodeids": [],
                    "passed_nodeids": ["tests/unit_tests/test_a.py::test_a"],
                }
            )
        )
    result = {
        "schema_version": "flagcicd.result/v1",
        "spec": spec,
        "run_id": "77",
        "run_attempt": 1,
        "exit_code": 0,
        "status": "passed",
        "reports": evidence.reports(directory, spec),
        "collections": evidence.validate_collection(directory, spec),
    }
    path = directory / "result.json"
    path.write_text(json.dumps(result))
    return directory, spec, path


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-rank",
        "broken-xml",
        "empty",
        "skipped",
        "failure",
        "count",
        "wrong-file",
        "wrong-target",
        "wrong-rank",
        "no-automated-pass",
        "missing-file",
        "omitted-case",
        "changed-after-record",
        "stale-source",
        "other-run",
        "future-attempt",
        "failed-exit",
    ],
)
def test_gate_rejects_invalid_execution_evidence(proof, mutation):
    directory, spec, path = proof
    if mutation == "missing-rank":
        (directory / "junit-rank-1.xml").unlink()
    if mutation == "broken-xml":
        (directory / "junit-rank-0.xml").write_text("invalid")
    variants = {
        "empty": '<testsuite tests="0"/>',
        "skipped": '<testsuite tests="1"><testcase><skipped/></testcase></testsuite>',
        "failure": '<testsuite tests="1"><testcase><failure/></testcase></testsuite>',
        "count": '<testsuite tests="9"><testcase/></testsuite>',
    }
    if mutation in variants:
        (directory / "junit-rank-0.xml").write_text(variants[mutation])
    if mutation in {
        "wrong-file",
        "wrong-target",
        "wrong-rank",
        "missing-file",
        "omitted-case",
        "no-automated-pass",
    }:
        f = directory / "collection-rank-0.json"
        d = json.loads(f.read_text())
        if mutation == "wrong-file":
            d["files"] = ["tests/unit_tests/wrong.py"]
        if mutation == "wrong-target":
            d["args"] = ["tests/unit_tests/"]
        if mutation == "wrong-rank":
            d["world_size"] = 8
        if mutation == "missing-file":
            d["files"] = []
        if mutation == "omitted-case":
            d["nodeids"].append("tests/unit_tests/test_a.py::test_omitted")
        f.write_text(json.dumps(d))
    if mutation == "no-automated-pass":
        f = directory / "collection-rank-0.json"
        data = json.loads(f.read_text())
        data["passed_nodeids"] = []
        f.write_text(json.dumps(data))
    if mutation == "changed-after-record":
        f = directory / "junit-rank-0.xml"
        f.write_text(f.read_text() + "\n")
    if mutation in {"stale-source", "other-run", "future-attempt", "failed-exit"}:
        d = json.loads(path.read_text())
        if mutation == "stale-source":
            d["spec"]["source_sha"] = "c" * 40
        if mutation == "other-run":
            d["run_id"] = "78"
        if mutation == "future-attempt":
            d["run_attempt"] = 2
        if mutation == "failed-exit":
            d["exit_code"] = 1
        path.write_text(json.dumps(d))
    with pytest.raises((ValueError, OSError, ET.ParseError)):
        evidence.verify_result(path, spec, run_id="77", attempt=1)


def small_plan(spec):
    return {
        "platforms": ["ascend"],
        "plan_hash": spec["plan_hash"],
        "source_sha": spec["source_sha"],
        "gate": {"require_unit_evidence": True, "require_functional_evidence": True},
        "slices": {
            "ascend": {
                "devices": ["d"],
                "unit_mode": "affected",
                "unit_targets": spec["targets"],
                "unit_files": {"d": spec["expected_files"]},
                "unit_allowed_files": {"d": spec["allowed_files"]},
                "unit_ranks": {"d": 2},
                "unit_deselect": {"d": spec["deselect"]},
                "unit_command_targets": {"d": spec["command_targets"]},
                "cases": [],
            }
        },
    }


def test_gate_checks_missing_duplicate_unexpected_and_attempt_precedence(proof, tmp_path):
    directory, spec, path = proof
    plan = small_plan(spec)
    evidence.verify_results(plan, directory, run_id="77", attempt=2)
    duplicate = directory / "duplicate"
    duplicate.mkdir()
    shutil.copy(path, duplicate / "result.json")
    with pytest.raises(ValueError, match="Duplicate"):
        evidence.verify_results(plan, directory, run_id="77", attempt=2)
    data = json.loads((duplicate / "result.json").read_text())
    data["run_attempt"] = 2
    data["status"] = "failed"
    (duplicate / "result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="did not pass"):
        evidence.verify_results(plan, directory, run_id="77", attempt=2)
    data["spec"]["id"] = "unit:unexpected"
    (duplicate / "result.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unexpected"):
        evidence.verify_results(plan, directory, run_id="77", attempt=2)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="Missing"):
        evidence.verify_results(plan, empty, run_id="77", attempt=2)
