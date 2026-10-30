# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise configured deselection with real pytest collection and JUnit evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"
sys.path.insert(0, str(SCRIPTS))
import flagcicd_evidence as evidence
import flagcicd_plan as planner

TEST_FILE = "tests/unit_tests/test_parallel_context.py"
EXCLUDED = TEST_FILE + "::test_parallel_config"


def run_pytest(root, spec, extra=()):
    directory = root / "reports"
    results = []
    for rank in range(spec["ranks"]):
        results.append(
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "flagcicd_pytest",
                    *spec["command_targets"],
                    *("--deselect=" + prefix for prefix in spec["deselect"]),
                    *extra,
                    "-q",
                ],
                cwd=root,
                env={
                    **os.environ,
                    "PYTHONPATH": str(SCRIPTS),
                    "PYTEST_ADDOPTS": "",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "FLAGSCALE_CI_JUNIT_DIR": str(directory),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(spec["ranks"]),
                },
                capture_output=True,
                text=True,
                timeout=20,
            )
        )
    return directory, results


@pytest.fixture(scope="module")
def nightly_plan():
    return planner.build_plan(ROOT, profile="nightly", changed=None, labels=[])


@pytest.mark.parametrize("platform", ["hygon", "kunlunxin", "metax", "musa", "ppu"])
def test_real_platform_plan_allows_configured_single_test_file_deselection(
    tmp_path, nightly_plan, platform
):
    device = nightly_plan["slices"][platform]["devices"][0]
    spec = planner.unit_spec(nightly_plan, platform, device)
    config = yaml.safe_load(
        (ROOT / f"tests/test_utils/config/platforms/{platform}.yaml").read_text()
    )
    assert spec["deselect"] == [p for p in config[device]["tests"]["unit"]["exclude"] if "::" in p]
    assert EXCLUDED in spec["deselect"]
    assert TEST_FILE in spec["expected_files"]
    # Use real platform targets/topology, replacing hardware tests with trivial bodies.
    for source in (ROOT / "tests/unit_tests").rglob("__init__.py"):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    for name in spec["expected_files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        function = "test_parallel_config" if name == TEST_FILE else "test_ok"
        target.write_text(f"def {function}(): pass\n")
    directory, results = run_pytest(tmp_path, spec)
    for result in results:
        assert result.returncode == 0, result.stdout + result.stderr
    evidence.reports(directory, spec)
    evidence.validate_collection(directory, spec)
    for rank in range(spec["ranks"]):
        data = json.loads((directory / f"collection-rank-{rank}.json").read_text())
        assert TEST_FILE not in data["files"]
        assert EXCLUDED in data["collected_nodeids"]
        assert data["deselected_nodeids"] == [EXCLUDED]


@pytest.fixture
def unit_case(tmp_path):
    target = tmp_path / TEST_FILE
    target.parent.mkdir(parents=True)
    target.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', [1, 2])\n"
        "def test_parallel_config(value): pass\n"
        "def test_remaining(): pass\n"
    )
    return {
        "kind": "unit",
        "ranks": 1,
        "command_targets": [TEST_FILE],
        "expected_files": [TEST_FILE],
        "allowed_files": [TEST_FILE],
        "deselect": [EXCLUDED],
    }


def test_partial_file_deselection_matches_parametrized_nodeid_prefix(tmp_path, unit_case):
    directory, results = run_pytest(tmp_path, unit_case)
    assert results[0].returncode == 0, results[0].stdout + results[0].stderr
    evidence.reports(directory, unit_case)
    evidence.validate_collection(directory, unit_case)
    data = json.loads((directory / "collection-rank-0.json").read_text())
    assert data["deselected_nodeids"] == [EXCLUDED + "[1]", EXCLUDED + "[2]"]
    assert data["nodeids"] == [TEST_FILE + "::test_remaining"]


@pytest.mark.parametrize("mutation", ["unauthorized", "omitted-node", "missing-file"])
def test_successful_pytest_cannot_hide_invalid_selection(tmp_path, unit_case, mutation):
    extra = []
    if mutation == "unauthorized":
        unit_case["deselect"] = []
        extra = ["--deselect=" + EXCLUDED]
        message = "without plan authorization"
    elif mutation == "omitted-node":
        unit_case["deselect"] = []
        (tmp_path / "conftest.py").write_text(
            "def pytest_collection_modifyitems(items):\n"
            "    items[:] = [i for i in items if 'test_remaining' not in i.nodeid]\n"
        )
        message = "not accounted for"
    else:
        unit_case["expected_files"].append("tests/unit_tests/test_missing.py")
        message = "Planned unit files"
    directory, results = run_pytest(tmp_path, unit_case, extra)
    assert results[0].returncode == 0, results[0].stdout + results[0].stderr
    evidence.reports(directory, unit_case)
    with pytest.raises(ValueError, match=message):
        evidence.validate_collection(directory, unit_case)


@pytest.mark.parametrize("mode", ["deselect", "skip"])
def test_no_executed_test_cannot_pass(tmp_path, unit_case, mode):
    if mode == "deselect":
        unit_case["deselect"].append(TEST_FILE + "::test_remaining")
    else:
        (tmp_path / TEST_FILE).write_text(
            "import pytest\n@pytest.mark.skip(reason='fixture')\ndef test_remaining(): pass\n"
        )
    directory, results = run_pytest(tmp_path, unit_case)
    assert results[0].returncode != 0
    with pytest.raises(ValueError, match="Empty or all-skipped"):
        evidence.reports(directory, unit_case)
