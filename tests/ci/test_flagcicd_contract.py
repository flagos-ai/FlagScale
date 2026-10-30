# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Policy-driven discovery, schema validation and missing execution adapters."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts/ci"))
import flagcicd_contract as contract


def edit(root, mutate):
    path = root / ".flagcicd/project.yaml"
    value = yaml.safe_load(path.read_text())
    mutate(value)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def test_single_policy_discovers_test_catalog():
    project, catalog = contract.load_contract(ROOT)
    assert not (ROOT / ".flagcicd/test-inventory.json").exists()
    assert not (ROOT / ".flagcicd/test-tiers.json").exists()
    entries = {e["path"]: e for e in catalog}
    assert entries["tests/unit_tests/test_basic.py"]["kind"] == "placeholder"
    assert entries["tests/unit_tests/test_utilities.py"]["kind"] == "helper"
    assert entries["tests/unit_tests/test_cli.py"]["group"] == "entrypoint"
    assert entries["tests/unit_tests/models/kerv/test_ops.py"]["suite"] == "accelerator-unit"
    assert set(project["profiles"]) == {"pr", "branch", "nightly", "requested"}


def test_new_test_in_existing_group_requires_no_inventory_edit(policy_repo):
    path = policy_repo / "tests/unit_tests/runner/test_new_launcher.py"
    path.write_text("def test_new(): pass\n")
    _, catalog = contract.load_contract(policy_repo)
    assert any(
        e["path"] == str(path.relative_to(policy_repo)) and e["group"] == "runner" for e in catalog
    )


@pytest.mark.parametrize(
    "path", ["tests/unit_tests/new_feature/test_new.py", "tests/new_suite/test_new.py"]
)
def test_unwired_test_blocks_even_when_other_files_trigger_full(policy_repo, path):
    p = policy_repo / path
    p.parent.mkdir(parents=True)
    p.write_text("def test_new(): pass\n")
    with pytest.raises(ValueError, match="Unowned test|execution adapter"):
        contract.load_contract(policy_repo)


@pytest.mark.parametrize("mutation", ["unknown", "conflict", "stale", "profile", "inert", "unsafe"])
def test_invalid_policy_rejected(policy_repo, mutation):
    def change(p):
        if mutation == "unknown":
            p["test_groups"]["runner"]["affects"] = ["absent"]
        if mutation == "conflict":
            p["test_groups"]["entrypoint"]["tests"].append("tests/unit_tests/runner/")
        if mutation == "stale":
            p["test_exceptions"]["tests/unit_tests/absent.py"] = {
                "kind": "helper",
                "reason": "missing",
            }
        if mutation == "profile":
            p["profiles"]["nightly"]["selection"] = "affected"
        if mutation == "inert":
            p["levels"] = {"L4": {"readiness": "complete"}}
        if mutation == "unsafe":
            p["test_groups"]["runner"]["sources"] = ["../outside/"]

    edit(policy_repo, change)
    with pytest.raises(ValueError):
        contract.load_contract(policy_repo)


def test_duplicate_yaml_key_rejected(policy_repo):
    p = policy_repo / ".flagcicd/project.yaml"
    p.write_text(p.read_text() + "\nrepository: Other\n")
    with pytest.raises(ValueError, match="Duplicate"):
        contract.load_contract(policy_repo)


@pytest.mark.parametrize(
    "path", ["tests/ci/nested/unwired.test.cjs", "tests/unit_tests/runner/unwired.test.cjs"]
)
def test_owned_test_with_wrong_language_or_glob_has_no_execution_adapter(policy_repo, path):
    target = policy_repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("// new test\n")
    with pytest.raises(ValueError, match="execution adapter"):
        contract.load_contract(policy_repo)
