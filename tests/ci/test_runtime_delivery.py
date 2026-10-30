# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Check the producer/consumer contract across the actual reusable workflows."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())


@pytest.mark.parametrize(
    "package,publish", [("megatron", "publish_megatron"), ("te_fl", "publish_te")]
)
def test_cache_hits_still_publish_validated_artifacts(package, publish):
    producer = workflow("prepare_dependencies")["jobs"]["prepare"]
    steps = producer["steps"]
    upload = next(step for step in steps if step.get("id") == publish)
    # Cache-hit must not suppress delivery; success, not an inferred cache state,
    # is the only way to tell a consumer that the fallback exists.
    assert upload["if"] == f"needs.resolve.outputs.{package}_enabled == 'true'"
    assert producer["outputs"][f"{package}_artifact_available"] == (
        "${{ steps." + publish + ".outcome == 'success' }}"
    )
    validator = next(
        step
        for step in steps
        if step["name"]
        == f"Validate prepared {'te' if package == 'te_fl' else 'megatron'} before publication"
    )
    assert steps.index(validator) < steps.index(upload)
    assert "continue-on-error" not in validator
    assert "validate_prepared_runtime.py" in validator["run"]
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["overwrite"] is True


@pytest.mark.parametrize(
    "name",
    [
        "unit_tests_common",
        "functional_tests_train",
        "functional_tests_hetero_train",
        "functional_tests_benchmark",
    ],
)
def test_every_platform_reaches_fallback_after_failed_cache_restore(name):
    steps = [step for job in workflow(name)["jobs"].values() for step in job.get("steps", [])]
    caches = [step for step in steps if step.get("uses") == "actions/cache/restore@v4"]
    recovery = [
        step for step in steps if step.get("uses", "").endswith("/restore-prepared-runtime")
    ]
    assert len(caches) == len(recovery) == 2
    assert not any(step.get("uses") == "actions/download-artifact@v4" for step in steps)
    for cache in caches:
        assert cache["continue-on-error"] is True
        assert cache["timeout-minutes"] == 15
    for step in recovery:
        assert "inputs.platform" not in step["if"]
        assert "cache-hit" not in step["if"]
        assert "${{ inputs.platform }}" in step["with"]["artifact-name"]
        assert "continue-on-error" not in step


def test_privileged_controller_has_bounded_code_policy_and_trusted_source():
    data = workflow("ci-infra-retry")
    job = data["jobs"]["inspect"]
    assert "vars." not in job["if"]
    assert "github.event.workflow_run.conclusion == 'failure'" in job["if"]
    assert "github.event.workflow_run.run_attempt == 1" in job["if"]
    assert "github.event.workflow_run.repository.full_name == github.repository" in job["if"]
    assert (
        "github.event.workflow_run.head_branch == github.event.repository.default_branch"
        in job["if"]
    )
    assert "github.event.workflow_run.event == 'push'" in job["if"]
    assert "github.event.workflow_run.event == 'schedule'" in job["if"]
    assert "CI_INFRA_RETRY_ENABLED" not in str(job)
    checkout = job["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert checkout["with"]["persist-credentials"] is False
    assert data["permissions"] == {"contents": "read", "actions": "write"}
    assert data["concurrency"]["cancel-in-progress"] is False
