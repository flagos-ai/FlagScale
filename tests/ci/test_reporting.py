# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Check credential reachability and execute the report preflight without accelerators."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"


def workflow(name):
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def test_reporting_secret_reaches_nested_uploads():
    """Walk all entry points, including legacy image-build/manual test calls."""
    files = {p.name: workflow(p.name) for p in WORKFLOWS.glob("*.yml")}
    leaves = {"unit_tests_common.yml", "functional_tests_benchmark.yml"}

    def reaches_upload(name, seen=frozenset()):
        if name in leaves:
            return True
        if name in seen:
            return False
        return any(
            reaches_upload(job["uses"].split("/")[-1], seen | {name})
            for job in files[name].get("jobs", {}).values()
            if job.get("uses", "").startswith("./.github/workflows/")
        )

    for name, content in files.items():
        for job in content.get("jobs", {}).values():
            target = job.get("uses", "")
            if not target.startswith("./.github/workflows/"):
                continue
            target = target.split("/")[-1]
            if not reaches_upload(target):
                continue
            if name == "ci-request.yml":
                # Explicit PR source runs remain credential-free even on workflow_dispatch.
                assert "secrets" not in job
                continue
            passed = job.get("secrets", {})
            assert passed == "inherit" or passed.get("FLAGCICD_API_TOKEN") == (
                "${{ secrets.FLAGCICD_API_TOKEN }}"
            ), (name, target)
            declaration = (
                files[target]["on"]["workflow_call"].get("secrets", {}).get("FLAGCICD_API_TOKEN")
            )
            if declaration is not None:
                assert declaration["required"] == "false"
            else:
                assert passed == "inherit"  # Existing image-build wrappers inherit secrets.


@pytest.mark.parametrize("report_exists", [False, True])
@pytest.mark.parametrize("token_available", [False, True])
def test_report_preflight(tmp_path, report_exists, token_available):
    steps = workflow("unit_tests_common.yml")["jobs"]["unit_test"]["steps"]
    preflight = next(step for step in steps if step.get("id") == "coverage_report")
    report = tmp_path / "coverage.json"
    if report_exists:
        report.write_text('{"meta": {"format": 3}, "files": {}, "totals": {}}')
    output, summary = tmp_path / "output", tmp_path / "summary"
    env = dict(
        os.environ,
        REPORT_PATH=str(report),
        GITHUB_OUTPUT=str(output),
        GITHUB_STEP_SUMMARY=str(summary),
        FLAGCICD_TOKEN_SET=str(token_available).lower(),
    )
    # Render only the static platform/device labels used in a summary message.
    script = (
        preflight["run"]
        .replace("${{ inputs.platform }}", "ascend")
        .replace("${{ inputs.device }}", "unit-910c")
    )
    result = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == f"available={str(report_exists).lower()}\n"
    messages = summary.read_text() if summary.exists() else ""
    assert ("FLAGCICD_API_TOKEN unavailable" in messages) == (not token_available)
    assert ("Coverage report unavailable" in messages) == (not report_exists)
