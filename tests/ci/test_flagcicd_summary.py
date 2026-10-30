# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import copy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts/ci"))
from flagcicd_plan import build_plan
from flagcicd_summary import render_summary


def test_readable_top_and_complete_identity_below():
    plan = build_plan(ROOT, profile="nightly", changed=None, labels=[])
    before = copy.deepcopy(plan)
    text = render_summary(plan)
    top, details = text.split("<details>", 1)
    assert (
        "| Platform | Device | Unit scope | Unit files | Processes | Train | Hetero | Benchmark |"
        in top
    )
    assert "nightly" in top and "configured" in top
    assert "schema_version" not in top and "plan_hash" not in top
    assert plan["plan_hash"] in details
    assert "test counts" in top
    assert plan == before


def test_summary_escapes_untrusted_reason_and_json():
    plan = build_plan(ROOT, profile="pr", changed=["README.md"], labels=[])
    plan["reasons"] = ["<script> | malicious\nheading", "</pre><script>"]
    text = render_summary(plan)
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert "&#124;" in text
    assert "None (common checks only)" in text


def test_functional_names_do_not_expand_execution_spec():
    for task in ("train", "hetero_train", "benchmark"):
        workflow = yaml.safe_load(
            (ROOT / f".github/workflows/functional_tests_{task}.yml").read_text()
        )
        for job in workflow["jobs"].values():
            assert job["name"] == (
                f"functional_test_{task} ("
                "${{ matrix.test_config.device }}, ${{ matrix.test_config.task }}, "
                "${{ matrix.test_config.model }}, ${{ matrix.test_config.case }})"
            )
        assert (
            "matrix.test_config.execution_spec"
            in (ROOT / f".github/workflows/functional_tests_{task}.yml").read_text()
        )
