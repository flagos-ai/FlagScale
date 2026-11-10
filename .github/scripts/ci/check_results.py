#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Validate the unified CI gate, including the intentional documentation skip."""

import hashlib
import json
import os
from pathlib import Path

from flagcicd_evidence import verify_results


def failures(needs: dict, platforms: list[str], required=("lint", "generate-matrix")) -> list[str]:
    errors = [name for name in required if needs[name]["result"] != "success"]
    expected = "success" if platforms else "skipped"
    if needs["platform-pipeline"]["result"] != expected:
        errors.append("platform-pipeline")
    return errors


def plan_failures(plan: dict, platforms: list[str], source_sha: str) -> list[str]:
    payload = {key: value for key, value in plan.items() if key != "plan_hash"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    errors = []
    if plan.get("schema_version") != "flagcicd.plan/v2" or digest != plan.get("plan_hash"):
        errors.append("plan identity")
    if plan.get("platforms") != platforms or plan.get("source_sha") != source_sha:
        errors.append("plan source/platforms")
    return errors


if __name__ == "__main__":
    platforms = json.loads(os.environ["PLATFORMS"])
    errors = failures(
        json.loads(os.environ["JOB_RESULTS"]),
        platforms,
        json.loads(os.environ.get("REQUIRED_JOBS", '["lint","generate-matrix"]')),
    )
    if "PLAN_JSON" in os.environ:
        errors += plan_failures(
            json.loads(os.environ["PLAN_JSON"]), platforms, os.environ["SOURCE_SHA"]
        )
    if errors:
        raise SystemExit(f"CI failed or did not finish: {', '.join(errors)}")
    if "PLAN_JSON" in os.environ:
        verify_results(json.loads(os.environ["PLAN_JSON"]), Path(os.environ["EVIDENCE_ROOT"]))
    print("All selected CI checks and execution evidence passed")
