#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Validate the unified CI gate, including the intentional documentation skip."""

import json
import os


def failures(needs: dict, platforms: list[str]) -> list[str]:
    errors = [name for name in ("lint", "generate-matrix") if needs[name]["result"] != "success"]
    expected = "success" if platforms else "skipped"
    if needs["platform-pipeline"]["result"] != expected:
        errors.append("platform-pipeline")
    return errors


if __name__ == "__main__":
    errors = failures(json.loads(os.environ["JOB_RESULTS"]), json.loads(os.environ["PLATFORMS"]))
    if errors:
        raise SystemExit(f"CI failed or did not finish: {', '.join(errors)}")
    print("All selected CI checks passed")
