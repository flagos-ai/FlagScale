# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def policy_repo(tmp_path):
    for path in (
        ".flagcicd",
        ".github/configs",
        "tests/ci",
        "tests/unit_tests",
        "tests/functional_tests",
        "tests/test_utils/config/platforms",
    ):
        shutil.copytree(
            ROOT / path,
            tmp_path / path,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
        )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return tmp_path
