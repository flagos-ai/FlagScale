# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence for planned unit runs; one JUnit file per torchrun rank."""

import json
import os
from pathlib import Path

import pytest

_passed = 0
_passed_nodeids = set()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    directory = Path(os.environ["FLAGSCALE_CI_JUNIT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("LOCAL_RANK", "0")
    if not rank.isdecimal():
        raise pytest.UsageError("Invalid LOCAL_RANK for unit report")
    config.option.xmlpath = str(directory / f"junit-rank-{rank}.xml")


def pytest_runtest_logreport(report):
    global _passed
    if report.when == "call" and report.passed:
        _passed += 1
        _passed_nodeids.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    directory = Path(os.environ["FLAGSCALE_CI_JUNIT_DIR"])
    path = directory / f"collection-rank-{os.environ.get('LOCAL_RANK', '0')}.json"
    if path.is_file():
        data = json.loads(path.read_text())
        data["passed_nodeids"] = sorted(_passed_nodeids)
        path.write_text(json.dumps(data, indent=2) + "\n")
    if exitstatus == 0 and _passed == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


_collection_skips = set()
_collected_nodeids = []
_deselected_nodeids = []


def pytest_itemcollected(item):
    _collected_nodeids.append(item.nodeid)


def pytest_deselected(items):
    _deselected_nodeids.extend(item.nodeid for item in items)


def pytest_collectreport(report):
    if report.skipped:
        _collection_skips.add(report.nodeid.split("::")[0])


def pytest_collection_finish(session):
    directory = Path(os.environ["FLAGSCALE_CI_JUNIT_DIR"])
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    data = {
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "args": session.config.args,
        "nodeids": sorted(item.nodeid for item in session.items),
        "collected_nodeids": sorted(_collected_nodeids),
        "deselected_nodeids": sorted(_deselected_nodeids),
        "files": sorted({str(item.path.relative_to(Path.cwd())) for item in session.items}),
        "collection_skips": sorted(_collection_skips),
    }
    (directory / f"collection-rank-{rank}.json").write_text(json.dumps(data, indent=2) + "\n")
