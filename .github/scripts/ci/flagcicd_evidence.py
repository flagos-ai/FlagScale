#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Produce and consume plan-bound execution evidence; standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reports(directory: Path, spec: dict) -> list[dict]:
    expected = (
        {f"junit-rank-{rank}.xml" for rank in range(spec["ranks"])}
        if spec["kind"] == "unit"
        else {"junit.xml"}
    )
    actual = {p.name for p in directory.glob("*.xml")}
    if actual != expected:
        raise ValueError(
            f"JUnit report set differs: expected {sorted(expected)}, got {sorted(actual)}"
        )
    result = []
    for name in sorted(expected):
        path = directory / name
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        if not suites:
            raise ValueError("No JUnit suite")
        if spec["kind"] == "functional":
            for field in ("platform", "device", "task", "model", "case"):
                values = [
                    p.get("value")
                    for p in root.findall(".//property")
                    if p.get("name") == "flagcicd." + field
                ]
                if not values or any(value != spec[field] for value in values):
                    raise ValueError(f"Functional comparator identity differs: {field}")
        passed = 0
        count = 0
        for suite in suites:
            cases = suite.findall("testcase")
            if int(suite.get("tests", "-1")) != len(cases):
                raise ValueError("JUnit testcase count is inconsistent")
            if int(suite.get("failures", "0")) or int(suite.get("errors", "0")):
                raise ValueError("JUnit reports a failure/error")
            for case in cases:
                if case.find("failure") is not None or case.find("error") is not None:
                    raise ValueError("JUnit contains failed testcase")
                if case.find("skipped") is None:
                    passed += 1
            count += len(cases)
        if not passed:
            raise ValueError("Empty or all-skipped JUnit cannot establish execution")
        result.append({"file": name, "sha256": digest(path), "tests": count, "passed": passed})
    return result


def validate_collection(directory: Path, spec: dict) -> list[dict]:
    expected = {f"collection-rank-{rank}.json" for rank in range(spec["ranks"])}
    if {p.name for p in directory.glob("collection-*.json")} != expected:
        raise ValueError("Missing or unexpected unit rank collection")
    result = []
    for rank in range(spec["ranks"]):
        name = f"collection-rank-{rank}.json"
        data = json.loads((directory / name).read_text())
        if data["rank"] != rank or data["world_size"] != spec["ranks"]:
            raise ValueError("Unit rank/topology mismatch")
        if data["args"] != spec["command_targets"]:
            raise ValueError("Executed pytest targets differ from plan")
        collected = set(data["collected_nodeids"])
        selected = set(data["nodeids"])
        deselected = set(data["deselected_nodeids"])
        for key in ("collected_nodeids", "nodeids", "deselected_nodeids"):
            if len(data[key]) != len(set(data[key])):
                raise ValueError("Duplicate collected or deselected testcase")
        if selected & deselected or collected != selected | deselected:
            raise ValueError("Collected testcases are not accounted for by selection/deselection")
        # pytest --deselect matches nodeid prefixes, not filesystem globs.
        if any(not node.startswith(tuple(spec["deselect"])) for node in deselected):
            raise ValueError("Unit testcase deselected without plan authorization")
        passed = data.get("passed_nodeids", [])
        if not set(passed) <= set(data["nodeids"]) or not any(
            node.split("::")[0] in spec["expected_files"] for node in passed
        ):
            raise ValueError("No planned automated unit testcase passed")
        files = set(data["files"])
        skipped = set(data["collection_skips"])
        if files != {node.split("::")[0] for node in selected}:
            raise ValueError("Unit files differ from selected testcases")
        deselected_files = {node.split("::")[0] for node in deselected}
        if not set(spec["expected_files"]) <= files | skipped | deselected_files:
            raise ValueError(
                "Planned unit files were not collected, collection-skipped or explicitly deselected"
            )
        if not files | skipped | deselected_files <= set(spec["allowed_files"]):
            raise ValueError("Unexpected unit file executed")
        reported = len(
            ET.parse(directory / f"junit-rank-{rank}.xml").getroot().findall(".//testcase")
        )
        if reported != len(data["nodeids"]) + len(skipped):
            raise ValueError("JUnit does not account for every collected/skipped testcase")
        result.append({"file": name, "sha256": digest(directory / name)})
    return result


def evidence_directory(root: Path, spec: dict) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", spec["platform"]) or not re.fullmatch(
        r"[A-Za-z0-9_.:-]+", spec["id"]
    ):
        raise ValueError("Unsafe execution identity")
    if (
        spec["kind"] not in {"unit", "functional"}
        or not spec["id"].startswith(spec["kind"] + ":")
        or any(part in {"", ".", ".."} for part in spec["id"].split(":"))
    ):
        raise ValueError("Invalid execution identity components")
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    if not re.fullmatch(r"[A-Za-z0-9-]+", run_id) or not attempt.isdecimal():
        raise ValueError("Invalid run identity")
    return (
        root / spec["platform"] / spec["id"].replace(":", "__") / f"run-{run_id}-attempt-{attempt}"
    )


def record(spec: dict, directory: Path, exit_code: int) -> dict:
    result = {
        "schema_version": "flagcicd.result/v1",
        "spec": spec,
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
        "exit_code": exit_code,
        "status": "failed",
        "reports": [],
        "collections": [],
    }
    try:
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != spec["source_sha"]
        ):
            raise ValueError("Execution source SHA differs from plan")
        if exit_code:
            raise ValueError(f"Command failed: {exit_code}")
        result["reports"] = reports(directory, spec)
        if spec["kind"] == "unit":
            result["collections"] = validate_collection(directory, spec)
        result["status"] = "passed"
    except (ValueError, KeyError, OSError, ET.ParseError, subprocess.CalledProcessError) as exc:
        result["error"] = str(exc)
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_result(path: Path, expected: dict, *, run_id: str, attempt: int) -> None:
    result = json.loads(path.read_text())
    if result.get("schema_version") != "flagcicd.result/v1" or result.get("spec") != expected:
        raise ValueError(f"Execution identity/targets differ: {path}")
    if result.get("run_id") != run_id or not 1 <= result.get("run_attempt", 0) <= attempt:
        raise ValueError("Result belongs to another run or future attempt")
    if result.get("status") != "passed" or result.get("exit_code") != 0:
        raise ValueError(f"Execution did not pass: {path}")
    if reports(path.parent, expected) != result["reports"]:
        raise ValueError("JUnit evidence changed after result recording")
    if (
        expected["kind"] == "unit"
        and validate_collection(path.parent, expected) != result["collections"]
    ):
        raise ValueError("Collection evidence changed after result recording")


def verify_results(
    plan: dict, directory: Path, *, platform=None, run_id=None, attempt=None
) -> None:
    # Imported only on the controller; hardware-side recording stays stdlib-only.
    from flagcicd_plan import functional_spec, unit_spec

    run_id = os.environ.get("GITHUB_RUN_ID", "local") if run_id is None else run_id
    attempt = int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")) if attempt is None else attempt
    platforms = [platform] if platform else plan["platforms"]
    expected = {}
    for name in platforms:
        s = plan["slices"][name]
        if plan["gate"]["require_unit_evidence"]:
            for device in s["devices"]:
                spec = unit_spec(plan, name, device)
                expected[(name, spec["id"])] = spec
        if plan["gate"]["require_functional_evidence"]:
            for case in s["cases"]:
                spec = functional_spec(plan, name, case)
                expected[(name, spec["id"])] = spec
    found = {}
    for path in directory.rglob("result.json"):
        result = json.loads(path.read_text())
        spec = result.get("spec", {})
        if platform and spec.get("platform") != platform:
            continue
        key = (spec.get("platform"), spec.get("id"))
        if key not in expected:
            raise ValueError(f"Unexpected execution result: {key}")
        current = result.get("run_attempt")
        if type(current) is not int or not 1 <= current <= attempt:
            raise ValueError("Invalid evidence attempt")
        versions = found.setdefault(key, {})
        if current in versions:
            raise ValueError(f"Duplicate execution result: {key}, attempt {current}")
        versions[current] = path
    if set(found) != set(expected):
        raise ValueError(f"Missing execution results: {sorted(set(expected) - set(found))}")
    # A later failed attempt cannot be replaced with an earlier successful one.
    # Successful jobs from earlier attempts of this same run remain valid.
    for key, versions in found.items():
        verify_result(versions[max(versions)], expected[key], run_id=run_id, attempt=attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["unit", "functional", "verify"])
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--spec-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("test-results/evidence"))
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--platform")
    args, command = parser.parse_known_args()
    if args.command == "verify":
        verify_results(
            json.loads(args.plan_file.read_text()), args.output_root, platform=args.platform
        )
        return
    if args.command == "unit":
        selection = json.loads(args.selection_file.read_text())
        if "specs" not in selection:  # explicit legacy diagnostic callers only
            return
        spec = selection["specs"][args.device]
        directory = evidence_directory(args.output_root, spec)
        directory.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.xml", "collection-*.json"):
            for path in args.report_dir.glob(pattern):
                shutil.copy2(path, directory / path.name)
        result = record(spec, directory, args.exit_code)
    else:
        spec = json.loads(args.spec_file.read_text())
        command = command[1:] if command[:1] == ["--"] else command
        if (
            len(command) < 2
            or command[0] != "bash"
            or Path(command[1]).resolve()
            != (Path.cwd() / "tests/test_utils/runners/run_tests.sh").resolve()
        ):
            raise ValueError("Functional execution must use the shared test runner")
        if (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != spec["source_sha"]
        ):
            raise ValueError("Functional source differs before execution")
        # The wrapper proves which configured case was requested, while the
        # original runner still owns CLI execution and numerical comparison.
        for key in ("platform", "device", "task", "model"):
            flag = "--" + key
            if command.count(flag) != 1 or command[command.index(flag) + 1] != spec[key]:
                raise ValueError(f"Functional command differs from plan: {key}")
        if command.count("--list") != 1 or command[command.index("--list") + 1] != spec["case"]:
            raise ValueError("Functional case differs from plan")
        directory = evidence_directory(args.output_root, spec)
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.xml"):
            path.unlink()
        code = subprocess.run(
            command,
            env={
                **os.environ,
                "FLAGSCALE_CI_FUNCTIONAL_JUNIT": str((directory / "junit.xml").resolve()),
            },
        ).returncode
        result = record(spec, directory, code)
    if result["status"] != "passed":
        raise SystemExit(result.get("error", "Invalid execution evidence"))


if __name__ == "__main__":
    main()
