#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Validate the single repository policy and discover executable test ownership."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently replacing policy."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"Duplicate policy key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def matches(path: str, pattern: str) -> bool:
    return path.startswith(pattern) if pattern.endswith("/") else fnmatch.fnmatchcase(path, pattern)


def safe_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith(("/", "-"))
        and not any(part in {"", ".", ".."} for part in path.rstrip("/").split("/"))
        and not any(c.isspace() for c in path)
        and "\\" not in path
    )


def tracked_tests(root: Path) -> set[str]:
    # Include nonignored new local tests too, so validation does not require staging.
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "tests"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    return {
        p
        for p in files
        if (root / p).is_file()
        and (
            p.endswith(".test.cjs")
            or p.endswith(".py")
            and (Path(p).name.startswith("test_") or Path(p).name.endswith("_test.py"))
        )
    }


def fields(value: dict, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Invalid {name} fields; expected {sorted(expected)}")


def paths(value, name: str) -> None:
    if (
        not isinstance(value, list)
        or any(not safe_path(p) for p in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"Invalid {name} paths")


def source_owner(project: dict, path: str) -> str | None:
    # Exact files and more specific directory roots override parent roots.
    hits = [
        (len(pattern.rstrip("/")), name)
        for name, group in project["test_groups"].items()
        for pattern in group["sources"]
        if matches(path, pattern)
    ]
    if not hits:
        return None
    specificity = max(n for n, _ in hits)
    owners = {name for n, name in hits if n == specificity}
    if len(owners) != 1:
        raise ValueError(f"Conflicting source ownership: {path}: {sorted(owners)}")
    return owners.pop()


def test_owner(project: dict, path: str) -> str | None:
    owners = [
        name
        for name, group in project["test_groups"].items()
        if any(matches(path, p) for p in group["tests"])
    ]
    if len(owners) > 1:
        raise ValueError(f"Conflicting test ownership: {path}: {owners}")
    return owners[0] if owners else None


def discover_tests(root: Path, project: dict) -> list[dict]:
    result = []
    for path in sorted(tracked_tests(root)):
        exception = project["test_exceptions"].get(path)
        if exception:
            result.append({"path": path, "kind": exception["kind"], "group": None, "suite": None})
            continue
        group = test_owner(project, path)
        if group is None:
            raise ValueError(
                f"Unowned test has no execution adapter: {path}; register it in project.yaml"
            )
        if path.startswith("tests/unit_tests/") and path.endswith(".py"):
            suite = "accelerator-unit"
        elif path.startswith("tests/ci/") and (path.endswith(".py") or len(Path(path).parts) == 3):
            suite = "ci-control"
        else:
            raise ValueError(f"No execution adapter for new test: {path}")
        result.append({"path": path, "kind": "automated", "group": group, "suite": suite})
    return result


def load_contract(root: Path) -> tuple[dict, list[dict]]:
    project = yaml.load((root / ".flagcicd/project.yaml").read_text(), Loader=UniqueLoader)
    fields(
        project,
        {
            "schema_version",
            "repository",
            "profiles",
            "functional_tasks",
            "shared_paths",
            "test_exceptions",
            "test_groups",
            "gate",
            "case_unit_groups",
        },
        "project",
    )
    if project["schema_version"] != "flagcicd.project/v2" or project["repository"] != "FlagScale":
        raise ValueError("Unsupported project schema/repository")
    if not isinstance(project["profiles"], dict) or set(project["profiles"]) != {
        "pr",
        "branch",
        "nightly",
        "requested",
    }:
        raise ValueError("Define pr/branch/nightly/requested profiles")
    for name, policy in project["profiles"].items():
        fields(policy, {"selection"}, f"profile {name}")
        allowed = (
            {"affected", "configured"}
            if name == "pr"
            else {"requested"}
            if name == "requested"
            else {"configured"}
        )
        if policy["selection"] not in allowed:
            raise ValueError(f"Invalid selection for {name}")
    tasks = project["functional_tasks"]
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(t, str) for t in tasks)
        or len(tasks) != len(set(tasks))
        or not set(tasks) <= {"train", "hetero_train", "benchmark"}
    ):
        raise ValueError("Unsupported or duplicate functional tasks")
    paths(project["shared_paths"], "shared")
    groups = project["test_groups"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Empty test groups")
    if (
        not isinstance(project["case_unit_groups"], list)
        or not project["case_unit_groups"]
        or not set(project["case_unit_groups"]) <= set(groups)
    ):
        raise ValueError("Invalid functional case unit groups")
    for name, group in groups.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError(f"Unsafe group: {name}")
        fields(group, {"tests", "sources", "affects", "functional", "gaps"}, f"group {name}")
        paths(group["tests"], f"{name} tests")
        paths(group["sources"], f"{name} sources")
        if any(not p.startswith("tests/") for p in group["tests"]):
            raise ValueError("Test rules must stay under tests/")
        if not isinstance(group["affects"], list) or not set(group["affects"]) <= set(groups):
            raise ValueError(f"Unknown affected group in {name}")
        if not isinstance(group["gaps"], list) or any(
            not isinstance(g, str) or not g for g in group["gaps"]
        ):
            raise ValueError(f"Invalid gaps in {name}")
        if not isinstance(group["functional"], list):
            raise ValueError(f"Invalid functional selectors in {name}")
        for selector in group["functional"]:
            if (
                not isinstance(selector, dict)
                or "task" not in selector
                or not set(selector) <= {"task", "model", "case", "backend"}
            ):
                raise ValueError(f"Invalid functional selector in {name}")
            if selector["task"] not in {
                "train",
                "hetero_train",
                "benchmark",
                "inference",
                "serve",
                "rl",
            }:
                raise ValueError("Unknown functional task")
            if any(
                not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.*-]+", v)
                for v in selector.values()
            ):
                raise ValueError("Unsafe functional selector")
    if not isinstance(project["test_exceptions"], dict):
        raise ValueError("Invalid test exceptions")
    for path, exception in project["test_exceptions"].items():
        fields(exception, {"kind", "reason"}, "test exception")
        if (
            not safe_path(path)
            or exception["kind"] not in {"helper", "placeholder"}
            or not isinstance(exception["reason"], str)
            or not exception["reason"]
        ):
            raise ValueError(f"Invalid exception: {path}")
        if not (root / path).is_file():
            raise ValueError(f"Stale exception: {path}")
    fields(project["gate"], {"require_unit_evidence", "require_functional_evidence"}, "gate")
    if any(type(v) is not bool or not v for v in project["gate"].values()):
        raise ValueError("Execution evidence is mandatory for planned runs")
    # Reject conflicting tracked source rules before any hardware work starts.
    for path in (
        subprocess.check_output(
            ["git", "ls-files", "-z", "flagscale", "tools", "examples"], cwd=root
        )
        .decode()
        .split("\0")
    ):
        if path:
            source_owner(project, path)
    return project, discover_tests(root, project)


if __name__ == "__main__":
    project, catalog = load_contract(Path.cwd())
    print(f"Validated project policy and {len(catalog)} discovered tests")
