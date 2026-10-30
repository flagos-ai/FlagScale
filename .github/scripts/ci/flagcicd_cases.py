# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Read configured cases and local Hydra defaults without importing FlagScale."""

import re
from pathlib import Path

import yaml


def config_dependencies(root: Path, path: str, seen=None) -> set[str]:
    seen = set() if seen is None else seen
    if path in seen:
        return seen
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError(f"Missing or escaping case configuration: {path}")
    seen.add(path)
    config = yaml.safe_load(target.read_text())
    for item in config.get("defaults", []):
        if item == "_self_":
            continue
        if isinstance(item, dict):
            for group, name in item.items():
                if name is None:
                    continue
                # Unresolved Hydra expressions must not be mistaken for a precise edge.
                if not isinstance(group, str) or not isinstance(name, str) or "${" in group + name:
                    continue
                child = target.parent / group / (name + ".yaml")
                if child.is_file():
                    config_dependencies(root, str(child.relative_to(root.resolve())), seen)
        elif isinstance(item, str) and "${" not in item:
            child = target.parent / (item + ".yaml")
            if child.is_file():
                config_dependencies(root, str(child.relative_to(root.resolve())), seen)
    return seen


def configured_cases(root: Path, platform: str, config: dict, tasks) -> list[dict]:
    result = []
    for device in config["device_types"]:
        for task in tasks:
            for model, cases in config[device]["tests"].get("functional", {}).get(task, {}).items():
                for case in cases:
                    if any(
                        not isinstance(v, str)
                        or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", v)
                        or ".." in v
                        for v in (device, task, model, case)
                    ):
                        raise ValueError("Unsafe functional identifier")
                    path = f"tests/functional_tests/{task}/{model}/conf/{case}.yaml"
                    target = root / path
                    # Disabled legacy declarations sometimes point at absent cases.
                    if not target.is_file():
                        raise ValueError(f"Missing configured case: {path}")
                    data = yaml.safe_load(target.read_text())
                    result.append(
                        {
                            "platform": platform,
                            "device": device,
                            "task": task,
                            "model": model,
                            "case": case,
                            "config": path,
                            "backend": data.get("experiment", {})
                            .get("task", {})
                            .get("backend", ""),
                            "dependencies": sorted(config_dependencies(root, path)),
                        }
                    )
    keys = [(c["device"], c["task"], c["model"], c["case"]) for c in result]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate functional cases on {platform}")
    return result


def affected_cases(path: str, cases: list[dict]) -> list[dict]:
    exact = []
    broad = []
    for case in cases:
        base = f"tests/functional_tests/{case['task']}/{case['model']}/"
        if (
            path in case["dependencies"]
            or path == base + f"gold_values/{case['case']}.json"
            or path.startswith(base + f"results_gold/{case['case']}/")
            or path == base + f"results_gold/{case['case']}"
        ):
            exact.append(case)
        elif path.startswith(base):
            broad.append(case)
    # For shared data/defaults, include every configured consumer. For an exact
    # case or golden path, the indexed support list defines the platform set.
    return exact if exact else broad
