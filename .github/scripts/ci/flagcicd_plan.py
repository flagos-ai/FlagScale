#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""One deterministic planner for PR, configured regression and requested tests."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import yaml
from flagcicd_cases import affected_cases, configured_cases
from flagcicd_contract import (
    fingerprint,
    load_contract,
    matches,
    safe_path,
    source_owner,
    test_owner,
)
from generate_matrix import changed_files, documentation, load_platforms, select_platforms

ACTIVE_TASKS = ("train", "hetero_train", "benchmark")
SCHEMA = "flagcicd.plan/v2"


def unit_files(
    root: Path, catalog: list[dict], config: dict, groups=None, *, automated=True
) -> list[str]:
    include = config.get("include", "*")
    include = [include] if isinstance(include, str) else include
    exclude = config.get("exclude", [])
    if not isinstance(include, list) or not include or not isinstance(exclude, list):
        raise ValueError("Invalid platform unit include/exclude")
    result = []
    for entry in catalog:
        path = entry["path"]
        if not path.startswith("tests/unit_tests/") or automated and entry["kind"] != "automated":
            continue
        if groups is not None and entry["group"] not in groups:
            continue
        if not any(
            p == "*" or path == p or matches(path, p.rstrip("/") + "/") or matches(path, p)
            for p in include
        ):
            continue
        if any(
            "::" not in p and (path == p or path.startswith(p.rstrip("/") + "/")) for p in exclude
        ):
            continue
        if not safe_path(path) or not (root / path).is_file():
            raise ValueError(f"Invalid test target: {path}")
        result.append(path)
    return sorted(result)


def functional_cases(root: Path, platform: str, config: dict, tasks) -> list[dict]:
    return configured_cases(root, platform, config, tasks)


def config_identity(root: Path, platforms: list[str], cases=()) -> str:
    paths = {".flagcicd/project.yaml"}
    for name in platforms:
        paths.update(
            (f".github/configs/{name}.yml", f"tests/test_utils/config/platforms/{name}.yaml")
        )
    for case in cases:
        paths.update(case["dependencies"])
    return fingerprint({p: (root / p).read_text() for p in sorted(paths)})


def case_matches(case: dict, selector: dict) -> bool:
    return all(matches(str(case.get(key, "")), pattern) for key, pattern in selector.items())


def case_id(case: dict) -> str:
    return ":".join(case[key] for key in ("device", "task", "model", "case"))


def unit_spec(plan: dict, platform: str, device: str) -> dict:
    s = plan["slices"][platform]
    return {
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "platform": platform,
        "kind": "unit",
        "id": f"unit:{device}",
        "device": device,
        "mode": s["unit_mode"],
        "targets": {device: s["unit_targets"][device]},
        "expected_files": s["unit_files"][device],
        "allowed_files": s["unit_allowed_files"][device],
        "ranks": s["unit_ranks"][device],
        "command_targets": s["unit_command_targets"][device],
        "deselect": s["unit_deselect"][device],
    }


def functional_spec(plan: dict, platform: str, case: dict) -> dict:
    return {
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "platform": platform,
        "kind": "functional",
        "id": "functional:" + case_id(case),
        **{k: case[k] for k in ("device", "task", "model", "case")},
    }


def build_plan(
    root: Path,
    *,
    profile: str,
    changed: list[str] | None,
    labels: list[str],
    source_sha="",
    control_sha="",
    base_sha="",
    requested_platforms=None,
) -> dict:
    root = root.resolve()
    project, catalog = load_contract(root)
    if profile not in project["profiles"]:
        raise ValueError("Invalid profile")
    selection = project["profiles"][profile]["selection"]
    source_sha = (
        source_sha
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    )
    control_sha = control_sha or source_sha
    if any(not re.fullmatch(r"[0-9a-f]{40}", v) for v in (source_sha, control_sha)):
        raise ValueError("Immutable source/control SHA required")
    if changed is not None and any(not safe_path(p) for p in changed):
        raise ValueError("Invalid changed path")
    platform_rules = load_platforms(root / ".github/configs")
    configs = {
        p: yaml.safe_load((root / f"tests/test_utils/config/platforms/{p}.yaml").read_text())
        for p in platform_rules
    }
    all_cases = {
        p: functional_cases(root, p, c, project["functional_tasks"]) for p, c in configs.items()
    }
    # A discovered test needs a real execution adapter, even when the same PR
    # also changes a shared file and would otherwise fall back to full.
    for entry in catalog:
        if entry["suite"] == "accelerator-unit" and not any(
            unit_files(root, [entry], c[d]["tests"]["unit"])
            for c in configs.values()
            for d in c["device_types"]
        ):
            raise ValueError(
                f"Test excluded on every platform: {entry['path']}; wire an execution adapter"
            )
    case_index = {
        path: sorted(
            {
                c["platform"]
                for c in affected_cases(path, [x for v in all_cases.values() for x in v])
            }
        )
        for path in changed or []
        if path.startswith("tests/functional_tests/")
    }
    platforms, reason = select_platforms(
        platform_rules,
        "pull_request" if profile == "pr" else "schedule",
        changed,
        labels,
        case_owners=case_index,
    )
    if profile == "requested":
        if (
            not requested_platforms
            or len(set(requested_platforms)) != len(requested_platforms)
            or not set(requested_platforms) <= set(platform_rules)
        ):
            raise ValueError("Requested platforms must be a nonempty configured set")
        platforms = sorted(requested_platforms)
    elif requested_platforms is not None:
        raise ValueError("Requested platforms require requested profile")
    code = [p for p in changed or [] if not documentation(p)]
    full = selection != "affected" or not changed
    reasons, gaps, direct, selected_groups = [reason], [], set(), set()
    selectors, changed_cases = [], []
    exceptions = project["test_exceptions"]
    for path in code:
        if path in exceptions:
            full = True
            reasons.append(
                f"Shared test {exceptions[path]['kind']}: {path}: {exceptions[path]['reason']}"
            )
            continue
        owner = (
            test_owner(project, path) if path.startswith("tests/") else source_owner(project, path)
        )
        if any(matches(path, p) for p in project["shared_paths"]):
            full = True
            reasons.append(f"Configured shared regression: {path}")
        consumers = (
            affected_cases(path, [c for values in all_cases.values() for c in values])
            if path.startswith("tests/functional_tests/")
            else []
        )
        if (
            path.startswith("tests/functional_tests/")
            and path.endswith((".yaml", ".yml"))
            and (root / path).is_file()
            and not any(path in c["dependencies"] for values in all_cases.values() for c in values)
        ):
            data = yaml.safe_load((root / path).read_text())
            if isinstance(data, dict) and data.get("experiment", {}).get("task"):
                raise ValueError(
                    f"Unconfigured functional test entrypoint: {path}; add platform execution support"
                )
        if consumers:
            selected_groups.update(project["case_unit_groups"])
            changed_cases.extend(consumers)
            reasons.append(f"Configured functional consumers: {path}")
        elif owner:
            direct.add(owner)
            selected_groups.add(owner)
            # Impact edges describe consequences of changing THIS owner's source.
            # Selecting its tests does not recursively pretend other sources changed.
            if not path.startswith("tests/"):
                rule = project["test_groups"][owner]
                selected_groups.update(rule["affects"])
                selectors.extend(rule["functional"])
                gaps.extend(rule["gaps"])
            reasons.append(f"Owner {owner}: {path}")
        else:
            full = True
            reasons.append(f"Conservative configured fallback: {path}")
    if not code and platforms:
        full = True
    requested_labels = {x.removeprefix("platform:") for x in labels if x.startswith("platform:")}
    slices = {}
    for platform in platforms:
        config = configs[platform]
        devices = config["device_types"]
        if not devices or len(devices) != len(set(devices)):
            raise ValueError(f"Invalid devices for {platform}")
        # Explicit platform requests ask for that platform's configured workload;
        # ordinary labels are never a test-selection control.
        platform_full = full or platform in requested_labels or "all" in requested_labels
        proposal = {
            d: unit_files(
                root,
                catalog,
                config[d]["tests"]["unit"],
                None if platform_full else selected_groups,
            )
            for d in devices
        }
        if not platform_full and any(not paths for paths in proposal.values()):
            platform_full = True
            gaps.append(f"{platform}: no supported affected unit targets; configured fallback")
        files = {
            d: unit_files(
                root,
                catalog,
                config[d]["tests"]["unit"],
                None if platform_full else selected_groups,
            )
            for d in devices
        }
        if any(not v for v in files.values()):
            raise ValueError(f"No executable unit tests on {platform}")
        cases = (
            all_cases[platform]
            if platform_full
            else [
                c
                for c in all_cases[platform]
                if any(case_matches(c, s) for s in selectors)
                or any(
                    case_id(c) == case_id(x) and c["platform"] == x["platform"]
                    for x in changed_cases
                )
            ]
        )
        for selector in selectors:
            if not any(case_matches(c, selector) for c in all_cases[platform]):
                gaps.append(
                    f"{platform}: no enabled configured functional case for {json.dumps(selector, sort_keys=True)}"
                )
        ranks = {d: config[d]["tests"]["unit"].get("nproc_per_node") for d in devices}
        if any(type(n) is not int or n <= 0 for n in ranks.values()):
            raise ValueError(
                f"Planned unit evidence requires explicit positive topology on {platform}"
            )
        slices[platform] = {
            "devices": devices,
            "unit_mode": "full" if platform_full else "affected",
            "unit_targets": {d: [] if platform_full else files[d] for d in devices},
            "unit_files": files,
            "unit_allowed_files": {
                d: unit_files(
                    root,
                    catalog,
                    config[d]["tests"]["unit"],
                    None if platform_full else selected_groups,
                    automated=False,
                )
                for d in devices
            },
            "unit_ranks": ranks,
            "unit_deselect": {
                d: [p for p in config[d]["tests"]["unit"].get("exclude", []) if "::" in p]
                for d in devices
            },
            "unit_command_targets": {
                d: (
                    ["tests/unit_tests/"]
                    if config[d]["tests"]["unit"].get("include", "*") == "*"
                    else (
                        [config[d]["tests"]["unit"]["include"]]
                        if isinstance(config[d]["tests"]["unit"]["include"], str)
                        else config[d]["tests"]["unit"]["include"]
                    )
                )
                if platform_full
                else files[d]
                for d in devices
            },
            "cases": cases,
        }
    plan = {
        "schema_version": SCHEMA,
        "profile": profile,
        "selection": selection,
        "source_sha": source_sha,
        "control_sha": control_sha,
        "base_sha": base_sha,
        "changed_files": sorted(changed) if changed is not None else None,
        "diff_fingerprint": fingerprint(sorted(changed) if changed is not None else None),
        "platforms": platforms,
        "direct_groups": sorted(direct),
        "selected_groups": sorted(selected_groups),
        "reasons": sorted(set(reasons)),
        "gaps": sorted(set(gaps)),
        "slices": slices,
        "gate": project["gate"],
        "config_fingerprint": config_identity(
            root, platforms, [c for s in slices.values() for c in s["cases"]]
        ),
    }
    plan["plan_hash"] = fingerprint(plan)
    return plan


def verify_plan(root: Path, plan: dict) -> None:
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if plan.get("schema_version") != SCHEMA or fingerprint(payload) != plan.get("plan_hash"):
        raise ValueError("Invalid plan schema or hash")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != plan["source_sha"]:
        raise ValueError("Plan source SHA does not match checkout")
    cases = [c for s in plan["slices"].values() for c in s["cases"]]
    if config_identity(root, plan["platforms"], cases) != plan["config_fingerprint"]:
        raise ValueError("Plan configuration does not match checkout")


def outputs(plan: dict, platform: str) -> dict[str, str]:
    s = plan["slices"][platform]
    result = {
        "unit_selection": json.dumps(
            {
                "plan_hash": plan["plan_hash"],
                "source_sha": plan["source_sha"],
                "mode": s["unit_mode"],
                "targets": s["unit_targets"],
                "specs": {d: unit_spec(plan, platform, d) for d in s["devices"]},
            },
            separators=(",", ":"),
        )
    }
    for task in ACTIVE_TASKS:
        result[f"{task}_test_matrix"] = json.dumps(
            [
                {k: c[k] for k in ("device", "task", "model", "case")}
                | {
                    "execution_spec": json.dumps(
                        functional_spec(plan, platform, c), separators=(",", ":")
                    )
                }
                for c in s["cases"]
                if c["task"] == task
            ],
            separators=(",", ":"),
        )
    return result


def check_platform(plan: dict, platform: str, needs: dict) -> list[str]:
    s = plan["slices"][platform]
    expected = {
        "checkout_and_config": "success",
        "cli_validation": "success",
        "unit_tests": "success",
    }
    for task in ACTIVE_TASKS:
        expected[f"functional_tests_{task}"] = (
            "success" if any(c["task"] == task for c in s["cases"]) else "skipped"
        )
    expected.update({f"functional_tests_{t}": "skipped" for t in ("inference", "serve")})
    return [name for name, state in expected.items() if needs.get(name, {}).get("result") != state]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "slice", "check"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", choices=["pr", "branch", "nightly", "requested"], default="pr")
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--control-sha", default="")
    parser.add_argument("--event-path", type=Path)
    parser.add_argument("--file", type=Path, default=Path("plan.json"))
    parser.add_argument("--platform", default="")
    parser.add_argument("--requested-platform", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "plan":
        event = json.loads(args.event_path.read_text()) if args.event_path else {}
        source = subprocess.check_output(
            ["git", "rev-parse", args.head_ref], cwd=root, text=True
        ).strip()
        requested = None
        if args.profile == "requested":
            requested = (
                sorted(load_platforms(root / ".github/configs"))
                if args.requested_platform == "all"
                else [args.requested_platform]
            )
        plan = build_plan(
            root,
            profile=args.profile,
            changed=changed_files(args.base_ref, args.head_ref, root=root)
            if args.profile == "pr"
            else None,
            labels=[x["name"] for x in event.get("pull_request", {}).get("labels", [])],
            source_sha=source,
            control_sha=args.control_sha,
            base_sha=event.get("pull_request", {}).get("base", {}).get("sha", ""),
            requested_platforms=requested,
        )
        args.file.write_text(json.dumps(plan, indent=2) + "\n")
        result = {
            "platforms": json.dumps(plan["platforms"]),
            "plan": json.dumps(plan, separators=(",", ":")),
            "reason": "; ".join(plan["reasons"]),
        }
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as f:
                from flagcicd_summary import render_summary

                f.write(render_summary(plan))
    else:
        plan = json.loads(args.file.read_text())
        verify_plan(root, plan)
        if args.command == "check":
            errors = check_platform(plan, args.platform, json.loads(os.environ["JOB_RESULTS"]))
            if errors:
                raise SystemExit(f"Required planned jobs did not finish: {errors}")
            result = {}
        else:
            result = outputs(plan, args.platform)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
