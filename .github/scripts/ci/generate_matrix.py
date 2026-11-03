#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select CI platforms conservatively; stdout contains GitHub job outputs only."""

from __future__ import annotations

import argparse
import fnmatch
import json
import pathlib
import subprocess
import sys

import yaml

_NON_PLATFORM_STEMS = {"template", "image_sources", "platforms"}


def load_platforms(configs_dir: pathlib.Path) -> dict[str, list[str]]:
    platforms = {}
    for path in sorted(configs_dir.glob("*.yml")):
        if path.stem in _NON_PLATFORM_STEMS:
            continue
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Invalid platform configuration: {path}")
        watch = config.get("paths_watch", [])
        if not isinstance(watch, list) or any(not isinstance(p, str) or not p for p in watch):
            raise ValueError(f"paths_watch must be a list of nonempty strings: {path}")
        platforms[path.stem] = watch
    if not platforms:
        raise ValueError(f"No platform configurations in {configs_dir}")
    return platforms


def matches(path: str, pattern: str) -> bool:
    """Directory prefixes end in /; other patterns are repository-relative globs."""
    return path.startswith(pattern) if pattern.endswith("/") else fnmatch.fnmatchcase(path, pattern)


def documentation(path: str) -> bool:
    return path.startswith("docs/") or path.endswith((".md", ".rst"))


def select_platforms(
    platforms: dict[str, list[str]],
    event: str,
    changed: list[str] | None,
    labels: list[str],
    *,
    case_owners: dict[str, list[str]] | None = None,
) -> tuple[list[str], str]:
    case_owners = case_owners or {}
    all_platforms = sorted(platforms)
    if event != "pull_request":
        return all_platforms, "non-PR event: full matrix"
    requested = {
        label.removeprefix("platform:") for label in labels if label.startswith("platform:")
    }
    unknown = requested - set(platforms) - {"all"}
    if unknown:
        raise ValueError(f"Unknown platform labels: {sorted(unknown)}")
    if not changed:
        return all_platforms, "missing or empty diff: full matrix"
    files = [path for path in changed if not documentation(path)]
    if not files:
        if requested:
            return (all_platforms if "all" in requested else sorted(requested)), "platform labels"
        return [], "documentation only"
    if any(not watch for watch in platforms.values()):
        return all_platforms, "missing paths_watch: full matrix"
    # Anything outside known platform-owned paths is shared or unknown. Never
    # narrow that change with a label: common code must keep cross-platform coverage.
    if any(
        not case_owners.get(path)
        and not any(matches(path, pattern) for watch in platforms.values() for pattern in watch)
        for path in files
    ):
        return all_platforms, "shared or unknown files: full matrix"
    if "all" in requested:
        return all_platforms, "platform:all"
    selected = [
        name
        for name, watch in platforms.items()
        if any(matches(path, pattern) for path in files for pattern in watch)
    ]
    selected.extend(p for path in files for p in case_owners.get(path, []))
    return sorted(set(selected) | requested), "platform-owned paths plus labels"


def changed_files(
    base_ref: str, head_ref: str, *, root: pathlib.Path | None = None
) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", "-z", f"{base_ref}...{head_ref}", "--"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return [path for path in result.stdout.split("\0") if path]
    except subprocess.CalledProcessError:
        print("::warning::Cannot determine changed files; running all platforms", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--configs-dir", type=pathlib.Path, default=pathlib.Path(".github/configs"))
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument(
        "--changed-files",
        default=None,
        help="Newline-separated paths; primarily for local validation",
    )
    parser.add_argument(
        "--event-path", type=pathlib.Path, help="GitHub event JSON containing PR labels"
    )
    args = parser.parse_args()
    try:
        platforms = load_platforms(args.configs_dir)
        payload = json.loads(args.event_path.read_text()) if args.event_path else {}
        labels = [label["name"] for label in payload.get("pull_request", {}).get("labels", [])]
        changed = None
        if args.event_name == "pull_request":
            changed = (
                args.changed_files.splitlines()
                if args.changed_files is not None
                else changed_files(args.base_ref, args.head_ref)
            )
        case_owners = {}
        if changed and any(path.startswith("tests/functional_tests/") for path in changed):
            from flagcicd_cases import affected_cases, configured_cases

            root = pathlib.Path.cwd()
            cases = []
            for name in platforms:
                path = root / f"tests/test_utils/config/platforms/{name}.yaml"
                if path.is_file():
                    config = yaml.safe_load(path.read_text())
                    cases.extend(
                        configured_cases(root, name, config, ("train", "hetero_train", "benchmark"))
                    )
            case_owners = {
                path: sorted({case["platform"] for case in affected_cases(path, cases)})
                for path in changed
            }
        selected, reason = select_platforms(
            platforms, args.event_name, changed, labels, case_owners=case_owners
        )
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.exit(1, f"::error::{exc}\n")
    print(f"platforms={json.dumps(selected)}")
    print(f"reason={reason}")


if __name__ == "__main__":
    main()
