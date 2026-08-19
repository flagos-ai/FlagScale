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

"""Resolve moving source policies to immutable Git revisions."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

SOURCE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SUPPORTED_POLICIES = {"branch", "latest_release"}


class SourceResolutionError(RuntimeError):
    """Raised when a source contract cannot be resolved."""


def validate_catalog(catalog: Any) -> dict[str, dict[str, str]]:
    if not isinstance(catalog, Mapping) or not catalog:
        raise SourceResolutionError("source catalog must be a non-empty mapping")

    validated: dict[str, dict[str, str]] = {}
    for name, raw_source in catalog.items():
        if not isinstance(name, str) or not SOURCE_NAME_RE.fullmatch(name):
            raise SourceResolutionError(f"invalid source name: {name!r}")
        if not isinstance(raw_source, Mapping):
            raise SourceResolutionError(f"source {name} must be a mapping")

        repository = raw_source.get("repository")
        policy = raw_source.get("policy")
        if not isinstance(repository, str) or not repository:
            raise SourceResolutionError(f"source {name} requires repository")
        if policy not in SUPPORTED_POLICIES:
            raise SourceResolutionError(f"source {name} has unsupported policy: {policy!r}")

        source = {"repository": repository, "policy": policy}
        if policy == "branch":
            branch = raw_source.get("branch")
            if not isinstance(branch, str) or not branch:
                raise SourceResolutionError(f"source {name} requires branch")
            source["branch"] = branch
        else:
            _github_slug(repository)

        validated[name] = source

    return validated


def _retry(operation, description: str, attempts: int = 3):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except (OSError, subprocess.SubprocessError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"Retrying {description} ({attempt}/{attempts})...",
                file=sys.stderr,
            )
            time.sleep(attempt * 2)
    raise SourceResolutionError(f"failed to {description}: {last_error}") from last_error


def _git_ls_remote(repository: str, *refs: str) -> dict[str, str]:
    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "git",
                "ls-remote",
                repository,
                *refs,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )

    try:
        result = _retry(run, f"query {repository}")
    except SourceResolutionError as exc:
        cause = exc.__cause__
        if isinstance(cause, subprocess.CalledProcessError) and cause.stderr:
            raise SourceResolutionError(cause.stderr.strip()) from cause
        raise
    resolved: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            resolved[fields[1]] = fields[0]
    return resolved


def _github_slug(repository: str) -> str:
    parsed = urlparse(repository)
    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise SourceResolutionError(
            f"latest_release requires an https://github.com repository: {repository}"
        )
    path = parsed.path.removesuffix(".git").strip("/")
    if len(path.split("/")) != 2:
        raise SourceResolutionError(f"invalid GitHub repository: {repository}")
    return path


def _latest_release_tag(repository: str) -> str:
    slug = _github_slug(repository)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{slug}/releases/latest",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "FlagScale-image-build",
        },
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")

    def fetch() -> dict[str, Any]:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)

    release = _retry(fetch, f"query latest release for {slug}")
    tag = release.get("tag_name")
    if release.get("draft") or release.get("prerelease"):
        raise SourceResolutionError(f"latest release for {slug} is not stable")
    if not isinstance(tag, str) or not tag:
        raise SourceResolutionError(f"latest release for {slug} has no tag")
    return tag


def _resolve_branch(repository: str, branch: str) -> tuple[str, str]:
    ref = f"refs/heads/{branch}"
    refs = _git_ls_remote(repository, ref)
    revision = refs.get(ref, "")
    if not SHA_RE.fullmatch(revision):
        raise SourceResolutionError(f"branch not found: {repository} {branch}")
    return branch, revision


def _resolve_latest_release(repository: str) -> tuple[str, str]:
    tag = _latest_release_tag(repository)
    tag_ref = f"refs/tags/{tag}"
    peeled_ref = f"{tag_ref}^{{}}"
    refs = _git_ls_remote(repository, tag_ref, peeled_ref)
    revision = refs.get(peeled_ref) or refs.get(tag_ref, "")
    if not SHA_RE.fullmatch(revision):
        raise SourceResolutionError(f"release tag not found: {repository} {tag}")
    return tag, revision


def resolve_catalog(catalog: Any) -> dict[str, dict[str, str]]:
    validated = validate_catalog(catalog)
    resolved: dict[str, dict[str, str]] = {}
    for name, source in validated.items():
        repository = source["repository"]
        policy = source["policy"]
        if policy == "branch":
            selector, revision = _resolve_branch(repository, source["branch"])
        else:
            selector, revision = _resolve_latest_release(repository)
        resolved[name] = {
            "repository": repository,
            "policy": policy,
            "selector": selector,
            "revision": revision,
        }
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    try:
        catalog = json.load(sys.stdin)
        result = validate_catalog(catalog) if args.validate_only else resolve_catalog(catalog)
    except (json.JSONDecodeError, SourceResolutionError) as exc:
        print(f"source resolution error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
