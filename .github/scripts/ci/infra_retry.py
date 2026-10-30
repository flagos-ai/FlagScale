# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Code-reviewed, one-attempt default-branch dependency-transfer recovery."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from github_api import GitHubAPI

RECOVERY = re.compile(r"Recover and validate prepared (megatron|te) runtime")
TRANSIENT = re.compile(
    r"\b(ETIMEDOUT|ECONNRESET|EAI_AGAIN)\b|socket hang up|connection timed out|request timed out|Unable to download artifact[^\n]*(?:500|502|503|504)",
    re.I,
)
INVALID = re.compile(
    r"RECORD mismatch|Wheel CRC failure|BadZipFile|Missing RECORD hash|Artifact not found|Unable to find any artifacts|\b(?:401|403|404)\b",
    re.I,
)


def timestamp(value):
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def classify(job, log):
    if job["conclusion"] != "failure":
        return "ignore"
    failed = [s for s in job.get("steps", []) if s["conclusion"] == "failure"]
    if not failed:
        return "unknown"
    aggregate = {
        "all_tests_complete": "Verify all tests passed",
        "CI Complete": "Verify required checks",
        "Nightly Complete": "Verify required checks",
    }.get(job["name"].split(" / ")[-1])
    if aggregate and all(s["name"] == aggregate for s in failed):
        return "aggregate"
    if not all(RECOVERY.fullmatch(s["name"]) for s in failed) or not isinstance(log, str):
        return "unknown"
    for step in failed:
        start, end = timestamp(step.get("started_at")), timestamp(step.get("completed_at"))
        if start is None or end is None or end < start:
            return "unknown"
        scoped = []
        for line in log.splitlines():
            match = re.match(r"^(\d{4}-\d\d-\d\dT\S+)", line)
            time = timestamp(match[1]) if match else None
            if time is not None and start <= time <= end + 0.999:
                scoped.append(line)
        text = "\n".join(scoped)
        if (
            not re.search(r"Z\s+CI_INFRA_RECOVERY_DOWNLOAD_FAILED\s*$", text, re.M)
            or INVALID.search(text)
            or not TRANSIENT.search(text)
        ):
            return "unknown"
    return "transient"


def retry(api, event):
    expected = event["workflow_run"]
    root = f"/repos/{event['repository']['full_name']}"
    run = api.get(f"{root}/actions/runs/{expected['id']}")
    branch = event["repository"]["default_branch"]
    if (
        run["run_attempt"] != 1
        or run["status"] != "completed"
        or run["conclusion"] != "failure"
        or run["event"] not in ("push", "schedule")
        or run["head_branch"] != branch
        or run["head_sha"] != expected["head_sha"]
        or run["repository"]["full_name"] != event["repository"]["full_name"]
    ):
        return "No retry: not an unchanged first failed default-branch push/schedule"
    workflow_file = "nightly.yml" if run["name"] == "Nightly" else "ci.yml"
    if workflow_file == "nightly.yml" and run["event"] != "schedule":
        return "No retry: Nightly requires schedule"
    workflow = api.get(f"{root}/actions/workflows/{workflow_file}")
    if run["workflow_id"] != workflow["id"]:
        return "No retry: workflow is not allowlisted"
    jobs = api.paginate(f"{root}/actions/runs/{run['id']}/jobs", "jobs", filter="latest")
    if any(
        j["conclusion"] in ("cancelled", "timed_out", "action_required", "startup_failure", "stale")
        for j in jobs
    ):
        return "No retry: cancellation, timeout or unresolved job"
    transient = 0
    for job in (j for j in jobs if j["conclusion"] == "failure"):
        log = ""
        if any(
            s["conclusion"] == "failure" and RECOVERY.fullmatch(s["name"])
            for s in job.get("steps", [])
        ):
            log = api.request("GET", f"{root}/actions/jobs/{job['id']}/logs", text=True)
        kind = classify(job, log)
        if kind == "unknown":
            return f"No retry: job {job['id']} is not a proven transfer failure"
        transient += kind == "transient"
    if not transient:
        return "No retry: no proven transient failure"
    current = api.get(f"{root}/actions/runs/{run['id']}")
    tip = api.get(f"{root}/branches/{quote(branch, safe='')}")
    if (
        current["run_attempt"] != 1
        or current["status"] != "completed"
        or current["conclusion"] != "failure"
        or current["head_sha"] != run["head_sha"]
        or tip["commit"]["sha"] != run["head_sha"]
    ):
        return "No retry: attempt or default-branch commit changed"
    api.post(f"{root}/actions/runs/{run['id']}/rerun-failed-jobs")
    return f"Requested one failed-jobs retry of run {run['id']}; {transient} transfer failure(s)"


def main():
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if event["repository"]["full_name"] != os.environ["GITHUB_REPOSITORY"]:
        raise ValueError("Unexpected workflow_run repository")
    print(retry(GitHubAPI(), event))


if __name__ == "__main__":
    main()
