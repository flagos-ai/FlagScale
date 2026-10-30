# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Trusted comment and requested-source controller; PR code is never executed."""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

from github_api import APIError, GitHubAPI


def parse_request(body):
    match = re.fullmatch(r"/ci run ([a-z][a-z0-9_-]*)", body.strip())
    if match:
        return {"kind": "run", "platform": match[1]}
    match = re.fullmatch(r"/ci rerun-job ([1-9][0-9]*)", body.strip())
    if match and int(match[1]) <= 2**53 - 1:
        return {"kind": "rerun-job", "job_id": int(match[1])}
    return None


def current_run(run, pr, repository, workflow_id):
    return (
        run.get("workflow_id") == workflow_id
        and run.get("event") == "pull_request"
        and (run.get("repository") or {}).get("full_name") == repository
        and (run.get("head_repository") or {}).get("full_name") == pr["head"]["repo"]["full_name"]
        and run.get("head_branch") == pr["head"]["ref"]
        and run.get("head_sha") == pr["head"]["sha"]
        and (
            not run.get("pull_requests")
            or any(p.get("number") == pr["number"] for p in run["pull_requests"])
        )
    )


def source(api, repository, number, sha):
    if not re.fullmatch(r"[1-9][0-9]*", number) or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("Expected a PR number and a full commit SHA")
    pr = api.get(f"/repos/{repository}/pulls/{number}")
    if pr["state"] != "open" or not pr["head"].get("repo") or pr["head"]["sha"] != sha:
        raise ValueError("PR is closed, unavailable, or its head changed; request CI again")
    repo = pr["head"]["repo"]["full_name"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid head repository")
    return {"repository": repo, "sha": sha}


def comment(api, event):
    request = parse_request(event["comment"]["body"])
    if request is None:
        return "Ignored: not a supported standalone command"
    repository = event["repository"]["full_name"]
    root = f"/repos/{repository}"
    number = event["issue"]["number"]
    pr = api.get(f"{root}/pulls/{number}")
    if pr["state"] != "open" or not pr["head"].get("repo"):
        raise ValueError("PR must be open and its head repository available")
    user = event["comment"]["user"]
    try:
        permission = api.get(f"{root}/collaborators/{quote(user['login'], safe='')}/permission")[
            "permission"
        ]
    except APIError as error:
        if error.status != 404:
            raise
        permission = "none"
    writer = permission in ("admin", "maintain", "write")
    author = (
        user.get("type") == "User"
        and type(user.get("id")) is int
        and user["id"] == pr.get("user", {}).get("id")
    )
    if not writer and not author:
        return "Ignored: repository write permission or PR authorship is required"
    workflow = api.get(f"{root}/actions/workflows/ci.yml")
    if workflow["state"] != "active":
        raise ValueError("CI must be enabled before requesting tests")
    if not writer and request["kind"] == "run":
        runs = api.paginate(
            f"{root}/actions/workflows/{workflow['id']}/runs",
            "workflow_runs",
            event="pull_request",
            head_sha=pr["head"]["sha"],
        )
        approved = False
        for run in runs:
            if (
                current_run(run, pr, repository, workflow["id"])
                and run["status"] == "completed"
                and run["conclusion"] in ("success", "failure")
            ):
                jobs = api.paginate(
                    f"{root}/actions/runs/{run['id']}/jobs", "jobs", filter="latest"
                )
                if any(
                    " / test / " in job["name"] and job["conclusion"] == "success" for job in jobs
                ):
                    approved = True
                    break
        if not approved:
            raise ValueError(
                "PR authors must wait for an approved current-head platform CI run to complete"
            )

    def unchanged():
        latest = api.get(f"{root}/pulls/{number}")
        return (
            latest["state"] == "open"
            and latest["head"].get("repo")
            and latest["head"]["repo"]["full_name"] == pr["head"]["repo"]["full_name"]
            and latest["head"]["sha"] == pr["head"]["sha"]
            and latest["head"]["ref"] == pr["head"]["ref"]
        )

    if request["kind"] == "rerun-job":
        job = api.get(f"{root}/actions/jobs/{request['job_id']}")
        run = api.get(f"{root}/actions/runs/{job['run_id']}")
        if (
            not current_run(run, pr, repository, workflow["id"])
            or run["status"] != "completed"
            or run["conclusion"] != "failure"
            or job["status"] != "completed"
            or job["conclusion"] != "failure"
        ):
            raise ValueError("Only failed current-head PR CI jobs can be rerun")
        jobs = api.paginate(f"{root}/actions/runs/{run['id']}/jobs", "jobs", filter="latest")
        if not any(item["id"] == job["id"] and item["conclusion"] == "failure" for item in jobs):
            raise ValueError("Job is from an old attempt; use the latest failed job ID")
        latest = api.get(f"{root}/actions/runs/{run['id']}")
        if (
            not unchanged()
            or latest["run_attempt"] != run["run_attempt"]
            or latest["status"] != "completed"
            or latest["conclusion"] != "failure"
            or not current_run(latest, pr, repository, workflow["id"])
        ):
            raise ValueError("PR head or run attempt changed during inspection")
        api.post(f"{root}/actions/jobs/{job['id']}/rerun")
        return f"Requested job {job['id']} and downstream jobs rerun"
    branch = event["repository"]["default_branch"]
    configs = api.get(f"{root}/contents/.github/configs", ref=branch)
    platforms = [
        f["name"][:-4]
        for f in configs
        if f["name"].endswith(".yml")
        and f["name"] not in ("template.yml", "image_sources.yml", "platforms.yml")
    ]
    target = request["platform"]
    if target != "all" and target not in platforms:
        raise ValueError(f"Unknown platform: {target}")
    if not unchanged():
        raise ValueError("PR head changed; send the command again")
    api.post(
        f"{root}/actions/workflows/ci-request.yml/dispatches",
        {
            "ref": branch,
            "inputs": {"pr_number": str(number), "head_sha": pr["head"]["sha"], "platform": target},
        },
    )
    api.post(
        f"{root}/issues/{number}/comments",
        {
            "body": f"CI request accepted: **{target}**, commit `{pr['head']['sha']}`. "
            f"[View requested runs]({event['repository']['html_url']}/actions/workflows/ci-request.yml). "
            "This supplemental run does not replace the CI Complete merge gate."
        },
    )
    return "Requested CI dispatched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("comment", "source"))
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid controller repository")
    api = GitHubAPI()
    if args.command == "source":
        result = source(api, repository, os.environ["PR_NUMBER"], os.environ["HEAD_SHA"])
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in result.items():
                output.write(f"{key}={value}\n")
    else:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if event["repository"]["full_name"] != repository or not event["issue"].get("pull_request"):
            raise ValueError("Unexpected comment repository or issue type")
        print(comment(api, event))


if __name__ == "__main__":
    main()
