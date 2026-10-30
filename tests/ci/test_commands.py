# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts/ci"))

from commands import comment, parse_request, source
from github_api import APIError

ROOT = "/repos/team/FlagScale"
SHA = "a" * 40


class API:
    def __init__(self, permission="write", body="/ci run ascend"):
        self.calls = []
        self.permission = permission
        self.event = {
            "repository": {
                "full_name": "team/FlagScale",
                "default_branch": "main",
                "html_url": "https://github.com/team/FlagScale",
            },
            "issue": {"number": 42, "pull_request": {}},
            "comment": {"body": body, "user": {"login": "member", "id": 1, "type": "User"}},
        }
        self.pr = {
            "number": 42,
            "user": {"id": 2},
            "state": "open",
            "head": {"sha": SHA, "ref": "feature", "repo": {"full_name": "fork/FlagScale"}},
        }
        self.run = {
            "id": 10,
            "workflow_id": 7,
            "event": "pull_request",
            "head_sha": SHA,
            "head_branch": "feature",
            "head_repository": {"full_name": "fork/FlagScale"},
            "repository": {"full_name": "team/FlagScale"},
            "pull_requests": [],
            "status": "completed",
            "conclusion": "failure",
            "run_attempt": 1,
        }
        self.job = {
            "id": 123,
            "run_id": 10,
            "status": "completed",
            "conclusion": "failure",
            "name": "Platform pipeline (ascend) / test / unit-910c",
        }
        self.jobs = [
            self.job,
            {
                "id": 124,
                "name": "Platform pipeline (ascend) / test / checkout",
                "conclusion": "success",
            },
        ]
        self.workflow = {"id": 7, "state": "active"}
        self.pr_reads = self.run_reads = 0
        self.race = None

    def get(self, path, **params):
        self.calls.append(("get", path, params))
        if "/pulls/" in path:
            self.pr_reads += 1
            result = copy.deepcopy(self.pr)
            if self.race == "head" and self.pr_reads > 1:
                result["head"]["sha"] = "b" * 40
            return result
        if "/permission" in path:
            if isinstance(self.permission, Exception):
                raise self.permission
            return {"permission": self.permission}
        if "/contents/" in path:
            assert params == {"ref": "main"}
            return [
                {"name": name + ".yml"}
                for name in ("ascend", "cuda", "ppu", "template", "image_sources", "platforms")
            ]
        if "/workflows/" in path:
            return copy.deepcopy(self.workflow)
        if "/jobs/" in path:
            return copy.deepcopy(self.job)
        if "/runs/" in path:
            self.run_reads += 1
            result = copy.deepcopy(self.run)
            if self.race == "attempt" and self.run_reads > 1:
                result["run_attempt"] = 2
            return result
        raise AssertionError(path)

    def paginate(self, path, key, **params):
        return copy.deepcopy([self.run] if key == "workflow_runs" else self.jobs)

    def post(self, path, data=None):
        self.calls.append(("post", path, data))

    @property
    def mutations(self):
        return [c for c in self.calls if c[0] == "post"]


@pytest.mark.parametrize(
    "body",
    [
        "/ci run",
        "/ci run ASCEND",
        "/ci run ../x",
        "/ci run all; ls",
        "please /ci run all",
        "/ci run all\n/ci run cuda",
        "/ci rerun-job 0",
        "/ci rerun-job 9007199254740992",
        "/ci rerun-job 123; ls",
        "/ci rerun-job 123\n/ci run all",
        "/ci run failed 123",
    ],
)
def test_strict_commands(body):
    assert parse_request(body) is None


@pytest.mark.parametrize("target", ["ascend", "all", "ppu"])
def test_writer_dispatch(target):
    api = API(body=f"/ci run {target}")
    comment(api, api.event)
    assert api.mutations[0] == (
        "post",
        ROOT + "/actions/workflows/ci-request.yml/dispatches",
        {"ref": "main", "inputs": {"pr_number": "42", "head_sha": SHA, "platform": target}},
    )
    assert "does not replace" in api.mutations[1][2]["body"]


@pytest.mark.parametrize("permission", ["read", "triage", "none"])
def test_unrelated_reader(permission):
    api = API(permission)
    comment(api, api.event)
    assert not api.mutations


@pytest.mark.parametrize("target", ["typo", "template", "image_sources", "platforms"])
def test_unknown_platform(target):
    api = API(body=f"/ci run {target}")
    with pytest.raises(ValueError, match="Unknown platform"):
        comment(api, api.event)
    assert not api.mutations


def test_author_and_noncollaborator_permission():
    api = API(APIError(404))
    api.event["comment"]["user"]["id"] = 2
    comment(api, api.event)
    assert len(api.mutations) == 2


@pytest.mark.parametrize(
    "mode", ["approval", "no-platform", "bot", "closed", "deleted", "disabled"]
)
def test_author_cannot_bypass_approval(mode):
    api = API("read")
    api.event["comment"]["user"]["id"] = 2
    if mode == "approval":
        api.run["conclusion"] = "action_required"
    elif mode == "no-platform":
        api.jobs = [{"name": "Lint", "conclusion": "success"}]
    elif mode == "bot":
        api.event["comment"]["user"]["type"] = "Bot"
    elif mode == "closed":
        api.pr["state"] = "closed"
    elif mode == "deleted":
        api.pr["head"]["repo"] = None
    else:
        api.workflow["state"] = "disabled_manually"
    try:
        comment(api, api.event)
    except ValueError:
        pass
    assert not api.mutations


def test_author_selected_job_rerun():
    api = API("read", "/ci rerun-job 123")
    api.event["comment"]["user"]["id"] = 2
    comment(api, api.event)
    assert api.mutations == [("post", ROOT + "/actions/jobs/123/rerun", None)]


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_sha", "b" * 40),
        ("head_repository", {"full_name": "other/repo"}),
        ("head_branch", "other"),
        ("repository", {"full_name": "other/repo"}),
        ("pull_requests", [{"number": 99}]),
        ("workflow_id", 999),
        ("event", "push"),
        ("status", "in_progress"),
        ("conclusion", "success"),
    ],
)
def test_reject_wrong_rerun_provenance(field, value):
    api = API(body="/ci rerun-job 123")
    api.run[field] = value
    with pytest.raises(ValueError):
        comment(api, api.event)
    assert not api.mutations


@pytest.mark.parametrize("mode", ["old-job", "successful-job", "head", "attempt"])
def test_reject_stale_job_and_races(mode):
    api = API(body="/ci rerun-job 123")
    if mode == "old-job":
        api.jobs = []
    elif mode == "successful-job":
        api.job["conclusion"] = "success"
    else:
        api.race = mode
    with pytest.raises(ValueError):
        comment(api, api.event)
    assert not api.mutations


def test_dispatch_rechecks_head():
    api = API()
    api.race = "head"
    with pytest.raises(ValueError):
        comment(api, api.event)
    assert not api.mutations


def test_permission_api_failures_are_not_authorization():
    api = API(APIError(403))
    with pytest.raises(APIError):
        comment(api, api.event)
    assert not api.mutations


def test_source_pin():
    assert source(API(), "team/FlagScale", "42", SHA) == {
        "repository": "fork/FlagScale",
        "sha": SHA,
    }


@pytest.mark.parametrize(
    "number,sha", [("0", SHA), ("1;ls", SHA), ("42", "main"), ("42", "b" * 40)]
)
def test_invalid_or_stale_source(number, sha):
    with pytest.raises(ValueError):
        source(API(), "team/FlagScale", number, sha)


def test_trusted_python_controllers():
    for name in ("slash-command", "ci-infra-retry", "ci-request"):
        text = (Path(__file__).resolve().parents[2] / f".github/workflows/{name}.yml").read_text()
        assert "actions/github-script" not in text
        assert "ref: ${{ github.event.repository.default_branch }}" in text
        assert "GH_TOKEN: ${{ github.token }}" in text
        assert "GITHUB_API_URL: ${{ github.api_url }}" in text
