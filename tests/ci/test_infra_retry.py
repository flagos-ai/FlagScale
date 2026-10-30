# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts/ci"))

from infra_retry import classify, main, retry


def root(name="Recover and validate prepared megatron runtime"):
    return {
        "id": 9,
        "name": "Platform pipeline (ppu) / test / unit-ppu",
        "conclusion": "failure",
        "steps": [
            {
                "name": name,
                "conclusion": "failure",
                "started_at": "2026-10-21T01:00:00Z",
                "completed_at": "2026-10-21T01:01:00Z",
            }
        ],
    }


def stamp(text):
    return "\n".join("2026-10-21T01:00:30.000Z " + line for line in text.splitlines())


LOG = stamp(
    "Error: read ECONNRESET\nCI_INFRA_RECOVERY_DOWNLOAD_FAILED\nInvalid runtime: missing files"
)
AGGREGATE = {
    "id": 10,
    "name": "CI Complete",
    "conclusion": "failure",
    "steps": [{"name": "Verify required checks", "conclusion": "failure"}],
}


class API:
    def __init__(self, **overrides):
        self.run = {
            "id": 1,
            "workflow_id": 2,
            "name": "CI",
            "event": "push",
            "head_branch": "main",
            "head_sha": "a",
            "repository": {"full_name": "org/repo"},
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "failure",
        } | overrides
        self.event = {
            "workflow_run": {"id": 1, "head_sha": "a"},
            "repository": {"full_name": "org/repo", "default_branch": "main"},
        }
        self.calls = []
        self.jobs = [root(), AGGREGATE]
        self.tip = "a"
        self.changed = False
        self.reads = 0
        self.log = LOG

    def get(self, path):
        if "/runs/" in path:
            self.reads += 1
            run = copy.deepcopy(self.run)
            if self.changed and self.reads > 1:
                run["run_attempt"] = 2
            return run
        if "/workflows/" in path:
            expected = "nightly.yml" if self.run["name"] == "Nightly" else "ci.yml"
            assert path.endswith(expected)
            return {"id": 2}
        if "/branches/" in path:
            return {"commit": {"sha": self.tip}}
        raise AssertionError(path)

    def paginate(self, path, key, **params):
        assert params == {"filter": "latest"}
        return self.jobs

    def request(self, method, path, **kwargs):
        if isinstance(self.log, Exception):
            raise self.log
        return self.log

    def post(self, path):
        self.calls.append(path)


def test_transfer_only_and_aggregate():
    assert classify(root(), LOG) == "transient"
    assert classify(AGGREGATE, "") == "aggregate"
    assert classify(root("Run unit tests"), LOG) == "unknown"


@pytest.mark.parametrize(
    "log",
    [
        "ECONNRESET",
        "CI_INFRA_RECOVERY_DOWNLOAD_FAILED",
        "ECONNRESET\nCI_INFRA_RECOVERY_DOWNLOAD_FAILED",
        "2026-10-21T00:59:00Z ECONNRESET\n" + stamp("CI_INFRA_RECOVERY_DOWNLOAD_FAILED"),
    ]
    + [
        LOG + "\n" + stamp(x)
        for x in ["RECORD mismatch", "Wheel CRC failure", "Artifact not found", "403 Forbidden"]
    ],
)
def test_unproven_or_invalid_transfer(log):
    assert classify(root(), log) == "unknown"


@pytest.mark.parametrize(
    "start,end", [(None, None), ("bad", "bad"), ("2026-10-21T01:02:00Z", "2026-10-21T01:01:00Z")]
)
def test_invalid_step_times(start, end):
    job = root()
    job["steps"][0].update(started_at=start, completed_at=end)
    assert classify(job, LOG) == "unknown"


def test_one_failed_jobs_retry():
    api = API()
    retry(api, api.event)
    assert api.calls == ["/repos/org/repo/actions/runs/1/rerun-failed-jobs"]


@pytest.mark.parametrize(
    "change",
    [
        {"event": "pull_request"},
        {"event": "workflow_dispatch"},
        {"run_attempt": 2},
        {"head_branch": "feature"},
        {"repository": {"full_name": "fork/repo"}},
        {"workflow_id": 999},
        {"head_sha": "new"},
        {"name": "Nightly", "event": "push"},
        {"name": "Nightly", "event": "schedule", "workflow_id": 999},
    ],
)
def test_ineligible_runs(change):
    api = API(**change)
    retry(api, api.event)
    assert not api.calls


@pytest.mark.parametrize(
    "jobs",
    [
        [root(), root("Run benchmark tests"), AGGREGATE],
        [AGGREGATE],
        [root(), dict(root(), conclusion="cancelled")],
        [dict(root(), steps=[])],
    ],
)
def test_mixed_unknown_and_cancelled(jobs):
    api = API()
    api.jobs = jobs
    retry(api, api.event)
    assert not api.calls


@pytest.mark.parametrize("race", ["attempt", "tip"])
def test_retry_races(race):
    api = API()
    if race == "attempt":
        api.changed = True
    else:
        api.tip = "new"
    retry(api, api.event)
    assert not api.calls


def test_logs_failure_stops_mutation():
    api = API()
    api.log = RuntimeError("unavailable")
    with pytest.raises(RuntimeError):
        retry(api, api.event)
    assert not api.calls


def test_nightly():
    api = API(name="Nightly", event="schedule")
    api.jobs = [root(), dict(AGGREGATE, name="Nightly Complete")]
    retry(api, api.event)
    assert len(api.calls) == 1


@pytest.mark.parametrize("old_setting", [None, "false", "true"])
def test_main_needs_no_repository_variable(monkeypatch, tmp_path, capsys, old_setting):
    api = API()
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(api.event))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    if old_setting is None:
        monkeypatch.delenv("CI_INFRA_RETRY_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CI_INFRA_RETRY_ENABLED", old_setting)
    monkeypatch.setattr("infra_retry.GitHubAPI", lambda: api)
    main()
    assert api.calls == ["/repos/org/repo/actions/runs/1/rerun-failed-jobs"]
    assert "Requested one failed-jobs retry" in capsys.readouterr().out


def test_main_rejects_unexpected_repository_before_api(monkeypatch, tmp_path):
    api = API()
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(api.event))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_REPOSITORY", "other/repo")

    def unexpected_client():
        pytest.fail("API client must not be created for an unexpected repository")

    monkeypatch.setattr("infra_retry.GitHubAPI", unexpected_client)
    with pytest.raises(ValueError, match="Unexpected workflow_run repository"):
        main()
    assert not api.calls
