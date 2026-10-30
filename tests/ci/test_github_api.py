# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts/ci"))
from github_api import APIError, GitHubAPI, SafeRedirect


def test_rest_headers_body_and_empty_response(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_API_URL", "https://enterprise.example/api/v3")
    api = GitHubAPI()
    calls = []

    def open_request(request, timeout):
        calls.append(request)
        assert timeout == 60
        return io.BytesIO(b"")

    monkeypatch.setattr(api.opener, "open", open_request)
    assert api.post("/repos/a/b/actions/workflows/ci.yml/dispatches", {"ref": "main"}) is None
    req = calls[0]
    assert req.full_url.startswith("https://enterprise.example/api/v3/repos/")
    assert req.get_header("Authorization") == "Bearer test-token"
    assert json.loads(req.data) == {"ref": "main"}
    assert req.method == "POST"


def test_cross_origin_redirect_drops_token():
    request = Request(
        "https://api.github.com/repos/a/b/logs", headers={"Authorization": "Bearer private"}
    )
    redirect = SafeRedirect()
    external = redirect.redirect_request(
        request, None, 302, "", {}, "https://blob.example/logs?sig=private"
    )
    assert external.get_header("Authorization") is None
    same = redirect.redirect_request(request, None, 302, "", {}, "https://api.github.com/other")
    assert same.get_header("Authorization") == "Bearer private"
    with pytest.raises(ValueError):
        redirect.redirect_request(request, None, 302, "", {}, "http://blob.example/logs")


def test_error_is_redacted_and_mutation_not_retried(monkeypatch):
    api = GitHubAPI("private")
    calls = []

    def fail(request, timeout):
        calls.append(request)
        raise HTTPError("https://api.github.com?secret=private", 403, "private", {}, None)

    monkeypatch.setattr(api.opener, "open", fail)
    with pytest.raises(APIError) as error:
        api.post("/repos/a/b/actions/runs/1/rerun-failed-jobs")
    assert "private" not in str(error.value)
    assert error.value.status == 403
    assert len(calls) == 1


def test_pagination_and_query(monkeypatch):
    api = GitHubAPI("private")
    calls = []

    def get(path, **params):
        calls.append(params)
        return {"jobs": list(range(100)) if params["page"] == 1 else [100]}

    monkeypatch.setattr(api, "get", get)
    assert api.paginate("/repos/a/b/actions/runs/1/jobs", "jobs", filter="latest") == list(
        range(101)
    )
    assert [c["page"] for c in calls] == [1, 2]
    assert all(c["filter"] == "latest" and c["per_page"] == 100 for c in calls)


@pytest.mark.parametrize("path", ["https://other.example", "//other.example"])
def test_cannot_send_token_to_absolute_path(path):
    with pytest.raises(ValueError):
        GitHubAPI("private").get(path)
