# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Small stdlib GitHub REST client. Mutations are never automatically retried."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class APIError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"GitHub API request failed (HTTP {status})")


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise ValueError("Refusing insecure API redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urlsplit(req.full_url).netloc != urlsplit(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


class GitHubAPI:
    def __init__(self, token=None, base_url=None):
        self.token = token if token is not None else os.environ["GH_TOKEN"]
        self.base_url = (
            base_url or os.environ.get("GITHUB_API_URL", "https://api.github.com")
        ).rstrip("/")
        if urlsplit(self.base_url).scheme != "https":
            raise ValueError("GITHUB_API_URL must use HTTPS")
        self.opener = build_opener(SafeRedirect())

    def request(self, method, path, data=None, params=None, text=False):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Expected repository-relative API path")
        url = self.base_url + path + ("?" + urlencode(params) if params else "")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "FlagScale-CI",
        }
        body = None if data is None else json.dumps(data).encode()
        if body is not None:
            headers["Content-Type"] = "application/json"
        try:
            with self.opener.open(
                Request(url, data=body, headers=headers, method=method), timeout=60
            ) as response:
                payload = response.read(32 * 1024 * 1024 + 1)
        except HTTPError as error:
            raise APIError(error.code) from None
        except URLError:
            raise RuntimeError(
                "GitHub API transport failed; no mutation is automatically retried"
            ) from None
        if len(payload) > 32 * 1024 * 1024:
            raise ValueError("API response exceeds controller limit")
        return payload.decode("utf-8") if text else (json.loads(payload) if payload else None)

    def get(self, path, **params):
        return self.request("GET", path, params=params)

    def post(self, path, data=None):
        return self.request("POST", path, data=data)

    def paginate(self, path, key=None, **params):
        result = []
        for page in range(1, 501):
            response = self.get(path, **(params | {"per_page": 100, "page": page}))
            items = response[key] if key else response
            if not isinstance(items, list):
                raise ValueError("Invalid paginated API response")
            result.extend(items)
            if len(items) < 100:
                return result
        raise ValueError("API pagination exceeds controller limit")
