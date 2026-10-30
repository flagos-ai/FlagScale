# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CI controller tests deliberately require no FlagScale/torch imports."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load("generate_matrix")
gate = load("check_results")
PLATFORMS = matrix.load_platforms(ROOT / ".github/configs")


@pytest.mark.parametrize("platform", sorted(PLATFORMS))
@pytest.mark.parametrize(
    "path",
    [
        "docker/{0}/Dockerfile.train",
        "requirements/{0}/train.txt",
        "tools/install/{0}/install_train.sh",
        ".github/configs/{0}.yml",
        ".github/scripts/set_env_{0}.sh",
        "tests/test_utils/config/platforms/{0}.yaml",
    ],
)
def test_platform_owned_change(platform, path):
    assert matrix.select_platforms(PLATFORMS, "pull_request", [path.format(platform)], [])[0] == [
        platform
    ]


@pytest.mark.parametrize(
    "files",
    [
        ["flagscale/train/foo.py"],
        ["tests/test_utils/runners/run_unit_tests.sh"],
        ["requirements/common.txt"],
        [".github/workflows/ci.yml"],
        ["tools/install/utils/pkg_utils.sh"],
        ["docker/ascend/Dockerfile.train", "flagscale/runner/foo.py"],
        ["new-unclassified-file"],
    ],
)
def test_shared_changes_cannot_be_narrowed_by_labels(files):
    assert matrix.select_platforms(PLATFORMS, "pull_request", files, ["platform:ascend"])[
        0
    ] == sorted(PLATFORMS)


def test_platform_union():
    assert matrix.select_platforms(
        PLATFORMS, "pull_request", ["docker/ascend/a", "docker/cuda/b"], []
    )[0] == ["ascend", "cuda"]


def test_labels_and_documentation():
    files = ["docker/ascend/a"]
    assert matrix.select_platforms(
        PLATFORMS, "pull_request", files, ["platform:cuda", "platform:ppu"]
    )[0] == ["ascend", "cuda", "ppu"]
    assert matrix.select_platforms(PLATFORMS, "pull_request", files, ["platform:all"])[0] == sorted(
        PLATFORMS
    )
    assert (
        matrix.select_platforms(PLATFORMS, "pull_request", ["docs/setup.txt", "README.md"], [])[0]
        == []
    )
    assert matrix.select_platforms(PLATFORMS, "pull_request", ["README.md"], ["platform:ascend"])[
        0
    ] == ["ascend"]
    with pytest.raises(ValueError, match="Unknown platform"):
        matrix.select_platforms(PLATFORMS, "pull_request", files, ["platform:typo"])


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_is_full(event):
    assert matrix.select_platforms(PLATFORMS, event, ["README.md"], ["platform:ascend"])[
        0
    ] == sorted(PLATFORMS)


@pytest.mark.parametrize("changed", [None, []])
def test_missing_diff_is_conservative(changed):
    assert matrix.select_platforms(PLATFORMS, "pull_request", changed, [])[0] == sorted(PLATFORMS)


def test_missing_watch_and_invalid_configs(tmp_path):
    assert matrix.select_platforms({"a": [], "b": ["b/"]}, "pull_request", ["b/file"], [])[0] == [
        "a",
        "b",
    ]
    (tmp_path / "a.yml").write_text("paths_watch: string")
    with pytest.raises(ValueError):
        matrix.load_platforms(tmp_path)
    (tmp_path / "a.yml").write_text("paths_watch: [")
    with pytest.raises(yaml.YAMLError):
        matrix.load_platforms(tmp_path)
    with pytest.raises(ValueError):
        matrix.load_platforms(tmp_path / "missing")


def test_path_boundaries_and_globs():
    assert matrix.matches("docker/ascend/file", "docker/ascend/")
    assert not matrix.matches("mydocker/ascend/file", "docker/ascend/")
    assert matrix.matches("a/b.py", "a/*.py")
    assert not matrix.matches(".github/configs/cuda.yml.backup", ".github/configs/cuda.yml")


@pytest.mark.parametrize("name", ["lint", "generate-matrix", "platform-pipeline"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_gate_cannot_hide_upstream_failure(name, result):
    needs = {key: {"result": "success"} for key in ["lint", "generate-matrix", "platform-pipeline"]}
    needs[name]["result"] = result
    assert name in gate.failures(needs, ["ascend"])


def test_documentation_gate():
    needs = {key: {"result": "success"} for key in ["lint", "generate-matrix", "platform-pipeline"]}
    needs["platform-pipeline"]["result"] = "skipped"
    assert not gate.failures(needs, [])
    assert gate.failures(needs, ["ascend"])


def test_diff_includes_both_sides_of_renames(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "CI test")
    (tmp_path / "shared.py").write_text("content")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "shared.py").rename(tmp_path / "ascend.py")
    git("add", ".")
    git("commit", "-qm", "rename")
    import os

    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert sorted(matrix.changed_files(base, "HEAD")) == ["ascend.py", "shared.py"]
        assert matrix.changed_files("missing-ref", "HEAD") is None
    finally:
        os.chdir(previous)


def test_workflow_source_identity_propagates_to_every_child():
    # BaseLoader preserves the YAML 'on' key rather than treating it as boolean.
    directory = ROOT / ".github/workflows"
    pending = ["_platform_pipeline.yml"]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        workflow = yaml.load((directory / name).read_text(), Loader=yaml.BaseLoader)
        assert {"source_repository", "source_sha"} <= workflow["on"]["workflow_call"][
            "inputs"
        ].keys()
        for job in workflow["jobs"].values():
            if job.get("uses", "").startswith("./.github/workflows/"):
                assert job["with"]["source_sha"] == "${{ inputs.source_sha }}"
                assert job["with"]["source_repository"] == "${{ inputs.source_repository }}"
                pending.append(Path(job["uses"]).name)
            for step in job.get("steps", []):
                if step.get("uses") == "actions/checkout@v4":
                    checkout = step.get("with", {})
                    # Other repositories (Megatron/TE) keep their own dependency refs.
                    if "needs.resolve.outputs." not in checkout.get("repository", ""):
                        assert "inputs.source_sha" in checkout.get("ref", ""), (name, step["name"])
                        assert "inputs.source_repository" in checkout.get("repository", "")
    assert "functional_tests_cli.yml" in seen
    assert "prepare_dependencies.yml" in seen


def test_matrix_cli_event_payload(tmp_path):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"labels": [{"name": "platform:ppu"}]}}))
    import sys

    result = subprocess.check_output(
        [
            sys.executable,
            str(SCRIPTS / "generate_matrix.py"),
            "--event-name",
            "pull_request",
            "--event-path",
            str(event),
            "--changed-files",
            "docker/ascend/a",
        ],
        cwd=ROOT,
        text=True,
    )
    assert json.loads(result.splitlines()[0].removeprefix("platforms=")) == ["ascend", "ppu"]


@pytest.mark.parametrize("changed_platform", sorted(PLATFORMS))
@pytest.mark.parametrize("label_platform", sorted(PLATFORMS))
def test_platform_label_never_removes_path_coverage(changed_platform, label_platform):
    files = [f"docker/{changed_platform}/Dockerfile"]
    selected, _ = matrix.select_platforms(PLATFORMS, "pull_request", files, [])
    labelled, _ = matrix.select_platforms(
        PLATFORMS, "pull_request", files, [f"platform:{label_platform}"]
    )
    assert set(selected) <= set(labelled)
    assert label_platform in labelled


def test_runtime_examples_are_code_inputs():
    assert matrix.select_platforms(PLATFORMS, "pull_request", ["examples/model/config.yaml"], [])[
        0
    ] == sorted(PLATFORMS)
    assert matrix.select_platforms(
        PLATFORMS, "pull_request", ["examples/model/config.yaml", "flagscale/runner.py"], []
    )[0] == sorted(PLATFORMS)


def test_requested_source_cannot_save_dependency_caches():
    workflow = yaml.load(
        (ROOT / ".github/workflows/prepare_dependencies.yml").read_text(), Loader=yaml.BaseLoader
    )
    saves = [
        step
        for step in workflow["jobs"]["prepare"]["steps"]
        if step.get("uses", "").startswith("actions/cache/save@")
    ]
    assert len(saves) == 2
    for step in saves:
        assert step["if"].startswith("inputs.source_sha == '' && ")


@pytest.mark.parametrize(
    "action,label,enabled",
    [
        ("opened", "", True),
        ("synchronize", "", True),
        ("labeled", "bug", False),
        ("unlabeled", "lgtm", False),
        ("labeled", "platform:ascend", True),
        ("unlabeled", "platform:ascend", True),
    ],
)
def test_label_events_do_not_cancel_or_replace_active_gate(action, label, enabled):
    import re

    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    # These targeted expressions use the same boolean/string operator subset as JS.
    # Execute the actual YAML expressions rather than copying their policy into the test.
    expressions = [
        workflow["jobs"]["lint"]["if"],
        workflow["jobs"]["ci-complete"]["if"],
        workflow["jobs"]["ci-complete"]["name"].removeprefix("${{ ").removesuffix(" }}"),
        re.findall(r"\$\{\{ (.*?) \}\}", workflow["concurrency"]["group"])[-1],
    ]
    context = {"event": {"action": action, "label": {"name": label}}, "run_id": 123}
    script = r"""
    const data = JSON.parse(process.argv[1]);
    const results = data.expressions.map(expression => new Function(
      'github', 'contains', 'fromJSON', 'startsWith', 'format', 'always',
      `return (${expression})`
    )(data.context, (xs, x) => xs.includes(x), JSON.parse,
      (s, prefix) => (s || '').startsWith(prefix),
      (s, value) => s.replace('{0}', value), () => true));
    process.stdout.write(JSON.stringify(results));
    """
    result = json.loads(
        subprocess.check_output(
            ["node", "-e", script, json.dumps({"expressions": expressions, "context": context})],
            text=True,
        )
    )
    assert result[:2] == [enabled, enabled]
    assert result[2] == ("CI Complete" if enabled else "Ignored label event")
    assert result[3] == ("" if enabled else "-ignored-label-123")


@pytest.mark.parametrize(
    "source_sha,source_repository", [("", ""), ("abc", ""), ("", "fork/repo"), ("abc", "fork/repo")]
)
@pytest.mark.parametrize(
    "enabled,cache_hit", [(True, False), (True, True), (False, False), (False, True)]
)
def test_cache_save_conditions_exclude_explicit_pr_sources(
    source_sha, source_repository, enabled, cache_hit
):
    workflow = yaml.load(
        (ROOT / ".github/workflows/prepare_dependencies.yml").read_text(), Loader=yaml.BaseLoader
    )
    saves = [
        step
        for step in workflow["jobs"]["prepare"]["steps"]
        if step.get("uses", "").startswith("actions/cache/save@")
    ]
    for step in saves:
        condition = step["if"]
        replacements = {
            "inputs.source_sha": source_sha,
            "inputs.source_repository": source_repository,
            "needs.resolve.outputs.megatron_enabled": str(enabled).lower(),
            "needs.resolve.outputs.te_fl_enabled": str(enabled).lower(),
            "steps.megatron_cache.outputs.cache-hit": str(cache_hit).lower(),
            "steps.te_fl_cache.outputs.cache-hit": str(cache_hit).lower(),
        }
        for expression, value in replacements.items():
            condition = condition.replace(expression, repr(value))
        # The actual conditions are string equality joined by boolean AND.
        actual = eval(condition.replace("&&", "and"), {"__builtins__": {}})
        assert actual == (not source_sha and not source_repository and enabled and not cache_hit)
