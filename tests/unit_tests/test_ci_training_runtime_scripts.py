# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMON_SCRIPT = ROOT / ".github/scripts/set_env_common.sh"
SCRIPT = ROOT / ".github/scripts/validate_prepared_runtime.py"
ACTION = ROOT / ".github/actions/restore-prepared-runtime/action.yml"
spec = importlib.util.spec_from_file_location("runtime_validator", SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def startup_hook_guard(tmp_path):
    """Detect accidental site initialization in standard-library subprocesses."""
    directory = tmp_path / "startup-hook"
    directory.mkdir()
    marker = tmp_path / "startup-hook-used"
    (directory / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\n"
    )
    return directory, marker


def run_common_helper(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "$1"\n{script}', "bash", str(COMMON_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def test_sanitize_training_pythonpath_removes_foreign_shadowing_paths(tmp_path):
    image_megatron = tmp_path / "image-megatron"
    image_te = tmp_path / "image-te"
    installed = tmp_path / "site-packages"
    prepared = tmp_path / "prepared-megatron"
    unrelated = tmp_path / "FlagCX"
    project = tmp_path / "project"

    for root, package in ((image_megatron, "megatron"), (image_te, "transformer_engine")):
        (root / package).mkdir(parents=True)
    for path in (installed, prepared, unrelated, project):
        path.mkdir(parents=True)
    (installed / "megatron").mkdir()
    (prepared / "megatron").mkdir()

    env = os.environ.copy()
    env.update(
        CI_PYTHON_BIN=sys.executable,
        MEGATRON_INSTALL_DIR=str(prepared),
        PYTHONPATH=os.pathsep.join(
            [
                str(image_megatron),
                str(installed),
                str(prepared),
                str(image_te),
                str(unrelated),
                str(installed),
            ]
        ),
    )

    result = run_common_helper(
        'CI_PROJECT_ROOT="$PROJECT_ROOT_OVERRIDE"\n'
        "ci_sanitize_training_pythonpath\n"
        'printf "RESULT=%s\\n" "$PYTHONPATH"',
        {**env, "PROJECT_ROOT_OVERRIDE": str(project)},
    )

    value = next(
        line.removeprefix("RESULT=")
        for line in result.stdout.splitlines()
        if line.startswith("RESULT=")
    )
    assert value.split(os.pathsep) == [str(prepared), str(unrelated)]
    assert str(image_megatron) in result.stderr
    assert str(installed) in result.stderr
    assert str(image_te) in result.stderr


def test_configure_training_pythonpath_places_flagscale_overlay_before_prepared_runtime(tmp_path):
    prepared = tmp_path / "prepared-megatron"
    (prepared / "megatron").mkdir(parents=True)

    env = os.environ.copy()
    env.update(
        CI_PYTHON_BIN=sys.executable,
        MEGATRON_INSTALL_DIR=str(prepared),
        PYTHONPATH="",
    )
    result = run_common_helper(
        'ci_configure_training_pythonpath\nprintf "RESULT=%s\\n" "$PYTHONPATH"',
        env,
    )

    value = next(
        line.removeprefix("RESULT=")
        for line in result.stdout.splitlines()
        if line.startswith("RESULT=")
    )
    paths = value.split(os.pathsep)
    assert paths[:3] == [
        str(ROOT / "flagscale/train"),
        str(prepared),
        str(ROOT),
    ]


def test_pythonpath_helpers_skip_vendor_startup_hooks(tmp_path):
    noisy_site = tmp_path / "noisy-site"
    noisy_site.mkdir()
    (noisy_site / "sitecustomize.py").write_text("print('vendor startup noise')\n")

    env = os.environ.copy()
    env.update(
        CI_PYTHON_BIN=sys.executable,
        PYTHONPATH=str(noisy_site),
        MEGATRON_INSTALL_DIR="",
    )
    result = run_common_helper(
        'CI_PROJECT_ROOT="$PROJECT_ROOT_OVERRIDE"\n'
        "ci_sanitize_training_pythonpath\n"
        'printf "RESULT=%s\\n" "$PYTHONPATH"',
        {**env, "PROJECT_ROOT_OVERRIDE": str(tmp_path / "project")},
    )

    assert "vendor startup noise" not in result.stdout
    assert "vendor startup noise" not in result.stderr


def test_flagscale_training_overlay_entrypoints_use_megatron_namespace():
    expected_imports = {
        "flagscale/train/megatron/training/arguments.py": (
            "from megatron.training.arguments_fs import add_flagscale_arguments",
        ),
        "flagscale/train/megatron/training/extra_valid.py": (
            "from megatron.training.global_vars import get_tensorboard_writer",
        ),
        "flagscale/train/megatron/training/global_vars.py": (
            "from megatron.training.tokenizer import build_tokenizer",
            "from megatron.training.spiky_loss import SpikyLossDetector",
        ),
        "flagscale/train/megatron/training/initialize.py": (
            "from megatron.training.global_vars import set_global_writers",
            "from megatron.backend_config import configure_backend_environment",
            "from megatron.training.arguments_fs import FSTrainArguments",
            "from megatron.training.global_vars import set_spiky_loss_detector",
        ),
        "flagscale/train/megatron/training/training.py": (
            "from megatron.training.global_vars import get_spiky_loss_detector",
        ),
    }

    for relative_path, imports in expected_imports.items():
        source = (ROOT / relative_path).read_text()
        for import_statement in imports:
            assert import_statement in source


def test_prepend_pythonpath_deduplicates_equivalent_paths(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    env = os.environ.copy()
    env.update(CI_PYTHON_BIN=sys.executable, PYTHONPATH=str(alias))
    result = run_common_helper(
        'ci_prepend_pythonpath "$TARGET"\nprintf "RESULT=%s\\n" "$PYTHONPATH"',
        {**env, "TARGET": str(target)},
    )

    value = next(
        line.removeprefix("RESULT=")
        for line in result.stdout.splitlines()
        if line.startswith("RESULT=")
    )
    assert value.split(os.pathsep) == [str(target)]


def test_apply_env_json_preserves_exact_scalar_values():
    payload = json.dumps({"TEXT": "value}", "COUNT": 3, "ENABLED": True})
    env = os.environ.copy()
    env["CI_PYTHON_BIN"] = sys.executable
    result = run_common_helper(
        'ci_apply_env_json "$PAYLOAD"\nprintf "RESULT=%s\\n" "$TEXT|$COUNT|$ENABLED"',
        {**env, "PAYLOAD": payload},
    )

    assert "RESULT=value}|3|True" in result.stdout


def test_sourcing_common_helpers_preserves_caller_shell_options():
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set +e +u\nset +o pipefail\nbefore="$-:$(set -o | grep pipefail)"\n'
            'source "$1"\nafter="$-:$(set -o | grep pipefail)"\n'
            'printf "BEFORE=%s\\nAFTER=%s\\n" "$before" "$after"',
            "bash",
            str(COMMON_SCRIPT),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    lines = result.stdout.splitlines()
    assert lines[0].removeprefix("BEFORE=") == lines[1].removeprefix("AFTER=")


def test_resolve_python_bin_uses_active_path(tmp_path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    env = os.environ.copy()
    env.pop("CI_PYTHON_BIN", None)
    env["PATH"] = os.pathsep.join((str(tmp_path), "/usr/bin", "/bin"))

    result = run_common_helper('ci_resolve_python_bin\nprintf "RESULT=%s\\n" "$CI_PYTHON_BIN"', env)

    assert f"RESULT={python}" in result.stdout


def test_candidate_image_tests_require_prepared_dependencies():
    workflow = (ROOT / ".github/workflows/build_image_common.yml").read_text()
    prepare_job, tests_job = workflow.split("\n  tests:\n", maxsplit=1)

    assert "needs.prepare_test_dependencies.result" not in prepare_job
    assert "needs.prepare_test_dependencies.result == 'success'" in tests_job


def test_te_runtime_uses_shared_install_argument_variable():
    script = (ROOT / ".github/scripts/install_te_fl_runtime.sh").read_text()

    assert 'install_pip_args_json="${CI_RUNTIME_PIP_INSTALL_ARGS_JSON:-[]}"' in script
    assert "TE_FL_INSTALL_PIP_ARGS_JSON" not in script


def test_runtime_reestablishes_pythonpath_after_environment_configuration():
    script = (ROOT / ".github/scripts/install_training_runtime.sh").read_text()

    last_environment_apply = script.rindex("ci_apply_env_json")
    configure = script.index("ci_configure_training_pythonpath")
    assert configure > last_environment_apply


def test_functional_runner_configures_prepared_pythonpath():
    script = (ROOT / "tests/test_utils/runners/run_functional_tests.sh").read_text()
    source_common = script.index('source "$PROJECT_ROOT/.github/scripts/set_env_common.sh"')
    resolve_python = script.index("ci_resolve_python_bin")
    configure_pythonpath = script.index("ci_configure_training_pythonpath")

    assert source_common < resolve_python < configure_pythonpath


def test_unit_runner_preserves_prepared_environment(tmp_path):
    script = (ROOT / "tests/test_utils/runners/run_unit_tests.sh").read_text()
    assert 'source "$PROJECT_ROOT/.github/scripts/set_env_common.sh"' in script
    assert "ci_resolve_python_bin" in script
    assert "ci_configure_training_pythonpath" in script
    assert 'export MEGATRON_INSTALL_DIR="$prepared_megatron_dir"' in script
    assert "Prepared Megatron-LM-FL runtime is required" in script
    assert (
        'export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/flagscale/train:${PYTHONPATH:-}"'
        not in script
    )


def test_training_workflows_use_shared_runtime_before_test_setup():
    for relative_path in (
        ".github/workflows/unit_tests_common.yml",
        ".github/workflows/functional_tests_train.yml",
        ".github/workflows/functional_tests_hetero_train.yml",
        ".github/workflows/functional_tests_benchmark.yml",
    ):
        workflow = (ROOT / relative_path).read_text()
        install_runtime = workflow.index("bash .github/scripts/install_training_runtime.sh")
        setup_tests = workflow.index("bash ./tests/test_utils/runners/setup_training_test_env.sh")

        assert workflow.count("bash .github/scripts/install_training_runtime.sh") == 1
        assert install_runtime < setup_tests


def test_parallel_context_uses_training_overlay_namespace():
    script = (ROOT / "tests/unit_tests/test_parallel_context.py").read_text()
    assert "from megatron.training.arguments_fs import FSTrainArguments" in script
    assert "flagscale.train.megatron.training" not in script


@pytest.mark.parametrize(
    "prepare_result,test_result,should_pass",
    [
        ("success", "success", True),
        ("skipped", "skipped", True),
        ("failure", "skipped", False),
        ("cancelled", "skipped", False),
        ("success", "failure", False),
        ("success", "cancelled", False),
        ("success", "skipped", False),
    ],
)
def test_all_tests_reports_preparation_and_test_failures(
    tmp_path, prepare_result, test_result, should_pass
):
    startup_directory, startup_marker = startup_hook_guard(tmp_path)
    workflow = yaml.safe_load((ROOT / ".github/workflows/all_tests.yml").read_text())
    jobs = workflow["jobs"]
    summary = jobs["all_tests"]
    dependencies = {name for name in jobs if name.endswith(("_prepare", "_tests"))} - {"all_tests"}
    assert set(summary["needs"]) == dependencies
    assert summary["if"] == "always()"
    step = summary["steps"][0]
    assert step["env"]["JOB_RESULTS"] == "${{ toJSON(needs) }}"

    # Exercise each platform independently while all other platforms are unselected.
    for prepare in sorted(name for name in dependencies if name.endswith("_prepare")):
        results = {name: {"result": "skipped"} for name in dependencies}
        results[prepare]["result"] = prepare_result
        tests = prepare.removesuffix("_prepare") + "_tests"
        results[tests]["result"] = test_result
        env = os.environ.copy()
        env["JOB_RESULTS"] = json.dumps(results)
        env["PYTHONPATH"] = str(startup_directory)
        result = subprocess.run(
            ["bash", "-e", "-c", step["run"]],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (result.returncode == 0) == should_pass, result.stdout + result.stderr
        assert not startup_marker.exists(), "Status checks must not initialize vendor site hooks"


def test_training_workflows_restore_prepared_dependencies_from_cache_or_artifact():
    for relative_path in (
        ".github/workflows/unit_tests_common.yml",
        ".github/workflows/functional_tests_train.yml",
        ".github/workflows/functional_tests_hetero_train.yml",
        ".github/workflows/functional_tests_benchmark.yml",
    ):
        workflow = (ROOT / relative_path).read_text()

        assert "actions/cache/restore@v4" in workflow
        assert "uses: ./FlagScale/.github/actions/restore-prepared-runtime" in workflow
        assert "name: Validate prepared training runtime" in workflow
        assert "Tests will run with image-provided dependencies" not in workflow
        assert "Expected exactly one TE-FL wheel" in workflow
        assert "name: Reset prepared runtime directories" in workflow


def test_prepare_workflow_uploads_same_run_dependency_artifacts():
    workflow = (ROOT / ".github/workflows/prepare_dependencies.yml").read_text()

    assert "actions/cache/save@v4" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "prepared-megatron-${{ inputs.platform }}-${{ github.run_id }}" in workflow
    assert "prepared-te-fl-${{ inputs.platform }}-${{ github.run_id }}" in workflow
    assert (
        "if: needs.resolve.outputs.megatron_enabled == 'true' && "
        "steps.megatron_cache.outputs.cache-hit != 'true'"
    ) in workflow
    assert (
        "if: needs.resolve.outputs.te_fl_enabled == 'true' && "
        "steps.te_fl_cache.outputs.cache-hit != 'true'"
    ) in workflow


def test_training_workflows_use_the_selected_python_interpreter():
    for relative_path in (
        ".github/workflows/unit_tests_common.yml",
        ".github/workflows/functional_tests_train.yml",
        ".github/workflows/functional_tests_hetero_train.yml",
        ".github/workflows/functional_tests_benchmark.yml",
        ".github/workflows/functional_tests_megatron_fl_trigger.yml",
    ):
        workflow = (ROOT / relative_path).read_text()
        assert "$(which python)" not in workflow
        assert "$(python --version)" not in workflow


def test_megatron_prepare_installs_build_dependencies_before_building():
    workflow = (ROOT / ".github/workflows/prepare_dependencies.yml").read_text()
    install = workflow.index('"pybind11==3.0.1"')
    build = workflow.index('"$python_bin" -m pip install --no-deps --no-build-isolation')

    assert install < build
    assert 'key="megatron-lm-fl-v3-' in workflow


def test_training_setup_exports_and_runners_use_ci_python_bin():
    setup_script = (ROOT / "tests/test_utils/runners/setup_training_test_env.sh").read_text()
    assert "ci_resolve_python_bin" in setup_script
    assert 'ci_export_env CI_PYTHON_BIN "$PYTHON_BIN"' in setup_script

    unit_runner = (ROOT / "tests/test_utils/runners/run_unit_tests.sh").read_text()
    assert 'source "$SCRIPT_DIR/utils.sh"' in unit_runner
    assert 'RUNNER_CMD=(--no-python "$CI_PYTHON_BIN"' in unit_runner
    assert '"$CI_PYTHON_BIN" -m coverage' in unit_runner

    functional_runner = (ROOT / "tests/test_utils/runners/run_functional_tests.sh").read_text()
    assert 'source "$SCRIPT_DIR/utils.sh"' in functional_runner
    assert "ci_resolve_python_bin" in functional_runner
    assert "python_bin=$(runner_python_bin)" in functional_runner
    assert '"$python_bin" -m pytest' in functional_runner


def test_platform_setup_scripts_only_activate_and_validate_runtime():
    for platform in ("ascend", "cuda", "enflame", "hygon", "kunlunxin", "metax", "musa", "ppu"):
        script = (ROOT / f".github/scripts/set_env_{platform}.sh").read_text()

        assert "ci_activate_python_environment" in script
        assert "pip uninstall" not in script
        assert "site-packages" not in script
        assert "install_megatron_runtime" not in script
        assert "install_te_fl_runtime" not in script


def test_enflame_setup_does_not_initialize_torch_gcu_before_platform_selection():
    script = (ROOT / ".github/scripts/set_env_enflame.sh").read_text()

    assert "efml-smi" in script
    assert "torch_gcu" not in script
    assert "torch.gcu" not in script


def test_inference_workflows_do_not_install_training_runtime():
    for relative_path in (
        ".github/workflows/functional_tests_inference.yml",
        ".github/workflows/functional_tests_serve.yml",
    ):
        assert "install_training_runtime" not in (ROOT / relative_path).read_text()


def test_ci_compatibility_fallback_does_not_enter_product_runtime():
    assert not (ROOT / "flagscale/train/compatibility_patches.py").exists()
    assert (
        "compatibility_patches" not in (ROOT / "flagscale/train/megatron/train_gpt.py").read_text()
    )
    assert "compatibility_patches" not in (ROOT / "tests/conftest.py").read_text()


def test_megatron_runtime_environment_uses_supported_contract():
    for config_path in sorted((ROOT / ".github/configs").glob("*.yml")):
        config = config_path.read_text()
        assert "MEGATRON_FL_PREFER" not in config


def record_for(files):
    stream = io.StringIO()
    writer = csv.writer(stream)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, f"sha256={digest}", len(data)])
    return stream.getvalue().encode()


def make_runtime(root, kind):
    root.mkdir(parents=True, exist_ok=True)
    if kind == "te":
        files = {
            "transformer_engine/__init__.py": b"# package\n",
            "transformer_engine-1.0.dist-info/METADATA": b"Name: transformer-engine\n",
            "transformer_engine-1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
        }
        files["transformer_engine-1.0.dist-info/RECORD"] = record_for(files)
        with zipfile.ZipFile(root / "transformer_engine-1.0-py3-none-any.whl", "w") as wheel:
            for name, data in files.items():
                wheel.writestr(name, data)
    else:
        files = {
            f"megatron/core/{part}__init__.py": b"# module\n"
            for part in ("", "tensor_parallel/", "transformer/", "pipeline_parallel/")
        }
        files["megatron_core-1.0.dist-info/RECORD"] = record_for(files)
        for name, data in files.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


@pytest.mark.parametrize("kind", ["te", "megatron"])
def test_accept_complete_runtime(tmp_path, kind):
    make_runtime(tmp_path, kind)
    validator.validate(kind, tmp_path)


def test_reject_truncated_wheel(tmp_path):
    make_runtime(tmp_path, "te")
    wheel = next(tmp_path.glob("*.whl"))
    wheel.write_bytes(wheel.read_bytes()[:100])
    with pytest.raises(zipfile.BadZipFile):
        validator.validate("te", tmp_path)


@pytest.mark.parametrize("change", ["missing", "corrupt"])
def test_reject_incomplete_megatron_with_core_init_present(tmp_path, change):
    make_runtime(tmp_path, "megatron")
    module = tmp_path / "megatron/core/tensor_parallel/__init__.py"
    if change == "missing":
        module.unlink()
    else:
        module.write_text("# changed\n")
    with pytest.raises((OSError, ValueError)):
        validator.validate("megatron", tmp_path)


def test_reject_wheel_with_valid_zip_but_wrong_record(tmp_path):
    make_runtime(tmp_path, "te")
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["transformer_engine/__init__.py"] = b"# changed\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="RECORD mismatch"):
        validator.validate("te", tmp_path)


def test_probe_reports_invalid_and_strict_mode_fails(tmp_path):
    startup_directory, startup_marker = startup_hook_guard(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(startup_directory)}
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(SCRIPT),
            "te",
            str(tmp_path),
            "--probe",
            "--output",
            str(output),
        ],
        capture_output=True,
        env=env,
    )
    assert result.returncode == 0
    assert output.read_text() == "valid=false\n"
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "te", str(tmp_path)], capture_output=True, env=env
    )
    assert result.returncode == 1
    assert not startup_marker.exists(), "Package validation must not initialize vendor site hooks"


def evaluate_action_condition(condition, context):
    """Evaluate only the comparison subset used here; reject all other syntax."""
    comparisons = []
    for clause in condition.split("&&"):
        match = re.fullmatch(
            r"\s*((?:inputs\.[\w-]+|steps\.[\w-]+\.outputs\.[\w-]+|steps\.[\w-]+\.outcome))"
            r"\s*(==|!=)\s*'([^']*)'\s*",
            clause,
        )
        if match is None:
            pytest.fail(f"Unsupported action condition: {condition!r}")
        comparisons.append(match.groups())
    # Validate every clause before evaluating: short-circuiting must not hide
    # unsupported syntax in a later clause.
    return all(
        (context.get(key, "") == value) if operator == "==" else (context.get(key, "") != value)
        for key, operator, value in comparisons
    )


@pytest.mark.parametrize(
    "cached,available,success_at,expected_downloads,passed",
    [
        (True, True, None, 0, True),
        (False, True, 1, 1, True),
        (False, True, 2, 2, True),
        (False, True, 3, 3, True),
        (False, True, None, 3, False),
        (False, True, "corrupt", 1, False),
        (False, False, None, 0, False),
    ],
)
@pytest.mark.parametrize("kind", ["te", "megatron"])
def test_action_recovery(tmp_path, cached, available, success_at, expected_downloads, passed, kind):
    """Execute composite shell steps; inject partial/successful artifact downloads."""
    startup_directory, startup_marker = startup_hook_guard(tmp_path)
    runtime = tmp_path / ("te-fl-wheel" if kind == "te" else "megatron-lm-fl-install")
    runtime.mkdir()
    if cached:
        make_runtime(runtime, kind)
    else:
        (runtime / "partial").write_text("left by failed cache restore")
    context = {
        "inputs.kind": kind,
        "inputs.artifact-name": "test",
        "inputs.artifact-available": str(available).lower(),
    }
    downloads = 0
    final_status = 0
    action = yaml.safe_load(ACTION.read_text())
    for index, step in enumerate(action["runs"]["steps"]):
        if "if" in step:
            if not evaluate_action_condition(step["if"], context):
                continue
        if "uses" in step:
            downloads += 1
            assert list(runtime.iterdir()) == [], "Every download needs a clean destination"
            if downloads == success_at:
                make_runtime(runtime, kind)
            else:
                (runtime / "partial").write_text("incomplete download")
            context[f"steps.{step['id']}.outcome"] = (
                "success"
                if downloads == success_at or (success_at == "corrupt" and downloads == 1)
                else "failure"
            )
            continue

        def resolve(value):
            return re.sub(r"\$\{\{\s*(.*?)\s*\}\}", lambda match: context[match.group(1)], value)

        output = tmp_path / f"output-{index}"
        env = {
            **os.environ,
            "GITHUB_WORKSPACE": str(tmp_path),
            "GITHUB_ACTION_PATH": str(ACTION.parent),
            "GITHUB_OUTPUT": str(output),
        }
        env.update({key: resolve(value) for key, value in step.get("env", {}).items()})
        # Use the test interpreter for the action's dependency-free validator.
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        env["PYTHONPATH"] = str(startup_directory)
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            final_status = result.returncode
            break
        if "id" in step and output.exists():
            for line in output.read_text().splitlines():
                key, value = line.split("=", 1)
                context[f"steps.{step['id']}.outputs.{key}"] = value
    assert downloads == expected_downloads
    assert (final_status == 0) == passed
    assert not startup_marker.exists(), "Recovery validation must not initialize vendor site hooks"


def test_installed_bytecode_without_hash_is_allowed(tmp_path):
    make_runtime(tmp_path, "megatron")
    record = next(tmp_path.glob("*.dist-info/RECORD"))
    with record.open("a") as output:
        output.write("megatron/core/__pycache__/__init__.cpython-312.pyc,,\n")
    validator.validate("megatron", tmp_path)


@pytest.mark.parametrize(
    "condition",
    [
        "inputs.kind != 'te' && contains(inputs.kind, 'te')",
        "inputs.kind == 'te' || inputs.kind == 'megatron'",
        "!(inputs.kind == 'te')",
        "inputs.kind >= 'te'",
        "${{ inputs.kind == 'te' }}",
        "inputs.kind == 'te' &&",
    ],
)
def test_action_condition_rejects_unsupported_syntax(condition):
    with pytest.raises(pytest.fail.Exception, match="Unsupported action condition"):
        evaluate_action_condition(condition, {"inputs.kind": "te"})


@pytest.mark.parametrize("missing_interpreter", [False, True])
def test_action_validation_uses_selected_interpreter(tmp_path, missing_interpreter):
    startup_directory, startup_marker = startup_hook_guard(tmp_path)
    runtime = tmp_path / "runtime"
    make_runtime(runtime, "te")
    interpreter = tmp_path / "selected python"
    marker = tmp_path / "interpreter-used"
    if not missing_interpreter:
        interpreter.write_text(
            '#!/bin/bash\nprintf "used\\n" >> "$INTERPRETER_MARKER"\nexec "$TEST_PYTHON" "$@"\n'
        )
        interpreter.chmod(0o755)
    action = yaml.safe_load(ACTION.read_text())
    validation_steps = [
        step
        for step in action["runs"]["steps"]
        if "validate_prepared_runtime.py" in step.get("run", "")
    ]
    assert len(validation_steps) == 5
    env = {
        **os.environ,
        "CI_PYTHON_BIN": str(interpreter),
        "KIND": "te",
        "RUNTIME_DIR": str(runtime),
        "GITHUB_ACTION_PATH": str(ACTION.parent),
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "INTERPRETER_MARKER": str(marker),
        "TEST_PYTHON": sys.executable,
        "PYTHONPATH": str(startup_directory),
    }
    for step in validation_steps:
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if missing_interpreter:
            assert result.returncode != 0
            assert "Python executable not found" in result.stdout + result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr
    if missing_interpreter:
        assert not marker.exists()
    else:
        assert marker.read_text().splitlines() == ["used"] * 5
    assert not startup_marker.exists()


@pytest.mark.parametrize(
    "value,valid",
    [
        ("1", True),
        ("20", True),
        ("30", True),
        ("360", True),
        ("0", False),
        ("361", False),
        ("-1", False),
        ("30.5", False),
        ("true", False),
        ("", False),
        ("030", False),
    ],
)
def test_unit_timeout_validation(value, valid):
    loader = ROOT / "tools/install/utils/load_platform_config.sh"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; validate_unit_timeout_minutes "$2" fixture.yml',
            "bash",
            str(loader),
            value,
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert (result.returncode == 0) == valid, result.stdout + result.stderr
    if not valid:
        assert "expected integer 1-360" in result.stderr


def test_unit_timeout_is_forwarded_through_common_workflow():
    common = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())
    unit = yaml.safe_load((ROOT / ".github/workflows/unit_tests_common.yml").read_text())
    trigger = unit.get("on", unit.get(True))
    setting = trigger["workflow_call"]["inputs"]["unit_timeout_minutes"]
    assert setting["type"] == "number" and setting["default"] == 20
    assert (
        common["jobs"]["checkout_and_config"]["outputs"]["unit_timeout_minutes"]
        == "${{ steps.config.outputs.unit_timeout_minutes }}"
    )
    assert (
        common["jobs"]["unit_tests"]["with"]["unit_timeout_minutes"]
        == "${{ fromJson(needs.checkout_and_config.outputs.unit_timeout_minutes) }}"
    )
    steps = unit["jobs"]["unit_test"]["steps"]
    run = next(step for step in steps if step["name"] == "Run unit tests")
    assert run["timeout-minutes"] == "${{ inputs.unit_timeout_minutes }}"
    enflame = yaml.safe_load((ROOT / ".github/configs/enflame.yml").read_text())
    assert enflame["test_matrix"]["unit"]["timeout_minutes"] == 30


def test_runtime_recovery_budget_is_forwarded_without_changing_test_budgets():
    common = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())
    key = "runtime_recovery_timeout_minutes"
    assert (
        common["jobs"]["checkout_and_config"]["outputs"][key]
        == "${{ steps.config.outputs.runtime_recovery_timeout_minutes }}"
    )
    consumers = {
        "unit_tests": "unit_tests_common",
        "functional_tests_train": "functional_tests_train",
        "functional_tests_hetero_train": "functional_tests_hetero_train",
        "functional_tests_benchmark": "functional_tests_benchmark",
    }
    for job, workflow in consumers.items():
        assert (
            common["jobs"][job]["with"][key]
            == "${{ fromJson(needs.checkout_and_config.outputs.runtime_recovery_timeout_minutes) }}"
        )
        data = yaml.safe_load((ROOT / f".github/workflows/{workflow}.yml").read_text())
        trigger = data.get("on", data.get(True))
        setting = trigger["workflow_call"]["inputs"][key]
        assert setting["type"] == "number" and setting["default"] == 60
        recovery = [
            step
            for definition in data["jobs"].values()
            for step in definition.get("steps", [])
            if step["name"].startswith("Recover and validate prepared")
        ]
        assert len(recovery) == 2
        for step in recovery:
            assert step["timeout-minutes"] == "${{ inputs.runtime_recovery_timeout_minutes }}"
            assert "inputs.platform" not in step["if"]
    cuda = yaml.safe_load((ROOT / ".github/configs/cuda.yml").read_text())
    assert cuda["timeouts"]["runtime_recovery_minutes"] == 60
    for platform in ("ascend", "enflame", "ppu", "metax", "musa", "hygon", "kunlunxin"):
        config = yaml.safe_load((ROOT / f".github/configs/{platform}.yml").read_text())
        assert config.get("timeouts", {}).get("runtime_recovery_minutes", 60) == 60
