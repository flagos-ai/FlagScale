# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMMON_SCRIPT = ROOT / ".github/scripts/set_env_common.sh"


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
def test_all_tests_reports_preparation_and_test_failures(prepare_result, test_result, should_pass):
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
        result = subprocess.run(
            ["bash", "-e", "-c", step["run"]],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (result.returncode == 0) == should_pass, result.stdout + result.stderr


def test_training_workflows_restore_prepared_dependencies_from_cache_or_artifact():
    for relative_path in (
        ".github/workflows/unit_tests_common.yml",
        ".github/workflows/functional_tests_train.yml",
        ".github/workflows/functional_tests_hetero_train.yml",
        ".github/workflows/functional_tests_benchmark.yml",
    ):
        workflow = (ROOT / relative_path).read_text()

        assert "actions/cache/restore@v4" in workflow
        assert "actions/download-artifact@v4" in workflow
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
    for platform in ("ascend", "cuda", "enflame", "hygon", "kunlunxin", "metax", "musa"):
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
