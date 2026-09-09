# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import importlib.machinery
import os
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from flagscale.inference.inference_kerv import main as inference_main
from flagscale.models.kerv.entrypoint import (
    KERVEntrypointError,
    _python_paths,
    build_inprocess_argv,
    run_kerv_entrypoint,
)
from flagscale.models.kerv.ops import bundled_runtime_path
from flagscale.train.train_kerv import main as train_main


def _write_function_entrypoint(root: Path, filename: str = "entry.py") -> Path:
    entrypoint = root / filename
    entrypoint.parent.mkdir(parents=True, exist_ok=True)
    entrypoint.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "def main():\n"
        "    output = os.environ.get('KERV_SMOKE_OUTPUT')\n"
        "    if output:\n"
        "        Path(output).write_text('|'.join(sys.argv[1:]), encoding='utf-8')\n"
        "    return {\n"
        "        'argv': list(sys.argv),\n"
        "        'cwd': os.getcwd(),\n"
        "        'marker': os.environ.get('KERV_SMOKE_MARKER'),\n"
        "    }\n",
        encoding="utf-8",
    )
    return entrypoint


def _config(root: Path, entrypoint: str, **overrides):
    values = {
        "stage": "inference",
        "source_root": str(root),
        "entrypoint": entrypoint,
        "launcher": "python",
        "arguments": {
            "enabled": True,
            "buckets": [224, 240],
            "ignored": None,
        },
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_build_inprocess_argv_matches_upstream_cli_contract():
    config = OmegaConf.create(
        {
            "arguments": {
                "use_spec": True,
                "tree_buckets": [224, 240, 256],
                "threshold": 4,
                "ignored": None,
            }
        }
    )

    assert build_inprocess_argv(config) == [
        "--use_spec",
        "True",
        "--tree_buckets",
        "224,240,256",
        "--threshold",
        "4",
    ]


def test_bundled_runtime_precedes_source_checkout(tmp_path):
    root = tmp_path / "KERV"
    external_runtime = root / "runtime_opt"
    external_package = external_runtime / "KERVRuntimeOptimization"
    external_package.mkdir(parents=True)
    (external_package / "__init__.py").write_text("ORIGIN = 'external'\n", encoding="utf-8")
    config = _config(
        root,
        "entry.py:main",
        python_paths=[str(external_runtime)],
    )

    paths = _python_paths(config, root)
    spec = importlib.machinery.PathFinder.find_spec("KERVRuntimeOptimization", paths)

    assert paths[0] == str(bundled_runtime_path())
    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).is_relative_to(bundled_runtime_path())


def test_kerv_entrypoint_imports_bundled_optimization_runtime(tmp_path):
    root = tmp_path / "KERV"
    external_runtime = root / "runtime_opt"
    external_package = external_runtime / "KERVRuntimeOptimization"
    external_package.mkdir(parents=True)
    (external_package / "__init__.py").write_text(
        "raise RuntimeError('external runtime must not be imported')\n",
        encoding="utf-8",
    )
    entrypoint = root / "entry.py"
    entrypoint.write_text(
        "def main():\n"
        "    import KERVRuntimeOptimization\n"
        "    from KERVRuntimeOptimization import adaptive_linear_fusion\n"
        "    from KERVRuntimeOptimization import adaptive_rms_norm\n"
        "    from KERVRuntimeOptimization import adaptive_rotary_fusion\n"
        "    from KERVRuntimeOptimization import fused_logsoftmax_topk\n"
        "    from KERVRuntimeOptimization import rotary_cache\n"
        "    from KERVRuntimeOptimization import tree_attention_mask\n"
        "    from KERVRuntimeOptimization import embodied_ops\n"
        "    return [module.__file__ for module in (\n"
        "        KERVRuntimeOptimization, adaptive_linear_fusion,\n"
        "        adaptive_rms_norm, adaptive_rotary_fusion,\n"
        "        fused_logsoftmax_topk, rotary_cache,\n"
        "        tree_attention_mask, embodied_ops)]\n",
        encoding="utf-8",
    )
    config = _config(
        root,
        "entry.py:main",
        python_paths=[str(external_runtime)],
        dependencies=["KERVRuntimeOptimization"],
    )

    module_paths = run_kerv_entrypoint(config)

    assert len(module_paths) == 8
    assert all(Path(path).is_relative_to(bundled_runtime_path()) for path in module_paths)


def test_script_function_runs_in_current_process_and_restores_context(tmp_path):
    root = tmp_path / "KERV"
    entrypoint = _write_function_entrypoint(root)
    work_dir = tmp_path / "work"
    original_argv = sys.argv
    original_cwd = Path.cwd()
    original_path = sys.path.copy()
    marker_before = os.environ.get("KERV_SMOKE_MARKER")
    config = _config(
        root,
        f"{entrypoint.name}:main",
        work_dir=str(work_dir),
        env={"KERV_SMOKE_MARKER": "in-process"},
    )

    result = run_kerv_entrypoint(config)

    assert result["argv"][1:] == ["--enabled", "True", "--buckets", "224,240"]
    assert Path(result["cwd"]) == work_dir
    assert result["marker"] == "in-process"
    assert sys.argv is original_argv
    assert Path.cwd() == original_cwd
    assert sys.path == original_path
    assert os.environ.get("KERV_SMOKE_MARKER") == marker_before


def test_dotted_module_function_runs_from_source_checkout(tmp_path):
    root = tmp_path / "KERV"
    _write_function_entrypoint(root, "native_entry.py")
    config = _config(root, "native_entry:main")

    result = run_kerv_entrypoint(config)

    assert result["argv"][0] == "native_entry:main"
    assert result["argv"][1:] == ["--enabled", "True", "--buckets", "224,240"]
    sys.modules.pop("native_entry", None)


def test_plain_script_executes_as_main_without_subprocess(tmp_path):
    root = tmp_path / "KERV"
    output = tmp_path / "plain-script.txt"
    script = root / "plain.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['KERV_SMOKE_OUTPUT']).write_text(\n"
        "    '|'.join(sys.argv[1:]), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    config = _config(root, script.name, env={"KERV_SMOKE_OUTPUT": str(output)})

    run_kerv_entrypoint(config)

    assert output.read_text(encoding="utf-8") == ("--enabled|True|--buckets|224,240")


def test_train_cli_executes_native_entrypoint(tmp_path, monkeypatch):
    root = tmp_path / "OpenVLA"
    script = _write_function_entrypoint(root)
    output = tmp_path / "train.txt"
    config = _config(
        root,
        f"{script.name}:main",
        stage="verifier_lora",
        env={"KERV_SMOKE_OUTPUT": str(output)},
    )
    config_path = tmp_path / "train.yaml"
    OmegaConf.save(OmegaConf.create({"kerv": config}), config_path)
    monkeypatch.setattr(sys, "argv", ["train_kerv", "--config-path", str(config_path)])

    train_main()

    assert output.read_text(encoding="utf-8") == ("--enabled|True|--buckets|224,240")


def test_inference_cli_executes_native_entrypoint(tmp_path, monkeypatch):
    root = tmp_path / "KERV"
    script = _write_function_entrypoint(root)
    output = tmp_path / "inference.txt"
    config = _config(
        root,
        f"{script.name}:main",
        env={"KERV_SMOKE_OUTPUT": str(output)},
    )
    config_path = tmp_path / "inference.yaml"
    OmegaConf.save(OmegaConf.create({"kerv": config}), config_path)
    monkeypatch.setattr(sys, "argv", ["inference_kerv", "--config-path", str(config_path)])

    inference_main()

    assert output.read_text(encoding="utf-8") == ("--enabled|True|--buckets|224,240")


def test_dry_run_validates_configuration_without_importing_model(tmp_path):
    root = tmp_path / "KERV"
    root.mkdir()
    config = _config(root, "unavailable_model.entrypoint:main")

    argv = run_kerv_entrypoint(config, dry_run=True)

    assert argv == ["--enabled", "True", "--buckets", "224,240"]


def test_missing_dependency_reports_stage_and_module(tmp_path):
    root = tmp_path / "KERV"
    script = _write_function_entrypoint(root)
    config = _config(
        root,
        f"{script.name}:main",
        stage="drafter",
        dependencies=["flagscale_kerv_dependency_that_does_not_exist"],
    )

    with pytest.raises(KERVEntrypointError) as error:
        run_kerv_entrypoint(config)

    message = str(error.value)
    assert "drafter" in message
    assert "flagscale_kerv_dependency_that_does_not_exist" in message


def test_inprocess_entrypoint_rejects_nested_arguments_and_external_launchers(tmp_path):
    root = tmp_path / "KERV"
    root.mkdir()
    with pytest.raises(TypeError, match="nested KERV argument"):
        build_inprocess_argv({"arguments": {"nested": {"value": 1}}})

    config = _config(root, "native_entry:main", launcher="torchrun")
    with pytest.raises(KERVEntrypointError, match="launcher='torchrun'"):
        run_kerv_entrypoint(config)
