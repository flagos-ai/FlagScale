# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process adapter for the public KERV and OpenVLA entrypoints.

KERV intentionally keeps its model implementation and checkpoints in their
upstream projects.  FlagScale owns orchestration: it loads the composed task
configuration, prepares the Python environment and invokes the configured
Python entrypoint in the current worker process.  This avoids a second,
unmanaged ``subprocess`` launcher while keeping the upstream source checkout
replaceable.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import runpy
import shlex
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from types import ModuleType


class KERVEntrypointError(RuntimeError):
    """Raised when FlagScale cannot invoke a configured KERV entrypoint."""


def load_kerv_task_config(config_path: str | Path) -> DictConfig:
    """Load either a composed FlagScale config or a standalone KERV section."""

    config = OmegaConf.load(config_path)
    if config.get("stage") and config.get("entrypoint"):
        kerv = config
    elif "kerv" in config:
        kerv = config.kerv
    elif config.get("train") and "kerv" in config.train:
        kerv = config.train.kerv
    elif config.get("inference") and "kerv" in config.inference:
        kerv = config.inference.kerv
    else:
        raise ValueError(
            "KERV configuration must be a KERV task section or contain kerv, "
            "train.kerv, or inference.kerv"
        )
    for required in ("stage", "source_root", "entrypoint"):
        if not kerv.get(required):
            raise ValueError(f"kerv.{required} is required")
    return kerv


def _render_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ",".join(str(item) for item in value)
    return str(value)


def build_inprocess_argv(kerv: Mapping[str, Any]) -> list[str]:
    """Translate ``kerv.arguments`` into the upstream command-line contract."""

    argv: list[str] = []
    for name, value in (kerv.get("arguments") or {}).items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            raise TypeError(f"nested KERV argument is not supported: {name}")
        argv.extend((f"--{name}", _render_value(value)))
    return argv


def _resolve_root(kerv: Mapping[str, Any]) -> Path:
    source_root = kerv.get("source_root")
    if not source_root:
        raise KERVEntrypointError("kerv.source_root is required")
    root = Path(str(source_root)).expanduser().resolve()
    if not root.is_dir():
        raise KERVEntrypointError(
            f"KERV source checkout does not exist: {root}. "
            "Set KERV_SOURCE_ROOT or OPENVLA_SOURCE_ROOT to a local checkout."
        )
    return root


def _split_entrypoint(entrypoint: str) -> tuple[str, str | None]:
    target, separator, function = entrypoint.rpartition(":")
    if not separator:
        return entrypoint, None
    if not target or not function:
        raise KERVEntrypointError(
            "kerv.entrypoint must be a script path or '<module-or-script>:<function>'"
        )
    return target, function


def _resolve_script(root: Path, target: str) -> Path | None:
    candidate = Path(target).expanduser()
    is_path = candidate.suffix == ".py" or candidate.is_absolute() or "/" in target
    if not is_path:
        return None
    script = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not script.is_relative_to(root):
        raise KERVEntrypointError("kerv.entrypoint must be located under kerv.source_root")
    if not script.is_file():
        raise KERVEntrypointError(
            f"KERV entrypoint does not exist: {script}. "
            "Use a released KERV/OpenVLA checkout or update kerv.entrypoint."
        )
    return script


def _python_paths(kerv: Mapping[str, Any], root: Path) -> list[str]:
    configured = [
        str(Path(str(value)).expanduser().resolve()) for value in kerv.get("python_paths", [])
    ]
    return list(dict.fromkeys((str(root), *configured)))


@contextmanager
def _execution_context(kerv: Mapping[str, Any], root: Path, argv0: str, argv: Sequence[str]):
    old_argv = sys.argv
    old_path = sys.path.copy()
    old_cwd = Path.cwd()
    environment = {str(key): _render_value(value) for key, value in (kerv.get("env") or {}).items()}
    python_paths = _python_paths(kerv, root)
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (*python_paths, os.environ.get("PYTHONPATH", "")) if value
    )
    old_environment = {key: os.environ.get(key) for key in environment}
    work_dir = Path(str(kerv.get("work_dir") or root)).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.argv = [argv0, *argv]
        sys.path[:0] = python_paths
        os.environ.update(environment)
        os.chdir(work_dir)
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        sys.path[:] = old_path
        for key, previous in old_environment.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _check_dependencies(kerv: Mapping[str, Any]) -> None:
    dependencies = kerv.get("dependencies") or []
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    for module_name in dependencies:
        try:
            importlib.import_module(str(module_name))
        except ModuleNotFoundError as error:
            raise KERVEntrypointError(
                f"KERV stage '{kerv.get('stage')}' requires Python module "
                f"'{module_name}', but it is not installed in {sys.executable}."
            ) from error


def _load_script_module(script: Path) -> ModuleType:
    module_name = f"_flagscale_kerv_{abs(hash(script))}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise KERVEntrypointError(f"cannot load KERV Python entrypoint: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _call_function(module: ModuleType, function_name: str, entrypoint: str) -> Any:
    function = getattr(module, function_name, None)
    if function is None or not callable(function):
        raise KERVEntrypointError(
            f"KERV entrypoint function '{function_name}' was not found in {entrypoint}"
        )
    result = function()
    if inspect.isawaitable(result):
        raise KERVEntrypointError("async KERV entrypoints are not supported")
    return result


def _invoke(entrypoint: str, root: Path) -> Any:
    target, function_name = _split_entrypoint(entrypoint)
    script = _resolve_script(root, target)
    if script is not None:
        if function_name is None:
            return runpy.run_path(str(script), run_name="__main__")
        return _call_function(_load_script_module(script), function_name, entrypoint)
    if function_name is None:
        raise KERVEntrypointError(
            "a dotted KERV module entrypoint must include a function, for example "
            "'experiments.robot.libero.run_kerv_libero:eval_libero'"
        )
    module = importlib.import_module(target)
    return _call_function(module, function_name, entrypoint)


def run_kerv_entrypoint(kerv: Mapping[str, Any], *, dry_run: bool = False) -> Any:
    """Run a KERV/OpenVLA Python entrypoint in the current FlagScale worker."""

    launcher = str(kerv.get("launcher") or "python").lower()
    if launcher != "python":
        raise KERVEntrypointError(
            f"in-process KERV execution does not accept launcher='{launcher}'. "
            "Configure launcher=python; FlagScale's runner is responsible for "
            "starting distributed workers."
        )
    root = _resolve_root(kerv)
    entrypoint = str(kerv.get("entrypoint") or "")
    if not entrypoint:
        raise KERVEntrypointError("kerv.entrypoint is required")
    target, _ = _split_entrypoint(entrypoint)
    script = _resolve_script(root, target)
    argv = build_inprocess_argv(kerv)
    display_entrypoint = str(script) if script is not None else entrypoint
    display = shlex.join([display_entrypoint, *argv])
    print(
        f"[FlagScale/KERV] stage={kerv.get('stage')} in_process={display}",
        flush=True,
    )
    if dry_run:
        return argv

    try:
        with _execution_context(kerv, root, display_entrypoint, argv):
            _check_dependencies(kerv)
            return _invoke(entrypoint, root)
    except ModuleNotFoundError as error:
        missing = error.name or "unknown"
        raise KERVEntrypointError(
            f"KERV stage '{kerv.get('stage')}' could not import '{missing}' while "
            f"loading '{entrypoint}'. Install the KERV/OpenVLA dependencies in "
            f"{sys.executable} and verify kerv.python_paths."
        ) from error
