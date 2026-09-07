# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "tests/test_utils/runners/run_functional_tests.sh"


@pytest.mark.parametrize("source_position", [0, 1, 2, None])
def test_functional_runner_preserves_dependency_paths(tmp_path, source_position):
    deps = tmp_path / "deps with spaces"
    source = deps / "Megatron-LM-FL"
    package = source / "megatron"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "core.py").write_text("SOURCE_ONLY = True\n")
    paths = [str(tmp_path / "first"), str(tmp_path / "last")]
    if source_position is not None:
        paths.insert(source_position, str(source))
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(paths),
        HF_HOME=str(tmp_path / "hf"),
        CHECK_PYTHON=sys.executable,
        CHECK_SOURCE=str(source) if source_position is not None else "",
    )
    if source_position is None:
        env.pop("FLAGSCALE_DEPS", None)
    else:
        env["FLAGSCALE_DEPS"] = str(deps)

    # --help executes the real environment setup without starting GPU work.
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
report_env() {
    "$CHECK_PYTHON" -S -c '
import json, os
if os.environ["CHECK_SOURCE"]:
    import megatron.core
    assert megatron.core.SOURCE_ONLY
print("RUNNER_ENV=" + json.dumps({key: os.environ[key] for key in
    ("PYTHONPATH", "HF_MODULES_CACHE")}))
'
}
trap report_env EXIT
source "$1" --help
""",
            "bash",
            str(RUNNER),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    output = next(line for line in result.stdout.splitlines() if line.startswith("RUNNER_ENV="))
    actual = json.loads(output.removeprefix("RUNNER_ENV="))
    assert actual["PYTHONPATH"].split(os.pathsep) == [str(ROOT), *paths]
    cache = Path(actual["HF_MODULES_CACHE"])
    assert cache.parent == tmp_path / "hf"
    assert cache.name.startswith("modules_")
    assert cache.is_dir()
