# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CPU checks for image bootstrap isolation and shared candidate validation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts/ci"))
from generate_matrix import load_platforms


def test_unvalidated_platform_does_not_join_automatic_ci(tmp_path):
    (tmp_path / "existing.yml").write_text("paths_watch: [existing/]\n")
    (tmp_path / "new.yml").write_text("ci_enabled: false\npaths_watch: [new/]\n")
    assert load_platforms(tmp_path) == {"existing": ["existing/"]}
    (tmp_path / "new.yml").write_text("ci_enabled: true\npaths_watch: [new/]\n")
    assert set(load_platforms(tmp_path)) == {"existing", "new"}
    (tmp_path / "new.yml").write_text('ci_enabled: "false"\n')
    with pytest.raises(ValueError, match="ci_enabled must be a boolean"):
        load_platforms(tmp_path)


@pytest.mark.parametrize("override", ["", "harbor.example/train@sha256:candidate"])
def test_blank_training_image_requires_candidate_override(tmp_path, override):
    # Mock YAML queries, not the loader's validation: no yq binary or GPU needed.
    config_dir = tmp_path / ".github/configs"
    config_dir.mkdir(parents=True)
    (config_dir / "iluvatar.yml").write_text('ci_train_image: ""\n')
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
/usr/local/bin/yq() {
    case "$*" in
        *ci_train_image*) echo '' ;;
        *timeout_minutes*) echo 20 ;;
        *runtime_recovery_minutes*) echo 60 ;;
        *runner_labels*) echo '["runner"]' ;;
        *) echo null ;;
    esac
}
python() { echo '[]'; }
jq() { cat >/dev/null; echo '[]'; }
load_platform_config iluvatar "$2"
""",
            "bash",
            str(ROOT / "tools/install/utils/load_platform_config.sh"),
            override,
        ],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(tmp_path / "output")},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) == bool(override), result.stdout + result.stderr
    if override:
        assert f"ci_train_image={override}" in (tmp_path / "output").read_text()
    else:
        assert "ci_train_image not found" in result.stdout
