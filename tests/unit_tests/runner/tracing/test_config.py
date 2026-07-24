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

import os

import pytest
from omegaconf import OmegaConf

from flagscale.runner.tracing.config import prepare_trace_launch_config


def _config(tmp_path, tracing):
    return OmegaConf.create(
        {
            "experiment": {
                "runner": {"tracing": tracing},
            },
            "train": {
                "system": {
                    "logging": {"log_dir": str(tmp_path / "logs")},
                }
            },
        }
    )


def test_disabled_tracing_is_a_noop(tmp_path):
    resolved = prepare_trace_launch_config(_config(tmp_path, {"enabled": False}), "run")
    assert resolved.enabled is False
    assert resolved.shell_setup_lines(0) == []
    assert resolved.command_body(0) == "$cmd; sync"


def test_enabled_tracing_renders_cpu_probe_and_analyzer_shell(tmp_path):
    probe = tmp_path / "probe.so"
    config = _config(
        tmp_path,
        {
            "enabled": True,
            "probe_library": str(probe),
            "collective_timeout_s": 12,
            "p2p_timeout_s": 12,
            "p2p_match_window_s": 4,
            "failure_grace_period_s": 12,
        },
    )

    resolved = prepare_trace_launch_config(config, "run-123")
    shell = "\n".join(resolved.shell_setup_lines(0))

    assert resolved.enabled is True
    assert resolved.trace_dir == os.path.join(str(tmp_path / "logs"), "tracing", "run-123")
    assert "FLAGSCALE_TRACE_ENABLE=1" in shell
    assert "FLAGSCALE_RANK_HEARTBEAT" not in shell
    assert "LD_PRELOAD=" in shell
    assert "flagscale.runner.tracing.monitor" in shell
    assert "--p2p-timeout 12" in shell
    assert "--p2p-match-window 4" in shell
    assert "rc=\\$?" in resolved.command_body(0)
    assert resolved.shell_setup_lines(1)
    assert "flagscale.runner.tracing.monitor" not in "\n".join(resolved.shell_setup_lines(1))


def test_heartbeat_settings_must_use_the_independent_component(tmp_path):
    config = _config(
        tmp_path,
        {
            "enabled": True,
            "heartbeat_timeout_s": 5,
        },
    )
    with pytest.raises(ValueError, match="moved to experiment.runner.heartbeat"):
        prepare_trace_launch_config(config, "run")


def test_p2p_match_window_cannot_exceed_timeout(tmp_path):
    config = _config(
        tmp_path,
        {
            "enabled": True,
            "p2p_timeout_s": 10,
            "p2p_match_window_s": 11,
        },
    )
    with pytest.raises(ValueError, match="p2p_match_window_s"):
        prepare_trace_launch_config(config, "run")


def test_no_shared_filesystem_is_rejected_for_cross_rank_analysis(tmp_path):
    config = _config(tmp_path, {"enabled": True})
    config.experiment.runner.no_shared_fs = True
    with pytest.raises(ValueError, match="shared filesystem"):
        prepare_trace_launch_config(config, "run")
