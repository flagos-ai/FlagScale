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

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from flagscale.runner.backend.backend_megatron import MegatronBackend


def _config(tmp_path: Path, action: str):
    return OmegaConf.create(
        {
            "action": action,
            "experiment": {
                "exp_dir": str(tmp_path),
                "task": {
                    "type": "train",
                    "backend": "megatron",
                    "entrypoint": "flagscale/train/megatron/train_gpt.py",
                },
                "runner": {
                    "hostfile": None,
                    "heartbeat": {
                        "enabled": True,
                        "publish_interval_s": 1,
                        "initial_process_timeout_s": 10,
                        "process_timeout_s": 10,
                        "initial_progress_timeout_s": 20,
                        "progress_timeout_s": 20,
                        "checkpoint_timeout_s": 30,
                        "failure_grace_period_s": 10,
                    },
                },
                "envs": {},
            },
            "train": {
                "system": {
                    "checkpoint": {
                        "save": str(tmp_path / "checkpoints"),
                        "load": str(tmp_path / "checkpoints"),
                    },
                    "logging": {
                        "log_dir": str(tmp_path / "logs"),
                        "scripts_dir": str(tmp_path / "logs" / "scripts"),
                        "pids_dir": str(tmp_path / "logs" / "pids"),
                        "details_dir": str(tmp_path / "logs" / "details"),
                        "tensorboard_dir": str(tmp_path / "tensorboard"),
                        "wandb_save_dir": str(tmp_path / "wandb"),
                        "straggler_dir": str(tmp_path / "logs" / "straggler"),
                    },
                    "straggler_log_dir": str(tmp_path / "logs" / "straggler"),
                },
                "model": {},
                "data": {},
            },
        }
    )


def _backend(config) -> MegatronBackend:
    with (
        patch(
            "flagscale.runner.backend.backend_megatron._get_args_megatron",
            return_value=[],
        ),
        patch("flagscale.runner.backend.backend_megatron._update_config_train"),
        patch(
            "flagscale.runner.backend.backend_megatron.parse_hostfile",
            return_value=None,
        ),
        patch("flagscale.runner.backend.backend_megatron.logger"),
    ):
        return MegatronBackend(config)


def test_run_script_atomically_publishes_active_diagnostics_run_id(tmp_path):
    config = _config(tmp_path, "run")
    backend = _backend(config)

    script_path = backend.generate_run_script(
        config,
        "localhost",
        0,
        "python train.py",
        background=True,
        pkg_dir=str(tmp_path),
    )
    script = Path(script_path).read_text(encoding="utf-8")

    assert "diagnostics.active_run_id" in script
    assert "$$.tmp" in script
    assert script.index("diagnostics.active_run_id") < script.index(
        "flagscale.runner.heartbeat.monitor"
    )


def test_stop_reuses_active_run_id_and_stops_monitor_before_worker(tmp_path):
    run_backend = _backend(_config(tmp_path, "run"))
    active_run_id = run_backend.diagnostics_run_id
    active_file = tmp_path / "logs" / "pids" / "diagnostics.active_run_id"
    active_file.parent.mkdir(parents=True)
    active_file.write_text(active_run_id, encoding="utf-8")

    backend = _backend(_config(tmp_path, "stop"))
    script_path = backend.generate_stop_script("localhost", 0)
    script = Path(script_path).read_text(encoding="utf-8")

    assert backend.diagnostics_run_id == active_run_id
    assert backend.heartbeat_config.run_id == active_run_id
    assert backend.heartbeat_config.heartbeat_dir.endswith(f"heartbeat/{active_run_id}")
    assert script.index("flagscale.runner.heartbeat.monitor") < script.index("pkill -P $pid")
    assert script.index("pkill -P $pid") < script.rindex("diagnostics.active_run_id")


def test_nonzero_node_waits_for_monitor_and_does_not_clear_active_run_id(tmp_path):
    active_file = tmp_path / "logs" / "pids" / "diagnostics.active_run_id"
    active_file.parent.mkdir(parents=True)
    active_file.write_text("original-run", encoding="utf-8")

    backend = _backend(_config(tmp_path, "stop"))
    script_path = backend.generate_stop_script("worker-1", 1)
    script = Path(script_path).read_text(encoding="utf-8")

    assert "[ ! -f" in script
    assert script.index("monitor.pid") < script.index("pkill -P $pid")
    assert "diagnostics.active_run_id" not in script


def test_stop_without_active_run_id_still_stops_worker(tmp_path):
    backend = _backend(_config(tmp_path, "stop"))
    script_path = backend.generate_stop_script("localhost", 0)
    script = Path(script_path).read_text(encoding="utf-8")

    assert backend.heartbeat_config.enabled is False
    assert "flagscale.runner.heartbeat.monitor" not in script
    assert "pkill -P $pid" in script
    assert "pkill -f 'torchrun'" in script
