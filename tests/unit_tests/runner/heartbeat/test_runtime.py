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

import json
import time

from flagscale.train import gpu_heartbeat
from flagscale.train.gpu_heartbeat import GpuProgressHeartbeat


def _runtime(tmp_path, interval=0.02):
    return GpuProgressHeartbeat(
        run_id="run",
        output_dir=tmp_path,
        rank=1,
        local_rank=1,
        world_size=2,
        publish_interval_s=interval,
    )


def test_progress_is_advanced_only_by_training_hook(tmp_path):
    runtime = _runtime(tmp_path)

    assert runtime.snapshot().progress_seq == 0
    runtime.set_phase("train")
    assert runtime.snapshot().progress_seq == 0

    runtime.mark_progress("train", iteration=17)
    snapshot = runtime.snapshot()
    assert snapshot.progress_seq == 1
    assert snapshot.iteration == 17
    assert snapshot.phase == "train"

    runtime.mark_progress("checkpointing")
    snapshot = runtime.snapshot()
    assert snapshot.progress_seq == 2
    assert snapshot.iteration == 17
    assert snapshot.phase == "checkpointing"


def test_megatron_training_progress_uses_completed_iteration(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path)
    monkeypatch.setattr(gpu_heartbeat, "_runtime", runtime)

    gpu_heartbeat.mark_training_progress(99)

    snapshot = runtime.snapshot()
    assert snapshot.progress_seq == 1
    assert snapshot.iteration == 100


def test_background_publisher_does_not_manufacture_gpu_progress(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.start()
    time.sleep(0.08)
    runtime.stop()

    records = [json.loads(line) for line in runtime.path.read_text().splitlines()]
    assert records[0]["event"] == "process_start"
    assert records[-1]["event"] == "process_end"
    assert any(record["event"] == "heartbeat" for record in records)
    assert {record["progress_seq"] for record in records} == {0}


def test_publisher_reports_progress_marked_by_training_hook(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.start()
    runtime.mark_progress("train", iteration=3)
    time.sleep(0.06)
    runtime.stop()

    records = [json.loads(line) for line in runtime.path.read_text().splitlines()]
    progressed = [record for record in records if record["progress_seq"] == 1]
    assert progressed
    assert progressed[-1]["iteration"] == 3
    assert progressed[-1]["phase"] == "train"
    assert progressed[-1]["assigned_gpu"] == {
        "visible_ordinal": 1,
        "device_token": "1",
    }
