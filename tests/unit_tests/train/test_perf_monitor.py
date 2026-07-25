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
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from flagscale.train.perf_monitor.hooks import (
    initialize_perf_monitor,
    perf_monitor_end_iteration,
    perf_monitor_end_training,
    perf_monitor_start_iteration,
)


def test_perf_monitor_smoke_writes_summary(tmp_path):
    args = SimpleNamespace(
        enable_perf_monitor=True,
        perf_log_interval=2,
        perf_log_dir=str(tmp_path),
        perf_console_output=False,
        perf_log_format="both",
        perf_memory_tracking=False,
        perf_breakdown=False,
        perf_max_log_files=10,
        perf_model_type="gpt",
        world_size=1,
        seq_length=512,
        hidden_size=1024,
        num_layers=8,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        padded_vocab_size=50257,
        micro_batch_size=1,
        num_micro_batches=1,
        swiglu=False,
    )

    callback = initialize_perf_monitor(args)
    assert callback is not None

    for iteration in range(1, 5):
        perf_monitor_start_iteration(iteration)
        time.sleep(0.001)
        perf_monitor_end_iteration(iteration)

    perf_monitor_end_training()

    realtime_log = Path(tmp_path) / "perf_realtime.log"
    summary_files = list(Path(tmp_path).glob("perf_summary_*.json"))

    is_rank0 = not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    if not is_rank0:
        assert not realtime_log.exists()
        assert not summary_files
        return

    assert realtime_log.exists()
    assert summary_files

    summary = json.loads(summary_files[0].read_text())
    assert summary["session_info"]["total_iterations"] == 4
