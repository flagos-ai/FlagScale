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

"""Training-loop hooks for the performance monitor."""

from __future__ import annotations

import torch

from flagscale.train.perf_monitor.perf_metrics import FLOPSMeasurementCallback

_perf_monitor_callback = None


def initialize_perf_monitor(args):
    """Initialize the global performance monitor if enabled."""
    global _perf_monitor_callback

    if not getattr(args, "enable_perf_monitor", False):
        _perf_monitor_callback = None
        return None

    log_interval = getattr(args, "perf_log_interval", 10)
    _perf_monitor_callback = FLOPSMeasurementCallback(args, log_interval=log_interval)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"[Performance Monitor] Initialized with log interval: {log_interval}")

    return _perf_monitor_callback


def get_perf_monitor():
    return _perf_monitor_callback


def perf_monitor_start_iteration(iteration):
    if _perf_monitor_callback is not None:
        _perf_monitor_callback.on_train_batch_start(iteration)


def perf_monitor_end_iteration(iteration, writer=None, wandb_writer=None):
    if _perf_monitor_callback is not None:
        _perf_monitor_callback.on_train_batch_end(iteration, writer, wandb_writer)


def perf_monitor_end_training(writer=None, wandb_writer=None):
    global _perf_monitor_callback
    if _perf_monitor_callback is not None:
        _perf_monitor_callback.on_train_end(writer, wandb_writer)
        _perf_monitor_callback = None
