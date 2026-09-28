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

from flagscale.runner.diagnostics import diagnostic_command_body
from flagscale.runner.heartbeat.config import HeartbeatLaunchConfig
from flagscale.runner.tracing.config import TraceLaunchConfig


def test_joint_diagnostics_preserve_completion_markers_and_hardware_cleanup():
    heartbeat = HeartbeatLaunchConfig(
        enabled=True,
        heartbeat_dir="/tmp/heartbeat",
        hardware_health_enabled=True,
    )
    tracing = TraceLaunchConfig(enabled=True, trace_dir="/tmp/tracing")

    node_zero = diagnostic_command_body(0, heartbeat, tracing)
    node_one = diagnostic_command_body(1, heartbeat, tracing)

    assert os.path.join("/tmp/heartbeat", "training.exit_code") in node_zero
    assert os.path.join("/tmp/tracing", "training.exit_code") in node_zero
    assert os.path.join("/tmp/heartbeat", "gpu_health_node_0.pid") in node_zero
    assert os.path.join("/tmp/heartbeat", "gpu_health_node_1.pid") in node_one
    assert os.path.join("/tmp/tracing", "training.exit_code") not in node_one


def test_joint_diagnostics_are_noop_when_disabled():
    command = diagnostic_command_body(
        0,
        HeartbeatLaunchConfig(enabled=False),
        TraceLaunchConfig(enabled=False),
    )

    assert command == "$cmd; sync"
