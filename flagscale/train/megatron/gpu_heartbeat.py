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

"""Expose the FlagScale GPU heartbeat runtime to Megatron integrations."""

from flagscale.train.gpu_heartbeat import (
    initialize_from_env,
    mark_progress,
    mark_training_progress,
    set_phase,
    shutdown,
)

__all__ = [
    "initialize_from_env",
    "mark_progress",
    "mark_training_progress",
    "set_phase",
    "shutdown",
]
