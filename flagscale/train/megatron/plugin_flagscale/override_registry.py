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

"""
Centralized override registry for FlagScale training
"""

from megatron.plugin.decorators import register


# =============================================================================
# DistSignalHandler - get_device
# =============================================================================

register(
    target="megatron.training.dist_signal_handler.get_device",
    impl="megatron.plugin_flagscale.dist_signal_handler.get_device",
)

register(
    target="megatron.training.dist_signal_handler.get_device",
    impl="megatron.plugin_flagscale.npu_plugin.get_device",
    vendor="npu",
)

register(
    target="megatron.training.utils.get_device_arch_version",
    impl="megatron.plugin_flagscale.npu_plugin.get_device_arch_version",
    vendor="npu",
)

register(
    target="megatron.training.initialize.initialize._compile_dependencies",
    impl="megatron.plugin_flagscale.npu_plugin._compile_dependencies",
    vendor="npu",
)
