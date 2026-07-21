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


def set_jiuding_platform_args(config, orig_config):
    """Set autotuner config by jiuding platform."""
    config.experiment.auto_tuner.platform.airs_switch = True

    if os.environ.get("AIRS_SIZE", None):
        config.experiment.auto_tuner.nnodes = int(os.environ["AIRS_SIZE"])
        # Set original config
        orig_config.experiment.runner.nnodes = int(os.environ["AIRS_SIZE"])
        # Set config
        config.experiment.runner.nnodes = int(os.environ["AIRS_SIZE"])

    if os.environ.get("AIRS_ACCELERATOR_COUNT", None):
        # Set config
        config.experiment.auto_tuner.nproc_per_node = (
            int(os.environ["AIRS_ACCELERATOR_COUNT"]) * 2
            if "luvatar_BI" in os.environ["AIRS_ACCELERATOR_MODEL"]
            else int(os.environ["AIRS_ACCELERATOR_COUNT"])
        )
        # Set original config
        orig_config.experiment.runner.nproc_per_node = (
            int(os.environ["AIRS_ACCELERATOR_COUNT"]) * 2
            if "luvatar_BI" in os.environ["AIRS_ACCELERATOR_MODEL"]
            else int(os.environ["AIRS_ACCELERATOR_COUNT"])
        )
        # Set config
        config.experiment.runner.nproc_per_node = (
            int(os.environ["AIRS_ACCELERATOR_COUNT"]) * 2
            if "luvatar_BI" in os.environ["AIRS_ACCELERATOR_MODEL"]
            else int(os.environ["AIRS_ACCELERATOR_COUNT"])
        )

    if os.environ.get("AIRS_FBMEM", None):
        config.experiment.auto_tuner.memory = int(os.environ["AIRS_FBMEM"])

    if os.environ.get("AIRS_HOSTFILE_PATH", None):
        # Set original config
        orig_config.experiment.runner.hostfile = os.environ["AIRS_HOSTFILE_PATH"]
        # Set config
        config.experiment.runner.hostfile = os.environ["AIRS_HOSTFILE_PATH"]
