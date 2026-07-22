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

import logging

logger = logging.getLogger("FlagScale-AutoTuner")


def prune_by_memory_model(config, strategy, history=[]):
    if "memory_model" in strategy:
        upper_bound = config.experiment.auto_tuner.memory_model.gpu_memory
        if strategy["memory_model"] > upper_bound:
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory {int(strategy['memory_model'])}MB (>{int(upper_bound)}MB)."
            )
            strategy["max_mem"] = "OOM"
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False


def prune_by_memory_model_util(config, strategy, history=[]):
    if "memory_model" in strategy:
        lower_bound = (
            config.experiment.auto_tuner.memory_model.gpu_memory * strategy["gpu_utilization"][0]
        )
        upper_bound = (
            config.experiment.auto_tuner.memory_model.gpu_memory * strategy["gpu_utilization"][1]
        )
        if strategy["memory_model"] > upper_bound or strategy["memory_model"] < lower_bound:
            logger.info(
                f"The strategy {strategy} has been pruned by modeling memory util and its memory is {int(strategy['memory_model'])}MB (>{int(upper_bound)}MB or <{int(lower_bound)}MB)."
            )
            strategy["max_mem"] = None
            strategy["performance"] = None
            strategy["pruned"] = True
            return True
    return False
