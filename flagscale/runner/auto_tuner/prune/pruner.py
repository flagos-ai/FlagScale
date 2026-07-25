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

from flagscale.runner.auto_tuner.prune.history import _HISTORY_BASED_PRUNE_FUNC
from flagscale.runner.auto_tuner.prune.memory import (
    prune_by_memory_model,
    prune_by_memory_model_util,
)


class Pruner:
    def __init__(self, config):
        self.config = config
        self.pruned_count = 0
        self.pruned_by_memory_model = 0

    def prune(self, strategy, history=[]):
        """Prune strategy based on history recorded strategies."""
        not_run = False
        if "memory_model" in self.config.experiment.auto_tuner:
            if prune_by_memory_model(self.config, strategy, history):
                not_run = True
                self.pruned_by_memory_model += 1
            elif prune_by_memory_model_util(self.config, strategy, history):
                not_run = True
                self.pruned_by_memory_model += 1

        if not not_run:
            for func in _HISTORY_BASED_PRUNE_FUNC:
                if func(self.config, strategy, history):
                    not_run = True
                    break

        history.append(strategy)
        if not_run:
            self.pruned_count += 1
        return not_run
