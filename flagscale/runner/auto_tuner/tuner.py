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
import time
from abc import ABC, abstractmethod

FLAGSCALE_USE_V1 = os.environ.get("FLAGSCALE_USE_V1", "1").lower() in ("1", "true")


class AutoTunerBase(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def tune(self, *args, **kwargs):
        raise NotImplementedError

    def gen(self):
        """Generate a task to run."""
        # 1. Get a strategy from searcher
        # 2. Whether prune by pruner
        # 3. If not pruned, generate the task by generator
        strategy = self.searcher.search()
        while strategy and (self.pruner is not None and self.pruner.prune(strategy, self.history)):
            strategy = self.searcher.search()
        if strategy:
            self.idx += 1
            strategy["idx"] = self.idx
            pruned_count = self.pruner.pruned_count if self.pruner is not None else 0
            pruned_by_memory_model = (
                self.pruner.pruned_by_memory_model if self.pruner is not None else 0
            )
            if "memory_model" in self.config.experiment.auto_tuner:
                self.logger.info(
                    f"Searching {self.idx + pruned_count} / {len(self.searcher.strategies)} strategy, Pruned {pruned_count} strategy, {pruned_by_memory_model} by memory model."
                )
            else:
                self.logger.info(
                    f"Searching {self.idx + pruned_count} / {len(self.searcher.strategies)} strategy, Pruned {pruned_count} strategy."
                )
            self.logger.info(f"Generate task_{self.idx}")
            self.cur_strategy = strategy
            self.cur_task = self.generator.gen(strategy)
        else:
            self.cur_strategy = None

    def need_stop(self):
        """Judge whether need to stop tuning."""
        end_time = time.time()
        # If the max time of tuner is reached, stop
        if self.max_time:
            if end_time - self.start_time > self.max_time:
                return True

        # If no task to tune, stop
        return bool(self.searcher.has_done())
