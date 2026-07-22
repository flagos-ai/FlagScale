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

from abc import ABC, abstractmethod

from flagscale.runner.auto_tuner.utils import (
    sort_by_memory,
    sort_by_memory_model,
    sort_by_performance,
)


class Algo(ABC):
    def __init__(self, strategies, config):
        self.strategies = strategies
        self.config = config

    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def has_done(self):
        pass


class GridAlgo(Algo):
    def __init__(self, strategies, config):
        super().__init__(strategies, config)
        self.idx = 0
        # Sort by modeling memory
        if "memory_model" in self.config.experiment.auto_tuner:
            self.checkout(mode="memory_model")

    def checkout(self, mode):
        if mode == "memory":
            if self.idx > 0 and self.idx < len(self.strategies):
                self.strategies = self.strategies[: self.idx] + sorted(
                    self.strategies[self.idx :], key=sort_by_memory
                )
        elif mode == "memory_model":
            self.strategies = sorted(self.strategies, key=sort_by_memory_model, reverse=True)
        elif mode == "performance":
            if self.idx > 0 and self.idx < len(self.strategies):
                self.strategies = self.strategies[: self.idx] + sorted(
                    self.strategies[self.idx :], key=sort_by_performance
                )

    def search(self):
        """Return a task iteratively."""
        strategy = None
        if self.idx < len(self.strategies):
            strategy = self.strategies[self.idx]
            self.idx += 1
        return strategy

    def has_done(self):
        """Return True if the task space is empty."""
        return self.idx >= len(self.strategies)
