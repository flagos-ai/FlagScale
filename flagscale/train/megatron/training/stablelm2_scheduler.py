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

from dataclasses import dataclass


@dataclass
class StableLM2SchedulerConfig:
    global_batch_size: int
    cosine_samples: int
    rsqrt_samples: int
    alpha: float
    beta: float
    cosine_lr: float = 0.0
    rsqrt_lr: float = 0.0
    cosine_max_lr: float = 0.0
    cosine_period_samples: int = 0
    decay_samples: int = 0

    def __eq__(self, other):
        return (
            self.global_batch_size == other.global_batch_size
            and self.cosine_samples == other.cosine_samples
            and self.rsqrt_samples == other.rsqrt_samples
            and self.alpha == other.alpha
            and self.beta == other.beta
            and self.cosine_max_lr == other.cosine_max_lr
            and self.cosine_period_samples == other.cosine_period_samples
            and self.decay_samples == other.decay_samples
        )
