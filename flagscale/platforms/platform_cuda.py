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

import torch

from .platform_base import PlatformBase


class PlatformCUDA(PlatformBase):
    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("cuda", device_index)

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def dist_backend(self) -> str:
        return "nccl"

    def manual_seed_all(self, seed):
        torch.cuda.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "cuda"
