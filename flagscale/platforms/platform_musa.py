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


class PlatformMUSA(PlatformBase):
    def name(self) -> str:
        return "musa"

    def is_available(self) -> bool:
        try:
            import torch_musa  # noqa: F401

            return torch.musa.is_available() and torch.musa.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        import torch_musa  # noqa: F401

        torch.musa.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("musa", device_index)

    def device_count(self) -> int:
        import torch_musa  # noqa: F401

        return torch.musa.device_count()

    def dist_backend(self) -> str:
        return "mccl"

    def manual_seed_all(self, seed):
        import torch_musa  # noqa: F401

        torch.musa.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "musa"

    def supports_distributions_on_device(self) -> bool:
        return False
