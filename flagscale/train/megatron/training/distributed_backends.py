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


def resolve_distributed_backend(backend):
    """Resolve framework backend names to PyTorch device/backend mappings."""
    if backend != "flagcx":
        return backend

    # Importing flagcx registers the backend and its supported device types.
    import flagcx  # noqa: F401

    devices = torch.distributed.Backend.backend_capability.get("flagcx", ())
    accelerator_devices = [device for device in devices if device != "cpu"]
    if not accelerator_devices:
        raise RuntimeError(
            "FlagCX was imported but did not register an accelerator device. "
            "Please install a FlagCX wheel matching the local accelerator."
        )

    return ",".join(
        ["cpu:gloo"] + [f"{device}:flagcx" for device in accelerator_devices]
    )
