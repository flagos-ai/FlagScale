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

import pynvml


class PynvmlManager:
    def __init__(self):
        pynvml.nvmlInit()

    def get_gpu_info(self):
        gpu_info = {}
        if pynvml.nvmlDeviceGetCount():
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info = {
                "name": pynvml.nvmlDeviceGetName(handle),
                "num": pynvml.nvmlDeviceGetCount(),
                "memory_total": f"{mem_info.total / 1024**3:.2f}GB",
                "memory_used": f"{mem_info.used / 1024**3:.2f}GB",
                "memory_free": f"{mem_info.free / 1024**3:.2f}GB",
            }
        return gpu_info


PYNVML_MANAGER = PynvmlManager()
