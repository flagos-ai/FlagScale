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

from vllm import LLM, SamplingParams


class Model1Logic:
    def __init__(self):
        self.llm = LLM(
            model="/models/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )

    def forward(self, prompt: str) -> str:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
        result = self.llm.generate([prompt], sampling_params=sampling_params)
        return result[0].outputs[0].text + "__model1__"


class Model2Logic:
    def __init__(self):
        self.llm = LLM(
            model="/models/Qwen2.5-7B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )

    def forward(self, prompt: str) -> str:
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=50)
        result = self.llm.generate([prompt], sampling_params=sampling_params)
        return result[0].outputs[0].text + "__model2__"


class Model3Logic:
    def forward(self, arg1: str, arg2: str) -> str:
        res = arg1 + arg2 + "__model3__"
        return res
