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

from util_models.util_model import fn


class ModelA:
    def forward(self, prompt, system_prompt="hello flagscale"):
        result = prompt + "__add_model_A_" + system_prompt
        return fn(result)


class ModelB:
    def forward(self, input_data):
        res = input_data + "__add_model_B"
        return res


if __name__ == "__main__":
    prompt = "introduce Bruce Lee"
    print(ModelA().forward(prompt))
