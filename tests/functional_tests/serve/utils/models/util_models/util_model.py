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


def fn(input_data):
    res = input_data + "__add_process_fn"
    return res


class ModelC:
    def forward(self, input_data):
        res = input_data + "__add_model_C"
        return res


class ModelD:
    def forward(self, input_data_B, input_data_C):
        output_data = input_data_B + input_data_C
        return output_data
