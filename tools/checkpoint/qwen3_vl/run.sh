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

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=../../../:$PYTHONPATH
# export MP_PP0_LAYERS=30
# bash hf2mcore_qwen3_vl_convertor.sh \
#     32B \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-32B-Instruct \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-32B-Instruct-tp4-pp2 \
#     4 \
#     2 \
#     false \
#     bf16 \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-32B-Instruct

bash hf2mcore_qwen3_vl_convertor.sh \
    8B \
    /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-8B-Instruct \
    /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-8B-Instruct-tp1 \
    1 \
    1 \
    false \
    bf16 \
    /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-8B-Instruct


# bash hf2mcore_qwen3_vl_convertor.sh \
#     4B \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-4B-Instruct-tp2 \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-4B-Instruct-tp2-hf \
#     2 \
#     1 \
#     true \
#     bf16 \
#     /share/project/lizhiyu/data/qwen3_vl/Qwen3-VL-4B-Instruct
