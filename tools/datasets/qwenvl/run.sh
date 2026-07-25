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

export PYTHONPATH=$PYTHONPATH:../../../
python convert_custom_dataset_to_wds_chatml_str.py \
    --dataset-root=/share/project/lizhiyu/LLaMA-Factory/data/sample_dataset/text_only/ \
    --output-root=/share/project/lizhiyu/LLaMA-Factory/data/sample_dataset/text_only/ \
    --json=text_only_samples_10_first.json \
    --train-split 1 \
    --val-split 0 \
    --images-key=image \
    --videos-key=video \
    --vision-root=/mnt/LLaVA-Pretrain \
    --max-samples-per-tar 100000000 \
    --dp-size 1 \
    --num-workers 20
