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

# Copied from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/toolkits/multimodal_data_preprocessing/replace_llava_image_key.py
import json
import os
from argparse import ArgumentParser


def process(in_file, out_file):
    d = os.path.dirname(out_file)
    os.makedirs(d, exist_ok=True)

    try:
        with open(in_file, "r") as f:
            data = json.load(f)
    except:
        with open(in_file, "r") as f:
            data = [json.loads(f) for l in f.readlines()]
    for i, sample in enumerate(data):
        if isinstance(sample, list):
            assert len(sample) == 1
            data[i] = sample[0]
            if "image" in data[i]:
                data[i]["images"] = [data[i].pop("image")]

    with open(out_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    argparser = ArgumentParser()

    argparser.add_argument("--input-file", type=str, required=True)
    argparser.add_argument("--output-file", type=str, default="dataset.json")

    args = argparser.parse_args()
    process(args.input_file, args.output_file)
