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

import json
import logging

import numpy as np
import PIL
import torch

from megatron.energon import DefaultTaskEncoder
from tools.datasets.vla.data.energon.chatml import ChatMLSample

dataset_logger = logging.getLogger(__name__)


class TaskEncoder(DefaultTaskEncoder[ChatMLSample, ChatMLSample, ChatMLSample, ChatMLSample]):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_root = config.vision_root
        return

    def encode_sample(self, sample: ChatMLSample) -> dict:
        conversation = (
            json.loads(sample.conversation)
            if isinstance(sample.conversation, (str, bytes))
            else sample.conversation
        )
        # For PI0 token <image> is useless, the position of image embeddings are fixed
        task = conversation["conversations"][0]["value"].replace("<image>", "")

        imgs = []
        for i in sample.imgs:
            image = PIL.Image.open(i)
            # image_tensor = transforms.ToTensor()(image)
            # imgs.append(image_tensor)
            imgs.append(image)

        action_paths = sample.metadata["action"][self.config.action_key]
        action = np.load(action_paths)
        # if action.shape[1] < self.config.action_horizon:
        #     pad_width = self.config.action_horizon - action.shape[1]
        #     action = np.pad(action, ((0, 0), (0, pad_width)), mode='constant')
        # elif action.shape[1] > self.config.action_horizon:
        #     action = action[:, : self.config.action_horizon]
        action = torch.from_numpy(action)
        # print(action.shape)
        batch = {
            "lang": task,
            "image": imgs,
            "action": action.to(torch.float32),
        }
        return batch

    def batch(self, samples: list[dict]) -> list[dict]:
        return samples

    def encode_batch(self, samples: dict) -> dict:
        return samples
