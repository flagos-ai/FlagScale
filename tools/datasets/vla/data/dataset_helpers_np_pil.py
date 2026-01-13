import json
import logging

import numpy as np
import PIL

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
            imgs.append(image)

        state_paths = sample.metadata["state"][self.config.state_key]
        state = np.load(state_paths)[0]
        if state.shape[0] < self.config.action_horizon:
            pad_width = self.config.action_horizon - state.shape[0]
            state = np.pad(state, (0, pad_width), mode="constant")
        elif state.shape[0] > self.config.action_horizon:
            state = state[: self.config.action_horizon]

        action_paths = sample.metadata["action"][self.config.action_key]
        action = np.load(action_paths)
        if action.shape[1] < self.config.action_horizon:
            pad_width = self.config.action_horizon - action.shape[1]
            action = np.pad(action, ((0, 0), (0, pad_width)), mode="constant")
        elif action.shape[1] > self.config.action_horizon:
            action = action[:, : self.config.action_horizon]

        batch = {
            "task": task,
            "observation.images.camera0": imgs[0],
            "observation.images.camera1": imgs[1],
            "observation.images.camera2": imgs[2],
            "observation.state": state.astype(np.float16),
            "action": action.astype(np.float16),
        }
        return batch

    def batch(self, samples: list[dict]) -> dict:
        return samples

    def encode_batch(self, samples: dict) -> dict:
        return samples
