# Copyright (c) 2025, BAAI. All rights reserved
#
# Adopted from: https://github.com/starVLA/starVLA/blob/starVLA/starVLA/model/modules/vlm/QWen2_5.py
# Below is the original copyright:
#   Copyright 2025 starVLA community. All rights reserved.
#   Licensed under the MIT License, Version 1.0 (the "License");
#   Implemented by [Jinhui YE / HKUST University] in [2025].
import json
import logging

import numpy as np
import PIL
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from megatron.energon import DefaultTaskEncoder
from tools.datasets.vla.data.energon.chatml import ChatMLSample

dataset_logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
_ACTION_TOKEN_MIN = 151665
_ACTION_TOKEN_MAX = 153712


class TaskEncoder(DefaultTaskEncoder[ChatMLSample, ChatMLSample, ChatMLSample, ChatMLSample]):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_root = config.datasets.task_encoder.vision_root

        model_id = config.get("tokenizer_model_id")
        if model_id is None:
            raise ValueError("tokenizer_model_id must be specified in the config.")
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "left"
        self.processor = processor
        return

    def encode_sample(self, sample: ChatMLSample) -> dict:
        return sample

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

        # image_inputs = list of PIL
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None:
            action_token_min = _ACTION_TOKEN_MIN  # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX  # here only for fast_tokenizer
            labels = batch_input["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning("action token are on in your tokenizer,")
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = -100  ## mask out pad tokens as well
            batch_input["labels"] = labels

        return batch_input

    def batch(self, samples: list[dict]) -> dict:
        return samples

    def encode_batch(self, samples: dict) -> dict:
        batch = []
        for sample in samples:
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

            state_paths = sample.metadata["state"][self.config.datasets.task_encoder.state_key]
            state = np.load(state_paths)[0]
            if state.shape[0] < self.config.datasets.task_encoder.action_horizon:
                pad_width = self.config.datasets.task_encoder.action_horizon - state.shape[0]
                state = np.pad(state, (0, pad_width), mode="constant")
            elif state.shape[0] > self.config.datasets.task_encoder.action_horizon:
                state = state[: self.config.datasets.task_encoder.action_horizon]

            action_paths = sample.metadata["action"][self.config.datasets.task_encoder.action_key]
            action = np.load(action_paths)
            if action.shape[1] < self.config.datasets.task_encoder.action_horizon:
                pad_width = self.config.datasets.task_encoder.action_horizon - action.shape[1]
                action = np.pad(action, ((0, 0), (0, pad_width)), mode="constant")
            elif action.shape[1] > self.config.datasets.task_encoder.action_horizon:
                action = action[:, : self.config.datasets.task_encoder.action_horizon]
            batch.append(
                {
                    "task": task,
                    "observation.images.camera0": imgs[0],
                    "observation.images.camera1": imgs[1],
                    "observation.images.camera2": imgs[2],
                    "observation.state": state.astype(np.float16),
                    "action": action.astype(np.float16),
                }
            )
        trimmed_batch = get_batch(batch=batch)
        batch_images = [example["image"] for example in trimmed_batch]  #  [B，[PLT]]
        instructions = [example["lang"] for example in trimmed_batch]  # [B, str]
        actions = [example["action"] for example in trimmed_batch]  # label [B， len, 7]
        state = (
            [example["state"] for example in trimmed_batch] if "state" in trimmed_batch[0] else None
        )  # [B, 1, state_dim]
        actions = np.stack(actions, axis=0)
        state = np.stack(state, axis=0)

        qwen_inputs = self.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        return {"qwen_inputs": qwen_inputs, "state": state, "actions": actions}


def get_batch(batch):
    rsp_batch = []
    for i_batch in batch:
        ab = {
            "action": i_batch["action"][:16, :7],
            "image": [i_batch["observation.images.camera0"], i_batch["observation.images.camera1"]],
            "lang": i_batch["task"],
            "state": i_batch["observation.state"][:7][None,],
        }
        rsp_batch.append(ab)
    return rsp_batch
