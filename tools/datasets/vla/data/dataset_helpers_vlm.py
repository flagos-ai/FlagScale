# Mainly Adopted from  https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/dataset_helpers.py.
# Below is the original copyright:

# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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
import dataclasses
import json
import logging
import math
import os
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import PIL
import torch
from PIL import Image

from megatron.energon import Batch, DefaultTaskEncoder, VQASample
from megatron.training import get_args
from megatron.training.global_vars import get_tokenizer
from tools.datasets.qwenvl.data.image_processing import get_visual_transform
from tools.datasets.vla.data.energon.chatml import ChatMLSample

from flagscale.logger import logger

dataset_logger = logging.getLogger(__name__)
FIRST_MAX_PADDING_FLAG = True
LAST_LARGE_IMG = False
CLEAR_CACHE_ITERATION = 200000
IGNORE_IDX = -100
MAX_IMG_THRESHOLD = 5000


# Type for intermediate batch, after batch()
@dataclass
class ImageTaskSample:
    __key__: str
    __subflavors__: dict

    imgs: list[np.ndarray]  # (c, h, w)
    videos: list[np.ndarray]  # (c, h, w)

    image_thw_grids: np.ndarray
    video_thw_grids: np.ndarray
    image_input_mask: np.ndarray
    video_input_mask: np.ndarray
    second_per_grid_ts: np.ndarray  # (n_videos, )

    text: np.ndarray
    target: np.ndarray


# Typing for the resulting batch data after encode_batch()
@dataclass
class VQATaskBatch(Batch):
    __keys__: list[str]
    __subflavors__: list[dict]
    # (num_tiles, c, h, w)
    imgs: torch.Tensor
    videos: torch.Tensor
    image_thw_grids: torch.Tensor
    video_thw_grids: torch.Tensor
    image_input_mask: torch.Tensor
    video_input_mask: torch.Tensor
    second_per_grid_ts: torch.Tensor  # (n_videos, ), read from metadata?

    # (n, seq_len)
    text: torch.Tensor
    # (n, seq_len)
    target: torch.Tensor


class InternalWarning(Warning): ...


def convert_to_qwen2vl_content(
    user_input: str, image_pattern: str = "<image>", video_pattern: str = "<video>"
):
    """
    Split user input into format Qwen2VL tokenizer accepts.
    """
    if not isinstance(user_input, str):
        raise TypeError(f"Input not a string. {type(user_input)} {user_input}")
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        text = user_input[cur:start]
        if text:
            contents.append({"type": "text", "text": text})

        contents.append(
            {
                "type": matched.string[start:end][1:-1],
                matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]]),
            }
        )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur : len(user_input)]})

    return contents


class TaskEncoder(
    DefaultTaskEncoder[VQASample | ChatMLSample, ImageTaskSample, VQATaskBatch, dict]
):
    """A simple task encoder for captioning."""

    ACTION_TOKEN_START_ID = 149595
    ACTION_TOKEN_END_ID = ACTION_TOKEN_START_ID + 2048

    def __init__(self):
        super().__init__()

        self.args = get_args()
        self.tp_size = self.args.tensor_model_parallel_size
        self.cp_size = self.args.context_parallel_size
        self.sequence_parallel = self.args.sequence_parallel

        self.tokenizer = get_tokenizer()

        self.temporal_patch_size = self.args.temporal_patch_size
        self.merge_size = self.args.spatial_merge_size
        self.patch_size = self.args.patch_size

        self.seq_len = self.args.max_padding_length

        self.vision_root = self.args.vision_root
        self._token_cache = self._build_token_cache()
        self._action_token_cache = self._build_action_token_cache()

        assert self.vision_root is not None, "Please specify the vision root."

    def encode_sample(self, sample: VQASample | ChatMLSample):
        if isinstance(sample, VQASample):
            is_llava_training = (
                sample.__subflavors__["is_llava_training"]
                if "is_llava_training" in sample.__subflavors__
                else False
            )
            if is_llava_training:
                raise NotImplementedError("Sample format not supported")
            else:
                yield self.encode_vqa(sample)
        elif isinstance(sample, ChatMLSample):
            yield self.encode_chatml(sample)
        else:
            raise NotImplementedError("Sample format not supported")

    def _flatten_visual_inputs(self, visuals, is_image: bool = True):
        """
        visuals: list of visual inputs, each input is a tensor of shape (c, h, w)
        """
        flattened = []
        thw_grids = []
        for visual in visuals:
            if is_image:
                resized_height, resized_width = visual.shape[-2:]
                patches = np.tile(np.array(visual), (self.temporal_patch_size, 1, 1, 1))
            else:
                assert len(visual) % self.temporal_patch_size == 0
                patches = np.array(visual)
                resized_height, resized_width = patches.shape[-2:]

            channel = patches.shape[1]
            grid_t = patches.shape[0] // self.temporal_patch_size
            grid_h, grid_w = (resized_height // self.patch_size, resized_width // self.patch_size)
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
            flattened.append(flatten_patches)
            thw_grids.append((grid_t, grid_h, grid_w))
        return flattened, np.array(thw_grids)

    def _preprocess_image(
        self, image: PIL.Image, image_max_pixels: int = 768 * 768, image_min_pixels: int = 32 * 32
    ) -> PIL.Image:
        """
        Pre-processes a single image.
        """
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))
        elif (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        return image

    def _safe_encode(self, text):
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def decode_token_ids_to_readable(self, token_ids, max_tokens=100):
        logger.info("=== Text recover: first {} tokens ===".format(max_tokens))

        result_text = ""
        boa_id = self._token_cache.get("boa")
        EOA_id = self._token_cache.get("EOA")
        for i, token_id in enumerate(token_ids[:max_tokens]):
            if token_id == self._token_cache["im_start"]:
                result_text += "\n<|im_start|>"
            elif token_id == self._token_cache["im_end"]:
                result_text += "<|im_end|>\n"
            elif boa_id is not None and token_id == boa_id:
                result_text += "<boa>"
            elif EOA_id is not None and token_id == EOA_id:
                result_text += "<EOA>"
            elif token_id == self._token_cache["image_pad"]:
                result_text += "<|image_pad|>"
            elif token_id == self._token_cache["newline"]:
                result_text += "\\n"
            elif token_id == self._token_cache["space"]:
                result_text += " "
            elif token_id == self._token_cache["user"]:
                result_text += "user"
            elif token_id == self._token_cache["assistant"]:
                result_text += "assistant"
            elif token_id == self._token_cache["system"]:
                result_text += "system"
            else:
                found_token = None
                for text, tid in self.tokenizer.vocab.items():
                    if tid == token_id:
                        found_token = text
                        break

                if found_token:
                    result_text += found_token
                else:
                    result_text += f"[UNK_{token_id}]"

        logger.info(result_text)
        return result_text

    def _build_token_cache(self):
        cache_start = time.time()

        token_cache = {
            "im_start": self.tokenizer.vocab["<|im_start|>"],
            "im_end": self.tokenizer.vocab["<|im_end|>"],
            "user": self.tokenizer.vocab["user"],
            "assistant": self.tokenizer.vocab["assistant"],
            "system": self.tokenizer.vocab["system"],
            "vision_start": self.tokenizer.vocab.get("<|vision_start|>"),
            "vision_end": self.tokenizer.vocab.get("<|vision_end|>"),
            "image_pad": self.tokenizer.vocab.get("<|image_pad|>"),
            "video_pad": self.tokenizer.vocab.get("<|video_pad|>"),
            "newline": self._safe_encode("\n")[0],
            "space": self._safe_encode(" ")[0],
            "boa": self.tokenizer.vocab.get("<boa>", 151665),
            "EOA": self.tokenizer.vocab.get("<EOA>", 151666),
            "action_split": self.tokenizer.vocab.get("<action_split>", 151667),
        }

        cache_end = time.time()
        logger.info(f"Token cache built in {(cache_end - cache_start) * 1000:.2f} ms")
        return token_cache

    def _build_action_token_cache(self):
        cache_start = time.time()
        action_cache = {}
        for action_id in range(2048):
            token_string = f"<action_token_{action_id}>"
            token_id = self.tokenizer.vocab.get(
                token_string, TaskEncoder.ACTION_TOKEN_START_ID + action_id
            )
            if token_id is not None:
                action_cache[action_id] = token_id

        cache_end = time.time()
        logger.info(
            f"Action token cache built in {(cache_end - cache_start) * 1000:.2f} ms with {len(action_cache)} tokens"
        )
        return action_cache

    def build_conversation_tokens(self, conversation, action_tokens_list):
        final_token_ids = []

        im_start_id = self._token_cache["im_start"]
        im_end_id = self._token_cache["im_end"]
        newline_id = self._token_cache["newline"]
        user_id = self._token_cache["user"]
        assistant_id = self._token_cache["assistant"]
        system_id = self._token_cache["system"]
        image_pad_id = self._token_cache["image_pad"]
        vision_start_id = self._token_cache["vision_start"]
        vision_end_id = self._token_cache.get("vision_end")

        for turn_idx, turn in enumerate(conversation):
            role = turn["role"]
            content = turn["content"]
            action_tokens = (
                action_tokens_list[turn_idx] if turn_idx < len(action_tokens_list) else []
            )
            final_token_ids.append(im_start_id)

            if role == "system":
                final_token_ids.append(system_id)
                final_token_ids.append(newline_id)
                if content.strip():
                    text_ids = self._safe_encode(content)
                    final_token_ids.extend(text_ids)

            elif role == "user":
                final_token_ids.append(user_id)
                final_token_ids.append(newline_id)
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "text":
                            if item["text"].strip():
                                text_ids = self._safe_encode(item["text"])
                                final_token_ids.extend(text_ids)
                        elif item["type"] == "image":
                            if vision_start_id:
                                final_token_ids.append(vision_start_id)
                            final_token_ids.append(image_pad_id)
                            if vision_end_id:
                                final_token_ids.append(vision_end_id)
                else:
                    if content.strip():
                        text_ids = self._safe_encode(content)
                        final_token_ids.extend(text_ids)

            elif role == "assistant":
                final_token_ids.append(assistant_id)
                final_token_ids.append(newline_id)
                if content.strip():
                    text_ids = self._safe_encode(content)
                    final_token_ids.extend(text_ids)
                if action_tokens and len(action_tokens) > 0:
                    boa_id = self._token_cache["boa"]
                    EOA_id = self._token_cache["EOA"]
                    action_split_id = self._token_cache["action_split"]
                    if content.strip():
                        final_token_ids.append(boa_id)
                    for i, action_id in enumerate(action_tokens):
                        if action_id == -1:
                            final_token_ids.append(action_split_id)
                        elif action_id == -2:
                            logger.info("EOA has been loaded successfully")
                            final_token_ids.append(EOA_id)
                        else:
                            correct_token_id = self._action_token_cache.get(action_id)
                            if correct_token_id is None:
                                raise ValueError(f"Action token {action_id} not found in cache.")
                            final_token_ids.append(correct_token_id)
            final_token_ids.append(im_end_id)
            final_token_ids.append(newline_id)
        result = np.array(final_token_ids, dtype=np.int64)
        return result

    def encode_chatml(self, sample: ChatMLSample):
        # Process images
        if sample.imgs is not None and len(sample.imgs) > 0:
            imgs = []
            for img in sample.imgs:
                img_path = os.path.join(self.vision_root, img)
                try:
                    image = PIL.Image.open(img_path)
                    image = self._preprocess_image(
                        image=image,
                        image_max_pixels=self.args.image_max_pixels,
                        image_min_pixels=self.args.image_min_pixels,
                    )
                    imgs.append(image)
                except Exception as e:
                    raise ValueError(
                        f"Failed to open image: {img_path}. Error: {e} of sample[{sample.__key__}]"
                    )
            imgs_info = self.tokenizer.processor.image_processor(imgs, return_tensors="np")
            flattened_imgs = imgs_info["pixel_values"]
            image_thw_grids = imgs_info["image_grid_thw"]
        else:
            flattened_imgs = []
            image_thw_grids = []

        # Process videos
        if sample.videos is not None and len(sample.videos) > 0:
            videos = [
                [PIL.Image.open(os.path.join(self.vision_root, frame)) for frame in video]
                for video in sample.videos
            ]
            for i, video in enumerate(videos):
                videos[i] = video[: len(video) // 2 * 2]
            videos_info = self.tokenizer.processor.image_processor(
                images=None, videos=videos, return_tensors="pt"
            )
            flattened_videos = videos_info["pixel_values_videos"]
            video_thw_grids = videos_info["video_grid_thw"]
        else:
            flattened_videos = []
            video_thw_grids = []

        conversation = (
            json.loads(sample.conversation)
            if isinstance(sample.conversation, (str, bytes))
            else sample.conversation
        )

        second_per_grid_ts = [1 / 2.0] * len(video_thw_grids)
        if "conversations" in conversation:
            second_per_grid_ts = conversation.get("second_per_grid_ts", second_per_grid_ts)
            second_per_grid_ts = [float(i) for i in second_per_grid_ts]
            conversation = conversation["conversations"]

        role_key = "from" if "from" in conversation[0] else "role"
        content_key = "value" if "from" in conversation[0] else "content"

        converted_conversation = []
        action_tokens_list = []  # Store action tokens for each turn
        if len(conversation) % 2 == 0:
            converted_conversation.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
            action_tokens_list.append([])
        else:
            dataset_logger.warning(
                f"The sample [{sample.__key__}] has an odd number of conversation turns. The first turn will be used as a system prompt, but this may be incorrect. Please check the sample."
            )
            converted_conversation.append(
                {"role": "system", "content": conversation[0][content_key]}
            )
            action_tokens_list.append([])
            conversation = conversation[1:]

        EXPECTED_ROLE = ["human", "gpt"]
        for turn_idx, turn in enumerate(conversation):
            role = turn[role_key]
            if role != EXPECTED_ROLE[turn_idx % len(EXPECTED_ROLE)]:
                raise InternalWarning(
                    f"Expect conversation organized in order: [sys] human gpt human gpt..., but got role '{role}' in turn {turn_idx}"
                )
            content = turn[content_key]
            current_action_tokens = []

            if role == "human":
                role = "user"
                content = convert_to_qwen2vl_content(content)

            elif role == "gpt":
                role = "assistant"
                if content.strip() == "<EOA>":
                    # Use a special marker (-2) to represent the EOA token.
                    current_action_tokens = [-2]
                    action_tokens_loaded = True
                    content = ""  # Clear content to prevent it from being tokenized as text
                elif "<action_token>" in content:
                    text_before_action = ""
                    action_part = content
                    if "<boa>" in content:
                        parts = content.split("<boa>", 1)
                        text_before_action = parts[0].strip()
                        action_part = parts[1] if len(parts) > 1 else ""
                        dataset_logger.debug(
                            f"Sample [{sample.__key__}]: Mixed content detected. Text: '{text_before_action}'"
                        )

                    current_action_tokens = []
                    action_tokens_loaded = False
                    action_token_paths = None
                    if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
                        action_token_data = sample.metadata.get("action_eepose_token")
                        if action_token_data is None:
                            action_token_data = sample.metadata.get("action_token")
                        if action_token_data is not None:
                            action_token_count = action_part.count("<action_token>")
                            if (
                                isinstance(action_token_data, list)
                                and len(action_token_data) > 0
                                and isinstance(action_token_data[0], list)
                            ):
                                dataset_logger.info(
                                    f"Sample [{sample.__key__}]: Loading action tokens directly from list"
                                )
                                logger.info("Loading action tokens directly from list")
                                if len(action_token_data) != action_token_count:
                                    dataset_logger.warning(
                                        f"Sample [{sample.__key__}]: action_token count mismatch. "
                                        f"Found {action_token_count} <action_token> tags but {len(action_token_data)} token lists."
                                    )
                                for i, token_list in enumerate(action_token_data):
                                    try:
                                        tokens = [int(token) for token in token_list]
                                        current_action_tokens.extend(tokens)
                                        action_tokens_loaded = True
                                        if i < len(action_token_data) - 1:
                                            logger.info("Add Separate token")
                                            current_action_tokens.append(-1)
                                    except Exception as e:
                                        dataset_logger.warning(
                                            f"Failed to process action token list at index {i}. Error: {e}"
                                        )
                                        current_action_tokens = []
                                        action_tokens_loaded = False
                                        break
                            else:
                                dataset_logger.warning(
                                    f"Sample [{sample.__key__}]: Loading action tokens from file paths (legacy mode). "
                                    f"Consider updating to use direct token lists for better performance."
                                )
                                action_token_paths = action_token_data
                                logger.info("Loading action tokens from paths")
                                if isinstance(action_token_paths, str):
                                    action_token_paths = [action_token_paths]
                                elif not isinstance(action_token_paths, list):
                                    dataset_logger.warning(
                                        f"Sample [{sample.__key__}]: Unexpected action_token type: {type(action_token_paths)}"
                                    )
                                    action_token_paths = []
                                if len(action_token_paths) != action_token_count:
                                    dataset_logger.warning(
                                        f"Sample [{sample.__key__}]: action_token count mismatch. "
                                        f"Found {action_token_count} <action_token> tags but {len(action_token_paths)} files."
                                    )
                                for i, action_token_path in enumerate(action_token_paths):
                                    full_action_token_path = os.path.join(
                                        self.vision_root, action_token_path
                                    )
                                    if os.path.exists(full_action_token_path):
                                        try:
                                            loaded_tokens = (
                                                np.load(full_action_token_path).flatten().tolist()
                                            )
                                            tokens = [int(token) for token in loaded_tokens]
                                            current_action_tokens.extend(tokens)
                                            action_tokens_loaded = True
                                            if i < len(action_token_paths) - 1:
                                                current_action_tokens.append(-1)
                                        except Exception as e:
                                            dataset_logger.warning(
                                                f"Failed to load action token file: {full_action_token_path}. Error: {e}"
                                            )
                                            current_action_tokens = []
                                            action_tokens_loaded = False
                                            break
                                    else:
                                        dataset_logger.warning(
                                            f"Action token file not found: {full_action_token_path}"
                                        )
                                        action_tokens_loaded = False
                                        break
                    if action_tokens_loaded and current_action_tokens:
                        content = text_before_action
                        dataset_logger.debug(
                            f"Sample [{sample.__key__}]: Loaded {len(current_action_tokens)} action tokens"
                        )
                    else:
                        should_have_action_tokens = (
                            hasattr(sample, "metadata")
                            and isinstance(sample.metadata, dict)
                            and "action_eepose_token" in sample.metadata
                        )

                        if should_have_action_tokens:
                            dataset_logger.error(
                                f"Sample [{sample.__key__}]: Failed to load action tokens despite having 'action_eepose_token' in metadata. "
                                f"File path might be incorrect: {sample.metadata.get('action_eepose_token')}. "
                                f"Converting to regular text generation."
                            )
                        else:
                            dataset_logger.debug(
                                f"Sample [{sample.__key__}]: No action tokens available, treating as regular text generation."
                            )
                        content = content.replace("<action_token>", "").replace(
                            "<action_split>", ""
                        )
                        content = re.sub(r"\s+", " ", content).strip()
                        if not content:
                            content = "I understand the task."

                        current_action_tokens = []

            converted_conversation.append({"role": role, "content": content})
            action_tokens_list.append(current_action_tokens)

        conversation = converted_conversation

        input_ids = self.build_conversation_tokens(converted_conversation, action_tokens_list)

        target = input_ids.copy()
        pad_token_id = IGNORE_IDX

        # Calculate system prompt length and set its mask
        if converted_conversation[0]["role"] == "system":
            system_tokens = self.build_conversation_tokens(
                [converted_conversation[0]], [action_tokens_list[0]]
            )
            system_prompt_prefix = len(system_tokens)
            target[:system_prompt_prefix] = pad_token_id
            start_idx = 1
        else:
            system_prompt_prefix = 0
            start_idx = 0

        offset = system_prompt_prefix

        for turn_idx in range(start_idx, len(converted_conversation)):
            turn = converted_conversation[turn_idx]
            action_tokens = (
                action_tokens_list[turn_idx] if turn_idx < len(action_tokens_list) else []
            )

            # Calculate the token length of the current turn
            turn_tokens = self.build_conversation_tokens([turn], [action_tokens])
            n_tokens = len(turn_tokens)

            if turn["role"] == "user":
                # Mask all user input
                target[offset : offset + n_tokens] = pad_token_id
            elif turn["role"] == "assistant":
                # Mask the assistant's prompt prefix, but not the generated content (text or action tokens)
                assistant_generation_prefix = 3  # <im_start>assistant\n
                target[offset : offset + assistant_generation_prefix] = pad_token_id

            offset += n_tokens

        merge_length = self.merge_size**2
        image_token_id, video_token_id = self.tokenizer.encode(["<|image_pad|>", "<|video_pad|>"])

        image_token_indices = np.where(input_ids == image_token_id)[0]
        assert len(image_token_indices) == len(image_thw_grids), (
            f"The sample [{sample.__key__}] has {len(image_thw_grids)} images, but {len(image_token_indices)} image placeholders!"
        )
        video_token_indices = np.where(input_ids == video_token_id)[0]
        assert len(video_token_indices) == len(video_thw_grids), (
            f"The sample [{sample.__key__}] has {len(video_thw_grids)} videos, but {len(video_token_indices)} video placeholders!"
        )
        image_thw_grids, video_thw_grids = (
            np.array(image_thw_grids, dtype=np.int64),
            np.array(video_thw_grids, dtype=np.int64),
        )

        target_length = (
            input_ids.shape[0]
            - image_thw_grids.shape[0]
            + image_thw_grids.prod(axis=-1).sum() // merge_length
            - video_thw_grids.shape[0]
            + video_thw_grids.prod(axis=-1).sum() // merge_length
        )
        if target_length > self.seq_len:
            dataset_logger.warning(
                f"Sample id [{sample.__key__}] has a long sequence with length {target_length}, which will be cutoff to the max length {self.seq_len} in the batching function."
            )
        final_input_ids = np.zeros(target_length, dtype=input_ids.dtype)
        final_input_masks = final_input_ids.copy()

        image_idx, video_idx = 0, 0
        indices = np.sort(np.concatenate([image_token_indices, video_token_indices]))
        cur_x, cur_y = 0, 0
        for idx in indices:
            token_id = input_ids[idx]
            if token_id == image_token_id:
                size = image_thw_grids[image_idx].prod() // merge_length
                image_idx += 1
            elif token_id == video_token_id:
                size = video_thw_grids[video_idx].prod() // merge_length
                video_idx += 1
            final_input_ids[cur_y : cur_y + idx - cur_x] = input_ids[cur_x:idx]
            final_input_masks[cur_y : cur_y + idx - cur_x] = target[cur_x:idx]
            cur_y += idx - cur_x
            final_input_ids[cur_y : cur_y + size] = token_id
            final_input_masks[cur_y : cur_y + size] = pad_token_id
            cur_y += size
            cur_x = idx + 1

        if cur_x < len(input_ids):
            final_input_ids[cur_y:] = input_ids[cur_x:]
            final_input_masks[cur_y:] = target[cur_x:]

        target = np.roll(final_input_masks, shift=-1)
        target[-1] = pad_token_id

        if (target == pad_token_id).all():
            raise InternalWarning(
                f"Sample id [{sample.__key__}] has all masked labels. The data is invalid and will be dropped!"
            )
        logger.info(f"DEBUG FINAL CHECK FOR sample [{sample.__key__}]:")
        action_token_indices = np.where(
            (final_input_ids >= self.ACTION_TOKEN_START_ID)
            & (final_input_ids < self.ACTION_TOKEN_END_ID)
        )[0]

        if len(action_token_indices) > 0:
            logger.info("  --- Action Token Verification ---")
            indices_to_check = list(action_token_indices[:3]) + list(action_token_indices[-3:])

            for idx in sorted(list(set(indices_to_check))):
                input_token = final_input_ids[idx]
                prev_input_token = final_input_ids[idx - 1]
                target_for_prev_token = target[idx - 1]

                logger.info(f"  - At index {idx - 1}:")
                logger.info(f"      Input Token: {prev_input_token}")
                logger.info(
                    f"      Target:      {target_for_prev_token}  <-- This should be the token to predict"
                )
                logger.info(f"  - At index {idx}:")
                logger.info(f"      Input Token: {input_token}  <-- This is the action token")

                if input_token == target_for_prev_token:
                    logger.info(
                        "      ✅ CHECK PASSED: Target correctly set to predict the action token."
                    )
                else:
                    logger.info(
                        f"      ❌ CHECK FAILED: Target is {target_for_prev_token}, but should be {input_token}."
                    )
            logger.info("  ---------------------------------")
        else:
            logger.info("  - No action tokens found in this sample for verification.")

        image_input_mask = final_input_ids == self.tokenizer.image_token_id
        video_input_mask = final_input_ids == self.tokenizer.video_token_id

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flattened_imgs,
            videos=flattened_videos,
            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            second_per_grid_ts=np.array(second_per_grid_ts, dtype=np.float32),
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            text=final_input_ids,
            target=target,
        )

    def encode_vqa(self, sample: VQASample):
        has_video = (
            sample.__subflavors__["has_video"] if "has_video" in sample.__subflavors__ else False
        )

        if has_video:
            raise NotImplementedError("You should use sharegpt dataset to train with videos.")
        else:
            imgs = get_visual_transform(sample.image)
            flatten_patches, thw_grids = self._flatten_visual_inputs(imgs, is_image=True)

        assert "<image>" in sample.context
        if isinstance(sample.answers, list):
            answer_list = sample.answers
            weight_list = np.array(sample.answer_weights).astype(np.float32)
            weight_list = weight_list / np.sum(weight_list)
            answer_idx = np.random.choice(weight_list.shape[0], 1, p=weight_list)[0]
            answer = answer_list[answer_idx]
        else:
            answer = sample.answers

        conversation = [
            {"role": "user", "content": convert_to_qwen2vl_content(sample.context)},
            {"role": "assistant", "content": answer},
        ]

        user_inputs = self.tokenizer.apply_chat_template(conversation[:-1], tokenize=False)
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        merge_length = self.merge_size**2
        image_token = "<|image_pad|>"
        assert len(thw_grids) == 1, "Only one image per sample is supported!"
        index = 0
        while image_token in text:
            grid_t, grid_h, grid_w = thw_grids[index]
            l = grid_t * grid_h * grid_w
            text = text.replace(image_token, "<|placeholder|>" * (l // merge_length), 1)
            user_inputs = user_inputs.replace(
                image_token, "<|placeholder|>" * (l // merge_length), 1
            )
            index += 1
        text = text.replace("<|placeholder|>", image_token)
        user_inputs = user_inputs.replace("<|placeholder|>", image_token)

        input_ids = self.tokenizer.tokenize(text)
        user_input_ids = self.tokenizer.tokenize(user_inputs)
        if len(input_ids) > self.seq_len:
            raise InternalWarning(f"Long sequence with length {len(input_ids)} found, dropped...")

        target = np.array([*input_ids[1:], IGNORE_IDX])
        if len(user_input_ids) >= len(input_ids):
            raise InternalWarning("Sample not supported, dropped...")
        if not (np.array(user_input_ids) == np.array(input_ids[: len(user_input_ids)])).all():
            raise InternalWarning("Sample not supported, dropped...")
        target[: len(user_input_ids) - 1] = IGNORE_IDX

        img_token_id = self.tokenizer.image_token_id
        image_input_mask = np.array(input_ids) == img_token_id

        return ImageTaskSample(
            __key__=sample.__key__,
            __subflavors__=sample.__subflavors__,
            imgs=flatten_patches,
            videos=list(),
            image_thw_grids=thw_grids,
            video_thw_grids=torch.empty([0, 3], dtype=torch.long),
            image_input_mask=image_input_mask,
            video_input_mask=None,
            second_per_grid_ts=np.zeros(0, dtype=np.float32),
            text=input_ids,
            target=target,
        )

    def batch(self, samples: list[ImageTaskSample]) -> VQATaskBatch:
        imgs = [s.imgs for s in samples if isinstance(s.imgs, np.ndarray) and s.imgs.size > 0]
        if len(imgs) > 0:
            imgs = torch.cat([torch.from_numpy(img) for img in imgs])
        else:
            imgs = torch.empty(
                [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size],
                dtype=torch.float32,
            )

        image_thw_grids = [thw_grids for s in samples for thw_grids in s.image_thw_grids]
        if len(image_thw_grids) > 0:
            image_thw_grids = torch.from_numpy(np.array(image_thw_grids)).long()
            assert image_thw_grids.prod(dim=-1).sum() == imgs.shape[0]
        else:
            image_thw_grids = torch.empty([0, 3], dtype=torch.long)

        videos = [
            s.videos for s in samples if isinstance(s.videos, np.ndarray) and s.videos.size > 0
        ]
        if len(videos) > 0:
            videos = torch.cat([torch.from_numpy(video) for video in videos])
        else:
            videos = torch.empty(
                [0, 3 * self.temporal_patch_size * self.patch_size * self.patch_size],
                dtype=torch.float32,
            )

        second_per_grid_ts = [
            second_per_grid for s in samples for second_per_grid in s.second_per_grid_ts
        ]
        if len(second_per_grid_ts) > 0:
            second_per_grid_ts = torch.from_numpy(np.array(second_per_grid_ts)).float()
        else:
            second_per_grid_ts = torch.empty([0], dtype=torch.float32)

        video_thw_grids = [thw_grids for s in samples for thw_grids in s.video_thw_grids]
        if len(video_thw_grids) > 0:
            video_thw_grids = torch.from_numpy(np.array(video_thw_grids)).long()
            assert video_thw_grids.prod(dim=-1).sum() == videos.shape[0]
        else:
            video_thw_grids = torch.empty([0, 3], dtype=torch.long)

        global FIRST_MAX_PADDING_FLAG, LAST_LARGE_IMG, MAX_IMG_THRESHOLD
        # MODIFIED START: Restore original cache clearing logic
        if self.args.curr_iteration > 0 and self.args.curr_iteration % CLEAR_CACHE_ITERATION == 0:
            FIRST_MAX_PADDING_FLAG = True

        if image_thw_grids.prod(axis=-1).sum() // 4 > MAX_IMG_THRESHOLD:
            MAX_IMG_THRESHOLD = image_thw_grids.prod(axis=-1).sum() // 4
            FIRST_MAX_PADDING_FLAG = True
            LAST_LARGE_IMG = True

        if not self.args.enable_variable_seq_lengths:
            max_seq_len = self.seq_len
        else:
            if FIRST_MAX_PADDING_FLAG:
                max_seq_len = self.seq_len
                FIRST_MAX_PADDING_FLAG = False
            else:
                max_seq_len = max(len(s.text) for s in samples)
                max_seq_len = min(max_seq_len, self.seq_len)
        if self.cp_size > 1 or self.sequence_parallel:
            max_seq_len = math.ceil(max_seq_len / (self.tp_size * self.cp_size)) * (
                self.tp_size * self.cp_size
            )
        text_mat = np.full((len(samples), max_seq_len), self.tokenizer.pad_token_id, dtype=np.int64)
        target_mat = np.full((len(samples), max_seq_len), IGNORE_IDX, dtype=np.int64)

        image_input_masks = np.zeros_like(text_mat, dtype=bool)
        video_input_masks = np.zeros_like(text_mat, dtype=bool)
        for i, s in enumerate(samples):
            text_len = min(max_seq_len, len(s.text))
            target_len = min(max_seq_len, len(s.target))
            text_mat[i, :text_len] = np.array(s.text)[:text_len]
            if s.image_input_mask is not None:
                image_input_masks[i, :text_len] = np.array(s.image_input_mask)[:text_len]
            if s.video_input_mask is not None:
                video_input_masks[i, :text_len] = np.array(s.video_input_mask)[:text_len]
            target_mat[i, :target_len] = np.array(s.target)[:target_len]

        batch = VQATaskBatch(
            __keys__=[s.__key__ for s in samples],
            __subflavors__=[s.__subflavors__ for s in samples],
            imgs=imgs,
            videos=videos,
            image_thw_grids=image_thw_grids,
            video_thw_grids=video_thw_grids,
            second_per_grid_ts=second_per_grid_ts,
            image_input_mask=torch.from_numpy(image_input_masks),
            video_input_mask=torch.from_numpy(video_input_masks),
            text=torch.from_numpy(text_mat),
            target=torch.from_numpy(target_mat),
        )
        return batch

    def encode_batch(self, batch: VQATaskBatch) -> dict:
        raw = dataclasses.asdict(batch)
        del raw["__subflavors__"]
        return raw


def print_error_handler(exc: Exception, key: str | None, debug=False):
    if not debug and isinstance(exc, InternalWarning):
        return
    logger.info(f"Exception occurred in the dataloader for sample {key} and is skipped")
    traceback.print_exc()
