import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    PretrainedConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from flagscale.train.train_config import TrainConfig
from flagscale.train.utils.image_tools import to_pil_preserve

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

_ACTION_TOKEN_MIN_QWEN25 = 151665
_ACTION_TOKEN_MAX_QWEN25 = 153712
_ACTION_TOKEN_MIN_QWEN3 = 151669
_ACTION_TOKEN_MAX_QWEN3 = 153716


class QwenVLBackbone(nn.Module):
    """
    Base class for Qwen VL backends.

    Args:
        config: TrainConfig object with config.model.qwenvl namespace.
    """

    def __init__(self, config: TrainConfig, **kwargs):
        super().__init__()
        qwenvl_config = config.model.qwenvl
        self.model_id = qwenvl_config.base_vlm

        # TODO: (yupu) The model loaded by `from_pretrained` is eval mode by default, is this expected? I removed `policy.train()` in train_qwen_gr00t.py to match starVLA, but not sure if this is the right way to do this.
        self.model = self._load_model(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # FIXME: Hard-coded padding side
        self.processor.tokenizer.padding_side = "left"
        self._config: TrainConfig = config

    def _load_model(self, model_id: str):
        raise NotImplementedError

    @property
    def model_config(self) -> PretrainedConfig:
        """HF config object (e.g., Qwen2VLConfig)."""
        return self.model.config

    def prepare_input(self, batch: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        # TODO: (yupu) We should output the original outputs, not just the hidden states.
        return {"hidden_states": outputs.hidden_states}


class Qwen25VLBackbone(QwenVLBackbone):
    """Qwen2.5-VL backend."""

    def __init__(self, config: TrainConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN_QWEN25
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX_QWEN25

    def _load_model(self, model_id: str):
        # WARNING: hard-coded attn_implementation and torch_dtype
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
        )

    def prepare_input(self, batch: dict) -> dict[str, torch.Tensor]:
        # TODO: (yupu) This is a hack, we should find a better way to handle this.
        image_keys = self._config.data.vla_data.image_features
        return self.build_qwenvl_inputs(examples=batch, image_keys=image_keys)

    def build_qwenvl_inputs(
        self,
        examples,
        images=None,
        instructions=None,
        image_keys=None,
        solutions=None,
        **kwargs,
    ):
        # TODO: (yupu) This is so ugly, we should find a better way to handle this.
        def _tensor_to_pil_list(batch_tensor):
            if not isinstance(batch_tensor, torch.Tensor):
                return batch_tensor
            if batch_tensor.ndim == 3:
                batch_tensor = batch_tensor.unsqueeze(0)
            if batch_tensor.ndim != 4:
                raise ValueError(f"Expected image tensor with 4 dims, got {batch_tensor.shape}")
            pil_images = []
            for item in batch_tensor:
                if item.shape[-1] in (1, 3, 4):
                    img = item
                else:
                    img = item.permute(1, 2, 0)
                pil_images.append(to_pil_preserve(img.detach().cpu().numpy()))
            return pil_images

        if examples is not None and (images is None or instructions is None):
            # TODO: (yupu) hard-code task key to "task"
            instructions = examples["task"]
            if isinstance(instructions, torch.Tensor):
                instructions = instructions.detach().cpu().tolist()
            if isinstance(instructions, str):
                instructions = [instructions]

            batch_images = None
            for key in image_keys:
                key_images = _tensor_to_pil_list(examples[key])
                if batch_images is None:
                    batch_images = [[img] for img in key_images]
                else:
                    for sample_images, img in zip(batch_images, key_images):
                        sample_images.append(img)

            for idx, sample_images in enumerate(batch_images):
                batch_images[idx] = [img for img in sample_images if img is not None]

            images = batch_images

        from qwen_vl_utils import process_vision_info

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions)
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self._config.data.vla_data:
                CoT_prompt = self._config.data.vla_data.get("CoT_prompt", "")
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

        # if solutions, mask out the non solution tokens in labels --> @JinhuiYE can we mask out system prompt?
        if solutions is not None:
            # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_input["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= self._ACTION_TOKEN_MIN) & (seq <= self._ACTION_TOKEN_MAX)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in your tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_input["labels"] = labels

        return batch_input.to(self.model.device)


class Qwen3VLBackbone(QwenVLBackbone):
    """Qwen3-VL backend."""

    def __init__(self, config: TrainConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Only for fast base model
        if "-Action" in self.model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN_QWEN3
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX_QWEN3

    def _load_model(self, model_id: str) -> Qwen3VLForConditionalGeneration:
        # FIXME: hard-coded attn_implementation and torch_dtype matches starVLA
        # TODO: (yupu): During inference/serving, it's required to load model twice, not only that, the original qwen model has to be loaded!
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        # Align dims qwen3 with qwen2.5, actually it's not needed in our case
        model.config.hidden_size = model.config.text_config.hidden_size
        return model

    def prepare_input(self, batch: dict) -> dict[str, torch.Tensor]:
        # TODO: (yupu) This is a hack, we should find a better way to handle this.
        # image_keys = self._config.data.vla_data.image_features.keys()
        image_keys = ["observation.images.image", "observation.images.wrist_image"]

        # Extract data in starVLA format (list of dicts)
        # examples = batch
        # batch_images = [example["image"] for example in examples]  # [B, [PIL]]
        # instructions = [example["lang"] for example in examples]  # [B, str]
        # actions = [example["action"] for example in examples]  # [B, T, action_dim]
        # state = None

        # return self.build_qwenvl_inputs(
        #     examples=None, images=batch_images, instructions=instructions
        # )

        return self.build_qwenvl_inputs(examples=batch, image_keys=image_keys)

    # TODO: (yupu) Refactor this args
    def build_qwenvl_inputs(
        self,
        examples,
        images=None,
        instructions=None,
        image_keys=None,
        solutions=None,
        **kwargs,
    ):
        # TODO: (yupu) This is so ugly, we should find a better way to handle this.
        def _tensor_to_pil_list(batch_tensor):
            if not isinstance(batch_tensor, torch.Tensor):
                return batch_tensor
            if batch_tensor.ndim == 3:
                batch_tensor = batch_tensor.unsqueeze(0)
            if batch_tensor.ndim != 4:
                raise ValueError(f"Expected image tensor with 4 dims, got {batch_tensor.shape}")
            pil_images = []
            for item in batch_tensor:
                if item.shape[-1] in (1, 3, 4):
                    img = item
                else:
                    img = item.permute(1, 2, 0)
                pil_images.append(to_pil_preserve(img.detach().cpu().numpy()))
            return pil_images

        if examples is not None and (images is None or instructions is None):
            # TODO: (yupu) hard-code task key to "task"
            instructions = examples["task"]
            if isinstance(instructions, torch.Tensor):
                instructions = instructions.detach().cpu().tolist()
            if isinstance(instructions, str):
                instructions = [instructions]

            batch_images = None
            for key in image_keys:
                key_images = _tensor_to_pil_list(examples[key])
                if batch_images is None:
                    batch_images = [[img] for img in key_images]
                else:
                    for sample_images, img in zip(batch_images, key_images):
                        sample_images.append(img)

            for idx, sample_images in enumerate(batch_images):
                batch_images[idx] = [img for img in sample_images if img is not None]

            images = batch_images

        # import numpy as np

        # torch.save(np.array([np.array(img) for img in images[0]]), "raw_images.pt")
        # assert False

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions)
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self._config.data.vla_data:
                CoT_prompt = self._config.data.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Preparation for inference
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # if solutions, mask out the solution tokens in labels
        # here only for fast_tokenizer now.
        if solutions is not None:
            # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs["input_ids"].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= self._ACTION_TOKEN_MIN) & (seq <= self._ACTION_TOKEN_MAX)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    # Mask out all tokens before the first action token.
                    seq[: nonzero_indices[0].item()] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning(
                        "action token are on in your tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md."
                    )

            # Mask out pad tokens as well
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)
