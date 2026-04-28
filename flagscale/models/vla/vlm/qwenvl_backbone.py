# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/starVLA/model/modules/vlm/QWen3.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    PretrainedConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from flagscale.logger import logger
from flagscale.models.vla.registry import register_vlm
from flagscale.platforms.platform_manager import get_platform


@dataclass
class QwenVLConfig:
    type: str = "qwen3-vl"
    base_vlm: str = ""
    load_pretrained: bool = True
    attn_implementation: str | None = None


def _to_pil(img):
    """Convert a single image (tensor, numpy, or PIL) to PIL.Image."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            return Image.fromarray(img)
        # float [0,1] → uint8
        return Image.fromarray((img * 255).clip(0, 255).astype(np.uint8))
    return img


class QwenVLBackbone(nn.Module):
    """
    Base class for Qwen VL backends.

    Args:
        vlm_config: QwenVLConfig with base_vlm, load_pretrained, attn_implementation.
        prompt_template: Optional prompt template with {instruction} placeholder.
    """

    def __init__(self, vlm_config: QwenVLConfig, prompt_template: str | None = None, **kwargs):
        super().__init__()
        self.model_id = vlm_config.base_vlm
        self._load_pretrained = vlm_config.load_pretrained
        self._attn_implementation = vlm_config.attn_implementation

        if not self._load_pretrained and not Path(self.model_id).exists():
            raise FileNotFoundError(
                f"VLM config directory not found: {self.model_id}. "
                "Ensure the checkpoint was saved with save_pretrained."
            )

        # TODO: (yupu) The model loaded by `from_pretrained` is eval mode by default, is this expected? I removed `policy.train()` in train_qwen_gr00t.py to match starVLA, but not sure if this is the right way to do this.
        self.model = self._load_model(self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        # FIXME: Hard-coded padding side
        self.processor.tokenizer.padding_side = "left"
        self._prompt_template = prompt_template

    def _load_model(self, model_id: str):
        raise NotImplementedError

    @property
    def model_config(self) -> PretrainedConfig:
        """HF config object (e.g., Qwen2VLConfig)."""
        return self.model.config

    def prepare_input(
        self, batch: dict, image_feature_keys: list[str]
    ) -> tuple[list[list[Image.Image]], list[str]]:
        # TODO: (yupu) hard-code task key to "task"
        instructions = batch["task"]
        if isinstance(instructions, torch.Tensor):
            instructions = instructions.detach().cpu().tolist()
        if isinstance(instructions, str):
            instructions = [instructions]

        logger.info(f"[prepare_input] image_feature_keys={image_feature_keys}")
        batch_images: list[list[Image.Image]] | None = None
        for key in image_feature_keys:
            imgs = batch[key]
            if isinstance(imgs, torch.Tensor):
                logger.info(
                    f"[prepare_input] key={key} tensor shape={imgs.shape} dtype={imgs.dtype}"
                )
            if isinstance(imgs, torch.Tensor) and imgs.ndim == 3:
                imgs = [imgs]
            key_images = [_to_pil(img) for img in imgs]
            if batch_images is None:
                batch_images = [[img] for img in key_images]
            else:
                for sample_images, img in zip(batch_images, key_images):
                    sample_images.append(img)

        for idx, sample_images in enumerate(batch_images):
            batch_images[idx] = [img for img in sample_images if img is not None]

        logger.info(
            f"[prepare_input] batch_size={len(batch_images)} images_per_sample={[len(s) for s in batch_images]} pil_size={batch_images[0][0].size if batch_images else None}"
        )
        return batch_images, instructions

    def build_qwenvl_inputs(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _build_messages(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> list[list[dict]]:
        messages = []
        assert len(images) == len(instructions)
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if self._prompt_template is not None:
                prompt = self._prompt_template.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        return messages

    def forward(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        logger.info(
            f"[VLM.forward] input keys={list(batch.keys())} "
            + " ".join(f"{k}={v.shape}" for k, v in batch.items() if isinstance(v, torch.Tensor))
        )
        with torch.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16):
            outputs = self.model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        logger.info(
            f"[VLM.forward] hidden_states: {len(outputs.hidden_states)} layers, last={outputs.hidden_states[-1].shape}"
        )
        # TODO: (yupu) We should output the original outputs, not just the hidden states.
        return {"hidden_states": outputs.hidden_states}

    def fsdp_units(self) -> list[nn.Module]:
        return list(self.model.model.visual.blocks) + list(self.model.model.language_model.layers)


@register_vlm("qwen2.5-vl")
class Qwen25VLBackbone(QwenVLBackbone):
    """Qwen2.5-VL backend."""

    def _load_model(self, model_id: str):
        attn_impl = self._attn_implementation or "flash_attention_2"
        if not self._load_pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_id, attn_implementation=attn_impl, torch_dtype="auto"
            )
            return Qwen2_5_VLForConditionalGeneration(hf_config)
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            torch_dtype="auto",
        )

    def build_qwenvl_inputs(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(images, instructions)

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

        logger.info(
            "[Qwen25.build_qwenvl_inputs] "
            + " ".join(
                f"{k}={v.shape}" for k, v in batch_input.items() if isinstance(v, torch.Tensor)
            )
        )

        # Use current CUDA device instead of self.model.device, which returns
        # a DTensor device under FSDP2 and causes mixed Tensor/DTensor errors.
        return batch_input.to(get_platform().device())


@register_vlm("qwen3-vl")
class Qwen3VLBackbone(QwenVLBackbone):
    """Qwen3-VL backend."""

    def _load_model(self, model_id: str) -> Qwen3VLForConditionalGeneration:
        attn_impl = self._attn_implementation or "flash_attention_2"
        if not self._load_pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_id, attn_implementation=attn_impl, torch_dtype=torch.bfloat16
            )
            model = Qwen3VLForConditionalGeneration(hf_config)
        else:
            # FIXME: hard-coded torch_dtype matches starVLA
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
            )

        return model

    def build_qwenvl_inputs(
        self, images: list[list[Image.Image]], instructions: list[str]
    ) -> dict[str, torch.Tensor]:
        messages = self._build_messages(images, instructions)

        # Preparation for inference
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        logger.info(
            "[Qwen3.build_qwenvl_inputs] "
            + " ".join(
                f"{k}={v.shape}" for k, v in batch_inputs.items() if isinstance(v, torch.Tensor)
            )
        )

        # Use current CUDA device instead of self.model.device, which returns
        # a DTensor device under FSDP2 and causes mixed Tensor/DTensor errors.
        return batch_inputs.to(get_platform().device())
