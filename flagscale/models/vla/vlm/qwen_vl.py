# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/starVLA/model/modules/vlm/QWen3.py

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

from flagscale.models.vla.registry import register_vlm
from flagscale.train.train_config import TrainConfig


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
        config: TrainConfig object with config.model.qwenvl namespace.
    """

    def __init__(self, config: TrainConfig, **kwargs):
        super().__init__()
        qwenvl_config = config.model.qwenvl
        self.model_id = qwenvl_config.base_vlm
        # When loading from checkpoint, base_vlm is resolved via OmegaConf
        # interpolation: "${_pretrained_dir}/vlm_config" → absolute path.
        self._load_pretrained = qwenvl_config.get("load_pretrained", True)
        self._attn_implementation = qwenvl_config.get("attn_implementation", None)

        if not self._load_pretrained and not Path(self.model_id).exists():
            raise FileNotFoundError(
                f"VLM config directory not found: {self.model_id}. "
                "Ensure the checkpoint was saved with save_pretrained_artifacts."
            )

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

    def prepare_input(
        self, batch: dict, image_feature_keys: list[str]
    ) -> tuple[list[list[Image.Image]], list[str]]:
        # TODO: (yupu) hard-code task key to "task"
        instructions = batch["task"]
        if isinstance(instructions, torch.Tensor):
            instructions = instructions.detach().cpu().tolist()
        if isinstance(instructions, str):
            instructions = [instructions]

        batch_images: list[list[Image.Image]] | None = None
        for key in image_feature_keys:
            imgs = batch[key]
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

            if "CoT_prompt" in self._config.data.vla_data:
                cot_prompt = self._config.data.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])
        return messages

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
            with torch.device("meta"):
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

        # Use current CUDA device instead of self.model.device, which returns
        # a DTensor device under FSDP2 and causes mixed Tensor/DTensor errors.
        return batch_input.to(f"cuda:{torch.cuda.current_device()}")


@register_vlm("qwen3-vl")
class Qwen3VLBackbone(QwenVLBackbone):
    """Qwen3-VL backend."""

    def _load_model(self, model_id: str) -> Qwen3VLForConditionalGeneration:
        attn_impl = self._attn_implementation or "flash_attention_2"
        if not self._load_pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_id, attn_implementation=attn_impl, torch_dtype=torch.bfloat16
            )
            with torch.device("meta"):
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

        # Use current CUDA device instead of self.model.device, which returns
        # a DTensor device under FSDP2 and causes mixed Tensor/DTensor errors.
        return batch_inputs.to(f"cuda:{torch.cuda.current_device()}")
