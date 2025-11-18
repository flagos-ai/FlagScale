# Mainly adopted from starVLA/starVLA:
# https://github.com/starVLA/starVLA/blob/starVLA/starVLA/model/framework/QwenGR00T.py
# Below is the original copyright:

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import os

from typing import List, Optional, Tuple

import numpy as np
import omegaconf
import torch

from omegaconf import OmegaConf
from PIL import Image
from transformers import PretrainedConfig, PreTrainedModel

from flagscale.logger import logger

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from flagscale.models.robotics.groot_action_header import FlowmatchingActionHead
from flagscale.models.robotics.qwen2_5 import _QWen_VL_Interface


class Qwen_GR00T(PreTrainedModel):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__(PretrainedConfig())
        self.config = config
        self.qwen_vl_interface = _QWen_VL_Interface(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model = FlowmatchingActionHead(full_config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # [B, T_full, action_dim]
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions[
                :, -(self.future_action_window_size + 1) :, :
            ]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def save_pretrained(self):
        action_model_path = os.path.join(self.config.output_directory + "/action_model.pt")
        backbone_path = os.path.join(self.config.output_directory + "/backbone")
        config_path = os.path.join(self.config.output_directory + "/config.yaml")
        if not os.path.exists(backbone_path):
            os.makedirs(self.config.output_directory, exist_ok=True)

        torch.save(self.action_model.state_dict(), action_model_path)
        self.qwen_vl_interface.model.save_pretrained(backbone_path)
        self.qwen_vl_interface.processor.save_pretrained(backbone_path)
        OmegaConf.save(self.config, config_path)

    @classmethod
    def from_pretrained(cls, pretrained_checkpoint: str, custom_config=None):
        config_path = os.path.join(pretrained_checkpoint + "/config.yaml")
        cfg = OmegaConf.load(config_path)
        if custom_config:
            if isinstance(custom_config, str):
                cfg_custom = OmegaConf.load(custom_config)
            elif isinstance(custom_config, omegaconf.dictconfig.DictConfig):
                cfg_custom = custom_config
            else:
                raise ValueError("custom_config must be str or dict")
            cfg = OmegaConf.merge(cfg, cfg_custom)
        model: Qwen_GR00T = Qwen_GR00T(cfg)

        action_model_path = os.path.join(pretrained_checkpoint + "/action_model.pt")
        action_model_state_dict = torch.load(action_model_path, map_location="cpu")
        logger.info(f"Loading action model weights from `{action_model_path}`")

        action_model_keys = set(model.action_model.state_dict().keys())
        checkpoint_keys = set(action_model_state_dict.keys())
        try:
            model.action_model.load_state_dict(action_model_state_dict, strict=True)
        except RuntimeError as e:
            # must keep all keys matched
            common_keys = action_model_keys.intersection(checkpoint_keys)
            missing_keys = action_model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
            raise e
        return model


def get_batch(batch):
    rsp_batch = []
    for i_batch in batch:
        ab = {
            "action": i_batch['action'][:16, :7],
            "image": [i_batch['observation.images.camera0'], i_batch['observation.images.camera1']],
            "lang": i_batch['task'],
            "state": i_batch['observation.state'][:7][None,],
        }
        rsp_batch.append(ab)
    return rsp_batch


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")
