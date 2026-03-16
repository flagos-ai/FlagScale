# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/starVLA/model/framework/QwenGR00T.py
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

from pathlib import Path

import torch

from flagscale.models.configs.types import FeatureType, PolicyFeature
from flagscale.models.utils.constants import ACTION, OBS_STATE, VLM_CONFIG_DIR
from flagscale.models.vla.base_policy import TrainablePolicy
from flagscale.models.vla.registry import build_action_model, build_vlm
from flagscale.train.train_config import TrainConfig


class QwenGr00t(TrainablePolicy):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(self, config: TrainConfig, **kwargs):
        super().__init__()
        self._config = config

        vlm_type = config.model.vlm.get("type", "qwen3-vl")
        self.vlm = build_vlm(vlm_type, config=config)

        action_model_type = config.model.action_model.get("type", "flow_matching")
        self.action_model = build_action_model(
            action_model_type,
            vlm_config=self.vlm.model_config,
            action_config={},
            full_config=config,
        )

        self.future_action_window_size = config.model.action_model.future_action_window_size
        self.use_state = config.model.action_model.get("use_state", False)

        # Deserialize input/output features from config (checkpoint load path).
        # At training time, make_policy sets these on the policy after construction.
        load_pretrained = config.model.qwenvl.get("load_pretrained", True)
        raw_input = config.model.get("input_features", {})
        raw_output = config.model.get("output_features", {})

        if not load_pretrained and (not raw_input or not raw_output):
            raise ValueError(
                "Checkpoint config missing input_features/output_features. "
                "Re-save the checkpoint with the latest training code."
            )

        if raw_input:
            self.input_features = {
                k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                for k, v in raw_input.items()
            }
        if raw_output:
            self.output_features = {
                k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                for k, v in raw_output.items()
            }

    def forward(self, batch: list[dict] | dict, **kwargs) -> dict[str, torch.Tensor]:
        """ """
        if isinstance(batch, list):  # wds: list of per-sample dicts
            images = [ex["image"] for ex in batch]
            instructions = [ex["lang"] for ex in batch]
            actions = [ex["action"] for ex in batch]
            if self.use_state and "state" in batch[0]:
                state = [ex["state"] for ex in batch]
            else:
                state = None
        else:  # lerobot: single dict with batched tensors
            images, instructions = self.vlm.prepare_input(
                batch, image_feature_keys=list(self.image_features.keys())
            )
            actions = [batch[ACTION][i] for i in range(batch[ACTION].shape[0])]
            state = batch.get(OBS_STATE) if self.use_state else None

        qwen_inputs = self.vlm.build_qwenvl_inputs(images, instructions)

        # TODO: (yupu) Hard-coded autocast and dtype, matches starVLA
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]

        target_horizon = self._config.model.action_model.action_horizon
        target_dim = self._config.model.action_model.action_dim

        padded_actions = []
        action_masks = []
        for a in actions:
            a = a.float()
            T_orig = a.shape[0]

            # 1. Align action dimension (padding or truncating)
            if a.shape[-1] != target_dim:
                aligned = torch.zeros(T_orig, target_dim, dtype=a.dtype)
                copy_dim = min(a.shape[-1], target_dim)
                aligned[:, :copy_dim] = a[:, :copy_dim]
                a = aligned

            # 2. Time dimension padding
            final_a = torch.zeros(target_horizon, target_dim, dtype=a.dtype)
            mask = torch.zeros(target_horizon, dtype=torch.bool)
            copy_T = min(T_orig, target_horizon)
            final_a[:copy_T] = a[:copy_T]
            mask[:copy_T] = True

            padded_actions.append(final_a)
            action_masks.append(mask)

        with torch.autocast("cuda", dtype=torch.float32):
            # TODO: (yupu) Is this a bug or a feature? The action dtype would stay as bf16 under this autocast.
            actions = torch.stack(padded_actions).to(
                device=last_hidden.device, dtype=last_hidden.dtype
            )
            action_masks = torch.stack(action_masks).to(device=last_hidden.device)

            # TODO: (yupu) I believe there is a bug in starVLA, the
            # `repeated_diffusion_steps` is not properly set in the config.
            repeated_diffusion_steps = self._config.model.action_model.get(
                "repeated_diffusion_steps", 4
            )

            actions_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            action_masks_repeated = action_masks.repeat(repeated_diffusion_steps, 1)

            state_repeated = None
            if state is not None:
                if isinstance(state, list):
                    state = torch.stack(state)
                state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            vlm_output_repeated = {"hidden_states": last_hidden_repeated}
            action_input = {
                "actions": actions_repeated,
                "state": state_repeated,
                "mask": action_masks_repeated,
            }

            output = self.action_model.forward(vlm_output_repeated, action_input)

        return {"loss": output["loss"]}

    @torch.inference_mode()
    def predict_action(self, batch: list[dict] | dict, **kwargs) -> dict:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if isinstance(batch, list):  # wds: list of per-sample dicts
            images = [ex["image"] for ex in batch]
            instructions = [ex["lang"] for ex in batch]
            if self.use_state and "state" in batch[0]:
                state = torch.stack([ex["state"] for ex in batch])
            else:
                state = None
        else:  # lerobot: single dict with batched tensors
            images, instructions = self.vlm.prepare_input(
                batch, image_feature_keys=list(self.image_features.keys())
            )
            state = batch.get(OBS_STATE) if self.use_state else None

        qwen_inputs = self.vlm.build_qwenvl_inputs(images, instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_output = self.vlm.forward(qwen_inputs, output_attentions=False)
            # last_hidden_state: [B, seq_len, H]
            last_hidden = vlm_output["hidden_states"][-1]  # [B, L, H]

        if state is not None:
            state = state.to(device=last_hidden.device, dtype=last_hidden.dtype)

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            vlm_output_for_action = {"hidden_states": last_hidden}
            action_input = {"state": state}
            output = self.action_model.predict_action(vlm_output_for_action, action_input)

        # Assume the output of the action model is dict mapping `ACTION` to the normalized actions
        return output

    def checkpoint_config_overrides(self) -> dict:
        return {
            "model": {
                "qwenvl": {
                    "load_pretrained": False,
                    "base_vlm": "${_pretrained_dir}/" + VLM_CONFIG_DIR,
                },
                "input_features": {
                    k: {"type": ft.type.value, "shape": list(ft.shape)}
                    for k, ft in self.input_features.items()
                },
                "output_features": {
                    k: {"type": ft.type.value, "shape": list(ft.shape)}
                    for k, ft in self.output_features.items()
                },
            }
        }

    def save_pretrained_artifacts(self, save_dir: Path) -> None:
        vlm_config_dir = save_dir / VLM_CONFIG_DIR
        vlm_config_dir.mkdir(parents=True, exist_ok=True)
        self.vlm.model.config.save_pretrained(vlm_config_dir)
        self.vlm.processor.save_pretrained(vlm_config_dir)
