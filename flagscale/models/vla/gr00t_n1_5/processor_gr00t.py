# Copied from https://github.com/huggingface/lerobot/blob/0db5f66d/src/lerobot/policies/groot/processor_groot.py
# Below is the original copyright:

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from transformers import AutoProcessor, ProcessorMixin

from flagscale.models.utils.constants import (
    ACTION,
    HF_LEROBOT_HOME,
    OBS_IMAGE,
    OBS_IMAGES,
    OBS_STATE,
)
from flagscale.train.processor.core import EnvTransition, TransitionKey
from flagscale.train.processor.pipeline import ProcessorStep, ProcessorStepRegistry

DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"


def _to_uint8_np_bhwc(img: torch.Tensor | np.ndarray) -> np.ndarray:
    # img: (B, C, H, W) float in [0,1] or uint8  — torch or numpy
    if isinstance(img, np.ndarray):
        if img.ndim == 3:  # (H, W, C) single image from inference
            img = img[np.newaxis]  # -> (B, H, W, C)
        if img.ndim == 4 and img.shape[-1] in (1, 3):  # already (B, H, W, C)
            return img.astype(np.uint8)
        # (B, C, H, W) layout
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0, 1) * 255.0
        return rearrange(img.astype(np.uint8), "b c h w -> b h w c")
    # torch path
    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
    return rearrange(img.cpu().numpy(), "b c h w -> b h w c")


def _build_eagle_processor(
    tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
) -> ProcessorMixin:
    cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"[GROOT] Eagle processor cache at '{cache_dir}' is not populated. "
            "Vendor files are copied during model creation. Create the policy/model first, "
            "or call ensure_eagle_cache_ready() before building processors."
        )

    proc = AutoProcessor.from_pretrained(
        str(cache_dir),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    proc.tokenizer.padding_side = "left"
    return proc


# Original GR00T-style collate: converts eagle_content -> eagle_* tensors
def collate(features: list[dict[str, Any]], eagle_processor: ProcessorMixin) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list: list[str] = []
            image_inputs: list[Any] = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            eagle_inputs = eagle_processor(
                text=text_list,
                images=image_inputs,
                images_kwargs={
                    "min_dynamic_tiles": 1,
                    "max_dynamic_tiles": 1,
                    "use_thumbnail": False,
                },
                return_tensors="pt",
                padding=True,
            )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


@dataclass
@ProcessorStepRegistry.register(name="groot_pack_inputs")
class GrootPackInputsStep(ProcessorStep):
    """Pack video/state/action/language/embodiment and apply optional min-max normalization before padding."""

    state_horizon: int = 1
    action_horizon: int = 16
    max_state_dim: int = 64
    max_action_dim: int = 32
    language_key: str = "task"
    formalize_language: bool = False
    embodiment_tag: str = "new_embodiment"
    embodiment_mapping: dict[str, int] = field(
        default_factory=lambda: {
            "new_embodiment": 31,  # Match original GR00T EMBODIMENT_TAG_MAPPING
            "oxe_droid": 17,
            "agibot_genie1": 26,
            "gr1": 24,
            "so100": 2,
            "unitree_g1": 3,
        }
    )
    # Min-max normalization (SO100-like) applied BEFORE padding
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        def _align_vec(vec: Any, target_dim: int, *, default: float) -> torch.Tensor:
            t = torch.as_tensor(vec)
            t = t.flatten().to(
                dtype=torch.float32,
                device=next(
                    (v.device for v in obs.values() if isinstance(v, torch.Tensor)),
                    torch.device("cpu"),
                ),
            )
            d = int(t.shape[-1]) if t.numel() > 0 else 0
            if d == target_dim:
                return t
            if d < target_dim:
                pad = torch.full((target_dim - d,), default, dtype=t.dtype, device=t.device)
                return torch.cat([t, pad], dim=0)
            return t[:target_dim]

        def _min_max_norm(x: torch.Tensor, key: str) -> torch.Tensor:
            if not self.normalize_min_max:
                return x
            if self.stats is None or key not in self.stats:
                return x
            stats_k = self.stats[key]
            last_dim = x.shape[-1]
            min_v = _align_vec(stats_k.get("min", torch.zeros(last_dim)), last_dim, default=0.0)
            max_v = _align_vec(stats_k.get("max", torch.ones(last_dim)), last_dim, default=1.0)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            mapped = 2 * (x - min_v) / safe_denom - 1
            return torch.where(mask, mapped, torch.zeros_like(mapped))

        # 1) Video (B, T=1, V, H, W, C) uint8
        img_keys = sorted(
            [k for k in obs if k.startswith(OBS_IMAGES) and not k.endswith("_is_pad")]
        )
        if not img_keys and OBS_IMAGE in obs:
            img_keys = [OBS_IMAGE]
        if img_keys:
            cams = [_to_uint8_np_bhwc(obs[k]) for k in img_keys]
            video = np.stack(cams, axis=1)  # (B, V, H, W, C)
            video = np.expand_dims(video, axis=1)  # (B, 1, V, H, W, C)
            # GR00T validates that video.shape[3] == 3 (channels), so reorder to (B, T, V, C, H, W)
            video = np.transpose(video, (0, 1, 2, 5, 3, 4))  # (B, 1, V, C, H, W)
            obs["video"] = video
            # Drop raw images to avoid confusion downstream
            for k in img_keys:
                obs.pop(k, None)

        # 2) Language (string)
        lang = comp.get(self.language_key)
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else None
        if not lang:
            lang = "Perform the task."
        if self.formalize_language:
            lang = (lang or "").lower()
            lang = "".join(ch for ch in lang if ch.isalnum() or ch.isspace())
        comp["language"] = lang

        # 3) State/state_mask -> (B, 1, max_state_dim)
        if OBS_STATE in obs:
            state = obs[OBS_STATE]  # (B, D) or numpy (D,)/(B, D) from inference
            if isinstance(state, np.ndarray):
                state = torch.as_tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)  # (D,) -> (1, D)
            if state.dim() != 2:
                raise ValueError(f"state must be (B, D), got {tuple(state.shape)}")
            bsz, d = state.shape
            # Normalize BEFORE padding
            if self.normalize_min_max:
                state = _min_max_norm(state, OBS_STATE)
            state = state.unsqueeze(1)  # (B, 1, D)
            if d > self.max_state_dim:
                state = state[:, :, : self.max_state_dim]
                d = self.max_state_dim
            elif d < self.max_state_dim:
                pad = torch.zeros(
                    bsz, 1, self.max_state_dim - d, dtype=state.dtype, device=state.device
                )
                state = torch.cat([state, pad], dim=2)
            state_mask = torch.zeros(
                bsz, 1, self.max_state_dim, dtype=torch.bool, device=state.device
            )
            state_mask[:, :, :d] = True
            obs["state"] = state
            obs["state_mask"] = state_mask

        # 4) Action/action_mask -> (B, action_horizon, max_action_dim)
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            # Normalize BEFORE temporal expansion/padding
            if self.normalize_min_max:
                if action.dim() == 2:
                    action = _min_max_norm(action, ACTION)
                elif action.dim() == 3:
                    b, t, d = action.shape
                    flat = action.reshape(b * t, d)
                    flat = _min_max_norm(flat, ACTION)
                    action = flat.view(b, t, d)
            if action.dim() == 2:
                action = action.unsqueeze(1).repeat(1, self.action_horizon, 1)
            elif action.dim() == 3:
                b, t, d = action.shape
                if t < self.action_horizon:
                    last = action[:, -1:, :]
                    pad = last.repeat(1, self.action_horizon - t, 1)
                    action = torch.cat([action, pad], dim=1)
                elif t > self.action_horizon:
                    action = action[:, : self.action_horizon, :]
            else:
                raise ValueError(f"action must be (B, D) or (B, T, D), got {tuple(action.shape)}")

            b, t, d = action.shape
            if d > self.max_action_dim:
                action = action[:, :, : self.max_action_dim]
                d = self.max_action_dim
            elif d < self.max_action_dim:
                pad = torch.zeros(
                    b, t, self.max_action_dim - d, dtype=action.dtype, device=action.device
                )
                action = torch.cat([action, pad], dim=2)
            action_mask = torch.zeros(
                b, t, self.max_action_dim, dtype=torch.bool, device=action.device
            )
            action_mask[:, :, :d] = True
            transition[TransitionKey.ACTION] = action
            comp["action_mask"] = action_mask

        # 5) Embodiment id as LongTensor (B,)
        emb_id = self.embodiment_mapping.get(self.embodiment_tag, 0)
        # Infer batch size/device from any tensor in obs or action
        bsz = None
        device = torch.device("cpu")
        for v in [*obs.values(), transition.get(TransitionKey.ACTION)]:
            if isinstance(v, torch.Tensor):
                bsz = v.shape[0]
                device = v.device
                break
        if bsz is None and "video" in obs and isinstance(obs["video"], np.ndarray):
            bsz = obs["video"].shape[0]
        if bsz is None:
            bsz = 1
        comp["embodiment_id"] = torch.full((bsz,), emb_id, dtype=torch.long, device=device)

        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    # Pipeline API requirement: declare how features change (we keep it simple)
    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
        return {
            "state_horizon": self.state_horizon,
            "action_horizon": self.action_horizon,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "language_key": self.language_key,
            "formalize_language": self.formalize_language,
            "embodiment_tag": self.embodiment_tag,
            "embodiment_mapping": self.embodiment_mapping,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed


@dataclass
@ProcessorStepRegistry.register(name="groot_eagle_encode")
class GrootEagleEncodeStep(ProcessorStep):
    """Encode video+language with Eagle VLM into intermediate eagle_content."""

    tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_eagle_processor(self.tokenizer_assets_repo)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}

        if "video" not in obs:
            return transition

        video = obs["video"]  # (B, T, V, H, W, C) uint8
        lang = comp.get("language", "Perform the task.")
        if isinstance(lang, list):
            lang = lang[0] if len(lang) > 0 else "Perform the task."

        bsz = video.shape[0]
        eagle_contents: list[dict[str, Any]] = []
        for b in range(bsz):
            vt = video[b]  # (T, V, C, H, W) after reorder
            if vt.ndim != 5:
                # Fallback: assume (T, V, H, W, C)
                t, v, h, w, c = vt.shape
                flat = rearrange(vt, "t v h w c -> (t v) h w c")
            else:
                t, v, c, h, w = vt.shape
                flat = rearrange(vt, "t v c h w -> (t v) h w c")
            images = [Image.fromarray(flat[i]) for i in range(t * v)]
            # Format language as string list representation to match Original GROOT
            lang_formatted = str([lang])
            text_content = [{"type": "text", "text": lang_formatted}]
            image_content = [{"type": "image", "image": img} for img in images]
            conv = [{"role": "user", "content": image_content + text_content}]
            text_list = [
                self.proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            ]
            img_inputs, vid_inputs = self.proc.process_vision_info(conv)
            eagle_contents.append(
                {
                    "text_list": text_list,
                    "image_inputs": img_inputs,
                    "video_inputs": vid_inputs,
                }
            )

        comp["eagle_content"] = eagle_contents
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    # Pipeline API requirement: declare how features change (no schema change here)
    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_eagle_collate")
class GrootEagleCollateStep(ProcessorStep):
    """Collate eagle_content into batched eagle_* tensors."""

    tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO
    _proc: ProcessorMixin | None = field(default=None, init=False, repr=False)

    @property
    def proc(self) -> ProcessorMixin:
        if self._proc is None:
            self._proc = _build_eagle_processor(self.tokenizer_assets_repo)
        return self._proc

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        comp = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
        contents = comp.get("eagle_content")
        if not contents:
            return transition

        # Build features list as original API expects: one dict per batch item
        features = [{"eagle_content": content} for content in contents]
        batched = collate(features, self.proc)

        # Inject eagle_* tensors and remove the temporary content and raw video to free memory
        for k, v in batched.items():
            comp[k] = v
        comp.pop("eagle_content", None)
        obs.pop(
            "video", None
        )  # The video has been fully encoded into eagle_* tensors, so we don't need the raw video anymore
        transition[TransitionKey.OBSERVATION] = obs
        transition[TransitionKey.COMPLEMENTARY_DATA] = comp
        return transition

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register(name="groot_action_unpack_unnormalize")
class GrootActionUnpackUnnormalizeStep(ProcessorStep):
    """Slice to env action dim and unnormalize to env scale (inverse of GrootPackInputsStep normalization)."""

    env_action_dim: int = 0
    # Apply inverse of min-max normalization if it was used in preprocessor
    normalize_min_max: bool = True
    stats: dict[str, dict[str, Any]] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        # Expect model outputs to be in TransitionKey.ACTION as (B, T, D_model)
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return transition

        # Slice to env action dimension (keep all timesteps for action chunking)
        if self.env_action_dim and action.shape[-1] >= self.env_action_dim:
            action = action[..., : self.env_action_dim]

        # Inverse min-max normalization mirroring _min_max_norm:
        # forward: y = 2 * (x - min) / denom - 1, with y=0 when denom==0
        # inverse: x = (y+1)/2 * denom + min, and when denom==0 -> x = min
        if self.normalize_min_max and self.stats is not None:
            stats_k = self.stats.get(ACTION, {})
            d = action.shape[-1]
            min_v = torch.as_tensor(
                stats_k.get("min", torch.zeros(d)), dtype=action.dtype, device=action.device
            )
            max_v = torch.as_tensor(
                stats_k.get("max", torch.ones(d)), dtype=action.dtype, device=action.device
            )
            if min_v.numel() != d:
                min_v = torch.nn.functional.pad(min_v.flatten()[:d], (0, max(0, d - min_v.numel())))
                min_v = min_v.to(action.device, dtype=action.dtype)
            if max_v.numel() != d:
                max_v = torch.nn.functional.pad(max_v.flatten()[:d], (0, max(0, d - max_v.numel())))
                max_v = max_v.to(action.device, dtype=action.dtype)
            denom = max_v - min_v
            mask = denom != 0
            safe_denom = torch.where(mask, denom, torch.ones_like(denom))
            inv = (action + 1.0) * 0.5 * safe_denom + min_v
            action = torch.where(mask, inv, min_v)

        transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(self, features):
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Excludes 'stats' since they are saved separately via state_dict().
        """
        return {
            "env_action_dim": self.env_action_dim,
            "normalize_min_max": self.normalize_min_max,
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Returns normalization statistics as a flat state dictionary.

        This enables saving stats to safetensors files, similar to normalizer_processor.
        """
        if not self.stats:
            return {}

        flat: dict[str, torch.Tensor] = {}
        for key, sub in self.stats.items():
            for stat_name, value in sub.items():
                tensor = torch.as_tensor(value).cpu()
                flat[f"{key}.{stat_name}"] = tensor
        return flat

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """
        Loads normalization statistics from a flat state dictionary.

        This enables loading stats from safetensors files during from_pretrained.
        """
        if not state:
            return

        reconstructed: dict[str, dict[str, Any]] = {}
        for flat_key, tensor in state.items():
            if "." in flat_key:
                key, stat_name = flat_key.rsplit(".", 1)
                if key not in reconstructed:
                    reconstructed[key] = {}
                reconstructed[key][stat_name] = tensor

        if reconstructed:
            self.stats = reconstructed
