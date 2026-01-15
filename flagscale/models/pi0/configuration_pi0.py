# Copied from https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/policies/pi0/configuration_pi0.py

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
from huggingface_hub.constants import CONFIG_NAME

# from lerobot.configs.policies import PreTrainedConfig
from flagscale.models.configs.types import FeatureType, NormalizationMode, PolicyFeature

# from lerobot.optim.optimizers import AdamWConfig
# from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
# from lerobot.policies.rtc.configuration_rtc import RTCConfig
from flagscale.models.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from flagscale.train.utils.hub import HubMixin

DEFAULT_IMAGE_SIZE = 224


# @PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(HubMixin):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # `use_amp` determines whether to use Automatic Mixed Precision (AMP) for training and evaluation. With AMP,
    # automatic gradient scaling is used.
    use_amp: bool = False

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    # rtc_config: RTCConfig | None = None
    # TODO(yupu): Impl
    rtc_config: None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    def __post_init__(self):
        # super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    # def get_optimizer_preset(self) -> AdamWConfig:
    #     return AdamWConfig(
    #         lr=self.optimizer_lr,
    #         betas=self.optimizer_betas,
    #         eps=self.optimizer_eps,
    #         weight_decay=self.optimizer_weight_decay,
    #         grad_clip_norm=self.optimizer_grad_clip_norm,
    #     )

    # def get_scheduler_preset(self):
    #     return CosineDecayWithWarmupSchedulerConfig(
    #         peak_lr=self.optimizer_lr,
    #         decay_lr=self.scheduler_decay_lr,
    #         num_warmup_steps=self.scheduler_warmup_steps,
    #         num_decay_steps=self.scheduler_decay_steps,
    #     )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @classmethod
    def from_pretrained(cls, config_dir: str, **kwargs: Any) -> "PI0Config":
        config_path = os.path.join(config_dir, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"config.json not found in {config_dir}")

        with open(config_path, "r") as f:
            config = json.load(f)

        _UNUSED_KEYS = ["type", "repo_id", "push_to_hub", "private", "tags", "license"]
        for key in _UNUSED_KEYS:
            config.pop(key, None)

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump(config, f)
            config_file = f.name

        cli_overrides = kwargs.pop("cli_overrides", [])
        with draccus.config_type("json"):
            return draccus.parse(cls, config_file, args=cli_overrides)

    # TODO(yupu): Copied from class PreTrainedConfig
    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.input_features.items():
            if ft.type is FeatureType.STATE and ft_name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for ft_name, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION and ft_name == ACTION:
                return ft
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)
