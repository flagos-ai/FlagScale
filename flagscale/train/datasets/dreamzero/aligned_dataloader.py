# Copyright (c) 2025, FlagScale Authors. All rights reserved.
"""
DreamZero Aligned Data Pipeline for FlagScale native training backend.

Uses reference (groot) ShardedLeRobotMixtureDataset with the FULL transform
chain to produce identical data to the reference training pipeline.

Transform chain:
  VideoToTensor -> VideoCrop(0.95) -> VideoResize(176,320) -> VideoColorJitter
  -> VideoToNumpy -> StateActionToTensor -> StateActionTransform(q99)
  -> ConcatTransform -> DreamTransform

Output per sample (after collation):
  - images: (B, 33, 352, 640, 3) uint8
  - action: (B, 96, 32) float32
  - action_mask: (B, 96, 32) bool
  - state: (B, 4, 64) float32
  - state_mask: (B, 4, 64) bool
  - text: (B, max_len) int64 (tokenized)
  - text_attention_mask: (B, max_len) int64
  - embodiment_id: (B,) int64
  - has_real_action: (B,) bool
"""

import logging
import os
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Ensure reference repo is importable
_REFERENCE_REPO = "/public-mixed/fengyupu/github/dreamzero"
if _REFERENCE_REPO not in sys.path:
    sys.path.insert(0, _REFERENCE_REPO)


def build_dataloader_aligned(
    data_path: str,
    tokenizer_path: str,
    batch_size: int = 1,
    num_workers: int = 1,
    num_frames: int = 25,
    action_horizon: int = 24,
    state_horizon: int = 1,
    action_dim: int = 32,
    max_state_dim: int = 64,
    embodiment_tag: str = "libero",
    embodiment_tag_mapping: dict[str, int] | None = None,
    deterministic: bool = False,
    image_size: tuple[int, int] = (176, 320),
    max_text_length: int = 512,
    num_views: int = 2,
    seed: int = 42,
    shard_sampling_rate: float = 0.1,
    max_chunk_size: int = 1,
) -> DataLoader:
    """Build DataLoader using the reference pipeline for data alignment.

    Uses groot's ShardedLeRobotMixtureDataset (IterableDataset) with the full
    transform chain. NO external sampler — dataset handles distributed splitting.
    """
    from groot.vla.data.dataset.lerobot_sharded import (
        ShardedLeRobotMixtureDataset,
        ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
    )
    from groot.vla.data.dataset.lerobot import ModalityConfig
    from groot.vla.data.transform.base import ComposedModalityTransform
    from groot.vla.data.transform.video import (
        VideoToTensor, VideoCrop, VideoResize, VideoColorJitter, VideoToNumpy,
    )
    from groot.vla.data.transform.state_action import (
        StateActionToTensor, StateActionTransform,
    )
    from groot.vla.data.transform.concat import ConcatTransform
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
        DefaultDataCollator, DreamTransform,
    )

    if embodiment_tag_mapping is None:
        embodiment_tag_mapping = {
            "agibot": 26, "oxe_droid": 17, "gr1_unified": 2,
            "mecka_hands": 27, "xdof": 30, "yam": 31,
            "dream": 100, "lapa": 101, "libero": 17,
        }

    # --- Modality keys ---
    video_keys = ["video.image", "video.wrist_image"]
    state_keys = ["state.joint_pos", "state.gripper_pos"]
    action_keys = ["action.joint_pos", "action.gripper_pos"]

    # --- Modality configs ---
    # NOTE: delta_indices must match the reference config (base_48_wan_fine_aug_relative.yaml)
    # The reference uses 25 video frames (0..24) and 24 action steps (0..23) in its delta_indices,
    # regardless of num_frames passed to the model. num_frames controls the model's sequence length,
    # but delta_indices controls which timesteps are loaded from the dataset.
    video_delta_indices = list(range(25))  # Reference hardcodes [0..24]
    action_delta_indices = list(range(action_horizon))  # [0..23]
    state_delta_indices = list(range(state_horizon))  # [0]
    
    modality_configs = {
        embodiment_tag: {
            "video": ModalityConfig(
                delta_indices=video_delta_indices,
                modality_keys=video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=state_delta_indices,
                modality_keys=state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=action_delta_indices,
                modality_keys=action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.task"],
            ),
        }
    }

    # --- Full transform pipeline (matching reference config exactly) ---
    h, w = image_size
    if deterministic:
        # Deterministic mode: disable random crop and color jitter for exact reproducibility.
        # Use center crop and no jitter so that the only source of randomness is
        # the shard/trajectory sampling (controlled by seed).
        # NOTE: VideoCrop.apply() uses self.training to choose RandomCrop vs CenterCrop.
        # The dataset's __init__ calls transforms.train() which resets training=True.
        # We subclass VideoCrop to lock it in eval mode (CenterCrop always).
        class DeterministicCrop(VideoCrop):
            """VideoCrop that always uses CenterCrop regardless of train/eval mode."""
            def train(self):
                pass  # ignore — stay in eval mode

            def apply(self, data):
                # Force eval path (CenterCrop)
                self.training = False
                return super().apply(data)

        crop = DeterministicCrop(apply_to=video_keys, scale=0.95)
        transform_list = [
            VideoToTensor(apply_to=video_keys),
            crop,
            VideoResize(apply_to=video_keys, height=h, width=w, interpolation="linear"),
            # No VideoColorJitter in deterministic mode
            VideoToNumpy(apply_to=video_keys),
        ]
    else:
        transform_list = [
            # Video transforms
            VideoToTensor(apply_to=video_keys),
            VideoCrop(apply_to=video_keys, scale=0.95, mode="random"),
            VideoResize(apply_to=video_keys, height=h, width=w, interpolation="linear"),
            VideoColorJitter(
                apply_to=video_keys,
                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08,
            ),
            VideoToNumpy(apply_to=video_keys),
        ]
    transform_list += [
        # State transforms
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(
            apply_to=state_keys,
            normalization_modes={
                "state.joint_pos": "q99",
                "state.gripper_pos": "q99",
            },
        ),
        # Action transforms
        StateActionToTensor(apply_to=action_keys),
        StateActionTransform(
            apply_to=action_keys,
            normalization_modes={
                "action.joint_pos": "q99",
                "action.gripper_pos": "q99",
            },
        ),
        # Consolidation: stack views, concat state/action dims
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=action_keys,
        ),
        # Model-specific final transform
        DreamTransform(
            num_views=num_views,
            action_horizon=action_horizon,
            state_horizon=state_horizon,
            max_action_dim=action_dim,
            max_state_dim=max_state_dim,
            embodiment_tag_mapping=embodiment_tag_mapping,
            training=True,
            default_instruction="",
            tokenizer_path=tokenizer_path,
        ),
    ]

    transforms = {
        embodiment_tag: ComposedModalityTransform(transforms=transform_list)
    }

    # --- Mixture spec ---
    mixture_spec = [{
        "dataset_path": {embodiment_tag: [data_path]},
        "dataset_weight": 1.0,
        "distribute_weights": True,
    }]

    # --- Instantiate dataset ---
    logger.info(f"Building ALIGNED ShardedLeRobotMixtureDataset from {data_path}")
    train_dataset = ShardedLeRobotMixtureDataset.from_mixture_spec(
        mixture_spec=mixture_spec,
        dataset_class=ShardedLeRobotSubLangSingleActionChunkDatasetDROID,
        all_modality_configs=modality_configs,
        all_transforms=transforms,
        metadata_versions={embodiment_tag: None},
        fps={embodiment_tag: None},
        dataset_kwargs={
            "video_backend": "decord",
            "use_global_metadata": False,
            "max_chunk_size": max_chunk_size,
            "relative_action": True,
            "relative_action_keys": ["joint_pos"],
            "relative_action_per_horizon": False,
        },
        mixture_kwargs={
            "training": True,
            "balance_dataset_weights": False,
            "seed": seed,
            "shard_sampling_rate": shard_sampling_rate,
        },
    )

    # --- Collator ---
    collator = DefaultDataCollator(
        tokenizer_path=tokenizer_path,
        max_length=max_text_length,
        num_views=num_views,
        embodiment_tag_mapping=embodiment_tag_mapping,
    )

    # --- DataLoader (no sampler for IterableDataset) ---
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    logger.info(
        f"Built ALIGNED DataLoader: IterableDataset, "
        f"batch_size={batch_size}, num_workers={num_workers}, seed={seed}"
    )
    return dataloader


def get_batch_aligned(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Move batch to device and cast dtypes for model consumption."""
    result = {}

    # Images stay uint8 (VAE handles conversion)
    if "images" in batch:
        result["images"] = batch["images"].to(device, non_blocking=True)

    # Actions/states to compute dtype
    for key in ["action", "state"]:
        if key in batch:
            result[key] = batch[key].to(device=device, dtype=compute_dtype, non_blocking=True)

    # Masks
    for key in ["action_mask", "state_mask"]:
        if key in batch:
            result[key] = batch[key].to(device, non_blocking=True)

    # Text tokens (int64)
    if "text" in batch:
        result["text"] = batch["text"].to(device, non_blocking=True)
    if "text_attention_mask" in batch:
        result["text_attention_mask"] = batch["text_attention_mask"].to(device, non_blocking=True)

    # Scalars
    if "has_real_action" in batch:
        result["has_real_action"] = batch["has_real_action"].to(device, non_blocking=True)
    if "embodiment_id" in batch:
        result["embodiment_id"] = batch["embodiment_id"].to(device, non_blocking=True)

    return result
