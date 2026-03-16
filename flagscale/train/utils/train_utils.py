# Copied from https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/utils/train_utils.py

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import load_model, save_file

from flagscale.logger import logger
from flagscale.models.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STEP,
)
from flagscale.train.datasets.utils import load_json, write_json


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(checkpoint_dir: Path, policy) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   ├── train_config.json  # train config
    │   ├── processor.json  # processor config (if preprocessor provided)
    │   └── step_*.safetensors  # processor state files (if any)
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)


def save_vla_checkpoint(
    checkpoint_dir: Path,
    model_or_state_dict,
    config,
    preprocessor=None,
    postprocessor=None,
) -> None:
    """Save model weights, config, and preprocessor state.

    Creates the following directory structure:
        005000/
        └── pretrained_model/
            ├── train_config.yaml              # train config (OmegaConf)
            ├── model.safetensors              # All weights (VLM + action head)
            ├── policy_preprocessor.json       # Preprocessor pipeline config
            └── policy_preprocessor_step_*.safetensors  # Norm stats

    Args:
        checkpoint_dir: Directory to save checkpoint (e.g., checkpoints/005000)
        model_or_state_dict: nn.Module or a pre-gathered state_dict (e.g. from FSDP2)
        config: Training config (OmegaConf, Pydantic, or dict)
        preprocessor: Optional PolicyProcessorPipeline
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # Save train config as YAML
    # Handle OmegaConf, Pydantic, and dict configs
    if hasattr(config, "model_dump"):
        config = OmegaConf.create(config.model_dump())
    elif not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    OmegaConf.save(config, pretrained_dir / "train_config.yaml")

    # Clone tensors to avoid safetensors errors with non-contiguous views.
    if isinstance(model_or_state_dict, dict):
        state_dict = {k: v.clone().contiguous() for k, v in model_or_state_dict.items()}
    else:
        state_dict = {
            k: v.clone().contiguous() for k, v in model_or_state_dict.state_dict().items()
        }
    save_file(state_dict, pretrained_dir / "model.safetensors")

    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)


def load_checkpoint(
    checkpoint_dir: str | Path,
    model_cls,
    device: str = "cpu",
):
    """Load config, model weights, and preprocessor from checkpoint.

    Args:
        checkpoint_dir: Checkpoint directory (e.g., checkpoints/005000)
        model_cls: Model class.
        device: Device to load weights to

    Returns:
        If model_cls provided: tuple of (model, preprocessor)
        If model_cls is None: tuple of (config, state_dict, preprocessor)

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist
    """
    from flagscale.train.processor import PolicyProcessorPipeline

    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

    if not pretrained_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {pretrained_dir}")

    config_path = pretrained_dir / "train_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    # Set _pretrained_dir so OmegaConf resolves ${_pretrained_dir} interpolations
    # (e.g., model.qwenvl.base_vlm saved as "${_pretrained_dir}/vlm_config")
    OmegaConf.update(config, "_pretrained_dir", str(pretrained_dir))

    model = model_cls(config)

    # Materialize any meta tensors (from torch.device("meta") init) before loading weights.
    has_meta = any(p.is_meta for p in model.parameters())
    if has_meta:
        model.to_empty(device=device)

    weights_path = pretrained_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    # strict=False to handle tied weights saved as separate entries
    missing_keys, unexpected_keys = load_model(model, weights_path, device=device, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                logger.warning(f"  - {key}")
        else:
            for key in missing_keys[:10]:
                logger.warning(f"  - {key}")
            logger.warning(f"  ... and {len(missing_keys) - 10} more")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                logger.warning(f"  - {key}")
        else:
            for key in unexpected_keys[:10]:
                logger.warning(f"  - {key}")
            logger.warning(f"  ... and {len(unexpected_keys) - 10} more")

    model.to(device)

    preprocessor = None
    preprocessor_config_path = pretrained_dir / "policy_preprocessor.json"
    if preprocessor_config_path.exists():
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_preprocessor.json",
        )

    postprocessor = None
    postprocessor_config_path = pretrained_dir / "policy_postprocessor.json"
    if postprocessor_config_path.exists():
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_postprocessor.json",
        )

    return model, preprocessor, postprocessor
