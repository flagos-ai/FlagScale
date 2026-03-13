# Adapted from https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/utils/train_utils.py

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
from typing import Any

from safetensors.torch import load_file, save_file
from torch.optim import Optimizer

from flagscale.models.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    PRETRAINED_MODEL_DIR,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from flagscale.train.datasets.utils import flatten_dict, load_json, unflatten_dict, write_json
from flagscale.train.utils.io_utils import deserialize_json_into_object
from flagscale.train.utils.random_utils import load_rng_state, save_rng_state


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


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    config,
    policy,
    optimizer_state_dict: dict | None = None,
    lr_scheduler=None,
    preprocessor=None,
    postprocessor=None,
    state_dict: dict | None = None,
) -> None:
    """Save a full training checkpoint.

    Creates the following directory structure:

    005000/
    ├── pretrained_model/
    │   ├── config.json                     # policy config
    │   ├── model.safetensors               # policy weights
    │   ├── vlm_config/                     # VLM artifacts (model-specific)
    │   ├── train_config.json               # training config
    │   ├── policy_preprocessor.json        # preprocessor config
    │   ├── policy_preprocessor_step_*.safetensors
    │   ├── policy_postprocessor.json       # postprocessor config
    │   └── policy_postprocessor_step_*.safetensors
    └── training_state/
        ├── training_step.json
        ├── optimizer_state.safetensors
        ├── optimizer_param_groups.json
        ├── scheduler_state.json
        └── rng_state.safetensors

    Args:
        checkpoint_dir: Root checkpoint directory (e.g., checkpoints/005000).
        step: Current training step.
        config: TrainConfig instance.
        policy: The policy model.
        optimizer_state_dict: Pre-gathered optimizer state dict (e.g. from
            get_optimizer_state_dict for FSDP2).
        lr_scheduler: Optional LR scheduler.
        preprocessor: Optional preprocessor pipeline.
        postprocessor: Optional postprocessor pipeline.
        state_dict: Pre-gathered model state dict (e.g. from FSDP2). If None, uses
            policy.state_dict().
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir, state_dict=state_dict)
    config._save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer_state_dict, lr_scheduler)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer_state_dict: dict | None = None,
    scheduler=None,
) -> None:
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer_state_dict is not None:
        save_optimizer_state(optimizer_state_dict, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path,
    optimizer: Optimizer,
    scheduler=None,
) -> tuple[int, Optimizer, Any]:
    """Load training state for resume.

    Returns:
        (step, optimizer, scheduler) with their state loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)
    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)
    return step, optimizer, scheduler


def save_optimizer_state(state_dict: dict, save_dir: Path) -> None:
    state_dict = dict(state_dict)
    param_groups = state_dict.pop("param_groups")
    flat_state = flatten_dict(state_dict)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def load_optimizer_state(optimizer: Optimizer, save_dir: Path) -> Optimizer:
    current_state_dict = optimizer.state_dict()
    flat_state = load_file(save_dir / OPTIMIZER_STATE)
    state = unflatten_dict(flat_state)

    if "state" in state:
        loaded_state_dict = {"state": {int(k): v for k, v in state["state"].items()}}
    else:
        loaded_state_dict = {"state": {}}

    if "param_groups" in current_state_dict:
        param_groups = deserialize_json_into_object(
            save_dir / OPTIMIZER_PARAM_GROUPS, current_state_dict["param_groups"]
        )
        loaded_state_dict["param_groups"] = param_groups

    optimizer.load_state_dict(loaded_state_dict)
    return optimizer


def save_scheduler_state(scheduler, save_dir: Path) -> None:
    write_json(scheduler.state_dict(), save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler, save_dir: Path):
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
