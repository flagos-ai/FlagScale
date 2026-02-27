"""Optimizer setup utilities.

Supports:
- Freezing parameters via regex patterns
- Per-module optimizer settings (lr, weight_decay, betas, etc.) via config
- LR scheduler via transformers.get_scheduler

Example config (YAML):
    model:
      optimizer:
        name: AdamW
        lr: 2.5e-5
        betas: [0.9, 0.95]
        eps: 1.0e-08
        weight_decay: 1.0e-08
        param_groups:
          vlm:
            lr: 1.0e-05
          action_model:
            lr: 1.0e-04
        scheduler:
          name: cosine_with_min_lr
          warmup_steps: 5000
          scheduler_kwargs:
            min_lr: 1.0e-06

      freeze:
        freeze_patterns:
          - "qwen_vl_interface\\..*"
        keep_patterns:
          - "qwen_vl_interface\\.model\\.visual\\.merger\\..*"
"""

import re
from collections import defaultdict
from collections.abc import Generator, Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from transformers import get_scheduler

from flagscale.logger import logger

if TYPE_CHECKING:
    from flagscale.train.train_config import (
        FreezeConfig,
        OptimizerConfig,
        SchedulerConfig,
        TrainConfig,
    )


class PatternMatcher:
    """Helper for matching parameter names against regex patterns with usage tracking."""

    def __init__(self, patterns: list[str]):
        self.patterns = patterns
        self.compiled = [re.compile(p) for p in patterns]
        self.match_counts = {p: 0 for p in patterns}

    def matches(self, name: str) -> bool:
        for i, pattern in enumerate(self.compiled):
            if pattern.search(name):
                self.match_counts[self.patterns[i]] += 1
                return True
        return False

    def get_unused_patterns(self) -> list[str]:
        return [p for p, count in self.match_counts.items() if count == 0]


def freeze_and_get_trainable_params(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    freeze_patterns: list[str] | None = None,
    keep_patterns: list[str] | None = None,
) -> Generator[torch.nn.Parameter, None, None]:
    """
    Freeze parameters matching patterns and yield only trainable parameters.

    Args:
        named_parameters: Output of model.named_parameters()
        freeze_patterns: Regex patterns for params to freeze
        keep_patterns: Regex patterns for params to keep trainable (overrides freeze_patterns)

    Yields:
        Only parameters that should be trained (for optimizer).
    """
    freeze_matcher = PatternMatcher(freeze_patterns or [])
    keep_matcher = PatternMatcher(keep_patterns or [])

    trainable_count, frozen_count = 0, 0
    previously_frozen_now_trainable = []

    for name, param in named_parameters:
        should_freeze = freeze_matcher.matches(name) and not keep_matcher.matches(name)

        if should_freeze:
            param.requires_grad = False
            frozen_count += param.numel()
        else:
            # Only force parameters to be trainable if freeze patterns are provided.
            # Otherwise, preserve the original requires_grad state.
            if freeze_patterns:
                if not param.requires_grad:
                    previously_frozen_now_trainable.append(name)
                param.requires_grad = True
            if param.requires_grad:
                trainable_count += param.numel()
                yield param
            else:
                frozen_count += param.numel()

    # Log summary
    total = trainable_count + frozen_count
    pct = trainable_count / total if total > 0 else 0
    logger.info(
        f"Parameters: trainable={trainable_count:,} ({pct:.2%}) | "
        f"frozen={frozen_count:,} | total={total:,}"
    )

    if previously_frozen_now_trainable:
        logger.warning(
            f"{len(previously_frozen_now_trainable)} parameter(s) were already frozen "
            f"(requires_grad=False) but don't match any freeze pattern and are being "
            f"made trainable. Add them to freeze_patterns if they should stay frozen:"
        )
        for name in previously_frozen_now_trainable:
            logger.warning(f"  unfrozen: {name}")

    # Warn about unused patterns
    unused_freeze = freeze_matcher.get_unused_patterns()
    if unused_freeze:
        logger.warning(f"Freeze patterns matched nothing: {unused_freeze}")

    unused_keep = keep_matcher.get_unused_patterns()
    if unused_keep:
        logger.warning(f"Keep patterns matched nothing: {unused_keep}")


def apply_freeze_config(model: nn.Module, freeze_config) -> list:
    """
    Apply freeze config and return list of trainable parameters for optimizer.

    Args:
        model: The model to freeze
        freeze_config: FreezeConfig with freeze_patterns and keep_patterns

    Returns:
        List of trainable parameters (pass directly to optimizer)
    """
    if freeze_config is None:
        return list(model.parameters())

    return list(
        freeze_and_get_trainable_params(
            model.named_parameters(),
            freeze_patterns=freeze_config.freeze_patterns,
            keep_patterns=freeze_config.keep_patterns,
        )
    )


def log_trainable_params(model: nn.Module) -> dict:
    """Log trainable/frozen parameter statistics by module."""
    trainable_by_module = defaultdict(int)
    frozen_by_module = defaultdict(int)

    for name, param in model.named_parameters():
        module_name = name.split(".")[0]
        if param.requires_grad:
            trainable_by_module[module_name] += param.numel()
        else:
            frozen_by_module[module_name] += param.numel()

    logger.info("=" * 60)
    logger.info("Parameter status by top-level module:")
    all_modules = set(trainable_by_module.keys()) | set(frozen_by_module.keys())
    for mod in sorted(all_modules):
        t = trainable_by_module.get(mod, 0)
        f = frozen_by_module.get(mod, 0)
        logger.info(f"  {mod}: {t:,} trainable, {f:,} frozen")
    logger.info("=" * 60)

    return {"trainable": dict(trainable_by_module), "frozen": dict(frozen_by_module)}


def print_param_names(model: nn.Module, pattern: str | None = None):
    """Debug helper: print parameter names (optionally filtered by pattern)."""
    for name, param in model.named_parameters():
        if pattern is None or re.search(pattern, name):
            status = "trainable" if param.requires_grad else "FROZEN"
            print(f"[{status}] {name}: {param.numel():,} params")


# TODO: (yupu) Freeze supports regex patterns, but param groups uses exact module paths. See if this is reasonable.
def build_optim_param_groups(
    model: nn.Module,
    optim_param_groups_config: dict[str, dict[str, Any]] | None = None,
) -> list[dict]:
    """
    Build optimizer param groups with per-module settings.

    Each module can have its own optimizer hyperparameters (lr, weight_decay, betas, etc.).
    Parameters not belonging to any specified module go into a default group.

    Args:
        model: The model to create param groups for.
        optim_param_groups_config: Dict mapping module names to optimizer kwargs.
            Example: {"encoder": {"lr": 1e-5}, "decoder": {"lr": 1e-3, "weight_decay": 0.01}}
            Supports nested paths like "action_head.mlp".

    Returns:
        List of param group dicts for optimizer.
    """
    if optim_param_groups_config is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]

    param_groups = []
    used_param_ids = set()

    for module_name, group_config in optim_param_groups_config.items():
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            logger.warning(
                f"build_optim_param_groups: Module '{module_name}' not found in model, skipping."
            )
            continue

        # All trainable params for this module (including descendants)
        module_params = [p for p in module.parameters() if p.requires_grad]
        if not module_params:
            logger.warning(
                f"build_optim_param_groups: Module '{module_name}' has no trainable parameters."
            )
            continue
        # Avoid assigning the same parameter to multiple param groups by
        # filtering out parameters that are already used by previous groups.
        params = [p for p in module_params if id(p) not in used_param_ids]
        if not params:
            # All trainable params for this module were already included in
            # previous param groups. This usually indicates overlapping
            # module paths in the optimizer config (e.g., both "encoder"
            # and "encoder.layer1").
            logger.warning(
                "build_optim_param_groups: All trainable parameters for module "
                f"'{module_name}' are already assigned to previous param groups. "
                "This suggests overlapping module paths in the optimizer "
                "configuration; this group will be skipped."
            )
            continue
        if len(params) < len(module_params):
            # Some, but not all, parameters were already assigned to previous
            # groups. Warn the user so they are aware of the partial overlap.
            logger.warning(
                "build_optim_param_groups: Some trainable parameters for module "
                f"'{module_name}' are already assigned to previous param groups "
                "(overlapping module paths). Only unassigned parameters will be "
                "included in this group."
            )

        used_param_ids.update(id(p) for p in params)
        param_groups.append({"params": params, "name": module_name, **group_config})

        param_count = sum(p.numel() for p in params)
        logger.info(f"Param group '{module_name}': {param_count:,} params, {group_config}")

    # Remaining params go to default group
    other_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in used_param_ids
    ]
    if other_params:
        param_groups.insert(0, {"params": other_params, "name": "default"})
        logger.info(f"Param group 'default': {sum(p.numel() for p in other_params):,} params")

    return param_groups


def setup_optimizer(
    model: nn.Module,
    optimizer_config: "OptimizerConfig",
    freeze_config: "FreezeConfig | None" = None,
) -> torch.optim.Optimizer:
    """
    Setup optimizer.

    Args:
        model: The model to optimize.
        optimizer_config: OptimizerConfig with name, lr, betas, eps, weight_decay, param_groups.
        freeze_config: FreezeConfig with freeze_patterns and keep_patterns.

    Returns:
        Configured optimizer instance.
    """
    if freeze_config is not None:
        apply_freeze_config(model, freeze_config)
        log_trainable_params(model)

    param_groups = build_optim_param_groups(model, optimizer_config.param_groups)
    total_params = sum(len(g["params"]) for g in param_groups)
    if not total_params:
        raise ValueError(
            "No trainable parameters found. All parameters may be frozen, "
            "or configured param groups have no trainable parameters."
        )

    optimizer_kwargs = {"params": param_groups, **optimizer_config.get_optimizer_kwargs()}

    # Get optimizer class by name
    optimizer_cls = _get_optimizer_class(optimizer_config.name)
    return optimizer_cls(**optimizer_kwargs)


# Supported optimizers
_OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "AdamW": torch.optim.AdamW,
}


def _get_optimizer_class(name: str) -> type[torch.optim.Optimizer]:
    """Get optimizer class by name."""
    if name not in _OPTIMIZER_REGISTRY:
        supported = list(_OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Unsupported optimizer: {name}. Supported: {supported}")
    return _OPTIMIZER_REGISTRY[name]


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: "SchedulerConfig",
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create LR scheduler using transformers' get_scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_config: Config with name, warmup_steps, scheduler_kwargs.
        num_training_steps: Total training steps.

    Returns:
        A learning rate scheduler instance.

    Raises:
        ValueError: If scheduler_config.name is None.
    """

    if scheduler_config.name is None:
        raise ValueError("scheduler_config.name must be specified to use setup_scheduler")

    return get_scheduler(
        name=scheduler_config.name,
        optimizer=optimizer,
        num_warmup_steps=scheduler_config.warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_config.scheduler_kwargs,
    )


def setup_optimizer_and_scheduler(
    model: nn.Module,
    train_config: "TrainConfig",
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    One-stop setup for both optimizer and scheduler from TrainConfig.

    Args:
        model: The model to optimize.
        train_config: TrainConfig containing model (optimizer, scheduler, freeze config)
            and system (train_steps).

    Returns:
        Tuple of (optimizer, lr_scheduler).

    Raises:
        ValueError: If scheduler_config.name is None.
    """
    optimizer = setup_optimizer(
        model,
        train_config.model.optimizer,
        freeze_config=train_config.model.freeze,
    )
    scheduler = setup_scheduler(
        optimizer,
        train_config.model.optimizer.scheduler,
        num_training_steps=train_config.system.train_steps,
    )
    return optimizer, scheduler
