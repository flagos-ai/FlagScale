# Mainly adopted from
# https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/scripts/lerobot_train.py

import argparse
import json
from pathlib import Path
from typing import Any, Iterator, TypedDict
import wandb
import os
import pathlib
import random
from dataclasses import dataclass
from typing_extensions import Unpack
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from flagscale.runner.utils import logger
from flagscale.train.datasets.transforms import ImageTransforms
from flagscale.train.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from flagscale.train.datasets.utils import dataset_to_policy_features
from flagscale.train.processor import PolicyAction, PolicyProcessorPipeline
from flagscale.train.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from flagscale.models.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from flagscale.models.configs.types import PolicyFeature
from flagscale.models.utils.constants import ACTION, OBS_PREFIX, REWARD
from flagscale.models.configs.types import FeatureType
from flagscale.models.pi0.configuration_pi0 import PI0Config
from flagscale.models.pi0.modeling_pi0 import PI0Policy
from flagscale.models.pi05.configuration_pi05 import PI05Config
from flagscale.models.pi05.modeling_pi05 import PI05Policy
from flagscale.train.utils.logging_utils import AverageMeter, MetricsTracker
from flagscale.train.utils.train_utils import (
    save_checkpoint,
    get_step_checkpoint_dir,
    update_last_checkpoint,
)

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True


def init_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return local_rank


# TODO: (yupu) Re-enable wandb
# def init_wandb(config, *, resuming: bool, log_code: bool = False, enabled: bool = True):
#     if not enabled:
#         wandb.init(mode="disabled")
#         return

#     ckpt_dir = pathlib.Path(config.checkpoint_dir)
#     if not ckpt_dir.exists():
#         raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
#     if resuming:
#         run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
#         wandb.init(id=run_id, resume="must", project=config.project_name)
#     else:
#         wandb.init(
#             name=config.exp_name, config=vars(config), project=config.project_name
#         )
#         (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

#     if log_code:
#         wandb.run.log_code(epath.Path(__file__).parent.parent)


def make_dataset(cfg, policy_config):
    # TODO: (yupu) Support image transforms
    cfg.enable_image_transform = False
    # TODO: (yupu) Remove hard-coded video backend
    cfg.video_backend = "pyav"

    image_transforms = (
        ImageTransforms(cfg.image_transforms) if cfg.enable_image_transform else None
    )
    # Leave the revision to None
    ds_meta = LeRobotDatasetMetadata(root=cfg.data_path, revision=None)
    delta_timestamps = resolve_delta_timestamps(policy_config, ds_meta)

    dataset = LeRobotDataset(
        root=cfg.data_path,
        episodes=None,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=None,
        video_backend=cfg.video_backend,
        tolerance_s=cfg.tolerance_s,
    )

    if cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    return dataset


def resolve_delta_timestamps(
    cfg, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg: The policy config (PI0Config or PI05Config) to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in cfg.observation_delta_indices
            ]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


# datasets/utils.py
def cycle(iterable: Any) -> Iterator[Any]:
    """Create a dataloader-safe cyclical iterator.

    This is an equivalent of `itertools.cycle` but is safe for use with
    PyTorch DataLoaders with multiple workers.
    See https://github.com/pytorch/pytorch/issues/23900 for details.

    Args:
        iterable: The iterable to cycle over.

    Yields:
        Items from the iterable, restarting from the beginning when exhausted.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def raise_feature_mismatch_error(
    provided_features: set[str],
    expected_features: set[str],
) -> None:
    """
    Raises a standardized ValueError for feature mismatches between dataset/environment and policy config.
    """
    missing = expected_features - provided_features
    extra = provided_features - expected_features
    # TODO (jadechoghari): provide a dynamic rename map suggestion to the user.
    raise ValueError(
        f"Feature mismatch between dataset/environment and policy config.\n"
        f"- Missing features: {sorted(missing) if missing else 'None'}\n"
        f"- Extra features: {sorted(extra) if extra else 'None'}\n\n"
        f"Please ensure your dataset and policy use consistent feature names.\n"
        f"If your dataset uses different observation keys (e.g., cameras named differently), "
        f"use the `--rename_map` argument, for example:\n"
        f'  --rename_map=\'{{"observation.images.left": "observation.images.camera1", '
        f'"observation.images.top": "observation.images.camera2"}}\''
    )


def validate_visual_features_consistency(
    cfg: PI0Config,
    features: dict[str, PolicyFeature],
) -> None:
    """
    Validates visual feature consistency between a policy config and provided dataset/environment features.

    Args:
        cfg (PreTrainedConfig): The model or policy configuration containing input_features and type.
        features (Dict[str, PolicyFeature]): A mapping of feature names to PolicyFeature objects.
    """
    expected_visuals = {
        k for k, v in cfg.input_features.items() if v.type == FeatureType.VISUAL
    }
    provided_visuals = {k for k, v in features.items() if v.type == FeatureType.VISUAL}
    if not provided_visuals.issubset(expected_visuals):
        raise_feature_mismatch_error(provided_visuals, expected_visuals)


def make_policy(
    cfg,
    ds_meta: LeRobotDatasetMetadata | None = None,
    rename_map: dict[str, str] | None = None,
    model_variant: str = "pi0",
):
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created (PI0Config or PI05Config).
             If `cfg.pretrained_path` is set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).
        model_variant: Model variant to use, either "pi0" or "pi0.5".

    Returns:
        An instantiated and device-placed policy model (PI0Policy or PI05Policy).
    """

    # Select policy class based on model variant
    if model_variant == "pi0.5":
        policy_cls = PI05Policy
    else:
        policy_cls = PI0Policy

    kwargs = {}
    features = dataset_to_policy_features(ds_meta.features)

    cfg.output_features = {
        # Changed from ft.type is FeatureType.ACTION to ft.type == FeatureType.ACTION
        # for different enum classes: flagscale.FeatureType vs lerobot.FeatureType
        key: ft
        for key, ft in features.items()
        if ft.type == FeatureType.ACTION
    }
    if not cfg.input_features:
        cfg.input_features = {
            key: ft for key, ft in features.items() if key not in cfg.output_features
        }
    kwargs["config"] = cfg

    # PI0 finetuning, so always load a pretrained policy.
    # Load a pretrained policy and override the config if needed (for example, if there are inference-time
    # hyperparameters that we want to vary).
    kwargs["pretrained_name_or_path"] = cfg.pretrained_path
    policy = policy_cls.from_pretrained(cfg.pretrained_path, config=cfg)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
    # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "preprocessor_config_filename",
                f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            ),
            overrides=kwargs.get("preprocessor_overrides", {}),
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_path,
            config_filename=kwargs.get(
                "postprocessor_config_filename",
                f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            ),
            overrides=kwargs.get("postprocessor_overrides", {}),
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


@dataclass
class CosineDecayWithWarmupSchedulerConfig:
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logger.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def update_policy(
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained (wrapped in DDP if not using Accelerator).
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get the policy model (unwrap DDP if needed) to access config
    policy_model = policy.module if isinstance(policy, DDP) else policy
    use_amp = getattr(policy_model.config, "use_amp", False)

    autocast_context = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    with autocast_context:
        loss, _= policy.forward(batch)
    # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    loss.backward()

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.module.parameters()
            if isinstance(policy, DDP)
            else policy.parameters(),
            grad_clip_norm,
        )
    else:
        # Compute grad norm even if not clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.module.parameters()
            if isinstance(policy, DDP)
            else policy.parameters(),
            float("inf"),
            error_if_nonfinite=False,
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()
    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(policy_model, "update"):
        policy_model.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    return train_metrics


def main(config: argparse.Namespace):
    set_seed(config.seed)

    model_variant = config.model_variant.lower()
    if model_variant not in ["pi0", "pi0.5"]:
        raise ValueError(
            f"Invalid model_variant: {model_variant}. Must be 'pi0' or 'pi0.5'"
        )

    if model_variant == "pi0.5":
        policy_config = PI05Config.from_pretrained(config.checkpoint_dir)
    else:
        policy_config = PI0Config.from_pretrained(config.checkpoint_dir)

    policy_config.pretrained_path = config.checkpoint_dir
    policy_config.n_action_steps = config.action_steps
    policy_config.tokenizer_max_length = config.tokenizer_max_length

    policy_config.use_amp = config.use_amp

    local_rank = init_ddp()
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()
    is_main_process = rank == 0 and local_rank == 0
    policy_config.device = device

    if is_main_process:
        logger.info(f"Policy config ({model_variant}): {policy_config}")

    dataset = make_dataset(config, policy_config)

    dist.barrier()

    # TODO: (yupu) This is so ugly
    rename_map = None
    if config.rename_map:
        rename_map_str = config.rename_map
        # Clean up the rename map string, remove outer quotes if present
        if (rename_map_str.startswith("'") and rename_map_str.endswith("'")) or (
            rename_map_str.startswith('"') and rename_map_str.endswith('"')
        ):
            rename_map_str = rename_map_str[1:-1]
            print(f"rename_map_str: {rename_map_str}")

        try:
            rename_map = json.loads(rename_map_str)
            if not isinstance(rename_map, dict):
                raise ValueError(
                    f"rename_map must be a dictionary, got {type(rename_map)}"
                )
        except json.JSONDecodeError as e:
            raise ValueError("Invalid JSON in --rename-map") from e

    policy = make_policy(
        cfg=policy_config,
        ds_meta=dataset.meta,
        rename_map=rename_map,
        model_variant=model_variant,
    )

    dist.barrier()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    # Only provide dataset_stats when not resuming from saved processor state
    processor_kwargs["dataset_stats"] = dataset.meta.stats

    if not config.use_quantiles and model_variant == "pi0.5":
        from flagscale.models.configs.types import NormalizationMode

        policy.config.normalization_mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }

    processor_kwargs["preprocessor_overrides"] = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {
                **policy.config.input_features,
                **policy.config.output_features,
            },
            "norm_map": policy.config.normalization_mapping,
        },
        "tokenizer_processor": {"tokenizer_name": config.tokenizer_path},
    }

    if rename_map is not None:
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": rename_map
        }
    postprocessor_kwargs["postprocessor_overrides"] = {
        "unnormalizer_processor": {
            "stats": dataset.meta.stats,
            "features": policy.config.output_features,
            "norm_map": policy.config.normalization_mapping,
        },
    }

    if is_main_process:
        logger.info(f"processor_kwargs: {processor_kwargs}")
        logger.info(f"postprocessor_kwargs: {postprocessor_kwargs}")

    preprocessor, _ = make_pre_post_processors(
        pretrained_path=policy_config.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    # Convert optimizer_betas to tuple if it's a list
    if isinstance(config.optimizer_betas, list):
        config.optimizer_betas = tuple(config.optimizer_betas)

    # TODO: (yupu) Should we let the user choose between config and policy preset?
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config.optimizer_lr,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )
    scheduler_config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=config.scheduler_warmup_steps,
        num_decay_steps=config.scheduler_decay_steps,
        peak_lr=config.optimizer_lr,
        decay_lr=config.scheduler_decay_lr,
    )
    lr_scheduler = scheduler_config.build(optimizer, config.train_steps)

    config.num_workers = 4
    shuffle = config.shuffle

    # DistributedSampler ensures each rank gets different data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        drop_last=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=False,  # Must be False when using sampler
        sampler=sampler,
        pin_memory=True,  # Assume all data is on GPU
        drop_last=False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )

    policy = DDP(
        policy,
        device_ids=[local_rank],
        find_unused_parameters=True,
        output_device=local_rank,
    )

    dist.barrier()

    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = config.batch_size * dist.get_world_size()

    step = 0

    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    # To ensures proper data shuffling across epochs in distributed training
    epoch = 0
    samples_per_epoch = None
    dataloader.sampler.set_epoch(epoch)
    samples_per_epoch = len(dataset) // effective_batch_size

    for _ in range(step, config.train_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            config.grad_clip_norm,
            lr_scheduler=lr_scheduler,
        )

        print(f"train_tracker at step {step}: {train_tracker}")

        step += 1
        train_tracker.step()

        # Update epoch counter for sampler.set_epoch() when we've processed one epoch worth of samples
        # This ensures proper data shuffling across epochs in distributed training
        if step % samples_per_epoch == 0:
            epoch += 1
            dataloader.sampler.set_epoch(epoch)

        if step % config.log_freq == 0 and is_main_process:
            logger.info(f"step: {step} loss: {train_tracker}")

        if config.save_checkpoint and step % config.save_freq == 0:
            # Synchronize all processes before checkpoint saving
            dist.barrier()

            if is_main_process:
                logger.info(f"Saving checkpoint at step {step}")
                output_dir = Path(config.output_directory)
                checkpoint_dir = get_step_checkpoint_dir(
                    output_dir, config.train_steps, step
                )
                policy_to_save = policy.module
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    policy=policy_to_save,
                )
                update_last_checkpoint(checkpoint_dir)

            # Synchronize all processes after checkpoint saving
            dist.barrier()

    if is_main_process:
        logger.info("Training completed")

    # Properly clean up the distributed process group
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ============================== System Configs ==============================
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=100000)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument(
        "--output-directory", type=str, default="", help="Path to the output directory"
    )
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--optimizer-lr", type=float, default=2.5e-5)
    parser.add_argument("--optimizer-betas", nargs=2, type=float, default=[0.9, 0.95])
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.01)
    parser.add_argument("--optimizer-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--scheduler-warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-decay-steps", type=int, default=30000)
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--project-name", type=str, default="default_project")
    parser.add_argument("--exp-name", type=str, default="default_exp")

    # ============================== Model Configs ==============================
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Path to the pretrained model checkpoint directory",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="pi0",
        choices=["pi0", "pi0.5"],
        help="Model variant to use: 'pi0' or 'pi0.5'",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default="", help="Path to the tokenizer"
    )
    parser.add_argument("--tokenizer-max-length", type=int, default=48)
    parser.add_argument("--action-steps", type=int, default=50)

    # ============================== Data Configs ==============================
    parser.add_argument("--enable-image-transform", action="store_true")
    parser.add_argument("--tolerance-s", type=float, default=0.0001)
    parser.add_argument("--use-imagenet-stats", action="store_true")
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument(
        "--data-path", type=str, default="", help="Path to the training dataset"
    )
    parser.add_argument(
        "--rename-map",
        type=str,
        default="",
        help=(
            "JSON string mapping dataset feature keys to policy feature keys, "
            'e.g., \'{"observation.images.cam_high": "observation.images.base_0_rgb"}\''
        ),
    )
    parser.add_argument("--use-quantiles", action="store_true")

    config = parser.parse_args()

    logger.info("=" * 100)
    logger.info(f"train_pi0_base.py config: {config}")
    main(config)
