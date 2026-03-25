# Mainly adopted from
# https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/scripts/lerobot_train.py

import argparse
import os
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, StateDictOptions
from torch.optim import Optimizer

from flagscale.logger import logger
from flagscale.train.train_config import TrainConfig
from flagscale.train.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from flagscale.train.datasets.utils import dataset_to_policy_features
from flagscale.train.processor import PolicyProcessorPipeline
from flagscale.models.utils.constants import ACTION, OBS_PREFIX
from flagscale.models.configs.types import FeatureType
from flagscale.train.utils.logging_utils import (
    AverageMeter,
    MetricsTracker,
    format_big_number,
)
from flagscale.train.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state_fsdp2,
    save_checkpoint,
    update_last_checkpoint,
)
from flagscale.train.utils.random_utils import serialize_rng_state, deserialize_rng_state
from flagscale.train.utils.optim_setup import setup_optimizer_and_scheduler
from flagscale.models.vla.base_policy import TrainablePolicy
from flagscale.models.vla.pretrained_config import PreTrainedConfig
import flagscale.models.vla.gr00t_n1_5.processor_gr00t  # noqa: F401  register GR00T processor steps
from flagscale.platform import get_platform

# Monkey-patch: transformers 4.57+ kernel-hub discovery can't find flash_attn on some platforms
# (e.g. MUSA), but direct import works fine. Replace _lazy_imports so transformers uses flash_attn directly.
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    import transformers.modeling_flash_attention_utils as _fa_utils
    _fa_utils._flash_fn = flash_attn_func
    _fa_utils._flash_varlen_fn = flash_attn_varlen_func
    _fa_utils._pad_fn = pad_input
    _fa_utils._unpad_fn = unpad_input
    def _patched_lazy_imports(implementation=None):
        return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input
    _fa_utils._lazy_imports = _patched_lazy_imports
except ImportError:
    pass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    get_platform().manual_seed_all(seed)
    if get_platform().name() == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False 
        torch.backends.cuda.matmul.allow_tf32 = False

def apply_fsdp2(policy, device_mesh):
    """Apply FSDP2 sharding to Gr00tN15.

    Uses a MixedPrecisionPolicy that matches DeepSpeed bf16 behavior:
      bf16.enabled=true + ZeRO-2 → param_dtype=bf16, reduce_dtype=bf16, reshard=False
    """
    # Cast everything to fp32 first so the root param group has uniform dtype.
    policy = policy.float()

    # TODO: (yupu) check `reduce_dtype=torch.float32`
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    fsdp_config = {"mesh": device_mesh, "mp_policy": mp_policy}

    # reshard_after_forward=False keeps params unsharded during forward+backward
    reshard = False

    for unit in policy.fsdp_units():
        fully_shard(unit, **fsdp_config, reshard_after_forward=reshard)

    fully_shard(policy, **fsdp_config)


def make_dataset(config: TrainConfig, policy_config: PreTrainedConfig):
    # TODO: (yupu) Remove hard-coded video backend
    # After not much testing, It feels like that `torchcodec` is more robust than `pyav`
    # `pyav` crashes sometimes
    # torchcodec depends on NVIDIA NVDEC which is not available on all platforms (e.g. MUSA);
    # fall back to pyav for non-CUDA platforms.
    video_backend = "torchcodec" if get_platform().name() == "cuda" else "pyav"

    # Leave the revision to None
    ds_meta = LeRobotDatasetMetadata(root=config.data.data_path, revision=None)
    delta_timestamps = _resolve_delta_timestamps(policy_config, ds_meta)

    dataset = LeRobotDataset(
        root=config.data.data_path,
        episodes=None,
        delta_timestamps=delta_timestamps,
        revision=None,
        video_backend=video_backend,
        tolerance_s=config.data.tolerance_s,
    )

    return dataset


def _resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg: The policy config (PreTrainedConfig) to read delta_indices from.
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
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

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


def format_train_tracker_step(train_tracker: MetricsTracker) -> str:
    def _format_meter_val(meter: AverageMeter) -> str:
        fmt = meter.fmt[1:] if meter.fmt.startswith(":") else meter.fmt
        return f"{meter.name}:{format(meter.val, fmt)}"

    display_list = [
        f"step:{format_big_number(train_tracker.steps)}",
        f"smpl:{format_big_number(train_tracker.samples)}",
        f"ep:{format_big_number(train_tracker.episodes)}",
        f"epch:{train_tracker.epochs:.2f}",
        *[_format_meter_val(m) for m in train_tracker.metrics.values()],
    ]
    return " ".join(display_list)



def make_policy(
    config: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
):
    features = dataset_to_policy_features(ds_meta.features)

    # Use == instead of `is` for FeatureType.ACTION comparison
    # because flagscale.FeatureType and lerobot.FeatureType are different enum classes
    output_features = {
        key: ft
        for key, ft in features.items()
        if ft.type == FeatureType.ACTION
    }
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    config.output_features = output_features
    config.input_features = input_features

    policy = TrainablePolicy.from_config(config)
    policy.to(get_platform().name())
    policy.train()

    return policy


def make_preprocessor_from_config(
    config: dict[str, Any] | list[str | dict[str, Any]],
    overrides: dict[str, Any] | None = None,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """
    Create a preprocessor pipeline from step configurations with optional overrides.

    This function creates a PolicyProcessorPipeline directly from step configurations,
    without requiring a pretrained path. It supports overriding step configurations
    similar to PolicyProcessorPipeline.from_pretrained().

    Args:
        config: Can be either:
            - A dict with "name" and "steps" fields (JSON format):
              {"name": "policy_preprocessor", "steps": [...]}
            - A list of step configurations (concise format):
              ["step_name", {"step_name": {...}}]
        overrides: Optional dictionary to override step configurations. Keys should
            match the step's registry_name. Example:
            {"device_processor": {"device": "cuda"},
             "normalizer_processor": {"stats": dataset.meta.stats}}

    Returns:
        A PolicyProcessorPipeline instance with the configured steps.

    Example (JSON format with overrides):
        ```python
        config = {
            "name": "policy_preprocessor",
            "steps": [
                {"registry_name": "device_processor", "config": {"device": "cpu"}},
                {"registry_name": "normalizer_processor", "config": {"eps": 1e-8}},
            ],
        }
        overrides = {
            "device_processor": {"device": "cuda"},
            "normalizer_processor": {"stats": dataset.meta.stats, "features": {...}},
        }
        preprocessor = make_preprocessor_from_config(config, overrides=overrides)
        # device_processor will use device="cuda" (overridden)
        # normalizer_processor will use eps=1e-8 (from config) and stats from overrides
        ```

    Example (concise list format):
        ```python
        steps = [
            "rename_observations_processor",
            "device_processor",
            {"normalizer_processor": {"eps": 1e-8}},
        ]
        preprocessor = make_preprocessor_from_config(steps)
        ```

    Raises:
        ValueError: If a step configuration is invalid or step cannot be instantiated.
        KeyError: If a registry name is not found.
    """
    from flagscale.train.processor.pipeline import ProcessorStepRegistry

    overrides = overrides or {}

    # Determine format and extract step configs
    if isinstance(config, (dict, DictConfig)) and "steps" in config:
        # JSON format: {"name": "...", "steps": [...]}
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        step_configs = config["steps"]
        pipeline_name = config.get("name", "policy_preprocessor")
    elif isinstance(config, list):
        # Concise list format
        step_configs = config
        pipeline_name = "policy_preprocessor"
    else:
        raise ValueError(f"Config must be a dict with 'steps' key or a list, got {type(config)}")

    steps = []
    for step_entry in step_configs:
        # Determine step format and normalize to standard dict
        if isinstance(step_entry, str):
            # Concise format: "step_name"
            step_dict = {"registry_name": step_entry, "config": {}}
        elif isinstance(step_entry, (dict, DictConfig)):
            if "registry_name" in step_entry:
                # JSON format: {"registry_name": "...", "config": {...}}
                if isinstance(step_entry, DictConfig):
                    step_entry = OmegaConf.to_container(step_entry, resolve=True)
                step_dict = step_entry
            elif len(step_entry) == 1:
                # Concise format: {"step_name": {...}}
                step_name = next(iter(step_entry.keys()))
                step_config = step_entry[step_name]
                if isinstance(step_config, DictConfig):
                    step_config = OmegaConf.to_container(step_config, resolve=True)
                step_dict = {"registry_name": step_name, "config": step_config}
            else:
                raise ValueError(
                    f"Step config dict must have either 'registry_name' or exactly one key, "
                    f"got {list(step_entry.keys())}"
                )
        else:
            raise ValueError(
                f"Step config must be str or dict, got {type(step_entry)}: {step_entry}"
            )

        # Get step class
        registry_name = step_dict["registry_name"]
        step_class = ProcessorStepRegistry.get(registry_name)

        # Merge config with overrides (overrides take precedence)
        try:
            base_config = step_dict.get("config", {})
            step_overrides = overrides.get(registry_name, {})
            merged_config = {**base_config, **step_overrides}

            step_instance = step_class(**merged_config)
            steps.append(step_instance)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate processor step '{registry_name}' "
                f"with config {merged_config}. Error: {e!s}"
            ) from e

    return PolicyProcessorPipeline(
        steps=steps,
        name=pipeline_name,
    )


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def update_policy(
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    optimizer: Optimizer,
    use_amp: bool,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=None,
) -> MetricsTracker:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained (FSDP2-sharded).
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        use_amp: Whether to use automatic mixed precision.
        grad_clip_norm: The maximum norm for gradient clipping.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        The updated MetricsTracker with new statistics for this step.
    """
    start_time = time.perf_counter()

    optimizer.zero_grad()

    autocast_context = (
        torch.amp.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16) if use_amp else nullcontext()
    )

    with autocast_context:
        output = policy(batch)
        loss = output["loss"]

    loss.backward()

    # Clip gradients (torch.nn.utils.clip_grad_norm_ works with DTensors in PyTorch ≥2.6)
    clip_value = grad_clip_norm if grad_clip_norm > 0 else float("inf")
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_value)

    with lock if lock is not None else nullcontext():
        optimizer.step()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.full_tensor().item() if hasattr(grad_norm, 'full_tensor') else grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time

    return train_metrics


def main(config: TrainConfig, seed: int):
    set_seed(seed)

    policy_config = PreTrainedConfig.from_train_config(config)
  
    local_rank = int(os.environ["LOCAL_RANK"])
    get_platform().set_device(local_rank)
    dist.init_process_group(backend=get_platform().dist_backend())
    device = get_platform().device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = rank == 0

    dataset = make_dataset(config, policy_config)
    dist.barrier()

    policy = make_policy(config=policy_config, ds_meta=dataset.meta)
    dist.barrier()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    preprocessor_overrides = {
        "device_processor": {"device": device.type},
        "normalizer_processor": {
            "stats": dataset.meta.stats,
            "features": {
                **policy.input_features,
                **policy.output_features,
            },
        },
        "groot_pack_inputs": {
            "stats": dataset.meta.stats,
            "normalize_min_max": True,
        },
    }

    num_workers = config.system.num_workers
    shuffle = config.system.shuffle

    # DistributedSampler ensures each rank gets different data
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=config.system.batch_size,
        shuffle=False,  # Must be False when using sampler
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Setup preprocessor
    preprocessor = None
    if config.data.preprocessor is not None:
        preprocessor = make_preprocessor_from_config(
            config.data.preprocessor, overrides=preprocessor_overrides
        )

    # Setup postprocessor (unnormalization for inference)
    postprocessor = None
    postprocessor_config = getattr(config.data, "postprocessor", None)
    if postprocessor_config is not None:
        postprocessor_overrides = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.input_features,
                    **policy.output_features,
                },
            },
            "groot_action_unpack_unnormalize": {
                "stats": dataset.meta.stats,
                "normalize_min_max": True,
            },
        }
        postprocessor = make_preprocessor_from_config(
            postprocessor_config, overrides=postprocessor_overrides
        )

    num_frames = dataset.num_frames
    num_episodes = dataset.num_episodes

    # --- Apply FSDP2 ---
    device_mesh = init_device_mesh(get_platform().name(), (world_size,))
    apply_fsdp2(policy, device_mesh)

    # Setup optimizer and scheduler (applies freeze config internally)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(policy, config)

    dl_iter = cycle(dataloader)

    step = 0
    resume_from = config.system.checkpoint.resume_from
    if resume_from:
        step = load_training_state_fsdp2(
            Path(resume_from), policy, optimizer, lr_scheduler,
        )
        saved_rng = serialize_rng_state()
        for _ in range(step):
            next(dl_iter)
        deserialize_rng_state(saved_rng)
        logger.info(f"Resumed from checkpoint at step {step}")

    dist.barrier()

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = config.system.batch_size * world_size

    train_tracker = MetricsTracker(
        effective_batch_size,
        num_frames,
        num_episodes,
        train_metrics,
        initial_step=step,
    )

    epoch = 0
    samples_per_epoch = num_frames // effective_batch_size
    if resume_from and samples_per_epoch > 0:
        epoch = step // samples_per_epoch
    sampler.set_epoch(epoch)

    for _ in range(step, config.system.train_steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        if isinstance(batch, dict):  # lerobot: move batched tensors to device
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

        if preprocessor is not None:
            batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            use_amp=config.system.use_amp,
            grad_clip_norm=config.system.grad_clip_norm,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()

        # Update epoch counter for sampler.set_epoch() when we've processed one epoch worth of samples
        if samples_per_epoch > 0 and step % samples_per_epoch == 0:
            epoch += 1
            sampler.set_epoch(epoch)

        if step % config.system.log_freq == 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info(f"[Rank {rank}] step: {step} {format_train_tracker_step(train_tracker)}")
            train_tracker.reset_averages()

        if (
            config.system.checkpoint.save_checkpoint
            and step % config.system.checkpoint.save_freq == 0
        ):
            dist.barrier()
            # get_model_state_dict and get_optimizer_state_dict are collectives — all ranks must call
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(policy, options=options)
            optimizer_state_dict = get_optimizer_state_dict(policy, optimizer, options=options)

            if is_main_process:
                logger.info(f"Saving checkpoint at step {step}")
                output_dir = Path(config.system.checkpoint.output_directory)
                checkpoint_dir = get_step_checkpoint_dir(
                    output_dir, config.system.train_steps, step
                )
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    config=config,
                    policy=policy,
                    optimizer_state_dict=optimizer_state_dict,
                    lr_scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    state_dict=state_dict,
                )
                update_last_checkpoint(checkpoint_dir)

            dist.barrier()

    if is_main_process:
        logger.info("Training completed")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GR00T N1.5 model. This script is typically called by the flagscale runner, not directly."
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    config_file_path = args.config_file

    # Load config from YAML file (Hydra-generated config.yaml contains both train and experiment)
    config = OmegaConf.load(config_file_path)

    logger.info(f"full config: {config}")

    # Extract train config and convert to Pydantic TrainConfig (preserves raw configs)
    train_config = TrainConfig.from_hydra_config(config)

    # Extract experiment config (seed, exp_dir, etc.)
    experiment_config = OmegaConf.to_container(config.experiment, resolve=True)
    seed = experiment_config.get("seed", 42)

    logger.info("=" * 100)
    logger.info(f"Experiment: {experiment_config}")
    logger.info(f"Train config: {train_config}")

    main(train_config, seed)
