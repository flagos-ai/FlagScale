#!/usr/bin/env python3
# Copyright (c) 2025, FlagScale Authors. All rights reserved.
"""
DreamZero Training Entrypoint for FlagScale native backend.

Uses FSDP2 for distributed training (matching original DeepSpeed ZeRO-2 strategy).
No TP/PP — pure data parallelism with sharded parameters.

Training loop: load data -> forward (VAE encode + noise + DiT predict) -> loss -> backward.
"""

import os
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

from omegaconf import OmegaConf, DictConfig
from flagscale.logger import logger
from flagscale.train.train_config import TrainConfig
from flagscale.train.utils.logging_utils import AverageMeter
from flagscale.train.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
    update_last_checkpoint,
)
from flagscale.train.utils.optim_setup import setup_optimizer_and_scheduler


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    # Raise Dynamo limits for compiled attention sub-methods.
    # The blockwise causal attention pattern has multiple code paths
    # that trigger recompiles — need higher limits to avoid FailOnRecompileLimitHit.
    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 512


def apply_fsdp2(policy, device_mesh):
    """Apply FSDP2 sharding to DreamZero.

    Strategy: shard every large sub-module individually to minimize peak memory.
    During forward, FSDP2 all-gathers ONE module's params at a time and reshards after.
    This is equivalent to ZeRO-3 (param + grad + optimizer sharding).

    Model structure:
      policy.action_head.text_encoder  — T5-XXL (~4.7B, frozen)
      policy.action_head.image_encoder — CLIP (~0.6B, frozen)
      policy.action_head.vae           — VAE (~0.1B, frozen)
      policy.action_head.model         — DiT (40 blocks, ~14B, trainable)
      policy.action_head.{action,state}_{encoder,decoder} — small projectors (trainable)
    """
    # Mixed precision policy for FSDP2.
    # param_dtype=bfloat16: store and all-gather sharded params in bf16 for memory efficiency.
    #   With 8-GPU sharding: 30.7GB model / 8 = 3.8GB per GPU (vs 7.7GB with fp32).
    #   The optimizer (AdamW) maintains fp32 master weights, matching DeepSpeed ZeRO-2.
    # reduce_dtype=float32: reduce-scatter gradients in fp32 for numerical accuracy.
    #   Tested bf16 reduce — no speed gain on single-node NVLink (bandwidth-saturated).
    # cast_forward_inputs=False: activations handled by autocast, not FSDP boundaries.
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )

    # Fix scalar parameters (FSDP2 requires ndim >= 1)
    for name, module in policy.named_modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            if param.ndim == 0:
                new_param = torch.nn.Parameter(
                    param.data.reshape(1), requires_grad=param.requires_grad
                )
                setattr(module, pname, new_param)

    ah = policy.action_head

    # FROZEN encoders (VAE, T5, CLIP): do NOT shard with FSDP2.
    # They have no gradients/optimizer states, so sharding gives no memory benefit
    # for optimizer. Also, VAE has 3D convolutions that are incompatible with DTensor.
    # Keep them replicated (same as ZeRO-2 behavior for frozen params).
    logger.info("Frozen encoders (text/image/vae) kept replicated (not sharded)")

    # 4. Shard DiT blocks individually (each ~400M params)
    # In full mode: action_head.model.blocks directly
    # In LoRA mode: action_head.model = PeftModel -> base_model -> model -> blocks
    blocks = None
    if hasattr(ah, "model") and hasattr(ah.model, "blocks"):
        blocks = ah.model.blocks
    elif hasattr(ah, "model") and hasattr(ah.model, "base_model"):
        base = ah.model.base_model
        if hasattr(base, "model") and hasattr(base.model, "blocks"):
            blocks = base.model.blocks
        elif hasattr(base, "blocks"):
            blocks = base.blocks

    if blocks is not None:
        logger.info(f"Sharding {len(blocks)} DiT blocks with FSDP2...")
        for i, block in enumerate(blocks):
            fully_shard(block, mesh=device_mesh, mp_policy=mp_policy, reshard_after_forward=True)
            if i == 0:
                logger.info(f"  Block 0 sharded successfully")
        logger.info(f"  All {len(blocks)} blocks sharded")
    else:
        logger.warning("Could not find DiT blocks for per-block FSDP2 sharding")

    # 5. Shard the DiT model wrapper (ah.model) — NOT the full policy
    # This catches remaining DiT flat params (head, embeddings) without sharding frozen encoders.
    # With blocks already sharded individually, this only processes root-level params (~200MB).

    # Debug: check what params are NOT yet sharded (managed by outer fully_shard)
    from torch.distributed._tensor import DTensor
    unsharded_size = 0
    sharded_size = 0
    for name, p in ah.model.named_parameters():
        if isinstance(p, DTensor):
            sharded_size += p.numel() * p.element_size()
        else:
            unsharded_size += p.numel() * p.element_size()
    logger.info(f"Before outer shard: unsharded={unsharded_size/1024**3:.2f}GB, already_sharded={sharded_size/1024**3:.2f}GB")

    logger.info("Sharding DiT model (PeftModel wrapper)...")
    fully_shard(ah.model, mesh=device_mesh, mp_policy=mp_policy)

    return policy


def apply_compile_and_reentrant(policy):
    """Apply torch.compile to attention sub-methods and switch to use_reentrant=True.

    Key optimizations from RLinf's DreamZero training:
    1. torch.compile(mode="reduce-overhead") on the 4 attention processing functions
       inside CausalWanSelfAttention — enables CUDA graphs for these hot inner loops.
    2. use_reentrant=True gradient checkpointing — avoids tensor pack/unpack hooks overhead
       (requires passing block args as positional, not keyword).
    """
    ah = policy.action_head
    dit = ah.model

    # 1. Compile attention sub-methods on each block's self_attn
    compiled_count = 0
    for block in dit.blocks:
        # The attention module is at block.self_attn (CausalWanSelfAttention)
        self_attn = getattr(block, "self_attn", None)
        if self_attn is None:
            continue
        for method_name in (
            "_process_clean_image_only",
            "_process_state_blocks",
            "_process_noisy_image_blocks",
            "_process_noisy_action_blocks",
        ):
            if hasattr(self_attn, method_name):
                original = getattr(self_attn, method_name)
                compiled = torch.compile(original, mode="reduce-overhead")
                setattr(self_attn, method_name, compiled)
                compiled_count += 1

    logger.info(f"Compiled {compiled_count} attention sub-methods with mode='reduce-overhead'")

    # 2. Switch to use_reentrant=True gradient checkpointing
    #    This is faster (no pack/unpack hooks) and compatible with CUDA graphs.
    if hasattr(dit, "gradient_checkpointing"):
        dit.gradient_checkpointing = True
    if hasattr(dit, "gradient_checkpointing_use_reentrant"):
        dit.gradient_checkpointing_use_reentrant = True
    else:
        dit.gradient_checkpointing_use_reentrant = True
    logger.info("Gradient checkpointing set to use_reentrant=True")


def safe_cycle(iterable) -> Iterator:
    """Cycle over iterable safely."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def get_model(config: TrainConfig):
    """Instantiate DreamZero model from config."""
    from flagscale.train.models.dreamzero import DreamZeroPolicy
    from flagscale.train.models.dreamzero.dreamzero_model import DreamZeroConfig

    model_cfg = config.model
    # pretrained_model_path is where the DreamZero checkpoint lives
    ckpt_dir = getattr(model_cfg, "pretrained_model_path", None) or model_cfg.checkpoint_dir
    dreamzero_config = DreamZeroConfig(
        model_path=ckpt_dir,
        tokenizer_path=getattr(model_cfg, "tokenizer_path", ckpt_dir),
        action_horizon=getattr(model_cfg, "action_horizon", 24),
        action_dim=getattr(model_cfg, "action_dim", 32),
        max_state_dim=getattr(model_cfg, "max_state_dim", 64),
        num_frames=getattr(model_cfg, "num_frames", 33),
        num_frame_per_block=getattr(model_cfg, "num_frame_per_block", 2),
        num_action_per_block=getattr(model_cfg, "num_action_per_block", 24),
        num_state_per_block=getattr(model_cfg, "num_state_per_block", 1),
        frame_seqlen=getattr(model_cfg, "frame_seqlen", 880),
        use_gradient_checkpointing=getattr(model_cfg, "use_gradient_checkpointing", True),
        train_architecture=getattr(model_cfg, "train_architecture", "full"),
        tune_diffusion_model=getattr(model_cfg, "tune_diffusion_model", True),
        tune_projector=getattr(model_cfg, "tune_projector", True),
        compute_dtype=getattr(model_cfg, "compute_dtype", "bfloat16"),
        embodiment_tag=getattr(model_cfg, "embodiment_tag", "libero"),
        lora_rank=getattr(model_cfg, "lora_rank", 4),
        lora_alpha=getattr(model_cfg, "lora_alpha", 4),
        lora_target_modules=getattr(model_cfg, "lora_target_modules", "q,k,v,o,ffn.0,ffn.2"),
        # Pretrained paths for component loading (Wan2.1 DIT, T5, CLIP, VAE)
        dit_version=getattr(model_cfg, "dit_version", None),
        text_encoder_pretrained_path=getattr(model_cfg, "text_encoder_pretrained_path", None),
        image_encoder_pretrained_path=getattr(model_cfg, "image_encoder_pretrained_path", None),
        vae_pretrained_path=getattr(model_cfg, "vae_pretrained_path", None),
    )
    policy = DreamZeroPolicy.from_pretrained_dreamzero(dreamzero_config)
    return policy


def get_dataloader(config: TrainConfig, seed: int = 42):
    """Build DreamZero DataLoader from config.

    Uses aligned_dataloader (reference transform chain) by default to ensure
    data pipeline matches the reference implementation exactly.
    """
    data_cfg = config.data
    model_cfg = config.model
    system_cfg = config.system

    use_aligned = getattr(data_cfg, "use_aligned_dataloader", True)

    if use_aligned:
        from flagscale.train.datasets.dreamzero.aligned_dataloader import build_dataloader_aligned

        dataloader = build_dataloader_aligned(
            data_path=data_cfg.data_path,
            tokenizer_path=getattr(data_cfg, "tokenizer_path", None) or getattr(model_cfg, "tokenizer_path", None),
            batch_size=system_cfg.batch_size,
            num_workers=system_cfg.num_workers,
            num_frames=getattr(model_cfg, "num_frames", 33),
            action_horizon=getattr(model_cfg, "action_horizon", 24),
            state_horizon=getattr(model_cfg, "state_horizon", 1),
            action_dim=getattr(model_cfg, "action_dim", 32),
            max_state_dim=getattr(model_cfg, "max_state_dim", 64),
            embodiment_tag=getattr(data_cfg, "embodiment_tag", "oxe_droid"),
            embodiment_tag_mapping=getattr(data_cfg, "embodiment_tag_mapping", None),
            image_size=tuple(getattr(data_cfg, "image_size", [176, 320])),
            max_text_length=getattr(data_cfg, "max_text_length", 512),
            num_views=getattr(model_cfg, "num_views", 3),
            max_chunk_size=getattr(data_cfg, "max_chunk_size", 4),
            seed=seed,
        )
    else:
        from flagscale.train.datasets.dreamzero import build_dataloader

        dataloader = build_dataloader(
            data_path=data_cfg.data_path,
            tokenizer_path=getattr(data_cfg, "tokenizer_path", None) or getattr(model_cfg, "tokenizer_path", None),
            batch_size=system_cfg.batch_size,
            num_workers=system_cfg.num_workers,
            max_chunk_size=getattr(data_cfg, "max_chunk_size", 4),
            macro_stride=getattr(data_cfg, "macro_stride", 24),
            action_horizon=getattr(model_cfg, "action_horizon", 24),
            state_horizon=getattr(model_cfg, "state_horizon", 1),
            action_dim=getattr(model_cfg, "action_dim", 32),
            max_state_dim=getattr(model_cfg, "max_state_dim", 64),
            embodiment_tag=getattr(data_cfg, "embodiment_tag", "libero"),
            embodiment_tag_mapping=getattr(data_cfg, "embodiment_tag_mapping", None),
            image_size=tuple(getattr(data_cfg, "image_size", [176, 320])),
            max_text_length=getattr(data_cfg, "max_text_length", 512),
            num_views=getattr(model_cfg, "num_views", 2),
            shuffle=system_cfg.shuffle,
            distributed=dist.is_initialized(),
            relative_action=getattr(data_cfg, "relative_action", True),
            crop_ratio=getattr(data_cfg, "crop_ratio", 0.95),
            color_jitter=getattr(data_cfg, "color_jitter", True),
            training=True,
        )
    return dataloader


def main(train_config: TrainConfig, seed: int = 42):
    """Main training loop for DreamZero."""
    set_seed(seed)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        logger.info(f"Starting DreamZero training: world_size={world_size}")

    # Build model (loads on CPU)
    policy = get_model(train_config)
    logger.info(f"[Rank {rank}] Model loaded on CPU. Starting FSDP2 sharding...")
    import sys; sys.stdout.flush()

    # AGENT_DEBUG: Weight dump disabled for performance (was writing 45GB to disk)
    # from flagscale.train.models.dreamzero.dump_init_weights import dump_trainable_weights  # AGENT_DEBUG
    # dump_trainable_weights(policy, _dump_path, rank=rank)  # AGENT_DEBUG

    # Apply FSDP2 BEFORE moving to GPU — model is too large for single GPU
    # FSDP2 shards parameters across ranks, so each rank only holds 1/N of params on GPU
    try:
        device_mesh = init_device_mesh("cuda", (world_size,))
        logger.info(f"[Rank {rank}] Device mesh created. Calling apply_fsdp2...")
        sys.stdout.flush()
        policy = apply_fsdp2(policy, device_mesh)
        logger.info(f"[Rank {rank}] FSDP2 applied successfully")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"[Rank {rank}] FSDP2 FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.stdout.flush()
        raise

    # Move frozen encoders to GPU in bf16 (matching reference _initialize behavior at line 1367-1368)
    # These are inference-only under autocast, so bf16 is correct and saves ~50% memory vs fp32.
    # The DiT (ah.model) stays fp32 via FSDP2 param_dtype — matching DeepSpeed ZeRO-2 master weights.
    ah = policy.action_head
    ah.text_encoder.to(device=device, dtype=torch.bfloat16)
    ah.image_encoder.to(device=device, dtype=torch.bfloat16)
    ah.vae.to(device=device, dtype=torch.bfloat16)
    # Move small projectors/encoders that live outside ah.model
    for name, module in ah.named_children():
        if name not in ("model", "text_encoder", "image_encoder", "vae"):
            module.to(device=device)
    if rank == 0:
        logger.info(f"Frozen encoders and projectors moved to {device}")

    # torch.compile: tested, gives <5% gain due to graph breaks in blockwise attention.
    # Keeping code for reference but disabled by default.
    if getattr(train_config.system, "use_torch_compile", False):
        import torch._inductor.config as inductor_config
        inductor_config.reorder_for_compute_comm_overlap = True
        if rank == 0:
            logger.info("Enabling torch.compile on DiT (mode=max-autotune-no-cudagraphs)")
            logger.info("Enabled inductor reorder_for_compute_comm_overlap=True")
        ah.model = torch.compile(ah.model, mode="max-autotune-no-cudagraphs")
        if rank == 0:
            logger.info("torch.compile applied to action_head.model")

    # RLinf-style optimization: compile attention sub-methods + use_reentrant=True
    # This is much more effective than whole-model compile because:
    # 1. The 4 attention methods have fixed shapes → CUDA graphs work
    # 2. use_reentrant=True avoids pack/unpack hooks overhead in gradient checkpointing
    if not getattr(train_config.system, "disable_attention_compile", False):
        apply_compile_and_reentrant(policy)
    else:
        if rank == 0:
            logger.info("Attention compile disabled by config")

    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(policy, train_config)

    # Dataloader with reference transforms (each rank gets different micro-batch via IterableDataset sharding)
    dataloader = get_dataloader(train_config, seed=seed)
    data_iter = safe_cycle(dataloader)

    # Training params
    train_steps = train_config.system.train_steps
    log_freq = train_config.system.log_freq
    grad_accum_steps = getattr(train_config.system, "gradient_accumulation_steps", 1)
    grad_clip_norm = getattr(train_config.system, "grad_clip_norm", 1.0)
    save_freq = getattr(train_config.system, "save_steps", 1000)
    output_dir = Path(train_config.system.checkpoint.output_directory)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    loss_meter = AverageMeter("loss")
    step_time_meter = AverageMeter("step_time")

    if rank == 0:
        eff_batch = train_config.system.batch_size * world_size * grad_accum_steps
        logger.info(
            f"Config: steps={train_steps}, micro_batch={train_config.system.batch_size}, "
            f"grad_accum={grad_accum_steps}, effective_batch={eff_batch}"
        )

    use_aligned = getattr(train_config.data, "use_aligned_dataloader", True)
    if use_aligned:
        from flagscale.train.datasets.dreamzero.aligned_dataloader import get_batch_aligned as get_batch
    else:
        from flagscale.train.datasets.dreamzero import get_batch

    # Barrier: ensure all ranks finished init before any forward pass
    dist.barrier()
    if rank == 0:
        logger.info("All ranks synchronized — starting training loop")

    # Training loop
    policy.train()
    optimizer.zero_grad(set_to_none=True)

    # DebugHooks for Level 4 alignment (capture fwd/bwd intermediates on step 1)
    _debug_hooks_enabled = os.environ.get("DREAMZERO_DEBUG_HOOKS", "") in ("1", "true", "True")
    if _debug_hooks_enabled and rank == 0:
        import sys
        sys.path.insert(0, "/public-mixed/fengyupu/github/null-space")
        from null_space.hamster.debug.hooks import DebugHooks
        _hook_log_path = os.path.join(str(output_dir), "debug_hooks_step1.log")
        _hook_log_file = open(_hook_log_path, "w")
        def _hook_print(msg):
            _hook_log_file.write(msg + "\n")
            _hook_log_file.flush()
        _debug_hooks = DebugHooks(policy.action_head.model, print_fn=_hook_print)
        _debug_hooks.register()
        logger.info(f"[DEBUG_HOOKS] Registered on action_head.model, logging to {_hook_log_path}")

    for step in range(1, train_steps + 1):
        step_start = time.time()
        accumulated_loss = 0.0
        # Signal to PyTorch that a new training iteration is starting.
        # Helps manage CUDA graph memory from compiled attention sub-methods.
        torch.compiler.cudagraph_mark_step_begin()

        for _ in range(grad_accum_steps):
            raw_batch = next(data_iter)
            batch = get_batch(raw_batch, device=device, compute_dtype=torch.bfloat16)

            # Debug: print batch shapes on first step
            if step == 1 and rank == 0:
                logger.info("[BATCH DEBUG] Batch shapes and stats:")
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        logger.info(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} min={v.min().item():.4f} max={v.max().item():.4f}")
                    else:
                        logger.info(f"  {k}: type={type(v)} value={v}")

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = policy(batch)
                loss = outputs["loss"] / grad_accum_steps
                loss.backward()
            accumulated_loss += loss.item()
            # Track component losses for logging
            if rank == 0:
                dyn_loss = outputs.get("dynamics_loss", torch.tensor(0.0)).item()
                act_loss = outputs.get("action_loss", torch.tensor(0.0)).item()

        # DebugHooks: remove after step 1
        if _debug_hooks_enabled and step == 1 and rank == 0:
            _debug_hooks.remove()
            _hook_log_file.close()
            logger.info(f"[DEBUG_HOOKS] Step 1 complete, hooks removed. Log: {_hook_log_path}")

        if grad_clip_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            total_norm = 0.0

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        policy.set_frozen_modules_to_eval()

        step_time = time.time() - step_start
        loss_meter.update(accumulated_loss)
        step_time_meter.update(step_time)

        # Logging
        if rank == 0 and step % log_freq == 0:
            lr = scheduler.get_last_lr()[0]
            grad_norm_val = total_norm.item() if torch.is_tensor(total_norm) else total_norm
            logger.info(
                f"step={step}/{train_steps} | loss={accumulated_loss:.4f} | "
                f"dynamics={dyn_loss:.4f} | action={act_loss:.4f} | "
                f"grad_norm={grad_norm_val:.4f} | "
                f"lr={lr:.2e} | time={step_time:.2f}s/step"
            )
            # JSON log for easy comparison parsing
            import json as _json
            _eff_batch = train_config.system.batch_size * world_size * grad_accum_steps
            _samples_per_sec = _eff_batch / step_time
            _log_entry = {
                "step": step, "loss": accumulated_loss,
                "dynamics_loss": dyn_loss, "action_loss": act_loss,
                "grad_norm": grad_norm_val, "learning_rate": lr,
                "step_time": round(step_time, 3),
                "samples_per_sec": round(_samples_per_sec, 3),
                "samples_per_sec_per_gpu": round(_samples_per_sec / world_size, 4),
            }
            _log_path = os.path.join(output_dir, "loss_log.jsonl")
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps(_log_entry) + "\n")
            loss_meter.reset()
            step_time_meter.reset()

        # Checkpoint
        if step % save_freq == 0 and rank == 0:
            ckpt_dir = get_step_checkpoint_dir(output_dir, train_steps, step)
            save_checkpoint(
                checkpoint_dir=ckpt_dir,
                step=step,
                config=train_config,
                policy=policy,
                optimizer_state_dict=optimizer.state_dict(),
                lr_scheduler=scheduler,
            )
            update_last_checkpoint(ckpt_dir)
            logger.info(f"Checkpoint saved: {ckpt_dir}")

    # Final checkpoint (disabled for alignment testing — FSDP2 save not yet supported)
    # if rank == 0:
    #     ckpt_dir = get_step_checkpoint_dir(output_dir, train_steps, train_steps)
    #     save_checkpoint(
    #         checkpoint_dir=ckpt_dir,
    #         step=train_steps,
    #         config=train_config,
    #         policy=policy,
    #         optimizer_state_dict=optimizer.state_dict(),
    #         lr_scheduler=scheduler,
    #     )
    #     update_last_checkpoint(ckpt_dir)
    #     logger.info(f"Training complete. Final checkpoint: {ckpt_dir}")

    if rank == 0:
        logger.info(f"Training complete after {train_steps} steps (checkpoint save disabled).")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train DreamZero model. Called by FlagScale runner, not directly."
    )
    parser.add_argument(
        "--config-file", type=str, required=True, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    # Load config from Hydra-generated YAML
    config = OmegaConf.load(args.config_file)

    logger.info(f"Full config: {OmegaConf.to_yaml(config)}")

    # Convert to Pydantic TrainConfig
    train_config = TrainConfig.from_hydra_config(config)

    # Extract seed from experiment config
    experiment_config = OmegaConf.to_container(config.experiment, resolve=True)
    seed = experiment_config.get("seed", 42)

    logger.info(f"Experiment: {experiment_config}")
    main(train_config, seed)
