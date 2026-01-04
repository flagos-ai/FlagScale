# Adopted from starVLA/starVLA:
# https://github.com/starVLA/starVLA/blob/starVLA/starVLA/training/train_starvla.py
# Below is the original copyright:

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import os
import pathlib
import platform
import random

from typing import Tuple

import epath
import numpy as np
import torch
import torch.distributed as dist

from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler

import wandb

from megatron.energon import WorkerConfig, get_loader, get_train_dataset
from tools.datasets.vla.data.dataset_helpers_np_pil import TaskEncoder

from flagscale.logger import logger
from flagscale.models.robobrain_x.qwen_groot import Qwen_GR00T, get_batch

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_param_lr_groups(model, cfg):
    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    used_params = set()
    param_groups = []

    for module_name, lr in lr_cfg.items():
        if module_name == "base":
            continue
        # try to find the module under vla by module_name (support nested paths)
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            params = list(module.parameters())
            param_groups.append({"params": params, "lr": lr, "name": module_name})
            used_params.update(id(p) for p in params)
        except AttributeError:
            ReferenceError(f"⚠️ module path `{module_name}` not found in vla")

    # assign base learning rate to the remaining unused parameters
    other_params = [p for p in model.parameters() if id(p) not in used_params]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    return param_groups


def setup_optimizer_and_scheduler(
    model, cfg
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(
                f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}"
            )

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


def init_ddp(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return local_rank


def init_wandb(config, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = pathlib.Path(config.checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=vars(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def main(cfg) -> None:
    # build model
    vla = Qwen_GR00T(cfg)
    # prepare data
    ds = get_train_dataset(
        cfg.datasets.data_path,
        batch_size=cfg.batch_size,
        shuffle_buffer_size=100,
        max_samples_per_sequence=100,
        worker_config=WorkerConfig.default_worker_config(num_workers=1, data_parallel_group=None),
        task_encoder=TaskEncoder(cfg.datasets.task_encoder),
        repeat=True,
    )
    vla_train_dataloader = get_loader(ds)
    data_iter = iter(vla_train_dataloader)
    batch = next(data_iter)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    # Run VLA Training
    local_rank = init_ddp(cfg.seed)
    if dist.get_rank() == 0 and local_rank == 0:
        logger.info(f"Running on: {platform.node()}")
        if cfg.batch_size % torch.cuda.device_count() != 0:
            raise ValueError(
                f"Batch size {cfg.batch_size} must be divisible by the number of devices {torch.cuda.device_count()}."
            )
        resuming = cfg.resume
        init_wandb(cfg, resuming=resuming, enabled=cfg.wandb_enabled)

    vla = vla.cuda()
    vla = DDP(vla, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)

    step = 0
    done = False
    while not done:
        batch = next(data_iter)
        batch = get_batch(batch)
        output_dict = vla.forward(batch)
        action_loss = output_dict["action_loss"]
        action_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % cfg.log_freq == 0:
            logger.info(f"step: {step} loss: {action_loss.item():.3f}")
        step += 1
        if step >= cfg.train_steps:
            done = True
            break

    if dist.get_rank() == 0 and local_rank == 0:
        vla.module.save_pretrained()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default="examples/robotics/conf/train/libero_qwengroot.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_path)
    main(cfg)
