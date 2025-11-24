import dataclasses
import functools
import logging
import os
import platform
import random

import etils.epath as epath
import numpy as np
import robotics.models.model as _model
import robotics.training.checkpoints as _checkpoints
import robotics.training.config as _config
import robotics.training.data_loader as _data_loader
import robotics.training.optimizer as _optimizer
import robotics.training.utils as training_utils
import torch
import torch.distributed as dist
import tqdm_loggable.auto as tqdm

from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from megatron.energon import WorkerConfig, get_loader, get_train_dataset
from tools.datasets.qwenvl.data.dataset_helpers_robotics import TaskEncoder


def init_ddp(config: _config.TrainConfig):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return local_rank


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def init_train_state(config: _config.TrainConfig, *, resume: bool, local_rank: int):
    model = config.model.create()
    model = config.weight_loader.load(model).to(device=f"cuda:{local_rank}")
    model.action_head.to(dtype=getattr(torch, config.training_dtype))
    if config.freeze_vlm:
        for n, p in model.named_parameters():
            if _model.is_vit(n, config.model.model_type) or _model.is_llm(
                n, config.model.model_type
            ):
                p.requires_grad = False

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        llm_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if _model.is_llm(n, config.model.model_type)
        )
        vit_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if _model.is_vit(n, config.model.model_type)
        )
        action_expert_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if _model.is_action_expert(n, config.model.model_type)
        )

        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        logging.info(f"LLM parameters: {llm_params:,} ({100 * llm_params / total_params:.2f}%)")
        logging.info(f"ViT parameters: {vit_params:,} ({100 * vit_params / total_params:.2f}%)")
        logging.info(
            f"Action Expert parameters: {action_expert_params:,} ({100 * action_expert_params / total_params:.2f}%)"
        )
        logging.info(
            f"Frozen parameters: {total_params - trainable_params:,} ({100 * (total_params - trainable_params) / total_params:.2f}%)"
        )

    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)
    optimizer, scheduler, clip_gradient_norm = _optimizer.create_optimizer(
        config.optimizer, config.lr_schedule, model.parameters()
    )

    train_state = training_utils.TrainState(
        step=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_gradient_norm=clip_gradient_norm,
        ema_decay=None,
        ema_params=None,
    )
    return train_state


def train_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
):
    model = state.model
    optimizer = state.optimizer
    scheduler = state.scheduler
    clip_gradient_norm = state.clip_gradient_norm
    ema_decay = state.ema_decay
    ema_params = state.ema_params

    model.train()
    observation, actions = batch
    optimizer.zero_grad()
    chunked_loss = model(
        observation.to(dtype=getattr(torch, config.training_dtype), device="cuda"),
        actions.to(dtype=getattr(torch, config.training_dtype), device="cuda"),
    )
    loss = torch.mean(chunked_loss)
    loss.backward()

    clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_norm)

    optimizer.step()
    scheduler.step()

    def is_kernel(name, param):
        return param.ndim > 1 and not any(
            x in name for x in ["bias", "scale", "pos_embedding", "input_embedding"]
        )

    kernel_params = [p for n, p in model.named_parameters() if is_kernel(n, p)]
    param_norm = float(torch.norm(torch.stack([torch.norm(p.detach()) for p in kernel_params])))
    grad_norm = float(
        torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in kernel_params if p.grad is not None])
        )
    )
    clip_norm = float(clip_norm)

    new_state = dataclasses.replace(state, step=state.step + 1)

    info = {
        "loss": float(loss.detach().cpu()),
        "lr": scheduler.get_last_lr()[0],
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "clip_norm": clip_norm,
    }
    return new_state, info


def main(config: _config.TrainConfig):
    local_rank = init_ddp(config)
    if dist.get_rank() == 0 and local_rank == 0:
        init_logging()
        logging.info(f"Running on: {platform.node()}")
        if config.batch_size % torch.cuda.device_count() != 0:
            raise ValueError(
                f"Batch size {config.batch_size} must be divisible by the number of devices {torch.cuda.device_count()}."
            )
        checkpoint_dir, resuming = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=config.ckpt_overwrite,
            resume=config.resume,
        )

        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    ENERGON_DATA_PATH = os.getenv('ENERGON_DATA_PATH')
    ds = get_train_dataset(
        ENERGON_DATA_PATH,
        batch_size=1,
        shuffle_buffer_size=0,
        max_samples_per_sequence=100,
        worker_config=WorkerConfig.default_worker_config(),
        task_encoder=TaskEncoder(config),
    )
    loader = get_loader(ds)
    data_iter = iter(loader)
    batch = next(data_iter)

    train_state = init_train_state(config, resume=config.resume, local_rank=local_rank)
    torch.cuda.synchronize()

    if config.resume:
        train_state = _checkpoints.restore_state(config.checkpoint_dir, train_state)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        train_state, info = train_step(config, train_state, batch)
        infos.append(info)
        if dist.get_rank() == 0 and local_rank == 0 and step % config.log_interval == 0:
            stacked_infos = {k: np.stack([info[k] for info in infos], axis=0) for k in infos[0]}
            reduced_info = {k: float(v.mean()) for k, v in stacked_infos.items()}
            info_str = ", ".join(f"{k}={v:.6f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (
            dist.get_rank() == 0
            and local_rank == 0
            and (step % config.save_interval == 0 and step > start_step)
            or step == config.num_train_steps - 1
        ):
            torch.save(
                train_state.model.state_dict(),
                os.path.join(config.checkpoint_dir, f"model_step_{step}.pt"),
            )
        dist.barrier()


if __name__ == "__main__":
    main(_config.cli())
