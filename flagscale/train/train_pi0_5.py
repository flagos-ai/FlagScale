import argparse
import os
import pathlib
import platform
import random

import etils.epath as epath
import numpy as np
import torch
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

from megatron.energon import WorkerConfig, get_loader, get_train_dataset
from tools.datasets.vla.data.dataset_helpers import TaskEncoder

from flagscale.models.pi0.modeling_pi0_5 import PI0_5_Policy, PI0_5_PolicyConfig
from flagscale.runner.utils import logger


def init_ddp(config):
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


def init_wandb(config, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = pathlib.Path(config.output_directory)
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


def main(config):
    device = torch.device("cuda")

    local_rank = init_ddp(config)
    if dist.get_rank() == 0 and local_rank == 0:
        logger.info(f"Running on: {platform.node()}")
        if config.batch_size % torch.cuda.device_count() != 0:
            raise ValueError(
                f"Batch size {config.batch_size} must be divisible by the number of devices {torch.cuda.device_count()}."
            )
        resuming = config.resume
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    ds = get_train_dataset(
        config.data_path,
        batch_size=config.batch_size,
        shuffle_buffer_size=10000,
        max_samples_per_sequence=100,
        worker_config=WorkerConfig.default_worker_config(num_workers=1, data_parallel_group=None),
        task_encoder=TaskEncoder(config),
        repeat=True,
    )
    loader = get_loader(ds)
    data_iter = iter(loader)

    # Load Pi0.5 config
    model_config = PI0_5_PolicyConfig.from_pretrained(config.checkpoint_dir)

    # Configure Pi0.5 specific parameters
    model_config.n_action_steps = config.action_steps
    model_config.tokenizer_max_length = config.tokenizer_max_length
    model_config.action_dim = getattr(config, 'action_dim', 32)
    model_config.action_horizon = getattr(config, 'action_horizon', 16)
    model_config.pi05 = getattr(config, 'pi05', True)
    model_config.discrete_state_input = getattr(config, 'discrete_state_input', True)
    model_config.max_token_len = getattr(config, 'max_token_len', 200)

    policy = PI0_5_Policy.from_pretrained(
        model_path=config.checkpoint_dir,
        tokenizer_path=config.tokenizer_path,
        stat_path=config.stat_path,
        config=model_config,
    )
    policy = policy.cuda()
    policy = DDP(policy, device_ids=[int(os.environ["LOCAL_RANK"])], find_unused_parameters=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    step = 0
    done = False
    while not done:
        batch = next(data_iter)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % config.log_freq == 0:
            logger.info(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= config.train_steps:
            done = True
            break
    if dist.get_rank() == 0 and local_rank == 0:
        policy.module.save_pretrained(config.output_directory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="/share/pi0_5/pi05_base")
    parser.add_argument("--project-name", type=str, default="pi05_project")
    parser.add_argument("--exp-name", type=str, default="pi05_exp")
    parser.add_argument("--data-path", type=str, default="/share/pi05_dataset/wds-2")
    parser.add_argument("--tokenizer-path", type=str, default="/share/paligemma-3b-pt-224")
    parser.add_argument("--state-key", type=str, default="observation.state")
    parser.add_argument("--action-key", type=str, default="action")
    parser.add_argument("--action-token-key", type=str, default="action_token")
    parser.add_argument("--stat-path", type=str, default="/share/pi05_dataset/stats.json")
    parser.add_argument("--output-directory", type=str, default="output_directory_not_set")
    parser.add_argument("--vision-root", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--action-steps", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--tokenizer-max-length", type=int, default=200)
    parser.add_argument("--pi05", action="store_true", default=True)
    parser.add_argument("--discrete-state-input", action="store_true", default=True)
    parser.add_argument("--max-token-len", type=int, default=200)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-overwrite", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")

    config = parser.parse_args()

    logger.info(f"train_pi0_5.py config: {config}")
    main(config)
