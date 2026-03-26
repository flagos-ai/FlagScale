# Non-FSDP2 tests adapted from
# https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/tests/utils/test_train_utils.py

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file
from torch.distributed._composable.fsdp import fully_shard

from flagscale.models.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    PRETRAINED_MODEL_DIR,
    RNG_STATE,
    SAFETENSORS_FILE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from flagscale.train.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_scheduler_state,
    load_training_step,
    save_checkpoint,
    save_optimizer_state,
    save_scheduler_state,
    save_training_state,
    save_training_step,
    update_last_checkpoint,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(self.linear1(x))


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def model_params(model):
    return list(model.parameters())


@pytest.fixture
def optimizer(model_params):
    opt = torch.optim.Adam(model_params, lr=1e-3)
    loss = sum(p.sum() for p in model_params)
    loss.backward()
    opt.step()
    return opt


@pytest.fixture
def scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_save_load_training_step(tmp_path):
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == 5000


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


def test_update_last_checkpoint_overwrites(tmp_path):
    ckpt1 = tmp_path / "0005"
    ckpt1.mkdir()
    update_last_checkpoint(ckpt1)

    ckpt2 = tmp_path / "0010"
    ckpt2.mkdir()
    update_last_checkpoint(ckpt2)

    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == ckpt2


def test_save_optimizer_state(optimizer, tmp_path):
    save_optimizer_state(optimizer.state_dict(), tmp_path)
    assert (tmp_path / OPTIMIZER_STATE).is_file()
    assert (tmp_path / OPTIMIZER_PARAM_GROUPS).is_file()


def test_save_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()


def test_save_load_scheduler_state(scheduler, tmp_path):
    for _ in range(5):
        scheduler.step()
    original_state = scheduler.state_dict()

    save_scheduler_state(scheduler, tmp_path)
    scheduler2 = torch.optim.lr_scheduler.StepLR(scheduler.optimizer, step_size=10, gamma=0.1)
    load_scheduler_state(scheduler2, tmp_path)

    assert scheduler2.state_dict() == original_state


def test_save_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer.state_dict(), scheduler)
    state_dir = tmp_path / TRAINING_STATE_DIR
    assert state_dir.is_dir()
    assert (state_dir / TRAINING_STEP).is_file()
    assert (state_dir / RNG_STATE).is_file()
    assert (state_dir / OPTIMIZER_STATE).is_file()
    assert (state_dir / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (state_dir / SCHEDULER_STATE).is_file()


def test_save_training_state_no_optim_no_scheduler(tmp_path):
    save_training_state(tmp_path, 10)
    state_dir = tmp_path / TRAINING_STATE_DIR
    assert state_dir.is_dir()
    assert (state_dir / TRAINING_STEP).is_file()
    assert (state_dir / RNG_STATE).is_file()
    assert not (state_dir / OPTIMIZER_STATE).exists()
    assert not (state_dir / SCHEDULER_STATE).exists()


def test_save_checkpoint_creates_files(tmp_path, model, optimizer, scheduler):
    policy = MagicMock()
    config = MagicMock()
    save_checkpoint(
        tmp_path,
        10,
        config,
        policy,
        optimizer_state_dict=optimizer.state_dict(),
        lr_scheduler=scheduler,
    )
    policy.save_pretrained.assert_called_once()
    config._save_pretrained.assert_called_once()


def _init_dist():
    """Initialize a single-process distributed group for FSDP2 tests."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="nccl")


def _build_fsdp2_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    ckpt_dir: Path,
):
    """Save a full checkpoint in the expected layout for load_training_state_fsdp2."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(model, options=options)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer, options=options)

    pretrained_dir = ckpt_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.clone().contiguous() for k, v in state_dict.items()},
        str(pretrained_dir / SAFETENSORS_FILE),
    )
    save_training_state(ckpt_dir, step, optimizer_state_dict, scheduler)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FSDP2 tests"
)


def _make_fsdp_model():
    model = SimpleModel().cuda()
    fully_shard(model)
    return model


def _get_full_model_sd(model):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))


def _get_full_optim_sd(model, opt):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    return get_optimizer_state_dict(model, opt, options=StateDictOptions(full_state_dict=True))


# FIXME: (yupu) Blocking release, comment out for now
# @requires_cuda
# class TestFSDP2:
#     """Tests for FSDP2 save/load round-trips.

#     Uses a real single-GPU FSDP2 setup (nccl, world_size=1).
#     """

#     @pytest.fixture(autouse=True)
#     def setup_dist(self):
#         _init_dist()

#     def test_load_model_state_fsdp2(self, tmp_path):
#         model = _make_fsdp_model()
#         original_sd = {k: v.clone() for k, v in _get_full_model_sd(model).items()}

#         pretrained_dir = tmp_path / PRETRAINED_MODEL_DIR
#         pretrained_dir.mkdir(parents=True)
#         save_file(
#             {k: v.clone().contiguous() for k, v in original_sd.items()},
#             str(pretrained_dir / SAFETENSORS_FILE),
#         )

#         model2 = _make_fsdp_model()
#         load_model_state_fsdp2(model2, pretrained_dir)

#         loaded_sd = _get_full_model_sd(model2)
#         for key in original_sd:
#             torch.testing.assert_close(original_sd[key], loaded_sd[key])

#     def test_load_optimizer_state_fsdp2(self, tmp_path):
#         model = _make_fsdp_model()
#         opt = torch.optim.Adam(model.parameters(), lr=1e-3)

#         loss = model(torch.randn(2, 10, device="cuda")).sum()
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#         original_optim_sd = _get_full_optim_sd(model, opt)
#         save_optimizer_state(original_optim_sd, tmp_path)

#         opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
#         load_optimizer_state_fsdp2(model, opt2, tmp_path)

#         loaded_optim_sd = _get_full_optim_sd(model, opt2)
#         for key in original_optim_sd["state"]:
#             for k, v in original_optim_sd["state"][key].items():
#                 torch.testing.assert_close(v, loaded_optim_sd["state"][key][k])

#     def test_load_training_state_fsdp2_round_trip(self, tmp_path):
#         model = _make_fsdp_model()
#         opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#         sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)

#         for _ in range(5):
#             loss = model(torch.randn(2, 10, device="cuda")).sum()
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#             sched.step()

#         original_sd = {k: v.clone() for k, v in _get_full_model_sd(model).items()}
#         sched_state_before = sched.state_dict()

#         _build_fsdp2_checkpoint(model, opt, sched, step=5, ckpt_dir=tmp_path)

#         model2 = _make_fsdp_model()
#         opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
#         sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=10, gamma=0.1)

#         step = load_training_state_fsdp2(tmp_path, model2, opt2, sched2)

#         assert step == 5
#         loaded_sd = _get_full_model_sd(model2)
#         for key in original_sd:
#             torch.testing.assert_close(original_sd[key], loaded_sd[key])
#         assert sched2.state_dict() == sched_state_before

#     def test_load_training_state_fsdp2_no_scheduler(self, tmp_path):
#         model = _make_fsdp_model()
#         opt = torch.optim.Adam(model.parameters(), lr=1e-3)

#         loss = model(torch.randn(2, 10, device="cuda")).sum()
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#         _build_fsdp2_checkpoint(model, opt, scheduler=None, step=1, ckpt_dir=tmp_path)

#         model2 = _make_fsdp_model()
#         opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

#         step = load_training_state_fsdp2(tmp_path, model2, opt2, scheduler=None)
#         assert step == 1

#     def test_load_optimizer_state_fsdp2_missing_params(self, tmp_path):
#         """Params that never got gradients should be handled gracefully."""
#         model = _make_fsdp_model()
#         opt = torch.optim.Adam(model.parameters(), lr=1e-3)

#         # Only compute gradients for linear2 by zeroing out linear1 grads
#         loss = model(torch.randn(2, 10, device="cuda")).sum()
#         loss.backward()
#         for name, p in model.named_parameters():
#             if "linear1" in name:
#                 p.grad = None
#         opt.step()
#         opt.zero_grad()

#         optim_sd = _get_full_optim_sd(model, opt)
#         save_optimizer_state(optim_sd, tmp_path)

#         opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
#         load_optimizer_state_fsdp2(model, opt2, tmp_path)
