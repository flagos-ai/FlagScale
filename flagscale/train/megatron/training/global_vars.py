# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import os
import signal
import sys
import torch

from datetime import timedelta

from megatron.core import Timers
from megatron.core.config import set_experimental_flag
from megatron.core.energy_monitor import EnergyMonitor
from megatron.core.jit import disable_jit_fuser
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.training.tokenizer import build_tokenizer  ########## FlagScale Add ##########
from megatron.training.dist_signal_handler import DistributedSignalHandler

########## FlagScale Begin ##########
from megatron.training.spiky_loss import SpikyLossDetector
from megatron.plugin.utils import get_device_type_for_comm
########## FlagScale End ##########

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None
_GLOBAL_ONE_LOGGER = None
_GLOBAL_ADLR_AUTORESUME = None
_GLOBAL_TIMERS = None
_GLOBAL_ENERGY_MONITOR = None
_GLOBAL_SIGNAL_HANDLER = None

########## FlagScale Begin ##########
_GLOBAL_SPIKY_LOSS_DETECTOR = None
_GLOBAL_EXTRA_VALID_DATASETS = None
########## FlagScale End ##########


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    return _GLOBAL_ARGS


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    return _GLOBAL_TOKENIZER


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def get_wandb_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_WANDB_WRITER


def get_one_logger():
    """Return one logger. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ONE_LOGGER


def get_adlr_autoresume():
    """ADLR autoresume object. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_ADLR_AUTORESUME


def get_timers():
    """Return timers."""
    _ensure_var_is_initialized(_GLOBAL_TIMERS, "timers")
    return _GLOBAL_TIMERS


def get_energy_monitor():
    """Return energy monitor."""
    _ensure_var_is_initialized(_GLOBAL_ENERGY_MONITOR, "energy monitor")
    return _GLOBAL_ENERGY_MONITOR


def get_signal_handler():
    _ensure_var_is_initialized(_GLOBAL_SIGNAL_HANDLER, "signal handler")
    return _GLOBAL_SIGNAL_HANDLER


def _set_signal_handler(exit_signal):
    global _GLOBAL_SIGNAL_HANDLER
    _ensure_var_is_not_initialized(_GLOBAL_SIGNAL_HANDLER, "signal handler")
    _GLOBAL_SIGNAL_HANDLER = DistributedSignalHandler(exit_signal).__enter__()


def _graceful_shutdown(signum, frame):
    """
    Signal handler for user-initiated termination (SIGINT / SIGTERM).

    This handler attempts a best-effort graceful shutdown:
      - Logs a single termination message from rank 0
      - Synchronizes all ranks (barrier)
      - Destroys the distributed process group
      - Exits the process cleanly
    """
    from megatron.training.utils import print_rank_0

    print_rank_0("\nTermination requested. Performing orderly shutdown.")

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # synchronize all ranks before exiting
            try:
                # avoid deadlock if ranks don't all reach here
                torch.distributed.barrier(timeout=timedelta(seconds=5))
            except Exception:
                pass

            torch.distributed.destroy_process_group()
    except Exception:
        pass

    sys.exit(0)


def set_global_variables(args, build_tokenizer=True):
    """Set args, tokenizer, tensorboard-writer, adlr-autoresume, and timers."""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
    set_args(args)

    init_num_microbatches_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
        args.decrease_batch_size_if_needed,
    )
    if build_tokenizer:
        _ = _build_tokenizer(args)
    _set_adlr_autoresume(args)
    _set_timers(args)
    _set_energy_monitor(args)

    if args.enable_experimental:
        set_experimental_flag(True)

    if args.exit_signal_handler:
        _set_signal_handler(args.exit_signal)

    if args.exit_signal_handler_for_training:
        signal.signal(signal.SIGINT, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)

    if args.disable_jit_fuser:
        disable_jit_fuser()


def set_global_writers(args):
    """Set tensorboard-writer and wandb writer.

    Note that this function should be called after calling finish_mpu_init.
    This is because we can know which rank is the last one after the rank mapping in finish_mpu_init.
    """
    from .utils import is_last_rank

    assert args is not None

    _ensure_var_is_initialized(_GLOBAL_ARGS, "args")
    _set_tensorboard_writer(args)
    _set_one_logger(args)

    # build wandb writers for all processes in the dp group of the last rank
    from megatron.core import mpu

    mp_groups = mpu.get_model_parallel_group()
    if not isinstance(mp_groups, list):
        mp_groups = [mp_groups]
    size = torch.distributed.get_world_size(mp_groups[-1])
    comm_device = get_device_type_for_comm(mp_groups)
    ranks_tensor = torch.tensor([0 for _ in range(size)], dtype=torch.int, device=comm_device)
    orig_ranks = torch.tensor([i for i in range(size)], dtype=torch.int, device=comm_device)
    if is_last_rank():
        ranks_list = torch.distributed.get_process_group_ranks(mp_groups[-1])
        ranks_tensor = torch.tensor(ranks_list, dtype=torch.int, device=comm_device)
    orig_ranks = ranks_tensor.clone().detach()
    for group in mp_groups:
        ranks_tensor = orig_ranks.clone()
        torch.distributed.all_reduce(ranks_tensor, group=group)
    if torch.distributed.get_rank() in ranks_tensor.tolist():
        _set_wandb_writer(args)


def unset_global_variables():
    """Unset global vars.

    Useful for multiple runs. See `tests/unit_tests/ckpt_converter/test_ckpt_converter.py` for an example.
    """

    global _GLOBAL_ARGS
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    global _GLOBAL_TOKENIZER
    global _GLOBAL_TENSORBOARD_WRITER
    global _GLOBAL_WANDB_WRITER
    global _GLOBAL_ONE_LOGGER
    global _GLOBAL_ADLR_AUTORESUME
    global _GLOBAL_TIMERS
    global _GLOBAL_ENERGY_MONITOR
    global _GLOBAL_SIGNAL_HANDLER

    _GLOBAL_ARGS = None
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
    _GLOBAL_TOKENIZER = None
    _GLOBAL_TENSORBOARD_WRITER = None
    _GLOBAL_WANDB_WRITER = None
    _GLOBAL_ONE_LOGGER = None
    _GLOBAL_ADLR_AUTORESUME = None
    _GLOBAL_TIMERS = None
    _GLOBAL_ENERGY_MONITOR = None
    _GLOBAL_SIGNAL_HANDLER = None

    unset_num_microbatches_calculator()


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, "tokenizer")
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def rebuild_tokenizer(args):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None
    return _build_tokenizer(args)


def _set_tensorboard_writer(args):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, "tensorboard writer")

    if (
        hasattr(args, "tensorboard_dir")
        and args.tensorboard_dir
        and args.rank == (args.world_size - 1)
    ):
        try:
            from torch.utils.tensorboard import SummaryWriter

            print("> setting tensorboard ...")
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=args.tensorboard_dir, max_queue=args.tensorboard_queue_size
            )
        except ModuleNotFoundError:
            print(
                "WARNING: TensorBoard writing requested but is not "
                "available (are you using PyTorch 1.1.0 or later?), "
                "no TensorBoard logs will be written.",
                flush=True,
            )


def _set_wandb_writer(args):
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER, "wandb writer")
    if getattr(args, "wandb_project", ""):
        if args.wandb_exp_name == "":
            raise ValueError("Please specify the wandb experiment name!")

        import wandb

        if args.wandb_save_dir:
            save_dir = args.wandb_save_dir
        else:
            # Defaults to the save dir.
            save_dir = os.path.join(args.save, "wandb")
        wandb_config = vars(args)
        if (
            "kitchen_config_file" in wandb_config
            and wandb_config["kitchen_config_file"] is not None
        ):
            # Log the contents of the config for discovery of what the quantization
            # settings were.
            with open(wandb_config["kitchen_config_file"], "r") as f:
                wandb_config["kitchen_config_file_contents"] = f.read()
        rank = torch.distributed.get_rank()
        save_dir = os.path.join(save_dir, "rank-{}".format(rank))
        wandb_id = f"{args.wandb_exp_name}-rank-{rank}"
        name = f"{args.wandb_exp_name}-rank-{rank}"
        group = args.wandb_exp_name
        wandb_kwargs = {
            "id": wandb_id,
            "dir": save_dir,
            "name": name,
            "group": group,
            "project": args.wandb_project,
            "mode": args.wandb_mode,
            "resume": "auto",
            "config": wandb_config,
        }
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        os.makedirs(wandb_kwargs["dir"], exist_ok=True)
        if args.wandb_mode == "online" or args.wandb_api_key:
            assert args.wandb_api_key, "wandb_api_key is required for online mode"
            wandb.login(key=args.wandb_api_key)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _set_one_logger(args):
    global _GLOBAL_ONE_LOGGER
    _ensure_var_is_not_initialized(_GLOBAL_ONE_LOGGER, "one logger")

    if args.enable_one_logger and args.rank == (args.world_size - 1):
        if args.one_logger_async or getattr(args, "wandb_project", ""):
            one_logger_async = True
        else:
            one_logger_async = False
        try:
            from one_logger import OneLogger

            config = {
                "project": args.one_logger_project,
                "name": args.one_logger_run_name,
                "async": one_logger_async,
            }
            one_logger = OneLogger(config=config)
            _GLOBAL_ONE_LOGGER = one_logger
        except Exception:
            print(
                "WARNING: one_logger package is required to enable e2e metrics "
                "tracking. please go to "
                "https://confluence.nvidia.com/display/MLWFO/Package+Repositories"
                " for details to install it"
            )


def _set_adlr_autoresume(args):
    """Initialize ADLR autoresume."""
    global _GLOBAL_ADLR_AUTORESUME
    _ensure_var_is_not_initialized(_GLOBAL_ADLR_AUTORESUME, "adlr autoresume")

    if args.adlr_autoresume:
        from megatron.training.utils import print_rank_0

        print_rank_0("enabling autoresume ...")
        sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
        try:
            from userlib.auto_resume import AutoResume
        except ImportError:
            print_rank_0("ADLR autoresume is not available, exiting ...")
            sys.exit()

        _GLOBAL_ADLR_AUTORESUME = AutoResume


def _set_timers(args):
    """Initialize timers."""
    global _GLOBAL_TIMERS
    _ensure_var_is_not_initialized(_GLOBAL_TIMERS, "timers")
    _GLOBAL_TIMERS = Timers(args.timing_log_level, args.timing_log_option)


def _set_energy_monitor(args):
    """Initialize energy monitor."""
    global _GLOBAL_ENERGY_MONITOR
    _ensure_var_is_not_initialized(_GLOBAL_ENERGY_MONITOR, "energy monitor")
    _GLOBAL_ENERGY_MONITOR = EnergyMonitor()


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


def destroy_global_vars():
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = None

    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = None

    global _GLOBAL_TENSORBOARD_WRITER
    _GLOBAL_TENSORBOARD_WRITER = None

    global _GLOBAL_WANDB_WRITER
    _GLOBAL_WANDB_WRITER = None

    global _GLOBAL_ONE_LOGGER
    _GLOBAL_ONE_LOGGER = None

    global _GLOBAL_ADLR_AUTORESUME
    _GLOBAL_ADLR_AUTORESUME = None

    global _GLOBAL_TIMERS
    _GLOBAL_TIMERS = None

    global _GLOBAL_ENERGY_MONITOR
    _GLOBAL_ENERGY_MONITOR = None

    global _GLOBAL_SIGNAL_HANDLER
    _GLOBAL_SIGNAL_HANDLER = None


########## FlagScale Begin ##########
def get_spiky_loss_detector():
    """Return spiky loss detector."""
    _ensure_var_is_initialized(_GLOBAL_SPIKY_LOSS_DETECTOR, "spiky loss detector")
    return _GLOBAL_SPIKY_LOSS_DETECTOR


def set_spiky_loss_detector(args):
    """Initialize spiky loss detector."""
    global _GLOBAL_SPIKY_LOSS_DETECTOR
    _ensure_var_is_not_initialized(_GLOBAL_SPIKY_LOSS_DETECTOR, "spiky loss detector")
    _GLOBAL_SPIKY_LOSS_DETECTOR = SpikyLossDetector(args.spiky_loss_threshold)


def get_extra_valid_datasets():
    """Return extra_valid datasets.""" ""
    return _GLOBAL_EXTRA_VALID_DATASETS


def set_extra_valid_datasets(extra_valid_datasets):
    """Set extra_valid datasets.""" ""
    global _GLOBAL_EXTRA_VALID_DATASETS
    _GLOBAL_EXTRA_VALID_DATASETS = extra_valid_datasets


########## FlagScale End ##########
