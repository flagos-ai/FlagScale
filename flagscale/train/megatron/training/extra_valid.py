import math

import torch

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.training.datasets.data_samplers import build_pretraining_data_loader
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.global_vars import get_tensorboard_writer, get_wandb_writer, get_extra_valid_datasets, set_extra_valid_datasets
from megatron.training.utils import is_last_rank, print_rank_last


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args, data_path):
    tokenizer = get_tokenizer()

    # Only build the validation dataset
    assert data_path is not None, "Please provide a valid data_path for extra validation dataset."
    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(data_path),
        blend_per_split=None,
        split="0,1,0",
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def extra_valid_datasets_provider(data_path, num_samples):
    """Build the train test and extra_validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and extra_validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args, data_path)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0(f"> building extra validation dataset ({data_path}, {num_samples}) for GPT ...")

    extra_train_ds, extra_valid_ds, extra_test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, [0, num_samples, 0], is_dataset_built_on_rank, config
    ).build()

    assert (
        extra_train_ds is None and extra_test_ds is None
    ), "train_ds and test_ds should be None for extra_valid_ds"

    print_rank_0("> finished creating GPT datasets ...")

    return extra_valid_ds


def build_extra_valid_datasets(build_extra_valid_dataset_provider):
    """Build extra_valid datasets."""

    args = get_args()

    if args.extra_valid_data_path is None:
        return [None]

    assert (
        len(args.extra_valid_data_path) % 2 == 0
    ), "extra_valid_data_path format should be a list of weight, prefix and tag."

    blend = args.extra_valid_data_path
    raw_num_tokens_per_dataset, raw_prefix_paths_per_dataset = zip(
        *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
    )

    num_samples_per_dataset = []
    valid_iters_per_dataset = []
    for rntpd in raw_num_tokens_per_dataset:
        try:
            num_tokens = int(rntpd)
        except ValueError:
            raise ValueError(f"Number of tokens {rntpd} is error.")

        assert num_tokens > 0, f"Number of tokens {num_tokens} should be greater than 0"
        # Make sure that the number of samples is a multiple of the sequence length
        num_samples = (num_tokens + args.seq_length - 1) // args.seq_length
        # Make sure that the number of samples is a multiple of the global batch size.
        eval_iters = (num_samples + args.global_batch_size - 1) // args.global_batch_size
        num_samples = eval_iters * args.global_batch_size
        num_samples_per_dataset.append(num_samples)
        valid_iters_per_dataset.append(eval_iters)

    args.extra_eval_iters_list = valid_iters_per_dataset
    args.extra_prefix_paths_list = raw_prefix_paths_per_dataset
    args.extra_num_samples_list = num_samples_per_dataset

    assert len(raw_prefix_paths_per_dataset) == len(
        num_samples_per_dataset
    ), f"Number of extra_valid data paths {len(raw_prefix_paths_per_dataset)} does not match number of extra_valid data samples {len(num_samples_per_dataset)}"

    extra_valid_datasets = []
    for path, num_samples in zip(raw_prefix_paths_per_dataset, num_samples_per_dataset):
        extra_valid_datasets.append(build_extra_valid_dataset_provider([path], num_samples))

    return extra_valid_datasets


def build_extra_valid_data_loaders(build_extra_valid_dataset_provider):
    """Build extra_valid data loaders."""

    args = get_args()

    extra_valid_dataloaders = [None]

    print_rank_0("> building extra validation datasets ...")
    print_rank_0("> extra validation consumed_samples is always 0.")

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_extra_valid_dataset_provider, "is_distributed", False)

    # Construct the data pipeline
    if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

        # Build datasets if necessary.
        if get_extra_valid_datasets() is None:
            extra_valid_datasets = build_extra_valid_datasets(build_extra_valid_dataset_provider)
            set_extra_valid_datasets(extra_valid_datasets)
        else:
            extra_valid_datasets = get_extra_valid_datasets()

        # Build dataloders.
        extra_valid_dataloaders = []
        for extra_valid_ds in extra_valid_datasets:
            extra_valid_dataloaders.append(build_pretraining_data_loader(extra_valid_ds, 0))

        # Flags to know if we need to do extra_validation.
        is_none = map(lambda _: _ is None, extra_valid_dataloaders)
        do_extra_valid = len(extra_valid_dataloaders) > 0 and not any(is_none)
        flags = torch.tensor([int(do_extra_valid)], dtype=torch.long, device="cuda")
    else:
        flags = torch.tensor([0], dtype=torch.long, device="cuda")

    torch.distributed.broadcast(flags, 0)

    args.do_extra_valid = getattr(args, "do_extra_valid", False) or flags[0].item()

    return extra_valid_dataloaders


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def build_extra_valid_data_iterators(build_extra_valid_dataset_provider):
    """Build pretraining data iterators."""
    if build_extra_valid_dataset_provider is None:
        return None

    args = get_args()

    # Build loaders.
    extra_valid_dataloaders = build_extra_valid_data_loaders(build_extra_valid_dataset_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ["single", "cyclic", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader]
            else:
                return RerunDataIterator(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    if extra_valid_dataloaders[0] is not None:
        extra_valid_data_iterators = [
            _get_iterator(dl_type, extra_valid_dataloader)
            for extra_valid_dataloader in extra_valid_dataloaders
        ]
    else:
        extra_valid_data_iterators = [None for _ in extra_valid_dataloaders]

    return extra_valid_data_iterators


def extra_evaluate_and_print_results(
    index,
    prefix,
    forward_step_func,
    data_iterator,
    model,
    iteration,
    process_non_loss_data_func,
    config,
    verbose=False,
    write_to_tensorboard=True,
    non_loss_data_func=None,
):
    """Helper function to evaluate and dump results on screen."""
    args = get_args()
    if write_to_tensorboard:
        writer = get_tensorboard_writer()
    else:
        writer = None

    wandb_writer = get_wandb_writer()

    from megatron.training.training import evaluate  # To avoid the circular import

    total_loss_dict, collected_non_loss_data, timelimit = evaluate(
        forward_step_func,
        data_iterator,
        model,
        process_non_loss_data_func,
        config,
        verbose,
        non_loss_data_func,
        index,
    )

    # Timelimit hit during evaluation
    if timelimit:
        return

    label = ""
    extra_prefix_paths_list = getattr(args, "extra_prefix_paths_list", None)
    if extra_prefix_paths_list:
        label = f"{extra_prefix_paths_list[index]}"

    comsumed_samples = ""
    extra_num_samples_list = getattr(args, "extra_num_samples_list", None)
    if extra_num_samples_list:
        comsumed_samples = extra_num_samples_list[index]

    string = f" extra validation loss at {prefix} {label} | "
    string += f"consumed samples: {comsumed_samples} | "
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar(
                "{} validation {}".format(key, label), total_loss_dict[key].item(), iteration
            )
            writer.add_scalar(
                "{} validation {} vs samples".format(key, label),
                total_loss_dict[key].item(),
                args.consumed_train_samples,
            )
            if args.log_validation_ppl_to_tensorboard:
                writer.add_scalar("{} validation {} ppl".format(key, label), ppl, iteration)
                writer.add_scalar(
                    "{} validation {} ppl vs samples".format(key, label),
                    ppl,
                    args.consumed_train_samples,
                )
            if wandb_writer and is_last_rank():
                wandb_writer.log(
                    {"{} validation {}".format(key, label): total_loss_dict[key].item()}, iteration
                )
                wandb_writer.log(
                    {"{} validation {} vs samples".format(key, label): args.consumed_train_samples},
                    iteration,
                )
                wandb_writer.log(
                    {"validation ppl/{} validation {} ppl".format(key, label): ppl}, iteration
                )
                wandb_writer.log(
                    {
                        "validation loss/{} validation {}".format(key, label): total_loss_dict[
                            key
                        ].item()
                    },
                    iteration,
                )

    if process_non_loss_data_func is not None and writer and is_last_rank():
        process_non_loss_data_func(collected_non_loss_data, iteration, writer)

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)
