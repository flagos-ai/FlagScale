import argparse
import os
from argparse import Namespace
from pathlib import Path

import torch

from megatron.core.dist_checkpointing import ShardedTensor, save
from megatron.core.dist_checkpointing.serialization import get_default_save_common_strategy


def convert_sfpt_ckpt_to_dist_ckpt(input_dir, output_dir):
    # Distributed checkpoint loading requires the distributed environment to be initialized
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    print(f"Rank: {rank}, World size: {world_size}")
    torch.distributed.init_process_group(backend="gloo", world_size=world_size, rank=rank)

    input_ckpt_dir = os.path.join(input_dir)
    if not os.path.isdir(input_ckpt_dir):
        raise ValueError(f"Checkpoint directory {input_ckpt_dir} does not exist")

    ckpt_output_dir = os.path.join(output_dir, "iter_0000000")
    if not os.path.exists(ckpt_output_dir):
        os.makedirs(ckpt_output_dir)

    for root, dirs, files in os.walk(input_ckpt_dir):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            state_dict = torch.load(file_path, weights_only=False)
            assert len(state_dict) == 1
            key = next(iter(state_dict.keys()))
            tensor = state_dict[key]
            sharded_state_dict = {}
            sharded_state_dict[key] = ShardedTensor.from_rank_offsets(key, tensor)
            save(sharded_state_dict, ckpt_output_dir)

    # Fake the minimal args for the checkpoint loading processing
    state_dict = {}
    args = Namespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    state_dict["args"] = args
    common_strategy = get_default_save_common_strategy()
    common_strategy.save_common(state_dict, Path(ckpt_output_dir))

    # add the latest_checkpointed_iteration file
    with open(os.path.join(output_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("0")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert single-file-per-tensor checkpoint to distributed checkpoint."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing the single-file-per-tensor checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the distributed checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_sfpt_ckpt_to_dist_ckpt(args.input_dir, args.output_dir)
