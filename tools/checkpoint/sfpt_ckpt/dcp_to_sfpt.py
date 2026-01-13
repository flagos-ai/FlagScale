import argparse
import os

import torch
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    FileSystemReader,
    Metadata,
    TensorStorageMetadata,
)

from megatron.core.dist_checkpointing import ShardedTensor, load
from megatron.core.dist_checkpointing.mapping import ShardedObject


def build_tensor_shared_state_dict(key, metadata: Metadata = None):
    # Based on load_tensors_metadata from FlagScale/megatron/megatron/core/dist_checkpointing/strategies/torch.py
    mcore_data = getattr(metadata, "mcore_data", {})
    sharded_state_dict = {}
    tp = metadata.state_dict_metadata[key]

    nd_orig_global_shape = mcore_data.get(key, {}).get("nd_reformulated_orig_global_shape")
    if nd_orig_global_shape is None:
        # Regular tensor
        sharded_state_dict[key] = ShardedTensor.from_rank_offsets(
            key, torch.empty(tp.size, **tp.properties.__dict__, device="cpu")
        )
    else:
        # N-D flattened tensor
        unflat_ten = torch.empty(nd_orig_global_shape, **tp.properties.__dict__, device="cpu")
        flat_ten = unflat_ten.flatten()
        sharded_state_dict[key] = ShardedTensor.from_rank_offsets_flat(
            key,
            flat_ten,
            unflat_ten.shape,
            flattened_range=slice(0, unflat_ten.numel()),  # whole slice
        )

    return sharded_state_dict


def build_sharded_state_dict(metadata_key, metadata):
    # Based on load_sharded_metadata from FlagScale/megatron/megatron/core/dist_checkpointing/strategies/torch.py
    storage_metadata = metadata.state_dict_metadata[metadata_key]
    if isinstance(storage_metadata, BytesStorageMetadata):
        sharded_state_dict = {}
        sh_obj = ShardedObject.empty_from_unique_key(metadata_key)
        sharded_state_dict[sh_obj.unique_key] = sh_obj
        return sharded_state_dict
    elif isinstance(storage_metadata, TensorStorageMetadata):
        sharded_state_dict = build_tensor_shared_state_dict(metadata_key, metadata)
        return sharded_state_dict


def convert_dist_ckpt_to_sfpt_ckpt(input_dir, output_dir):
    # Distributed checkpoint loading requires the distributed environment to be initialized
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    print(f"Rank: {rank}, World size: {world_size}")
    torch.distributed.init_process_group(backend="gloo", world_size=world_size, rank=rank)

    fs_reader = FileSystemReader(input_dir)
    metadata = fs_reader.read_metadata()
    state_dict_metadata = metadata.state_dict_metadata
    for metadata_key, storage_metadata in state_dict_metadata.items():
        # Skip optimizer state_dict
        if "optimizer" not in metadata_key and isinstance(storage_metadata, TensorStorageMetadata):
            print(f"Processing {metadata_key}")
            sharded_state_dict = build_sharded_state_dict(metadata_key, metadata)
            loaded_state_dict = load(sharded_state_dict, input_dir)
            sharded_tensor = loaded_state_dict[metadata_key]
            unshared_tensor = sharded_tensor.data
            path = os.path.join(output_dir, metadata_key)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(f"{path}.pt", "wb") as f:
                torch.save({metadata_key: unshared_tensor}, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert distributed checkpoint to single-file-per-tensor checkpoint."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing the distributed checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the single-file-per-tensor checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dist_ckpt_to_sfpt_ckpt(args.input_dir, args.output_dir)
