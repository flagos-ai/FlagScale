# Copyright (c) 2024, FlagScale CORPORATION. All rights reserved.
import argparse
import os

import torch


def convert(input_path, output_path, tensor_parallel_size):
    state_dict = torch.load(input_path, weights_only=False)

    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    for name, tensor in state_dict.items():
        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        chunk_dim = None

        # This is used for chunking some tensors to target tensor parallel size.
        if name == "model.mm_projector.0.weight":
            new_name = "encoder.linear_fc1.weight"
            chunk_dim = 0
        elif name == "model.mm_projector.0.bias":
            new_name = "encoder.linear_fc1.bias"
            chunk_dim = 0
        elif name == "model.mm_projector.2.weight":
            new_name = "encoder.linear_fc2.weight"
            chunk_dim = 1
        elif name == "model.mm_projector.2.bias":
            new_name = "encoder.linear_fc2.bias"

        assert new_name != "", f"unexpected name {name}"

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp, exist_ok=True)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert LLaVA MLP weights to megatron format.


Example usage:
python mlp_converter.py --input /some/input/folder/mm_projector.bin --output /some/output/folder --tensor-parallel-size 2
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True, help="The mlp weights with hf format")
    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )

    args = parser.parse_args()

    convert(args.input, args.output, args.tensor_parallel_size)

    print("done.")
