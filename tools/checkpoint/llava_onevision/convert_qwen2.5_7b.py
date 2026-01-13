# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import json
import os

import torch
from safetensors.torch import load_file


def convert(input_path, output_path, tensor_parallel_size, use_te):
    device = "cuda"
    # index.json
    index_path = None
    for file in os.listdir(input_path):
        if file.endswith("index.json"):
            index_path = os.path.join(input_path, file)
            break
    assert index_path is not None, "index.json not found in input path"

    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    caches = {}
    for name in weight_map:
        file_name = weight_map[name]
        if file_name not in caches:
            caches[file_name] = load_file(os.path.join(input_path, file_name), device=device)

    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    hidden_dim = 3584
    num_heads = 28
    assert hidden_dim % num_heads == 0
    kv_channels = hidden_dim // num_heads
    # GQA Process
    num_query_groups = 4
    kv_projection_size = kv_channels * num_query_groups
    indices = []
    assert num_heads % num_query_groups == 0
    for i in range(num_query_groups):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(
            torch.arange(
                num_heads // num_query_groups * kv_channels * i,
                num_heads // num_query_groups * kv_channels * (i + 1),
                dtype=torch.int,
            )
        )
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(
            torch.arange(
                (hidden_dim + kv_projection_size) + lb,
                (hidden_dim + kv_projection_size) + ub,
                dtype=torch.int,
            )
        )

    indices = torch.cat(indices)

    gate_up_indices = []
    ffn_hidden_size = 18944
    assert ffn_hidden_size % tensor_parallel_size == 0
    interval = ffn_hidden_size // tensor_parallel_size
    for i in range(tensor_parallel_size):
        lb = i * interval
        ub = (i + 1) * interval
        gate_up_indices.append(torch.arange(lb, ub, dtype=torch.int))
        gate_up_indices.append(
            torch.arange(ffn_hidden_size + lb, ffn_hidden_size + ub, dtype=torch.int)
        )
    gate_up_indices = torch.cat(gate_up_indices)

    for name in weight_map:
        file_name = weight_map[name]
        tensor = caches[file_name][name]

        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        qkv_params = set()
        gate_up_params = set()
        if "embed_tokens.weight" in name:
            new_name = "embedding.word_embeddings.weight"
            chunk_dim = 0
        elif "lm_head.weight" in name:
            new_name = "output_layer.weight"
            chunk_dim = 0
        # the norm after last layer
        elif "norm.weight" in name and "layernorm.weight" not in name:
            new_name = "decoder.final_layernorm.weight"
        elif "model.layers" in name:
            layer_idx = name.split(".")[2]
            base = f"decoder.layers.{layer_idx}"
            if (
                "self_attn.q_proj.weight" in name
                or "self_attn.k_proj.weight" in name
                or "self_attn.v_proj.weight" in name
            ):
                new_name = f"{base}.self_attention.linear_qkv.weight"
                if new_name not in qkv_params:
                    # q_proj, k_proj, v_proj
                    split_name = name.split(".")
                    split_name[-2] = "q_proj"
                    q_name = ".".join(split_name)
                    file_name = weight_map[q_name]
                    q_tensor = caches[file_name][q_name]

                    split_name[-2] = "k_proj"
                    k_name = ".".join(split_name)
                    file_name = weight_map[k_name]
                    k_tensor = caches[file_name][k_name]

                    split_name[-2] = "v_proj"
                    v_name = ".".join(split_name)
                    file_name = weight_map[v_name]
                    v_tensor = caches[file_name][v_name]

                    # concat and dim = 0
                    # q,k,v concat in the first dim
                    new_tensor = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

                    # reorder
                    new_tensor = new_tensor[indices]

                    chunk_dim = 0
                    qkv_params.add(new_name)
                else:
                    continue

            elif (
                "self_attn.q_proj.bias" in name
                or "self_attn.k_proj.bias" in name
                or "self_attn.v_proj.bias" in name
            ):
                new_name = f"{base}.self_attention.linear_qkv.bias"
                if new_name not in qkv_params:
                    # q_proj, k_proj, v_proj
                    split_name = name.split(".")
                    split_name[-2] = "q_proj"
                    q_name = ".".join(split_name)
                    file_name = weight_map[q_name]
                    q_tensor = caches[file_name][q_name]

                    split_name[-2] = "k_proj"
                    k_name = ".".join(split_name)
                    file_name = weight_map[k_name]
                    k_tensor = caches[file_name][k_name]

                    split_name[-2] = "v_proj"
                    v_name = ".".join(split_name)
                    file_name = weight_map[v_name]
                    v_tensor = caches[file_name][v_name]

                    # concat and dim = 0
                    new_tensor = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

                    # reorder
                    new_tensor = new_tensor[indices]

                    chunk_dim = 0
                    qkv_params.add(new_name)
                else:
                    continue
            elif "self_attn.o_proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif "input_layernorm.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif "mlp.gate_proj.weight" in name or "mlp.up_proj.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                if new_name not in gate_up_params:
                    # gate, up
                    split_name = name.split(".")
                    split_name[-2] = "gate_proj"
                    gate_name = ".".join(split_name)
                    file_name = weight_map[gate_name]
                    gate_tensor = caches[file_name][gate_name]

                    split_name = name.split(".")
                    split_name[-2] = "up_proj"
                    up_name = ".".join(split_name)
                    file_name = weight_map[up_name]
                    up_tensor = caches[file_name][up_name]

                    # concat and dim = 0
                    new_tensor = torch.cat([gate_tensor, up_tensor], dim=0)
                    new_tensor = new_tensor[gate_up_indices]
                    gate_up_params.add(new_name)
                    chunk_dim = 0
                else:
                    continue
            elif "mlp.down_proj.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif "post_attention_layernorm.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"

        assert new_name != "", f"unexpected layer name {name}"

        if chunk_dim is None:
            new_tensors = [new_tensor for _ in range(tensor_parallel_size)]
        else:
            new_tensors = torch.chunk(new_tensor, tensor_parallel_size, dim=chunk_dim)

        for i in range(tensor_parallel_size):
            # chunk() creates a view of a bigger tensor. clone() is used here to avoid excessive storage.
            new_state_dicts[i]["model"][new_name] = new_tensors[i].clone()

            # TE sets _extra_state (for FP8 purposes), so set an empty one here for compatibility.
            extra_state_layers = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
            is_extra_state_layer = any([l in new_name for l in extra_state_layers])
            if use_te and is_extra_state_layer:
                layer = new_name.split(".")[-2]
                if layer in extra_state_layers:
                    extra_state_name = (
                        new_name[: new_name.rfind(".") + 1] + "_extra_state"
                    )  # Replace the weight name.
                    new_state_dicts[i]["model"][extra_state_name] = None
        print(f"{new_name} processed")

    for i in range(tensor_parallel_size):
        output_dir_tp = os.path.join(output_path, "iter_0000001", f"mp_rank_0{i}")
        os.makedirs(output_dir_tp)
        output_path_tp = os.path.join(output_dir_tp, "model_optim_rng.pt")
        torch.save(new_state_dicts[i], output_path_tp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert Qwen weights to megatron format.


Example usage:
python qwen_converter.py --input /some/input/folder --output /some/output/folder --tensor-parallel-size 4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True, help="Qwen folder")
    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")

    args = parser.parse_args()

    convert(args.input, args.output, args.tensor_parallel_size, args.use_te)

    print("done.")
