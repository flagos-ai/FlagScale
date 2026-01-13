# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import torch
from safetensors.torch import load_file


def convert(input_path, output_path, tensor_parallel_size, use_te):
    device = "cuda"

    state_dict = load_file(input_path, device=device)
    for name, tensor in state_dict.items():
        print(name, tensor.dtype, tensor.shape)
    new_state_dicts = [{"model": dict()} for _ in range(tensor_parallel_size)]

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 72
    hidden_dim = 1152
    num_heads = 16
    indices = []
    for i in range(num_heads):
        lb = i * kv_channels
        ub = (i + 1) * kv_channels
        indices.append(torch.arange(lb, ub, dtype=torch.int))
        indices.append(torch.arange(hidden_dim + lb, hidden_dim + ub, dtype=torch.int))
        indices.append(torch.arange(2 * hidden_dim + lb, 2 * hidden_dim + ub, dtype=torch.int))

    indices = torch.cat(indices)

    for name, tensor in state_dict.items():
        # Skip text model, logit_bias, logit_scale
        if "vision_model" not in name:
            print(f"{name} skipped")
            continue

        # Skip final layers not used in our model.
        if "head" in name:
            print(f"{name} skipped")
            continue

        # Map parameter names to ones used in megatron.
        new_name = ""
        new_tensor = tensor
        if new_tensor.dtype == torch.float16:
            new_tensor = new_tensor.to(torch.float32)

        # This is used for chunking some tensors to target tensor parallel size.
        chunk_dim = None

        qkv_params = set()
        if "position_embedding" in name:
            new_name = "position_embeddings.weight"
        elif "post_layernorm.weight" in name:
            new_name = "ln_post.weight"
        elif "post_layernorm.bias" in name:
            new_name = "ln_post.bias"
        elif "patch_embedding.weight" in name:
            new_name = "conv1.weight"
        elif "patch_embedding.bias" in name:
            new_name = "conv1.bias"
        elif "encoder.layers" in name:
            layer_idx = name.split(".")[3]
            base = f"decoder.layers.{layer_idx}"
            if "encoder.layers.26" in name:
                print(f"{name} skipped due to the last layer")
                continue

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
                    q_tensor = state_dict[q_name]

                    split_name[-2] = "k_proj"
                    k_name = ".".join(split_name)
                    k_tensor = state_dict[k_name]

                    split_name[-2] = "v_proj"
                    v_name = ".".join(split_name)
                    v_tensor = state_dict[v_name]

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
                    q_tensor = state_dict[q_name]

                    split_name[-2] = "k_proj"
                    k_name = ".".join(split_name)
                    k_tensor = state_dict[k_name]

                    split_name[-2] = "v_proj"
                    v_name = ".".join(split_name)
                    v_tensor = state_dict[v_name]

                    # concat and dim = 0
                    new_tensor = torch.cat([q_tensor, k_tensor, v_tensor], dim=0)

                    # reorder
                    new_tensor = new_tensor[indices]

                    chunk_dim = 0
                    qkv_params.add(new_name)
                else:
                    continue
            elif "attn.out_proj.weight" in name:
                new_name = f"{base}.self_attention.linear_proj.weight"
                chunk_dim = 1
            elif "attn.out_proj.bias" in name:
                new_name = f"{base}.self_attention.linear_proj.bias"
            elif "layer_norm1.weight" in name:
                new_name = f"{base}.input_layernorm.weight"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_weight"
            elif "layer_norm1.bias" in name:
                new_name = f"{base}.input_layernorm.bias"
                if use_te:
                    new_name = f"{base}.self_attention.linear_qkv.layer_norm_bias"
            elif "mlp.fc1.weight" in name:
                new_name = f"{base}.mlp.linear_fc1.weight"
                chunk_dim = 0
            elif "mlp.fc1.bias" in name:
                new_name = f"{base}.mlp.linear_fc1.bias"
                chunk_dim = 0
            elif "mlp.fc2.weight" in name:
                new_name = f"{base}.mlp.linear_fc2.weight"
                chunk_dim = 1
            elif "mlp.fc2.bias" in name:
                new_name = f"{base}.mlp.linear_fc2.bias"
            elif "layer_norm2.weight" in name:
                new_name = f"{base}.pre_mlp_layernorm.weight"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_weight"
            elif "layer_norm2.bias" in name:
                new_name = f"{base}.pre_mlp_layernorm.bias"
                if use_te:
                    new_name = f"{base}.mlp.linear_fc1.layer_norm_bias"

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
Convert SigLIP VIT weights to megatron format.


Example usage:
python siglip_converter.py --input /some/input/folder --output /some/output/folder --tensor-parallel-size 4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True, help="SigLIP weights folder")
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
