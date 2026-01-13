# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import argparse
import os

import torch
from safetensors.torch import load_file, save_file

convert_count = 0


def check_model(name, model):
    global convert_count
    convert_count += 1
    assert name in model, f"unexpected name {name}"


def is_chunked(name, chunk_names):
    for chunk_name in chunk_names:
        if chunk_name in name:
            return True
    return False


def convert(input_path, output_path, use_te, tensor_parallel_size=2):
    mc_models = []
    for i in range(tensor_parallel_size):
        mc_model = torch.load(
            os.path.join(input_path, f"mp_rank_{i:02d}", "model_optim_rng.pt"),
            map_location="cuda:0",
            weights_only=False,
        )
        mc_models.append(mc_model)
    mc_model = {}
    mc_model_0 = mc_models[0]
    llm_chunk_dim_0 = [
        "embedding.word_embeddings.weight",
        "self_attention.linear_qkv.weight",
        "self_attention.linear_qkv.bias",
        "mlp.linear_fc1.weight",
    ]
    llm_chunk_dim_1 = ["self_attention.linear_proj.weight", "mlp.linear_fc2.weight"]
    vision_chunk_dim_0 = [
        "self_attention.linear_qkv.weight",
        "self_attention.linear_qkv.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
    ]
    vision_chunk_dim_1 = ["self_attention.linear_proj.weight", "mlp.linear_fc2.weight"]
    mlp_chunk_dim_0 = ["encoder.linear_fc1.weight", "encoder.linear_fc1.bias"]
    mlp_chunk_dim_1 = ["encoder.linear_fc2.weight"]
    for name, param in mc_model_0["model"].items():
        # column parallel
        if "_extra_state" in name:
            continue
        params = [model["model"][name] for model in mc_models]
        if (
            is_chunked(name, llm_chunk_dim_0)
            or is_chunked(name, vision_chunk_dim_0)
            or is_chunked(name, mlp_chunk_dim_0)
        ):
            print(f"{name} concat in dim 0")

            mc_model[name] = torch.cat(params, dim=0)
        elif (
            is_chunked(name, llm_chunk_dim_1)
            or is_chunked(name, vision_chunk_dim_1)
            or is_chunked(name, mlp_chunk_dim_1)
        ):
            print(f"{name} concat in dim 1")
            mc_model[name] = torch.cat(params, dim=1)
        else:
            print(f"{name} without concat")
            mc_model[name] = param

    hf_dict = load_file(os.path.join(output_path, "model.safetensors"), device="cuda")
    hf_model = hf_dict

    mc_args = mc_model_0["args"]
    hidden_dim = mc_args.hidden_size
    ffn_hidden_size = mc_args.ffn_hidden_size
    num_heads = mc_args.num_attention_heads
    kv_channels = hidden_dim // num_heads
    num_query_groups = mc_args.num_query_groups
    kv_projection_size = kv_channels * num_query_groups

    assert hidden_dim % num_heads == 0
    assert kv_channels == mc_args.kv_channels
    assert num_heads % num_query_groups == 0

    indices = []
    # Q
    start = 0
    interval = kv_channels * num_heads // num_query_groups + 2 * kv_channels
    for i in range(num_query_groups):
        offset = i * interval
        indices.append(
            torch.arange(
                start + offset,
                start + offset + kv_channels * num_heads // num_query_groups,
                dtype=torch.int,
            )
        )
    # K
    start = kv_channels * num_heads // num_query_groups
    for i in range(num_query_groups):
        offset = i * interval
        indices.append(torch.arange(start + offset, start + offset + kv_channels, dtype=torch.int))
    # V
    start = kv_channels * num_heads // num_query_groups + kv_channels
    for i in range(num_query_groups):
        offset = i * interval
        indices.append(torch.arange(start + offset, start + offset + kv_channels, dtype=torch.int))
    indices = torch.cat(indices)
    deorder_indices = indices

    gate_up_indices = []
    # Gate
    start = 0
    for i in range(tensor_parallel_size):
        offset = i * (ffn_hidden_size // tensor_parallel_size * 2)
        gate_up_indices.append(
            torch.arange(
                start + offset,
                start + offset + ffn_hidden_size // tensor_parallel_size,
                dtype=torch.int,
            )
        )

    # UP
    start = ffn_hidden_size // tensor_parallel_size
    for i in range(tensor_parallel_size):
        offset = i * (ffn_hidden_size // tensor_parallel_size * 2)
        gate_up_indices.append(
            torch.arange(
                start + offset,
                start + offset + ffn_hidden_size // tensor_parallel_size,
                dtype=torch.int,
            )
        )
    gate_up_indices = torch.cat(gate_up_indices)
    deorder_gate_up_indices = gate_up_indices

    input_layer_norm_weight = (
        "input_layernorm.weight" if not use_te else "self_attention.linear_qkv.layer_norm_weight"
    )
    input_layer_norm_bias = (
        "input_layernorm.bias" if not use_te else "self_attention.linear_qkv.layer_norm_bias"
    )

    post_attention_layer_norm_weight = (
        "pre_mlp_layernorm.weight" if not use_te else "mlp.linear_fc1.layer_norm_weight"
    )

    layer_norm_2_weight = (
        "pre_mlp_layernorm.weight" if not use_te else "mlp.linear_fc1.layer_norm_weight"
    )
    layer_norm_2_bias = "pre_mlp_layernorm.bias" if not use_te else "mlp.linear_fc1.layer_norm_bias"

    for mc_name in mc_model:
        print("mc_layer:", mc_name)
        mc_tensor = mc_model[mc_name]

        # Language model mappings
        if "image_newline" in mc_name:
            check_model("model.image_newline", hf_model)
            hf_model["model.image_newline"] = mc_tensor

        if "language_model.embedding.word_embeddings.weight" in mc_name:
            check_model("model.embed_tokens.weight", hf_model)
            hf_model["model.embed_tokens.weight"] = mc_tensor
        elif "language_model.output_layer.weight" in mc_name:
            check_model("lm_head.weight", hf_model)
            hf_model["lm_head.weight"] = mc_tensor
        elif "language_model.decoder.final_layernorm.weight" in mc_name:
            check_model("model.norm.weight", hf_model)
            hf_model["model.norm.weight"] = mc_tensor
        elif "language_model.decoder.layers" in mc_name:
            layer_idx = mc_name.split(".")[3]
            base = f"model.layers.{layer_idx}"
            if "self_attention.linear_qkv.weight" in mc_name:
                # deorder_indices
                mc_tensor = mc_tensor[deorder_indices]
                qkv_weight = torch.split(
                    mc_tensor, [hidden_dim, kv_projection_size, kv_projection_size]
                )
                check_model(f"{base}.self_attn.q_proj.weight", hf_model)
                check_model(f"{base}.self_attn.k_proj.weight", hf_model)
                check_model(f"{base}.self_attn.v_proj.weight", hf_model)
                hf_model[f"{base}.self_attn.q_proj.weight"] = qkv_weight[0]
                hf_model[f"{base}.self_attn.k_proj.weight"] = qkv_weight[1]
                hf_model[f"{base}.self_attn.v_proj.weight"] = qkv_weight[2]
            elif "self_attention.linear_qkv.bias" in mc_name:
                # deorder_indices
                mc_tensor = mc_tensor[deorder_indices]
                qkv_bias = torch.split(
                    mc_tensor, [hidden_dim, kv_projection_size, kv_projection_size]
                )
                check_model(f"{base}.self_attn.q_proj.bias", hf_model)
                check_model(f"{base}.self_attn.k_proj.bias", hf_model)
                check_model(f"{base}.self_attn.v_proj.bias", hf_model)
                hf_model[f"{base}.self_attn.q_proj.bias"] = qkv_bias[0]
                hf_model[f"{base}.self_attn.k_proj.bias"] = qkv_bias[1]
                hf_model[f"{base}.self_attn.v_proj.bias"] = qkv_bias[2]
            elif "self_attention.linear_proj.weight" in mc_name:
                check_model(f"{base}.self_attn.o_proj.weight", hf_model)
                hf_model[f"{base}.self_attn.o_proj.weight"] = mc_tensor
            elif "self_attention.linear_proj.bias" in mc_name:
                check_model(f"{base}.self_attn.o_proj.bias", hf_model)
                hf_model[f"{base}.self_attn.o_proj.bias"] = mc_tensor
            elif input_layer_norm_weight in mc_name:
                check_model(f"{base}.input_layernorm.weight", hf_model)
                hf_model[f"{base}.input_layernorm.weight"] = mc_tensor
            elif "mlp.linear_fc1.weight" in mc_name:
                mc_tensor = mc_tensor[deorder_gate_up_indices]
                gate_up_weight = torch.split(mc_tensor, [ffn_hidden_size, ffn_hidden_size])
                check_model(f"{base}.mlp.gate_proj.weight", hf_model)
                check_model(f"{base}.mlp.up_proj.weight", hf_model)
                hf_model[f"{base}.mlp.gate_proj.weight"] = gate_up_weight[0]
                hf_model[f"{base}.mlp.up_proj.weight"] = gate_up_weight[1]
            elif "mlp.linear_fc2.weight" in mc_name:
                check_model(f"{base}.mlp.down_proj.weight", hf_model)
                hf_model[f"{base}.mlp.down_proj.weight"] = mc_tensor

            elif post_attention_layer_norm_weight in mc_name:
                check_model(f"{base}.post_attention_layernorm.weight", hf_model)
                hf_model[f"{base}.post_attention_layernorm.weight"] = mc_tensor

            else:
                raise ValueError(f"{name} is not converted.")

    # Indices from mapping pytorch multihead attention to megatron.
    kv_channels = 72
    hidden_dim = 1152
    num_heads = 16
    num_query_groups = 16
    kv_projection_size = kv_channels * num_query_groups
    indices = []
    # Q
    start = 0
    interval = kv_channels * 3
    for i in range(num_query_groups):
        offset = interval * i
        indices.append(torch.arange(start + offset, start + offset + kv_channels, dtype=torch.int))
    # K
    start = kv_channels
    for i in range(num_query_groups):
        offset = interval * i
        indices.append(torch.arange(start + offset, start + offset + kv_channels, dtype=torch.int))
    # V
    start = kv_channels * 2
    for i in range(num_query_groups):
        offset = interval * i
        indices.append(torch.arange(start + offset, start + offset + kv_channels, dtype=torch.int))
    indices = torch.cat(indices)
    deorder_indices = indices

    for mc_name in mc_model:
        print("mc_layer:", mc_name)
        mc_tensor = mc_model[mc_name]

        # vision_model
        hf_base_name = "model.vision_tower.vision_tower.vision_model"

        if "vision_model.position_embeddings.weight" in mc_name:
            check_model(f"{hf_base_name}.embeddings.position_embedding.weight", hf_model)
            hf_model[f"{hf_base_name}.embeddings.position_embedding.weight"] = mc_tensor

        elif "vision_model.ln_post.weight" in mc_name:
            check_model(f"{hf_base_name}.post_layernorm.weight", hf_model)
            hf_model[f"{hf_base_name}.post_layernorm.weight"] = mc_tensor

        elif "vision_model.ln_post.bias" in mc_name:
            check_model(f"{hf_base_name}.post_layernorm.bias", hf_model)
            hf_model[f"{hf_base_name}.post_layernorm.bias"] = mc_tensor

        elif "vision_model.conv1.weight" in mc_name:
            check_model(f"{hf_base_name}.embeddings.patch_embedding.weight", hf_model)
            hf_model[f"{hf_base_name}.embeddings.patch_embedding.weight"] = mc_tensor

        elif "vision_model.conv1.bias" in mc_name:
            check_model(f"{hf_base_name}.embeddings.patch_embedding.bias", hf_model)
            hf_model[f"{hf_base_name}.embeddings.patch_embedding.bias"] = mc_tensor

        elif "vision_model.decoder.layers" in mc_name:
            layer_idx = mc_name.split(".")[3]
            base = f"model.vision_tower.vision_tower.vision_model.encoder.layers.{layer_idx}"

            if "self_attention.linear_qkv.weight" in mc_name:
                mc_tensor = mc_tensor[deorder_indices]
                qkv_weight = torch.split(
                    mc_tensor, [hidden_dim, kv_projection_size, kv_projection_size]
                )
                check_model(f"{base}.self_attn.q_proj.weight", hf_model)
                check_model(f"{base}.self_attn.k_proj.weight", hf_model)
                check_model(f"{base}.self_attn.v_proj.weight", hf_model)
                hf_model[f"{base}.self_attn.q_proj.weight"] = qkv_weight[0]
                hf_model[f"{base}.self_attn.k_proj.weight"] = qkv_weight[1]
                hf_model[f"{base}.self_attn.v_proj.weight"] = qkv_weight[2]

            elif "self_attention.linear_qkv.bias" in mc_name:
                mc_tensor = mc_tensor[deorder_indices]
                qkv_bias = torch.split(
                    mc_tensor, [hidden_dim, kv_projection_size, kv_projection_size]
                )
                check_model(f"{base}.self_attn.q_proj.bias", hf_model)
                check_model(f"{base}.self_attn.k_proj.bias", hf_model)
                check_model(f"{base}.self_attn.v_proj.bias", hf_model)
                hf_model[f"{base}.self_attn.q_proj.bias"] = qkv_bias[0]
                hf_model[f"{base}.self_attn.k_proj.bias"] = qkv_bias[1]
                hf_model[f"{base}.self_attn.v_proj.bias"] = qkv_bias[2]

            elif "self_attention.linear_proj.weight" in mc_name:
                check_model(f"{base}.self_attn.out_proj.weight", hf_model)
                hf_model[f"{base}.self_attn.out_proj.weight"] = mc_tensor

            elif "self_attention.linear_proj.bias" in mc_name:
                check_model(f"{base}.self_attn.out_proj.bias", hf_model)
                hf_model[f"{base}.self_attn.out_proj.bias"] = mc_tensor

            elif input_layer_norm_weight in mc_name:
                check_model(f"{base}.layer_norm1.weight", hf_model)
                hf_model[f"{base}.layer_norm1.weight"] = mc_tensor

            elif input_layer_norm_bias in mc_name:
                check_model(f"{base}.layer_norm1.bias", hf_model)
                hf_model[f"{base}.layer_norm1.bias"] = mc_tensor

            elif "mlp.linear_fc1.weight" in mc_name:
                check_model(f"{base}.mlp.fc1.weight", hf_model)
                hf_model[f"{base}.mlp.fc1.weight"] = mc_tensor

            elif "mlp.linear_fc1.bias" in mc_name:
                check_model(f"{base}.mlp.fc1.bias", hf_model)
                hf_model[f"{base}.mlp.fc1.bias"] = mc_tensor

            elif "mlp.linear_fc2.weight" in mc_name:
                check_model(f"{base}.mlp.fc2.weight", hf_model)
                hf_model[f"{base}.mlp.fc2.weight"] = mc_tensor

            elif "mlp.linear_fc2.bias" in mc_name:
                check_model(f"{base}.mlp.fc2.bias", hf_model)
                hf_model[f"{base}.mlp.fc2.bias"] = mc_tensor

            elif layer_norm_2_weight in mc_name:
                check_model(f"{base}.layer_norm2.weight", hf_model)
                hf_model[f"{base}.layer_norm2.weight"] = mc_tensor

            elif layer_norm_2_bias in mc_name:
                check_model(f"{base}.layer_norm2.bias", hf_model)
                hf_model[f"{base}.layer_norm2.bias"] = mc_tensor

            else:
                raise ValueError(f"{name} is not converted.")

    # vision_projection
    check_model("model.mm_projector.0.weight", hf_model)
    check_model("model.mm_projector.0.bias", hf_model)
    check_model("model.mm_projector.2.weight", hf_model)
    check_model("model.mm_projector.2.bias", hf_model)
    hf_model["model.mm_projector.0.weight"] = mc_model[
        "vision_projection.encoder.linear_fc1.weight"
    ]
    hf_model["model.mm_projector.0.bias"] = mc_model["vision_projection.encoder.linear_fc1.bias"]
    hf_model["model.mm_projector.2.weight"] = mc_model[
        "vision_projection.encoder.linear_fc2.weight"
    ]
    hf_model["model.mm_projector.2.bias"] = mc_model["vision_projection.encoder.linear_fc2.bias"]

    metadata = {"format": "pt"}
    save_file(hf_model, output_path + "/model_new.safetensors", metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert Qwen2 1.5b weights to hugging face format.


Example usage:
python convert_to_hf_qwen2.5_1.5b.py --input /some/input/folder --output /some/output/folder --tensor-parallel-size 4
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", type=str, required=True, help="megatron ckpt folder")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output directory for hugging face state dict file(s)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="model tensor parallel size"
    )
    parser.add_argument("--use-te", action="store_true", help="Use Transformer Engine")

    args = parser.parse_args()

    convert(args.input, args.output, args.use_te, args.tensor_parallel_size)

    print("done.")
