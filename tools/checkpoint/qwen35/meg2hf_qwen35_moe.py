#!/usr/bin/env python3
"""
Convert Megatron TP/PP checkpoint to HuggingFace format for Qwen3.5-MoE models.

Usage:
    python meg2hf_qwen35_moe.py \
        --yaml /path/to/35b_a3b.yaml \
        --meg-ckpt-dir /path/to/checkpoints/iter_0000001 \
        --save-dir /path/to/output \
        [--hf-ref-dir /path/to/hf/qwen35_35ba3b]
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

# Layer norm adjustment for zero-centered gamma.
# Controls LLM and MTP layernorms only. Vision layernorms are never adjusted.
# By default enabled. Use --no-adjust-ln CLI flag to disable.
LN_ADJUSTMENT = True


def restore_ln_weight(weight):
    return weight + 1.0 if LN_ADJUSTMENT else weight


# ============================================================
# Config
# ============================================================


class Config:
    def __init__(self, yaml_path):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        system = raw.get("system", {})
        model = raw.get("model", {})

        self.tp = system.get("tensor_model_parallel_size", 1)
        self.pp = system.get("pipeline_model_parallel_size", 1)

        self.num_layers = model["num_layers"]
        self.hidden_size = model["hidden_size"]
        self.ffn_hidden_size = model.get("ffn_hidden_size", model.get("moe_ffn_hidden_size", 512))
        self.num_attention_heads = model["num_attention_heads"]
        self.num_query_groups = model["num_query_groups"]
        self.kv_channels = model["kv_channels"]
        self.untie = model.get("untie_embeddings_and_output_weights", False)
        self.attention_output_gate = model.get("attention_output_gate", False)

        self.linear_attention_freq = model.get("linear_attention_freq", 4)
        self.linear_key_head_dim = model.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = model.get("linear_value_head_dim", 128)
        self.linear_num_key_heads = model.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = model.get("linear_num_value_heads", 16)
        self.qk_dim = self.linear_key_head_dim * self.linear_num_key_heads
        self.v_dim = self.linear_value_head_dim * self.linear_num_value_heads

        # MoE params
        self.num_experts = model.get("num_experts", 256)
        self.moe_ffn_hidden_size = model.get("moe_ffn_hidden_size", 512)
        self.moe_shared_expert_intermediate_size = model.get(
            "moe_shared_expert_intermediate_size", self.moe_ffn_hidden_size
        )

        self.vision_num_layers = model.get("vision_num_layers", 12)
        self.vision_hidden_size = model.get("vision_hidden_size", 768)
        self.vision_num_attention_heads = model.get("vision_num_attention_heads", 12)
        self.vision_ffn_hidden_size = model.get("vision_ffn_hidden_size", 3072)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="Path to training yaml config")
    p.add_argument("--meg-ckpt-dir", required=True, help="Path to Megatron checkpoint directory")
    p.add_argument(
        "--hf-ref-dir", default=None, help="Reference HF model dir for shape validation (optional)"
    )
    p.add_argument("--save-dir", required=True, help="Output directory for HF checkpoint")
    p.add_argument(
        "--no-adjust-ln",
        action="store_true",
        help="Disable layer norm adjustment (zero-centered gamma)",
    )
    return p.parse_args()


# ============================================================
# Load & merge
# ============================================================


def load_shard(ckpt_dir, tp_rank, pp_rank):
    # Try PP suffix format first (TP+PP > 1)
    path_pp = os.path.join(ckpt_dir, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}", "model_optim_rng.pt")
    if os.path.exists(path_pp):
        sd = torch.load(path_pp, map_location="cpu", weights_only=False)
        return sd["model"]
    # Fallback to no-PP format (PP == 1)
    if pp_rank == 0:
        path_no_pp = os.path.join(ckpt_dir, f"mp_rank_{tp_rank:02d}", "model_optim_rng.pt")
        if os.path.exists(path_no_pp):
            sd = torch.load(path_no_pp, map_location="cpu", weights_only=False)
            return sd["model"]
    raise FileNotFoundError(f"Cannot find checkpoint for TP={tp_rank}, PP={pp_rank} in {ckpt_dir}")


def is_gdn_layer(idx, freq):
    return (idx % freq) != (freq - 1)


def merge_tp_shards(shards, cfg):
    tp = cfg.tp
    merged = {}
    all_keys = set()
    for s in shards:
        all_keys.update(s.keys())

    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads
    vis_gps = vis_qg // tp

    for k in sorted(all_keys):
        if "extra_state" in k:
            continue
        vals = [s[k] for s in shards if k in s]
        if not isinstance(vals[0], torch.Tensor):
            continue

        # ---- Vision model ----
        if "vision_model" in k:
            if "patch_embed" in k or "pos_embed" in k:
                merged[k] = vals[0]
            elif "linear_qkv.weight" in k:
                viewed = [x.view(vis_gps, -1, vis_head_dim, vis_h) for x in vals]
                merged[k] = torch.cat(viewed, dim=0).view(-1, vis_h)
            elif "linear_qkv.bias" in k:
                viewed = [x.view(vis_gps, -1, vis_head_dim) for x in vals]
                merged[k] = torch.cat(viewed, dim=0).view(-1)
            elif "linear_proj.weight" in k or "linear_fc2.weight" in k:
                merged[k] = torch.cat(vals, dim=1)
            elif "linear_fc1.weight" in k and "projection" not in k:
                merged[k] = torch.cat(vals, dim=0)
            elif "linear_fc1.bias" in k and "projection" not in k:
                merged[k] = torch.cat(vals, dim=0)
            elif "linear_proj.bias" in k or "linear_fc2.bias" in k:
                merged[k] = vals[0]
            elif "layer_norm" in k or "final_layernorm" in k:
                merged[k] = vals[0]
            elif "projection.encoder" in k:
                if "linear_fc1" in k:
                    merged[k] = torch.cat(vals, dim=0)
                elif "linear_fc2.weight" in k:
                    merged[k] = torch.cat(vals, dim=1)
                else:
                    merged[k] = vals[0]
            else:
                merged[k] = vals[0]
            continue

        # ---- LLM ----
        if "embedding" in k:
            merged[k] = torch.cat(vals, dim=0)
        elif k == "language_model.output_layer.weight":
            merged[k] = torch.cat(vals, dim=0)
        elif "final_layernorm" in k:
            merged[k] = vals[0]
        elif "layer_norm_weight" in k or "layer_norm_bias" in k:
            merged[k] = vals[0]
        # pre_mlp_layernorm (MoE): replicated
        elif "pre_mlp_layernorm" in k:
            merged[k] = vals[0]
        # GDN layers
        elif "in_proj.weight" in k:
            qk_dim = cfg.qk_dim
            v_dim = cfg.v_dim
            num_v = cfg.linear_num_value_heads
            q_per = qk_dim // tp
            k_per = qk_dim // tp
            v_per = v_dim // tp
            z_per = v_dim // tp
            b_per = num_v // tp
            a_per = num_v // tp

            q_all, k_all, v_all, z_all, b_all, a_all = [], [], [], [], [], []
            for val in vals:
                off = 0
                q_all.append(val[off : off + q_per])
                off += q_per
                k_all.append(val[off : off + k_per])
                off += k_per
                v_all.append(val[off : off + v_per])
                off += v_per
                z_all.append(val[off : off + z_per])
                off += z_per
                b_all.append(val[off : off + b_per])
                off += b_per
                a_all.append(val[off : off + a_per])

            merged[k] = torch.cat(
                [
                    torch.cat(q_all, dim=0),
                    torch.cat(k_all, dim=0),
                    torch.cat(v_all, dim=0),
                    torch.cat(z_all, dim=0),
                    torch.cat(b_all, dim=0),
                    torch.cat(a_all, dim=0),
                ],
                dim=0,
            )
        elif "conv1d.weight" in k:
            merged[k] = torch.cat(vals, dim=0)
        elif "A_log" in k or "dt_bias" in k:
            merged[k] = torch.cat(vals, dim=0)
        elif "out_norm" in k:
            merged[k] = vals[0]
        elif "out_proj.weight" in k:
            merged[k] = torch.cat(vals, dim=1)
        # Attention layers
        elif "linear_qkv.weight" in k:
            # Handle both num_qg >= tp (standard) and num_qg < tp cases
            # When num_qg < tp, some ranks may have empty shards; filter them out
            non_empty = [x for x in vals if x.shape[0] > 0]
            merged[k] = torch.cat(non_empty, dim=0)
        elif "linear_proj.weight" in k:
            merged[k] = torch.cat(vals, dim=1)
        elif "q_layernorm" in k or "k_layernorm" in k:
            merged[k] = vals[0]
        # MLP / MoE linear_fc1 (includes experts, shared_experts, dense MLP)
        elif "linear_fc1.weight" in k:
            viewed = [x.view(2, -1, cfg.hidden_size) for x in vals]
            merged[k] = torch.cat(viewed, dim=1).view(-1, cfg.hidden_size)
        elif "linear_fc2.weight" in k:
            merged[k] = torch.cat(vals, dim=1)
        # Router (not sharded)
        elif "router.weight" in k:
            merged[k] = vals[0]
        # MTP eh_proj (ColumnParallel)
        elif "eh_proj.weight" in k:
            merged[k] = torch.cat(vals, dim=0)
        else:
            merged[k] = vals[0]

    return merged


def merge_pp_stages(pp_merged, cfg):
    layers_per_pp = cfg.num_layers // cfg.pp
    full_sd = {}

    for pp_rank in range(cfg.pp):
        sd = pp_merged[pp_rank]
        for k, v in sd.items():
            if "decoder.layers." in k and "vision" not in k:
                parts = k.split("decoder.layers.")
                rest = parts[1]
                local_idx = int(rest.split(".")[0])
                global_idx = pp_rank * layers_per_pp + local_idx
                new_k = (
                    parts[0]
                    + "decoder.layers."
                    + str(global_idx)
                    + "."
                    + ".".join(rest.split(".")[1:])
                )
                full_sd[new_k] = v
            else:
                full_sd[k] = v

    return full_sd


# ============================================================
# Convert Megatron -> HF
# ============================================================


def split_gdn_in_proj(in_proj_weight, cfg):
    hidden = cfg.hidden_size
    num_qk_heads = cfg.linear_num_key_heads
    num_v_heads = cfg.linear_num_value_heads
    qk_head_dim = cfg.linear_key_head_dim
    v_head_dim = cfg.linear_value_head_dim
    v_per_group = num_v_heads // num_qk_heads
    tp_size = cfg.tp
    heads_per_tp = num_qk_heads // tp_size

    per_rank_dim = in_proj_weight.shape[0] // tp_size
    rank_weights = in_proj_weight.reshape(tp_size, per_rank_dim, hidden)

    q_parts, k_parts, v_parts, z_parts, b_parts, a_parts = [], [], [], [], [], []
    for rank in range(tp_size):
        rw = rank_weights[rank]
        q_end = heads_per_tp * qk_head_dim
        k_end = q_end + heads_per_tp * qk_head_dim
        v_end = k_end + heads_per_tp * v_per_group * v_head_dim
        z_end = v_end + heads_per_tp * v_per_group * v_head_dim
        b_end = z_end + heads_per_tp * v_per_group
        a_end = b_end + heads_per_tp * v_per_group
        q_parts.append(rw[:q_end])
        k_parts.append(rw[q_end:k_end])
        v_parts.append(rw[k_end:v_end])
        z_parts.append(rw[v_end:z_end])
        b_parts.append(rw[z_end:b_end])
        a_parts.append(rw[b_end:a_end])

    q_g = torch.cat(q_parts, dim=0).reshape(num_qk_heads, qk_head_dim, hidden)
    k_g = torch.cat(k_parts, dim=0).reshape(num_qk_heads, qk_head_dim, hidden)
    v_g = torch.cat(v_parts, dim=0).reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    z_g = torch.cat(z_parts, dim=0).reshape(num_qk_heads, v_per_group * v_head_dim, hidden)
    b_g = torch.cat(b_parts, dim=0).reshape(num_qk_heads, v_per_group, hidden)
    a_g = torch.cat(a_parts, dim=0).reshape(num_qk_heads, v_per_group, hidden)

    qkv = torch.cat(
        [
            q_g.reshape(-1, hidden),
            k_g.reshape(-1, hidden),
            v_g.reshape(-1, hidden),
        ],
        dim=0,
    )
    z_flat = z_g.reshape(-1, hidden)
    b_flat = b_g.reshape(-1, hidden)
    a_flat = a_g.reshape(-1, hidden)

    return qkv, z_flat, b_flat, a_flat


def convert_to_hf(full_sd, cfg):
    hf_sd = {}
    hidden = cfg.hidden_size
    freq = cfg.linear_attention_freq

    # Embedding
    emb_key = "language_model.embedding.word_embeddings.weight"
    if emb_key in full_sd:
        hf_sd["model.language_model.embed_tokens.weight"] = full_sd[emb_key]

    # LLM layers
    for layer_idx in range(cfg.num_layers):
        mg_pfx = f"language_model.decoder.layers.{layer_idx}"
        hf_pfx = f"model.language_model.layers.{layer_idx}"

        if is_gdn_layer(layer_idx, freq):
            mk = f"{mg_pfx}.self_attention.in_proj.layer_norm_weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.input_layernorm.weight"] = restore_ln_weight(full_sd[mk])

            mk = f"{mg_pfx}.self_attention.in_proj.weight"
            if mk in full_sd:
                qkv, z, b, a = split_gdn_in_proj(full_sd[mk], cfg)
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_qkv.weight"] = qkv
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_z.weight"] = z
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_b.weight"] = b
                hf_sd[f"{hf_pfx}.linear_attn.in_proj_a.weight"] = a

            mk = f"{mg_pfx}.self_attention.conv1d.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.conv1d.weight"] = full_sd[mk]

            mk = f"{mg_pfx}.self_attention.out_proj.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.out_proj.weight"] = full_sd[mk]

            mk = f"{mg_pfx}.self_attention.out_norm.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.norm.weight"] = restore_ln_weight(full_sd[mk])

            mk = f"{mg_pfx}.self_attention.A_log"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.A_log"] = full_sd[mk]
            mk = f"{mg_pfx}.self_attention.dt_bias"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.linear_attn.dt_bias"] = full_sd[mk]

        else:
            mk = f"{mg_pfx}.self_attention.linear_qkv.layer_norm_weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.input_layernorm.weight"] = restore_ln_weight(full_sd[mk])

            mk = f"{mg_pfx}.self_attention.linear_qkv.weight"
            if mk in full_sd:
                num_qg = cfg.num_query_groups
                kv_ch = cfg.kv_channels
                heads_per_group = cfg.num_attention_heads // num_qg
                if cfg.attention_output_gate:
                    total_hpg = 2 * heads_per_group + 2
                else:
                    total_hpg = heads_per_group + 2
                qkv = full_sd[mk].view(num_qg, total_hpg, kv_ch, hidden)
                if cfg.attention_output_gate:
                    q_heads = qkv[:, :heads_per_group]
                    z_heads = qkv[:, heads_per_group : 2 * heads_per_group]
                    k_heads = qkv[:, -2:-1]
                    v_heads = qkv[:, -1:]
                    q_combined = torch.cat([q_heads, z_heads], dim=1)
                    hf_sd[f"{hf_pfx}.self_attn.q_proj.weight"] = q_combined.reshape(-1, hidden)
                else:
                    q, k_heads, v_heads = torch.split(qkv, [heads_per_group, 1, 1], dim=1)
                    hf_sd[f"{hf_pfx}.self_attn.q_proj.weight"] = q.reshape(-1, hidden)
                hf_sd[f"{hf_pfx}.self_attn.k_proj.weight"] = k_heads.reshape(-1, hidden)
                hf_sd[f"{hf_pfx}.self_attn.v_proj.weight"] = v_heads.reshape(-1, hidden)

            mk = f"{mg_pfx}.self_attention.linear_proj.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.self_attn.o_proj.weight"] = full_sd[mk]

            mk = f"{mg_pfx}.self_attention.q_layernorm.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.self_attn.q_norm.weight"] = restore_ln_weight(full_sd[mk])
            mk = f"{mg_pfx}.self_attention.k_layernorm.weight"
            if mk in full_sd:
                hf_sd[f"{hf_pfx}.self_attn.k_norm.weight"] = restore_ln_weight(full_sd[mk])

        # --- MoE (replaces dense MLP) ---
        # pre_mlp_layernorm -> post_attention_layernorm
        mk = f"{mg_pfx}.pre_mlp_layernorm.weight"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.post_attention_layernorm.weight"] = restore_ln_weight(full_sd[mk])

        # Router -> gate
        mk = f"{mg_pfx}.mlp.router.weight"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.mlp.gate.weight"] = full_sd[mk]

        # Experts linear_fc1 -> gate_up_proj
        gate_up_list = []
        for e in range(cfg.num_experts):
            mk = f"{mg_pfx}.mlp.experts.linear_fc1.weight{e}"
            if mk in full_sd:
                gate_up_list.append(full_sd[mk])
        if gate_up_list:
            hf_sd[f"{hf_pfx}.mlp.experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0)

        # Experts linear_fc2 -> down_proj
        down_list = []
        for e in range(cfg.num_experts):
            mk = f"{mg_pfx}.mlp.experts.linear_fc2.weight{e}"
            if mk in full_sd:
                down_list.append(full_sd[mk])
        if down_list:
            hf_sd[f"{hf_pfx}.mlp.experts.down_proj"] = torch.stack(down_list, dim=0)

        # Shared expert: linear_fc1 -> gate_proj + up_proj
        mk = f"{mg_pfx}.mlp.shared_experts.linear_fc1.weight"
        if mk in full_sd:
            gate, up = torch.split(full_sd[mk], cfg.moe_ffn_hidden_size, dim=0)
            hf_sd[f"{hf_pfx}.mlp.shared_expert.gate_proj.weight"] = gate
            hf_sd[f"{hf_pfx}.mlp.shared_expert.up_proj.weight"] = up

        # Shared expert: linear_fc2 -> down_proj
        mk = f"{mg_pfx}.mlp.shared_experts.linear_fc2.weight"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.mlp.shared_expert.down_proj.weight"] = full_sd[mk]

        # Shared expert gate
        mk = f"{mg_pfx}.mlp.shared_experts.gate_weight"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.mlp.shared_expert_gate.weight"] = full_sd[mk]

    # Final layernorm
    mk = "language_model.decoder.final_layernorm.weight"
    if mk in full_sd:
        hf_sd["model.language_model.norm.weight"] = restore_ln_weight(full_sd[mk])

    # Output layer
    if cfg.untie:
        mk = "language_model.output_layer.weight"
        if mk in full_sd:
            hf_sd["lm_head.weight"] = full_sd[mk]

    # Vision model
    convert_vision(full_sd, hf_sd, cfg)

    # MTP
    convert_mtp(full_sd, hf_sd, cfg)

    return hf_sd


def convert_vision(full_sd, hf_sd, cfg):
    mg_pfx = "vision_model"
    hf_pfx = "model.visual"
    vis_h = cfg.vision_hidden_size
    vis_heads = cfg.vision_num_attention_heads
    vis_head_dim = vis_h // vis_heads
    vis_qg = vis_heads

    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.patch_embed.proj.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.patch_embed.proj.{s}"] = full_sd[mk]

    mk = f"{mg_pfx}.pos_embed.weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.pos_embed.weight"] = full_sd[mk]

    for i in range(cfg.vision_num_layers):
        mg_blk = f"{mg_pfx}.decoder.layers.{i}"
        hf_blk = f"{hf_pfx}.blocks.{i}"

        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.self_attention.linear_qkv.layer_norm_{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.norm1.{s}"] = full_sd[mk]

        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.mlp.linear_fc1.layer_norm_{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.norm2.{s}"] = full_sd[mk]

        mk = f"{mg_blk}.self_attention.linear_qkv.weight"
        if mk in full_sd:
            qkv_w = full_sd[mk].view(vis_qg, 3, vis_head_dim, vis_h)
            converted = qkv_w.transpose(0, 1).reshape(-1, vis_h).contiguous()
            hf_sd[f"{hf_blk}.attn.qkv.weight"] = converted

        mk = f"{mg_blk}.self_attention.linear_qkv.bias"
        if mk in full_sd:
            qkv_b = full_sd[mk].view(vis_qg, 3, vis_head_dim)
            converted = qkv_b.transpose(0, 1).reshape(-1).contiguous()
            hf_sd[f"{hf_blk}.attn.qkv.bias"] = converted

        for s in ["weight", "bias"]:
            mk = f"{mg_blk}.self_attention.linear_proj.{s}"
            if mk in full_sd:
                hf_sd[f"{hf_blk}.attn.proj.{s}"] = full_sd[mk]

        for s in ["weight", "bias"]:
            for layer in ["linear_fc1", "linear_fc2"]:
                mk = f"{mg_blk}.mlp.{layer}.{s}"
                if mk in full_sd:
                    hf_sd[f"{hf_blk}.mlp.{layer}.{s}"] = full_sd[mk]

    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.projection.encoder.linear_fc1.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.linear_fc1.{s}"] = full_sd[mk]
        mk = f"{mg_pfx}.projection.encoder.linear_fc2.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.linear_fc2.{s}"] = full_sd[mk]

    for s in ["weight", "bias"]:
        mk = f"{mg_pfx}.decoder.final_layernorm.{s}"
        if mk in full_sd:
            hf_sd[f"{hf_pfx}.merger.norm.{s}"] = full_sd[mk]


def convert_mtp(full_sd, hf_sd, cfg):
    hidden = cfg.hidden_size

    # Direct mappings (MoE version: post_attention_layernorm -> pre_mlp_layernorm)
    mtp_direct = {
        "language_model.mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
        "language_model.mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "language_model.mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "language_model.mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.pre_mlp_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight": "mtp.layers.0.input_layernorm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
    }
    for mk, hk in mtp_direct.items():
        if mk in full_sd:
            hf_sd[hk] = restore_ln_weight(full_sd[mk])

    # MTP QKV
    mk = "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"
    if mk in full_sd:
        num_qg = cfg.num_query_groups
        kv_ch = cfg.kv_channels
        heads_per_group = cfg.num_attention_heads // num_qg
        if cfg.attention_output_gate:
            total_hpg = 2 * heads_per_group + 2
        else:
            total_hpg = heads_per_group + 2
        qkv = full_sd[mk].view(num_qg, total_hpg, kv_ch, hidden)
        if cfg.attention_output_gate:
            q_heads = qkv[:, :heads_per_group]
            z_heads = qkv[:, heads_per_group : 2 * heads_per_group]
            k_heads = qkv[:, -2:-1]
            v_heads = qkv[:, -1:]
            q_combined = torch.cat([q_heads, z_heads], dim=1)
            hf_sd["mtp.layers.0.self_attn.q_proj.weight"] = q_combined.reshape(-1, hidden)
        else:
            q, k_heads, v_heads = torch.split(qkv, [heads_per_group, 1, 1], dim=1)
            hf_sd["mtp.layers.0.self_attn.q_proj.weight"] = q.reshape(-1, hidden)
        hf_sd["mtp.layers.0.self_attn.k_proj.weight"] = k_heads.reshape(-1, hidden)
        hf_sd["mtp.layers.0.self_attn.v_proj.weight"] = v_heads.reshape(-1, hidden)

    # --- MTP MoE ---
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.router.weight"
    if mk in full_sd:
        hf_sd["mtp.layers.0.mlp.gate.weight"] = full_sd[mk]

    # MTP experts: linear_fc1.weightN -> individual gate_proj + up_proj
    for e in range(cfg.num_experts):
        mk = f"language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1.weight{e}"
        if mk in full_sd:
            gate, up = torch.split(full_sd[mk], cfg.moe_ffn_hidden_size, dim=0)
            hf_sd[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = gate
            hf_sd[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = up

    # MTP experts: linear_fc2.weightN -> individual down_proj
    for e in range(cfg.num_experts):
        mk = f"language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2.weight{e}"
        if mk in full_sd:
            hf_sd[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = full_sd[mk]

    # MTP shared expert
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc1.weight"
    if mk in full_sd:
        gate, up = torch.split(full_sd[mk], cfg.moe_ffn_hidden_size, dim=0)
        hf_sd["mtp.layers.0.mlp.shared_expert.gate_proj.weight"] = gate
        hf_sd["mtp.layers.0.mlp.shared_expert.up_proj.weight"] = up

    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2.weight"
    if mk in full_sd:
        hf_sd["mtp.layers.0.mlp.shared_expert.down_proj.weight"] = full_sd[mk]

    # MTP shared expert gate
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.gate_weight"
    if mk in full_sd:
        hf_sd["mtp.layers.0.mlp.shared_expert_gate.weight"] = full_sd[mk]


# ============================================================
# Compare
# ============================================================


def load_hf_shapes(hf_dir):
    if hf_dir is None:
        return None
    shapes = {}
    for st_file in sorted(Path(hf_dir).glob("*.safetensors")):
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = list(f.get_tensor(key).shape)
    return shapes


def compare_models(hf_sd, ref_shapes, cfg):
    print("\n" + "=" * 100)
    print("Shape Comparison: Converted vs Reference HF Model")
    print("=" * 100)

    if ref_shapes is None or len(ref_shapes) == 0:
        print("No reference shapes provided, skipping comparison.")
        return True

    # Keys that are expected to differ (Megatron doesn't have these)
    expected_missing_in_converted = set()

    # If the reference HF model has more layers than the Megatron checkpoint
    # specified by cfg.num_layers, layers beyond cfg.num_layers are expected
    # to be missing from the converted output.
    for k in list(ref_shapes.keys()):
        # Match patterns like model.language_model.layers.XX. or mtp.layers.XX.
        import re

        m = re.search(r"layers\.(\d+)\.", k)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx >= cfg.num_layers:
                expected_missing_in_converted.add(k)

    all_keys = sorted(set(list(hf_sd.keys()) + list(ref_shapes.keys())))
    mismatches, missing, extra, matched = [], [], [], 0

    for k in all_keys:
        in_conv = k in hf_sd
        in_ref = k in ref_shapes
        if in_conv and in_ref:
            cs = list(hf_sd[k].shape)
            rs = ref_shapes[k]
            if cs == rs:
                matched += 1
            else:
                mismatches.append((k, cs, rs))
        elif in_conv:
            extra.append(k)
        else:
            if k not in expected_missing_in_converted:
                missing.append(k)

    print(f"\nMatched: {matched}/{len(all_keys)}")

    if mismatches:
        print(f"\n❌ Shape mismatches ({len(mismatches)}):")
        for k, cs, rs in mismatches:
            print(f"  {k:80s} converted={cs} ref={rs}")
    if missing:
        print(f"\n❌ Missing in converted ({len(missing)}):")
        for k in missing:
            print(f"  {k:80s} ref_shape={ref_shapes[k]}")
    if extra:
        print(f"\n❌ Extra in converted ({len(extra)}):")
        for k in extra:
            print(f"  {k:80s} shape={list(hf_sd[k].shape)}")

    if expected_missing_in_converted:
        print(
            f"\n⚠️  Expected missing (Megatron has no equivalent): {len(expected_missing_in_converted)}"
        )
        for k in sorted(expected_missing_in_converted):
            print(f"  {k}")

    conv_total = sum(t.numel() for t in hf_sd.values())
    ref_total = sum(__import__("math").prod(shape) if shape else 1 for shape in ref_shapes.values())

    print(f"\nConverted total params: {conv_total:>15,}")
    print(f"Reference total params: {ref_total:>15,}")
    print(f"Difference:             {conv_total - ref_total:>15,}")

    print("\n" + "=" * 100)
    if mismatches or missing or extra:
        print("❌ VALIDATION FAILED")
        print(f"   - {len(mismatches)} shape mismatches")
        print(f"   - {len(missing)} missing params")
        print(f"   - {len(extra)} extra params")
        success = False
    else:
        print("✅ VALIDATION PASSED")
        print(f"   - All {matched} parameters match")
        print(f"   - Total params: {conv_total:,}")
        if expected_missing_in_converted:
            print(
                f"   - {len(expected_missing_in_converted)} params expected to differ (Megatron lacks equivalent)"
            )
        success = True
    print("=" * 100)

    return success


# ============================================================
# Main
# ============================================================


def main():
    args = parse_args()
    cfg = Config(args.yaml)

    # Apply CLI override for layer norm adjustment
    global LN_ADJUSTMENT
    if args.no_adjust_ln:
        LN_ADJUSTMENT = False

    print(
        f"Config: TP={cfg.tp}, PP={cfg.pp}, layers={cfg.num_layers}, "
        f"hidden={cfg.hidden_size}, ffn={cfg.ffn_hidden_size}"
    )
    print(f"MoE: experts={cfg.num_experts}, moe_ffn={cfg.moe_ffn_hidden_size}")
    print(f"GDN: freq={cfg.linear_attention_freq}, qk_dim={cfg.qk_dim}, v_dim={cfg.v_dim}")
    print(f"LN adjustment: {LN_ADJUSTMENT}")

    if cfg.pp > 1:
        print("\n❌ ERROR: PP (Pipeline Parallelism) > 1 is not yet supported.")
        print("   Please set pipeline_model_parallel_size=1 in your yaml config.")
        import sys

        sys.exit(1)

    os.makedirs(args.save_dir, exist_ok=True)

    pp_merged = {}
    for pp_rank in range(cfg.pp):
        print(f"\nLoading PP stage {pp_rank}...")
        shards = []
        for tp_rank in range(cfg.tp):
            print(f"  Loading TP shard {tp_rank}...")
            shards.append(load_shard(args.meg_ckpt_dir, tp_rank, pp_rank))
        pp_merged[pp_rank] = merge_tp_shards(shards, cfg)
        del shards

    print("\nMerging PP stages...")
    full_sd = merge_pp_stages(pp_merged, cfg)
    del pp_merged

    print(f"\nMerged Megatron keys: {len(full_sd)}")
    for k in sorted(full_sd.keys()):
        if isinstance(full_sd[k], torch.Tensor):
            print(f"  {k:90s} {list(full_sd[k].shape)}")

    print("\nConverting to HF format...")
    hf_sd = convert_to_hf(full_sd, cfg)
    del full_sd

    print(f"\nConverted HF keys: {len(hf_sd)}")
    for k in sorted(hf_sd.keys()):
        print(f"  {k:80s} {list(hf_sd[k].shape)}")

    print(f"\nSaving to {args.save_dir}...")
    save_file(hf_sd, os.path.join(args.save_dir, "model.safetensors"))

    if args.hf_ref_dir:
        print("\nComparing with reference HF model...")
        ref_shapes = load_hf_shapes(args.hf_ref_dir)
        success = compare_models(hf_sd, ref_shapes, cfg)
        import sys

        sys.exit(0 if success else 1)
    else:
        print("\nConversion complete (no reference provided for validation).")


if __name__ == "__main__":
    main()
