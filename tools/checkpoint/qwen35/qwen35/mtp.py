"""MTP (Multi-Token Prediction) conversion helpers (HF <-> Megatron)."""

import torch

import qwen35.constants as _constants
from qwen35.attention import merge_attention_qkv, split_attention_qkv


def _adjust_ln(weight):
    """Subtract 1.0 when LN adjustment is enabled."""
    return weight - 1.0 if _constants.LN_ADJUSTMENT else weight


def _restore_ln(weight):
    """Add 1.0 when LN adjustment is enabled."""
    return weight + 1.0 if _constants.LN_ADJUSTMENT else weight


# HF -> Megatron direct mappings.
# Dense and MoE checkpoints use different names for the post-attention
# layernorm that feeds the MLP.
_MTP_DIRECT_HF2MEG_BASE = {
    "mtp.fc.weight": "language_model.mtp.layers.0.eh_proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "language_model.mtp.layers.0.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "language_model.mtp.layers.0.hnorm.weight",
    "mtp.norm.weight": "language_model.mtp.layers.0.final_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight": "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc2.weight",
    "mtp.layers.0.input_layernorm.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight",
    "mtp.layers.0.self_attn.q_norm.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight",
    "mtp.layers.0.self_attn.o_proj.weight": "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight",
}

_MTP_POST_ATTN_LN_DENSE = (
    "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.layer_norm_weight"
)
_MTP_POST_ATTN_LN_MOE = "language_model.mtp.layers.0.mtp_model_layer.pre_mlp_layernorm.weight"

_MTP_LN_KEYS_HF2MEG = {
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
}


def _get_mtp_mappings(is_moe):
    """Return (hf2meg, meg2hf, ln_keys_meg2hf) for dense or MoE MTP naming."""
    post_ln = _MTP_POST_ATTN_LN_MOE if is_moe else _MTP_POST_ATTN_LN_DENSE
    hf2meg = {
        **_MTP_DIRECT_HF2MEG_BASE,
        "mtp.layers.0.post_attention_layernorm.weight": post_ln,
    }
    meg2hf = {v: k for k, v in hf2meg.items()}
    ln_keys_meg2hf = {
        "language_model.mtp.layers.0.enorm.weight",
        "language_model.mtp.layers.0.hnorm.weight",
        "language_model.mtp.layers.0.final_layernorm.weight",
        post_ln,
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.layer_norm_weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.q_layernorm.weight",
        "language_model.mtp.layers.0.mtp_model_layer.self_attention.k_layernorm.weight",
    }
    return hf2meg, meg2hf, ln_keys_meg2hf


# Backwards-compatible module-level constants (dense defaults).
_MTP_DIRECT_HF2MEG, _MTP_DIRECT_MEG2HF, _MTP_LN_KEYS_MEG2HF = _get_mtp_mappings(is_moe=False)


def _convert_mtp_mlp_hf2meg(hf_sd, meg_sd, cfg):
    """Convert MTP MLP: dense by default, MoE extensions when present."""
    # Dense MLP
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight"
    gate = hf_sd.get("mtp.layers.0.mlp.gate_proj.weight")
    up = hf_sd.get("mtp.layers.0.mlp.up_proj.weight")
    if gate is not None and up is not None:
        meg_sd[mk] = torch.cat([gate, up], dim=0)

    # MoE extensions (only if MoE-specific keys present)
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.router.weight"
    hk = "mtp.layers.0.mlp.gate.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]

    for e in range(cfg.num_experts):
        mk = f"language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1.weight{e}"
        gate = hf_sd.get(f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight")
        up = hf_sd.get(f"mtp.layers.0.mlp.experts.{e}.up_proj.weight")
        if gate is not None and up is not None:
            meg_sd[mk] = torch.cat([gate, up], dim=0)

    for e in range(cfg.num_experts):
        mk = f"language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2.weight{e}"
        hk = f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"
        if hk in hf_sd:
            meg_sd[mk] = hf_sd[hk]

    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc1.weight"
    gate = hf_sd.get("mtp.layers.0.mlp.shared_expert.gate_proj.weight")
    up = hf_sd.get("mtp.layers.0.mlp.shared_expert.up_proj.weight")
    if gate is not None and up is not None:
        meg_sd[mk] = torch.cat([gate, up], dim=0)

    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2.weight"
    hk = "mtp.layers.0.mlp.shared_expert.down_proj.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]

    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.gate_weight"
    hk = "mtp.layers.0.mlp.shared_expert_gate.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]


def convert_mtp_hf2meg(hf_sd, meg_sd, cfg):
    """Convert MTP parameters from HF to Megatron."""
    hf2meg, _, _ = _get_mtp_mappings(cfg.is_moe)
    for hk, mk in hf2meg.items():
        if hk in hf_sd:
            meg_sd[mk] = _adjust_ln(hf_sd[hk]) if hk in _MTP_LN_KEYS_HF2MEG else hf_sd[hk]

    mk = "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"
    q = hf_sd.get("mtp.layers.0.self_attn.q_proj.weight")
    k = hf_sd.get("mtp.layers.0.self_attn.k_proj.weight")
    v = hf_sd.get("mtp.layers.0.self_attn.v_proj.weight")
    if q is not None and k is not None and v is not None:
        meg_sd[mk] = merge_attention_qkv(q, k, v, cfg)

    _convert_mtp_mlp_hf2meg(hf_sd, meg_sd, cfg)


def _convert_mtp_mlp_meg2hf(full_sd, hf_sd, cfg):
    """Convert MTP MLP: dense by default, MoE extensions when present."""
    # Dense MLP
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight"
    if mk in full_sd:
        gate, up = torch.split(full_sd[mk], cfg.ffn_hidden_size, dim=0)
        hf_sd["mtp.layers.0.mlp.gate_proj.weight"] = gate
        hf_sd["mtp.layers.0.mlp.up_proj.weight"] = up

    # MoE extensions
    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.router.weight"
    if mk in full_sd:
        hf_sd["mtp.layers.0.mlp.gate.weight"] = full_sd[mk]

    for e in range(cfg.num_experts):
        mk = f"language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc1.weight{e}"
        if mk in full_sd:
            gate, up = torch.split(full_sd[mk], cfg.moe_ffn_hidden_size, dim=0)
            hf_sd[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = gate
            hf_sd[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = up

    for e in range(cfg.num_experts):
        mk = f"language_model.mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2.weight{e}"
        if mk in full_sd:
            hf_sd[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = full_sd[mk]

    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc1.weight"
    if mk in full_sd:
        gate, up = torch.split(full_sd[mk], cfg.moe_shared_expert_intermediate_size, dim=0)
        hf_sd["mtp.layers.0.mlp.shared_expert.gate_proj.weight"] = gate
        hf_sd["mtp.layers.0.mlp.shared_expert.up_proj.weight"] = up

    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.linear_fc2.weight"
    if mk in full_sd:
        hf_sd["mtp.layers.0.mlp.shared_expert.down_proj.weight"] = full_sd[mk]

    mk = "language_model.mtp.layers.0.mtp_model_layer.mlp.shared_experts.gate_weight"
    if mk in full_sd:
        hf_sd["mtp.layers.0.mlp.shared_expert_gate.weight"] = full_sd[mk]


def convert_mtp_meg2hf(full_sd, hf_sd, cfg):
    """Convert MTP parameters from Megatron to HF."""
    _, meg2hf, ln_keys_meg2hf = _get_mtp_mappings(cfg.is_moe)
    for mk, hk in meg2hf.items():
        if mk in full_sd:
            hf_sd[hk] = _restore_ln(full_sd[mk]) if mk in ln_keys_meg2hf else full_sd[mk]

    mk = "language_model.mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight"
    if mk in full_sd:
        q, k, v = split_attention_qkv(full_sd[mk], cfg)
        hf_sd["mtp.layers.0.self_attn.q_proj.weight"] = q
        hf_sd["mtp.layers.0.self_attn.k_proj.weight"] = k
        hf_sd["mtp.layers.0.self_attn.v_proj.weight"] = v

    _convert_mtp_mlp_meg2hf(full_sd, hf_sd, cfg)
