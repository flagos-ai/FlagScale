# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLP conversion helpers (dense + MoE) shared across directions."""

import torch

import qwen35.constants as _constants


def _adjust_ln(weight):
    """Subtract 1.0 when legacy LN adjustment is enabled."""
    return weight - 1.0 if _constants.LN_ADJUSTMENT else weight


def _restore_ln(weight):
    """Add 1.0 when legacy LN adjustment is enabled."""
    return weight + 1.0 if _constants.LN_ADJUSTMENT else weight


# -----------------------------------------------------------------------------
# HF -> Megatron
# -----------------------------------------------------------------------------
def convert_dense_mlp_hf2meg(hf_sd, meg_sd, layer_idx, hf_pfx, mg_pfx):
    """Convert dense MLP parameters for a single LLM layer (HF -> Megatron)."""
    mk = f"{mg_pfx}.mlp.linear_fc1.weight"
    gate = hf_sd.get(f"{hf_pfx}.mlp.gate_proj.weight")
    up = hf_sd.get(f"{hf_pfx}.mlp.up_proj.weight")
    if gate is not None and up is not None:
        meg_sd[mk] = torch.cat([gate, up], dim=0)

    mk = f"{mg_pfx}.mlp.linear_fc2.weight"
    hk = f"{hf_pfx}.mlp.down_proj.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]

    hk = f"{hf_pfx}.post_attention_layernorm.weight"
    mk = f"{mg_pfx}.mlp.linear_fc1.layer_norm_weight"
    if hk in hf_sd:
        meg_sd[mk] = _adjust_ln(hf_sd[hk])


def convert_moe_mlp_hf2meg(hf_sd, meg_sd, layer_idx, hf_pfx, mg_pfx, cfg):
    """Convert MoE MLP parameters for a single LLM layer (HF -> Megatron)."""
    # pre-MoE layer norm
    hk = f"{hf_pfx}.post_attention_layernorm.weight"
    mk = f"{mg_pfx}.pre_mlp_layernorm.weight"
    if hk in hf_sd:
        meg_sd[mk] = _adjust_ln(hf_sd[hk])

    # Router
    hk = f"{hf_pfx}.mlp.gate.weight"
    mk = f"{mg_pfx}.mlp.router.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]

    # Stacked expert format: [num_experts, 2*moe_ffn, hidden]
    stacked_gu = hf_sd.get(f"{hf_pfx}.mlp.experts.gate_up_proj")
    stacked_down = hf_sd.get(f"{hf_pfx}.mlp.experts.down_proj")
    if stacked_gu is not None:
        for e in range(cfg.num_experts):
            meg_sd[f"{mg_pfx}.mlp.experts.linear_fc1.weight{e}"] = stacked_gu[e].clone()
    else:
        for e in range(cfg.num_experts):
            mk = f"{mg_pfx}.mlp.experts.linear_fc1.weight{e}"
            gate = hf_sd.get(f"{hf_pfx}.mlp.experts.{e}.gate_proj.weight")
            up = hf_sd.get(f"{hf_pfx}.mlp.experts.{e}.up_proj.weight")
            if gate is not None and up is not None:
                meg_sd[mk] = torch.cat([gate, up], dim=0)

    if stacked_down is not None:
        for e in range(cfg.num_experts):
            meg_sd[f"{mg_pfx}.mlp.experts.linear_fc2.weight{e}"] = stacked_down[e].clone()
    else:
        for e in range(cfg.num_experts):
            mk = f"{mg_pfx}.mlp.experts.linear_fc2.weight{e}"
            hk = f"{hf_pfx}.mlp.experts.{e}.down_proj.weight"
            if hk in hf_sd:
                meg_sd[mk] = hf_sd[hk]

    # Shared expert
    mk = f"{mg_pfx}.mlp.shared_experts.linear_fc1.weight"
    gate = hf_sd.get(f"{hf_pfx}.mlp.shared_expert.gate_proj.weight")
    up = hf_sd.get(f"{hf_pfx}.mlp.shared_expert.up_proj.weight")
    if gate is not None and up is not None:
        meg_sd[mk] = torch.cat([gate, up], dim=0)

    mk = f"{mg_pfx}.mlp.shared_experts.linear_fc2.weight"
    hk = f"{hf_pfx}.mlp.shared_expert.down_proj.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]

    mk = f"{mg_pfx}.mlp.shared_experts.gate_weight"
    hk = f"{hf_pfx}.mlp.shared_expert_gate.weight"
    if hk in hf_sd:
        meg_sd[mk] = hf_sd[hk]


def convert_layer_mlp_hf2meg(hf_sd, meg_sd, layer_idx, hf_pfx, mg_pfx, cfg):
    """Route to dense or MoE MLP conversion based on config."""
    if cfg.is_moe:
        convert_moe_mlp_hf2meg(hf_sd, meg_sd, layer_idx, hf_pfx, mg_pfx, cfg)
    else:
        convert_dense_mlp_hf2meg(hf_sd, meg_sd, layer_idx, hf_pfx, mg_pfx)


# -----------------------------------------------------------------------------
# Megatron -> HF
# -----------------------------------------------------------------------------
def convert_dense_mlp_meg2hf(full_sd, hf_sd, layer_idx, hf_pfx, mg_pfx, cfg):
    """Convert dense MLP parameters for a single LLM layer (Megatron -> HF)."""
    mk = f"{mg_pfx}.mlp.linear_fc1.weight"
    if mk in full_sd:
        gate, up = torch.split(full_sd[mk], cfg.ffn_hidden_size, dim=0)
        hf_sd[f"{hf_pfx}.mlp.gate_proj.weight"] = gate
        hf_sd[f"{hf_pfx}.mlp.up_proj.weight"] = up

    mk = f"{mg_pfx}.mlp.linear_fc2.weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.mlp.down_proj.weight"] = full_sd[mk]

    mk = f"{mg_pfx}.mlp.linear_fc1.layer_norm_weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.post_attention_layernorm.weight"] = _restore_ln(full_sd[mk])


def convert_moe_mlp_meg2hf(full_sd, hf_sd, layer_idx, hf_pfx, mg_pfx, cfg):
    """Convert MoE MLP parameters for a single LLM layer (Megatron -> HF)."""
    mk = f"{mg_pfx}.pre_mlp_layernorm.weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.post_attention_layernorm.weight"] = _restore_ln(full_sd[mk])

    mk = f"{mg_pfx}.mlp.router.weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.mlp.gate.weight"] = full_sd[mk]

    gate_up_list = []
    down_list = []
    per_expert_gate_up = []
    per_expert_down = []
    all_use_stacked = True

    for e in range(cfg.num_experts):
        mk_fc1 = f"{mg_pfx}.mlp.experts.linear_fc1.weight{e}"
        mk_fc2 = f"{mg_pfx}.mlp.experts.linear_fc2.weight{e}"
        if mk_fc1 in full_sd:
            gate_up_list.append(full_sd[mk_fc1])
        else:
            all_use_stacked = False
        if mk_fc2 in full_sd:
            down_list.append(full_sd[mk_fc2])
        else:
            all_use_stacked = False

        # Collect per-expert individual projections when available
        if mk_fc1 in full_sd:
            gate, up = torch.split(full_sd[mk_fc1], cfg.moe_ffn_hidden_size, dim=0)
            per_expert_gate_up.append((gate, up))
        if mk_fc2 in full_sd:
            per_expert_down.append(full_sd[mk_fc2])

    # Prefer stacked output format when all experts are present
    if gate_up_list:
        if all_use_stacked:
            hf_sd[f"{hf_pfx}.mlp.experts.gate_up_proj"] = torch.stack(gate_up_list, dim=0)
        elif len(per_expert_gate_up) == cfg.num_experts:
            for e, (gate, up) in enumerate(per_expert_gate_up):
                hf_sd[f"{hf_pfx}.mlp.experts.{e}.gate_proj.weight"] = gate
                hf_sd[f"{hf_pfx}.mlp.experts.{e}.up_proj.weight"] = up

    if down_list:
        if all_use_stacked:
            hf_sd[f"{hf_pfx}.mlp.experts.down_proj"] = torch.stack(down_list, dim=0)
        elif len(per_expert_down) == cfg.num_experts:
            for e, down in enumerate(per_expert_down):
                hf_sd[f"{hf_pfx}.mlp.experts.{e}.down_proj.weight"] = down

    mk = f"{mg_pfx}.mlp.shared_experts.linear_fc1.weight"
    if mk in full_sd:
        gate, up = torch.split(full_sd[mk], cfg.moe_shared_expert_intermediate_size, dim=0)
        hf_sd[f"{hf_pfx}.mlp.shared_expert.gate_proj.weight"] = gate
        hf_sd[f"{hf_pfx}.mlp.shared_expert.up_proj.weight"] = up

    mk = f"{mg_pfx}.mlp.shared_experts.linear_fc2.weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.mlp.shared_expert.down_proj.weight"] = full_sd[mk]

    mk = f"{mg_pfx}.mlp.shared_experts.gate_weight"
    if mk in full_sd:
        hf_sd[f"{hf_pfx}.mlp.shared_expert_gate.weight"] = full_sd[mk]


def convert_layer_mlp_meg2hf(full_sd, hf_sd, layer_idx, hf_pfx, mg_pfx, cfg):
    """Route to dense or MoE MLP conversion based on config."""
    if cfg.is_moe:
        convert_moe_mlp_meg2hf(full_sd, hf_sd, layer_idx, hf_pfx, mg_pfx, cfg)
    else:
        convert_dense_mlp_meg2hf(full_sd, hf_sd, layer_idx, hf_pfx, mg_pfx, cfg)
