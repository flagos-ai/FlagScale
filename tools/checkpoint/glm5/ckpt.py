"""
GLM5 checkpoint conversion plugin.

Handles weight mapping between HuggingFace GlmMoeDsaForCausalLM and
Megatron-Core GPTModel with MLA + MoE + DSA.

Supports: TP (tensor parallel), PP (pipeline parallel), EP (expert parallel),
          ETP (expert tensor parallel).
"""

import sys

import torch

sys.path.append("..")
from utils import (
    get_expert_tensor_parallel_model_groups,
    get_expert_tensor_parallel_models,
    get_expert_tensor_parallel_size,
    get_tensor_model_parallel_rank,
    get_tensor_parallel_models,
    padding_vocab_size,
)


def _get_parallel_size(args):
    return (
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
        args.expert_model_parallel_size,
        args.virtual_pipeline_model_parallel_size or 1,
    )


# =============================================================================
# HF -> Megatron (load from HuggingFace checkpoint)
# =============================================================================


def get_hf_attn_ckpt(message, model, layer_id, args):
    """Extract MLA attention weights from HF model layer."""
    tf_layer = model.model.layers[layer_id]

    # MLA q path: q_a_proj -> q_a_layernorm -> q_b_proj
    if args.q_lora_rank is not None:
        message["q a weight"] = tf_layer.self_attn.q_a_proj.weight.data
        message["q a norm weight"] = tf_layer.self_attn.q_a_layernorm.weight.data
        message["q b weight"] = tf_layer.self_attn.q_b_proj.weight.data
    else:
        message["q weight"] = tf_layer.self_attn.q_proj.weight.data

    # MLA kv path: kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj
    message["kv a weight"] = tf_layer.self_attn.kv_a_proj_with_mqa.weight.data
    message["kv a norm weight"] = tf_layer.self_attn.kv_a_layernorm.weight.data
    message["kv b weight"] = tf_layer.self_attn.kv_b_proj.weight.data

    # Output projection
    message["o weight"] = tf_layer.self_attn.o_proj.weight.data

    # Layer norms
    message["input norm weight"] = tf_layer.input_layernorm.weight.data

    # post_norm_weight: for MoE layers it's stored in attn message,
    # for dense layers it's stored in MLP message (fused into linear_fc1)
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if layer_id >= first_k_dense_replace:
        message["post norm weight"] = tf_layer.post_attention_layernorm.weight.data

    # DSA indexer weights (only for "full" layers, skip "shared" layers)
    # MTP layers (layer_id >= num_layers) always have an indexer
    indexer_types = getattr(args, "indexer_types", None)
    layer_has_indexer = (
        indexer_types is None or layer_id >= len(indexer_types) or indexer_types[layer_id] == "full"
    )
    if layer_has_indexer and hasattr(tf_layer.self_attn, "indexer"):
        indexer = tf_layer.self_attn.indexer
        message["indexer wq_b weight"] = indexer.wq_b.weight.data
        message["indexer wk weight"] = indexer.wk.weight.data
        message["indexer k_norm weight"] = indexer.k_norm.weight.data
        message["indexer k_norm bias"] = indexer.k_norm.bias.data
        message["indexer weights_proj weight"] = indexer.weights_proj.weight.data


def get_hf_mlp_ckpt(message, model, layer_id, args):
    """Extract MLP weights from HF model layer (dense or MoE)."""
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if layer_id < first_k_dense_replace:
        _get_hf_dense_mlp_ckpt(message, model, layer_id, args)
    else:
        _get_hf_moe_mlp_ckpt(message, model, layer_id, args)


def _get_hf_dense_mlp_ckpt(message, model, layer_id, args):
    """Dense MLP layers (first_k_dense_replace layers)."""
    tf_layer = model.model.layers[layer_id]
    message["post norm weight"] = tf_layer.post_attention_layernorm.weight.data
    message["gate weight"] = tf_layer.mlp.gate_proj.weight.data
    message["up weight"] = tf_layer.mlp.up_proj.weight.data
    message["down weight"] = tf_layer.mlp.down_proj.weight.data


def _get_hf_moe_mlp_ckpt(message, model, layer_id, args):
    """MoE MLP layers."""
    tf_layer = model.model.layers[layer_id]

    # Router
    message["router weight"] = tf_layer.mlp.gate.weight.data
    if hasattr(tf_layer.mlp.gate, "e_score_correction_bias"):
        message["router expert bias"] = tf_layer.mlp.gate.e_score_correction_bias.data

    # Shared experts
    message["shared expert gate weight"] = tf_layer.mlp.shared_experts.gate_proj.weight.data
    message["shared expert up weight"] = tf_layer.mlp.shared_experts.up_proj.weight.data
    message["shared expert down weight"] = tf_layer.mlp.shared_experts.down_proj.weight.data

    # Routed experts
    # GlmMoeDsaNaiveMoe stores all expert weights as fused 3D tensors:
    #   gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
    #   down_proj:    [num_experts, hidden_dim, intermediate_dim]
    experts = tf_layer.mlp.experts
    intermediate_dim = experts.intermediate_dim
    for expert_id in range(args.num_experts):
        gate_up = experts.gate_up_proj.data[expert_id]  # [2 * intermediate_dim, hidden_dim]
        message[f"expert{expert_id} gate weight"] = gate_up[:intermediate_dim, :]
        message[f"expert{expert_id} up weight"] = gate_up[intermediate_dim:, :]
        message[f"expert{expert_id} down weight"] = experts.down_proj.data[expert_id]


def get_hf_mtp_ckpt(message, model, mtp_layer_id, args):
    """Extract MTP (Multi-Token Prediction) layer weights."""
    mtp_layer = model.model.layers[args.num_layers + mtp_layer_id]

    message["mtp enorm weight"] = mtp_layer.enorm.weight.data
    message["mtp hnorm weight"] = mtp_layer.hnorm.weight.data
    message["mtp eh weight"] = mtp_layer.eh_proj.weight.data
    message["mtp shared head norm weight"] = mtp_layer.shared_head.norm.weight.data

    # MTP layer has the same attn + mlp structure
    get_hf_attn_ckpt(message, model, args.num_layers + mtp_layer_id, args)
    _get_hf_moe_mlp_ckpt(message, model, args.num_layers + mtp_layer_id, args)


# =============================================================================
# HF -> Megatron: set weights into Megatron model (with TP/EP/ETP splitting)
# =============================================================================


def set_embedding_ckpt(message, models, md, args):
    """Set embedding weights into Megatron models (split across TP)."""
    tp_size, _, _, _ = _get_parallel_size(args)

    pos_embed = None
    if md.position_embedding_type == "learned_absolute":
        pos_embed = message.pop("position embeddings")
    orig_word_embed = message.pop("word embeddings")
    full_word_embed = padding_vocab_size(orig_word_embed, md, args)

    out_word_embed = torch.chunk(full_word_embed, tp_size, dim=0)
    for tp_ep_rank, model in enumerate(models):
        tp_rank = get_tensor_model_parallel_rank(tp_ep_rank, args)
        model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        if pos_embed is not None:
            model.embedding.position_embeddings.weight.data.copy_(pos_embed)


def set_attn_ckpt(message, models, layer_id, md, args):
    """Set MLA attention weights into Megatron models (split across TP)."""
    tp_size, _, _, _ = _get_parallel_size(args)

    # MLA q path
    if args.q_lora_rank is not None:
        q_a_weight = message.pop("q a weight")
        q_a_norm_weight = message.pop("q a norm weight")
        q_b_weight = torch.chunk(message.pop("q b weight"), tp_size, dim=0)
    else:
        q_weight = torch.chunk(message.pop("q weight"), tp_size, dim=0)

    # MLA kv path
    kv_a_weight = message.pop("kv a weight")
    kv_a_norm_weight = message.pop("kv a norm weight")
    kv_b_weight = torch.chunk(message.pop("kv b weight"), tp_size, dim=0)

    # Output projection (split along input dim)
    o_weight = torch.chunk(message.pop("o weight"), tp_size, dim=1)

    # Layer norms (replicated)
    input_norm_weight = message.pop("input norm weight")

    # post_norm_weight only exists for MoE layers (dense layers store it in mlp.linear_fc1.layer_norm_weight)
    first_k_dense_replace = args.moe_layer_freq.index(1)
    post_norm_weight = None
    if args.total_layer_num >= first_k_dense_replace:
        post_norm_weight = message.pop("post norm weight")

    # DSA indexer weights (replicated, not TP-split)
    # Only present for "full" layers; "share" layers have no indexer
    indexer_wq_b_weight = message.pop("indexer wq_b weight", None)
    indexer_wk_weight = message.pop("indexer wk weight", None)
    indexer_k_norm_weight = message.pop("indexer k_norm weight", None)
    indexer_k_norm_bias = message.pop("indexer k_norm bias", None)
    indexer_weights_proj_weight = message.pop("indexer weights_proj weight", None)

    for tp_ep_rank, model in enumerate(models):
        tp_rank = get_tensor_model_parallel_rank(tp_ep_rank, args)
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer  # for mtp

        # MLA q path
        if args.q_lora_rank is not None:
            tf_layer.self_attention.linear_q_down_proj.weight.data.copy_(q_a_weight)
            tf_layer.self_attention.linear_q_up_proj.layer_norm_weight.data.copy_(q_a_norm_weight)
            tf_layer.self_attention.linear_q_up_proj.weight.data.copy_(q_b_weight[tp_rank])
        else:
            tf_layer.self_attention.linear_q_proj.weight.data.copy_(q_weight[tp_rank])

        # MLA kv path
        tf_layer.self_attention.linear_kv_down_proj.weight.data.copy_(kv_a_weight)
        tf_layer.self_attention.linear_kv_up_proj.layer_norm_weight.data.copy_(kv_a_norm_weight)
        tf_layer.self_attention.linear_kv_up_proj.weight.data.copy_(kv_b_weight[tp_rank])

        # Output projection
        tf_layer.self_attention.linear_proj.weight.data.copy_(o_weight[tp_rank])

        # Layer norms
        tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)
        if post_norm_weight is not None:
            tf_layer.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)

        # DSA indexer (inside core_attention in Megatron model)
        # Only "full" layers have an indexer; "share" layers have indexer=None
        if indexer_wq_b_weight is not None and hasattr(tf_layer.self_attention, "core_attention"):
            core_attn = tf_layer.self_attention.core_attention
            if hasattr(core_attn, "indexer") and core_attn.indexer is not None:
                indexer = core_attn.indexer
                indexer.linear_wq_b.weight.data.copy_(indexer_wq_b_weight)
                indexer.linear_wk.weight.data.copy_(indexer_wk_weight)
                indexer.k_norm.weight.data.copy_(indexer_k_norm_weight)
                indexer.k_norm.bias.data.copy_(indexer_k_norm_bias)
                indexer.linear_weights_proj.weight.data.copy_(indexer_weights_proj_weight)


def set_mlp_ckpt(message, models, layer_id, md, args):
    """Set MLP weights into Megatron models (dense or MoE)."""
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num < first_k_dense_replace:
        _set_dense_mlp_ckpt(message, models, layer_id, md, args)
    else:
        _set_moe_mlp_ckpt(message, models, layer_id, md, args)


def _set_dense_mlp_ckpt(message, models, layer_id, md, args):
    """Set dense MLP weights with TP splitting."""
    tp_size, _, _, _ = _get_parallel_size(args)

    post_norm_weight = message.pop("post norm weight")
    gate_weight = torch.chunk(message.pop("gate weight"), tp_size, dim=0)
    up_weight = torch.chunk(message.pop("up weight"), tp_size, dim=0)
    down_weight = torch.chunk(message.pop("down weight"), tp_size, dim=1)

    for tp_ep_rank, model in enumerate(models):
        tp_rank = get_tensor_model_parallel_rank(tp_ep_rank, args)
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer

        # Megatron stores gate+up as concatenated fc1; layernorm weight is fused
        linear1_weight = torch.cat([gate_weight[tp_rank], up_weight[tp_rank]], dim=0)
        tf_layer.mlp.linear_fc1.layer_norm_weight.data.copy_(post_norm_weight)
        tf_layer.mlp.linear_fc1.weight.data.copy_(linear1_weight)
        tf_layer.mlp.linear_fc2.weight.data.copy_(down_weight[tp_rank])


def _set_moe_mlp_ckpt(message, models, layer_id, md, args):
    """Set MoE MLP weights with TP/EP/ETP splitting."""
    tp_size, _, ep_size, _ = _get_parallel_size(args)
    etp_size = get_expert_tensor_parallel_size(args)

    # Router (replicated)
    router_weight = message.pop("router weight")
    router_expert_bias = message.pop("router expert bias", None)

    # Shared experts (TP-split)
    shared_gate_weight = torch.chunk(message.pop("shared expert gate weight"), tp_size, dim=0)
    shared_up_weight = torch.chunk(message.pop("shared expert up weight"), tp_size, dim=0)
    shared_down_weight = torch.chunk(message.pop("shared expert down weight"), tp_size, dim=1)

    for tp_ep_rank, model in enumerate(models):
        tp_rank = get_tensor_model_parallel_rank(tp_ep_rank, args)
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer

        # Router
        tf_layer.mlp.router.weight.data.copy_(router_weight)
        if router_expert_bias is not None and hasattr(tf_layer.mlp.router, "expert_bias"):
            tf_layer.mlp.router.expert_bias.data.copy_(router_expert_bias)

        # Shared experts
        shared_fc1 = torch.cat([shared_gate_weight[tp_rank], shared_up_weight[tp_rank]], dim=0)
        tf_layer.mlp.shared_experts.linear_fc1.weight.data.copy_(shared_fc1)
        tf_layer.mlp.shared_experts.linear_fc2.weight.data.copy_(shared_down_weight[tp_rank])

    # Routed experts (EP + ETP split)
    assert args.num_experts % ep_size == 0
    num_local_experts = args.num_experts // ep_size

    for expert_id in range(num_local_experts):
        for ep_rank in range(ep_size):
            global_expert_id = ep_rank * num_local_experts + expert_id

            gate_weight = torch.chunk(
                message.pop(f"expert{global_expert_id} gate weight"), etp_size, dim=0
            )
            up_weight = torch.chunk(
                message.pop(f"expert{global_expert_id} up weight"), etp_size, dim=0
            )
            linear1_weight = [torch.cat([g, u], dim=0) for g, u in zip(gate_weight, up_weight)]
            linear2_weight = torch.chunk(
                message.pop(f"expert{global_expert_id} down weight"), etp_size, dim=1
            )

            etp_model_groups = get_expert_tensor_parallel_model_groups(models, args, ep_rank)
            for etp_rank, model_group in enumerate(etp_model_groups):
                for model in model_group:
                    if hasattr(model, "decoder"):
                        tf_layer = model.decoder.layers[layer_id]
                    else:
                        tf_layer = model.transformer_layer

                    if not args.moe_grouped_gemm:
                        expert = tf_layer.mlp.experts.local_experts[expert_id]
                        expert.linear_fc1.weight.data.copy_(linear1_weight[etp_rank])
                        expert.linear_fc2.weight.data.copy_(linear2_weight[etp_rank])
                    else:
                        expert_fc1 = getattr(
                            tf_layer.mlp.experts.linear_fc1, f"weight{expert_id}", None
                        )
                        expert_fc2 = getattr(
                            tf_layer.mlp.experts.linear_fc2, f"weight{expert_id}", None
                        )
                        expert_fc1.data.copy_(linear1_weight[etp_rank])
                        expert_fc2.data.copy_(linear2_weight[etp_rank])


def set_final_norm_ckpt(message, models, md, args):
    """Set final layernorm weights (replicated)."""
    final_norm_weight = message.pop("weight")
    for model in models:
        model.decoder.final_layernorm.weight.data.copy_(final_norm_weight)


def set_output_layer_ckpt(message, models, md, args):
    """Set output layer weights (TP-split)."""
    tp_size, _, _, _ = _get_parallel_size(args)

    orig_output_layer_weight = message.pop("weight")
    full_output_layer_weight = padding_vocab_size(orig_output_layer_weight, md, args)
    output_layer_weight = torch.chunk(full_output_layer_weight, tp_size, dim=0)
    for tp_ep_rank, model in enumerate(models):
        tp_rank = get_tensor_model_parallel_rank(tp_ep_rank, args)
        model.output_layer.weight.data.copy_(output_layer_weight[tp_rank])


def set_mtp_ckpt(message, models, md, mtp_layer_id, args):
    """Set MTP layer weights into Megatron models."""
    tp_size, _, _, _ = _get_parallel_size(args)

    # MTP layers are always MoE layers, ensure total_layer_num reflects this
    args.total_layer_num = args.num_layers

    mtp_layers = []
    for tp_ep_rank, model in enumerate(models):
        mtp_layer = model.mtp.layers[mtp_layer_id]
        mtp_layers.append(mtp_layer)

    # Set transformer (attn + mlp) weights
    set_attn_ckpt(message, mtp_layers, 0, md, args)
    _set_moe_mlp_ckpt(message, mtp_layers, 0, md, args)

    # Set MTP-specific weights
    mtp_enorm_weight = message.pop("mtp enorm weight")
    mtp_hnorm_weight = message.pop("mtp hnorm weight")
    mtp_eh_weight = torch.chunk(message.pop("mtp eh weight"), tp_size, dim=0)
    mtp_shared_head_norm_weight = message.pop("mtp shared head norm weight")

    for tp_ep_rank, model in enumerate(models):
        tp_rank = get_tensor_model_parallel_rank(tp_ep_rank, args)
        mtp_layer = model.mtp.layers[mtp_layer_id]
        mtp_layer.enorm.weight.data.copy_(mtp_enorm_weight)
        mtp_layer.hnorm.weight.data.copy_(mtp_hnorm_weight)
        mtp_layer.eh_proj.weight.data.copy_(mtp_eh_weight[tp_rank])
        mtp_layer.final_layernorm.weight.data.copy_(mtp_shared_head_norm_weight)


# =============================================================================
# Megatron -> HF: get weights from Megatron model (gather across TP/EP/ETP)
# =============================================================================


def get_embedding_ckpt(message, models, args):
    """Gather embedding weights from TP-split Megatron models."""
    word_embeddings = []
    for model in get_tensor_parallel_models(models, args):
        word_embeddings.append(model.embedding.word_embeddings.weight.data)
    message["word embeddings"] = torch.cat(word_embeddings, dim=0)


def get_attn_ckpt(message, models, layer_id, args):
    """Gather MLA attention weights from TP-split Megatron models."""
    tp_size, _, _, _ = _get_parallel_size(args)

    q_a_weight = None
    q_a_norm_weight = None
    q_b_weight = []
    q_weight = []
    kv_a_weight = None
    kv_a_norm_weight = None
    kv_b_weight = []
    o_weight = []
    input_norm_weight = None
    post_norm_weight = None

    # DSA indexer (non-parallel, take from first TP rank)
    indexer_wq_b_weight = None
    indexer_wk_weight = None
    indexer_k_norm_weight = None
    indexer_k_norm_bias = None
    indexer_weights_proj_weight = None

    first_k_dense_replace = args.moe_layer_freq.index(1)

    for model in get_tensor_parallel_models(models, args):
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer

        if args.q_lora_rank is not None:
            q_a_weight = tf_layer.self_attention.linear_q_down_proj.weight.data
            q_a_norm_weight = tf_layer.self_attention.linear_q_up_proj.layer_norm_weight.data
            q_b_weight.append(tf_layer.self_attention.linear_q_up_proj.weight.data)
        else:
            q_weight.append(tf_layer.self_attention.linear_q_proj.weight.data)

        kv_a_weight = tf_layer.self_attention.linear_kv_down_proj.weight.data
        kv_a_norm_weight = tf_layer.self_attention.linear_kv_up_proj.layer_norm_weight.data
        kv_b_weight.append(tf_layer.self_attention.linear_kv_up_proj.weight.data)
        o_weight.append(tf_layer.self_attention.linear_proj.weight.data)
        input_norm_weight = tf_layer.input_layernorm.weight.data

        # post_norm_weight only for MoE layers (from pre_mlp_layernorm)
        if args.total_layer_num >= first_k_dense_replace:
            post_norm_weight = tf_layer.pre_mlp_layernorm.weight.data

        # DSA indexer (inside core_attention, take from first model)
        # Only "full" layers have an indexer; "share" layers have indexer=None
        if indexer_wq_b_weight is None and hasattr(tf_layer.self_attention, "core_attention"):
            core_attn = tf_layer.self_attention.core_attention
            if hasattr(core_attn, "indexer") and core_attn.indexer is not None:
                indexer = core_attn.indexer
                indexer_wq_b_weight = indexer.linear_wq_b.weight.data
                indexer_wk_weight = indexer.linear_wk.weight.data
                indexer_k_norm_weight = indexer.k_norm.weight.data
                indexer_k_norm_bias = indexer.k_norm.bias.data
                indexer_weights_proj_weight = indexer.linear_weights_proj.weight.data

    if args.q_lora_rank is not None:
        message["q a weight"] = q_a_weight
        message["q a norm weight"] = q_a_norm_weight
        message["q b weight"] = torch.cat(q_b_weight, dim=0)
    else:
        message["q weight"] = torch.cat(q_weight, dim=0)

    message["kv a weight"] = kv_a_weight
    message["kv a norm weight"] = kv_a_norm_weight
    message["kv b weight"] = torch.cat(kv_b_weight, dim=0)
    message["o weight"] = torch.cat(o_weight, dim=1)
    message["input norm weight"] = input_norm_weight

    if args.total_layer_num >= first_k_dense_replace:
        message["post norm weight"] = post_norm_weight

    if indexer_wq_b_weight is not None:
        message["indexer wq_b weight"] = indexer_wq_b_weight
        message["indexer wk weight"] = indexer_wk_weight
        message["indexer k_norm weight"] = indexer_k_norm_weight
        message["indexer k_norm bias"] = indexer_k_norm_bias
        message["indexer weights_proj weight"] = indexer_weights_proj_weight


def get_mlp_ckpt(message, models, layer_id, args):
    """Gather MLP weights from Megatron models."""
    first_k_dense_replace = args.moe_layer_freq.index(1)
    if args.total_layer_num < first_k_dense_replace:
        _get_dense_mlp_ckpt(message, models, layer_id, args)
    else:
        _get_moe_mlp_ckpt(message, models, layer_id, args)


def _get_dense_mlp_ckpt(message, models, layer_id, args):
    """Gather dense MLP weights from TP-split Megatron models."""
    tp_size, _, _, _ = _get_parallel_size(args)

    post_norm_weight = None
    linear1_weight = []
    linear2_weight = []

    for model in get_tensor_parallel_models(models, args):
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer
        post_norm_weight = tf_layer.mlp.linear_fc1.layer_norm_weight.data
        linear1_weight.append(tf_layer.mlp.linear_fc1.weight.data)
        linear2_weight.append(tf_layer.mlp.linear_fc2.weight.data)

    # Split gate and up from concatenated fc1
    for tp_rank in range(tp_size):
        linear1_weight[tp_rank] = torch.chunk(linear1_weight[tp_rank], 2, dim=0)

    message["post norm weight"] = post_norm_weight
    message["gate weight"] = torch.cat([w[0] for w in linear1_weight], dim=0)
    message["up weight"] = torch.cat([w[1] for w in linear1_weight], dim=0)
    message["down weight"] = torch.cat(linear2_weight, dim=1)


def _get_moe_mlp_ckpt(message, models, layer_id, args):
    """Gather MoE MLP weights from EP/ETP-split Megatron models."""
    tp_size, _, ep_size, _ = _get_parallel_size(args)
    etp_size = get_expert_tensor_parallel_size(args)

    assert args.num_experts is not None and args.num_experts % ep_size == 0
    num_local_experts = args.num_experts // ep_size

    # Router (replicated, take from first model)
    first_model = models[0]
    if hasattr(first_model, "decoder"):
        first_tf_layer = first_model.decoder.layers[layer_id]
    else:
        first_tf_layer = first_model.transformer_layer
    message["router weight"] = first_tf_layer.mlp.router.weight.data
    if hasattr(first_tf_layer.mlp.router, "expert_bias"):
        message["router expert bias"] = first_tf_layer.mlp.router.expert_bias.data

    # Shared experts (gather from TP)
    shared_gate_weight = []
    shared_up_weight = []
    shared_down_weight = []
    for model in get_tensor_parallel_models(models, args):
        if hasattr(model, "decoder"):
            tf_layer = model.decoder.layers[layer_id]
        else:
            tf_layer = model.transformer_layer
        shared_fc1 = tf_layer.mlp.shared_experts.linear_fc1.weight.data
        gate_w, up_w = torch.chunk(shared_fc1, 2, dim=0)
        shared_gate_weight.append(gate_w)
        shared_up_weight.append(up_w)
        shared_down_weight.append(tf_layer.mlp.shared_experts.linear_fc2.weight.data)

    message["shared expert gate weight"] = torch.cat(shared_gate_weight, dim=0)
    message["shared expert up weight"] = torch.cat(shared_up_weight, dim=0)
    message["shared expert down weight"] = torch.cat(shared_down_weight, dim=1)

    # Routed experts (gather from EP x ETP)
    for expert_id in range(num_local_experts):
        for ep_rank in range(ep_size):
            global_expert_id = ep_rank * num_local_experts + expert_id

            expert_linear1_weight = []
            expert_linear2_weight = []
            for model in get_expert_tensor_parallel_models(models, args, ep_rank):
                if hasattr(model, "decoder"):
                    tf_layer = model.decoder.layers[layer_id]
                else:
                    tf_layer = model.transformer_layer

                if not args.moe_grouped_gemm:
                    expert = tf_layer.mlp.experts.local_experts[expert_id]
                    expert_linear1_weight.append(expert.linear_fc1.weight.data)
                    expert_linear2_weight.append(expert.linear_fc2.weight.data)
                else:
                    expert_linear1_weight.append(
                        getattr(
                            tf_layer.mlp.experts.linear_fc1, f"weight{expert_id}", None
                        ).detach()
                    )
                    expert_linear2_weight.append(
                        getattr(
                            tf_layer.mlp.experts.linear_fc2, f"weight{expert_id}", None
                        ).detach()
                    )

            # Split gate/up per ETP rank, then concatenate across ETP
            for etp_rank in range(etp_size):
                expert_linear1_weight[etp_rank] = torch.chunk(
                    expert_linear1_weight[etp_rank], 2, dim=0
                )

            message[f"expert{global_expert_id} gate weight"] = torch.cat(
                [w[0] for w in expert_linear1_weight], dim=0
            )
            message[f"expert{global_expert_id} up weight"] = torch.cat(
                [w[1] for w in expert_linear1_weight], dim=0
            )
            message[f"expert{global_expert_id} down weight"] = torch.cat(
                expert_linear2_weight, dim=1
            )


def get_final_norm_ckpt(message, models, args):
    """Get final layernorm weight."""
    message["weight"] = models[0].decoder.final_layernorm.weight.data


def get_output_layer_ckpt(message, models, args):
    """Gather output layer weights from TP-split models."""
    output_layer_weight = []
    for model in get_tensor_parallel_models(models, args):
        output_layer_weight.append(model.output_layer.weight.data)
    message["weight"] = torch.cat(output_layer_weight, dim=0)


def get_mtp_ckpt(message, models, mtp_layer_id, args):
    """Gather MTP layer weights from Megatron models."""
    tp_size, _, _, _ = _get_parallel_size(args)

    # MTP layers are always MoE layers, ensure total_layer_num reflects this
    args.total_layer_num = args.num_layers

    mtp_layers = []
    for tp_ep_rank, model in enumerate(models):
        mtp_layer = model.mtp.layers[mtp_layer_id]
        mtp_layers.append(mtp_layer)

    # Gather transformer (attn + mlp) weights
    get_attn_ckpt(message, mtp_layers, 0, args)
    _get_moe_mlp_ckpt(message, mtp_layers, 0, args)

    # Gather MTP-specific weights
    mtp_eh_weight = []
    for model in get_tensor_parallel_models(models, args):
        mtp_layer = model.mtp.layers[mtp_layer_id]
        mtp_enorm_weight = mtp_layer.enorm.weight.data
        mtp_hnorm_weight = mtp_layer.hnorm.weight.data
        mtp_eh_weight.append(mtp_layer.eh_proj.weight.data)
        mtp_norm_weight = mtp_layer.final_layernorm.weight.data

    message["mtp enorm weight"] = mtp_enorm_weight
    message["mtp hnorm weight"] = mtp_hnorm_weight
    message["mtp eh weight"] = torch.cat(mtp_eh_weight, dim=0)
    message["mtp shared head norm weight"] = mtp_norm_weight


# =============================================================================
# Megatron -> HF: set weights into HuggingFace model
# =============================================================================


def set_hf_embedding_ckpt(message, hf_model, md, margs):
    """Set embedding weights into HF model."""
    word_embeddings = message.pop("word embeddings")
    hf_model.model.embed_tokens.weight.data.copy_(word_embeddings[: margs.vocab_size, :])


def set_hf_attn_ckpt(message, hf_model, layer_id, md, margs):
    """Set MLA attention weights into HF model layer."""
    tf_layer = hf_model.model.layers[layer_id]

    if margs.q_lora_rank is not None:
        q_a_weight = message.pop("q a weight")
        q_a_norm_weight = message.pop("q a norm weight")
        q_b_weight = message.pop("q b weight")
        tf_layer.self_attn.q_a_proj.weight.data.copy_(q_a_weight)
        tf_layer.self_attn.q_a_layernorm.weight.data.copy_(q_a_norm_weight)
        tf_layer.self_attn.q_b_proj.weight.data.copy_(q_b_weight)
    else:
        q_weight = message.pop("q weight")
        tf_layer.self_attn.q_proj.weight.data.copy_(q_weight)

    kv_a_weight = message.pop("kv a weight")
    kv_a_norm_weight = message.pop("kv a norm weight")
    kv_b_weight = message.pop("kv b weight")
    o_weight = message.pop("o weight")
    input_norm_weight = message.pop("input norm weight")

    tf_layer.self_attn.kv_a_proj_with_mqa.weight.data.copy_(kv_a_weight)
    tf_layer.self_attn.kv_a_layernorm.weight.data.copy_(kv_a_norm_weight)
    tf_layer.self_attn.kv_b_proj.weight.data.copy_(kv_b_weight)
    tf_layer.self_attn.o_proj.weight.data.copy_(o_weight)
    tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)

    # post_norm_weight: for MoE layers stored in attn message, for dense in MLP message
    first_k_dense_replace = margs.moe_layer_freq.index(1)
    if margs.total_layer_num >= first_k_dense_replace:
        post_norm_weight = message.pop("post norm weight")
        tf_layer.post_attention_layernorm.weight.data.copy_(post_norm_weight)

    # DSA indexer weights (only for "full" layers)
    indexer_wq_b = message.pop("indexer wq_b weight", None)
    indexer_wk = message.pop("indexer wk weight", None)
    indexer_k_norm_w = message.pop("indexer k_norm weight", None)
    indexer_k_norm_b = message.pop("indexer k_norm bias", None)
    indexer_wp = message.pop("indexer weights_proj weight", None)

    if indexer_wq_b is not None and hasattr(tf_layer.self_attn, "indexer"):
        indexer = tf_layer.self_attn.indexer
        if indexer is not None:
            indexer.wq_b.weight.data.copy_(indexer_wq_b)
            indexer.wk.weight.data.copy_(indexer_wk)
            indexer.k_norm.weight.data.copy_(indexer_k_norm_w)
            indexer.k_norm.bias.data.copy_(indexer_k_norm_b)
            indexer.weights_proj.weight.data.copy_(indexer_wp)


def set_hf_mlp_ckpt(message, hf_model, layer_id, md, margs):
    """Set MLP weights into HF model layer."""
    first_k_dense_replace = margs.moe_layer_freq.index(1)
    if margs.total_layer_num < first_k_dense_replace:
        _set_hf_dense_mlp_ckpt(message, hf_model, layer_id, md, margs)
    else:
        _set_hf_moe_mlp_ckpt(message, hf_model, layer_id, md, margs)


def _set_hf_dense_mlp_ckpt(message, hf_model, layer_id, md, margs):
    """Set dense MLP weights into HF model."""
    tf_layer = hf_model.model.layers[layer_id]
    tf_layer.post_attention_layernorm.weight.data.copy_(message.pop("post norm weight"))
    tf_layer.mlp.gate_proj.weight.data.copy_(message.pop("gate weight"))
    tf_layer.mlp.up_proj.weight.data.copy_(message.pop("up weight"))
    tf_layer.mlp.down_proj.weight.data.copy_(message.pop("down weight"))


def _set_hf_moe_mlp_ckpt(message, hf_model, layer_id, md, margs):
    """Set MoE MLP weights into HF model."""
    tf_layer = hf_model.model.layers[layer_id]

    router_weight = message.pop("router weight")
    tf_layer.mlp.gate.weight.data.copy_(router_weight)
    if "router expert bias" in message:
        router_expert_bias = message.pop("router expert bias")
        if hasattr(tf_layer.mlp.gate, "e_score_correction_bias"):
            tf_layer.mlp.gate.e_score_correction_bias.data.copy_(router_expert_bias)

    # Shared experts
    tf_layer.mlp.shared_experts.gate_proj.weight.data.copy_(
        message.pop("shared expert gate weight")
    )
    tf_layer.mlp.shared_experts.up_proj.weight.data.copy_(message.pop("shared expert up weight"))
    tf_layer.mlp.shared_experts.down_proj.weight.data.copy_(
        message.pop("shared expert down weight")
    )

    # Routed experts
    # GlmMoeDsaNaiveMoe stores all expert weights as fused 3D tensors:
    #   gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
    #   down_proj:    [num_experts, hidden_dim, intermediate_dim]
    experts = tf_layer.mlp.experts
    for expert_id in range(margs.num_experts):
        gate_weight = message.pop(f"expert{expert_id} gate weight")
        up_weight = message.pop(f"expert{expert_id} up weight")
        down_weight = message.pop(f"expert{expert_id} down weight")
        experts.gate_up_proj.data[expert_id] = torch.cat([gate_weight, up_weight], dim=0)
        experts.down_proj.data[expert_id] = down_weight


def set_hf_final_norm_ckpt(message, hf_model, md, margs):
    """Set final layernorm weight into HF model."""
    hf_model.model.norm.weight.data.copy_(message.pop("weight"))


def set_hf_output_layer_ckpt(message, hf_model, md, margs):
    """Set output layer (lm_head) weight into HF model."""
    output_weight = message.pop("weight")
    hf_model.lm_head.weight.data.copy_(output_weight[: margs.vocab_size, :])


def set_hf_mtp_ckpt(message, hf_model, mtp_layer_id, md, margs):
    """Set MTP layer weights into HF model."""
    layer_id = margs.num_layers + mtp_layer_id
    # MTP layers are always MoE layers, ensure total_layer_num reflects this
    margs.total_layer_num = margs.num_layers
    set_hf_attn_ckpt(message, hf_model, layer_id, md, margs)
    _set_hf_moe_mlp_ckpt(message, hf_model, layer_id, md, margs)

    mtp_layer = hf_model.model.layers[layer_id]
    mtp_layer.enorm.weight.data.copy_(message.pop("mtp enorm weight"))
    mtp_layer.hnorm.weight.data.copy_(message.pop("mtp hnorm weight"))
    mtp_layer.eh_proj.weight.data.copy_(message.pop("mtp eh weight"))
    mtp_layer.shared_head.norm.weight.data.copy_(message.pop("mtp shared head norm weight"))
