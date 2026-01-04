"""
Computes theoretical memory footprint for model training referring to megatron.
Reference: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/theoretical_memory_usage.py
Activation memory is optimized with adding block recompute formula.
"""

import math
import os

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_activated_weight_number(args, verbose=False):
    if args.num_experts is None:
        return

    # Part 1: Attention ======================================================================
    if args.multi_latent_attention:
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        if args.q_lora_rank is None:
            attn_params = args.hidden_size * args.num_attention_heads * q_head_dim
        else:
            attn_params = (
                args.hidden_size * args.q_lora_rank
                + args.q_lora_rank * args.num_attention_heads * q_head_dim
            )

        attn_params += args.hidden_size * (
            args.kv_lora_rank + args.qk_pos_emb_head_dim
        ) + args.kv_lora_rank * args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
        # out proj
        attn_params += args.v_head_dim * args.num_attention_heads * args.hidden_size
        # pre attn layernorm
        attn_params += 2 * args.hidden_size

        if args.qk_layernorm and args.q_lora_rank is None:
            attn_params += args.kv_lora_rank
        elif args.qk_layernorm:
            attn_params += args.kv_lora_rank + args.q_lora_rank
    else:
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        kv_projection_size = args.kv_channels * args.num_query_groups

        # qkv proj
        attn_params = args.hidden_size * (query_projection_size + 2 * kv_projection_size)
        # out proj
        attn_params += query_projection_size * args.hidden_size
        # pre attn layernorm
        attn_params += 2 * args.hidden_size

        if args.qk_layernorm:
            if not args.qk_layernorm_hidden_dim:
                attn_params += 2 * query_projection_size // args.num_attention_heads
            else:
                attn_params += query_projection_size
                attn_params += kv_projection_size

    # Part 2: MLP or MoE =====================================================================
    moe_ffn_hidden_size = (
        args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
    )
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    # SwiGLU.
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    # MoE.
    if args.num_experts is None:
        # Every Transformer MLP is dense.
        num_dense_layers = args.num_layers
        num_moe_layers = 0
        num_experts = 0
        num_experts_routed_to = 0
    else:
        # Calculate number of dense and MoE Transformer MLPs.
        if isinstance(args.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
            ]
        elif isinstance(args.moe_layer_freq, list):
            moe_layer_pattern = args.moe_layer_freq
        else:
            raise RuntimeError("Illegal --moe-layer-freq argument provided!")
        assert len(moe_layer_pattern) == args.num_layers
        num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
        num_dense_layers = args.num_layers - num_moe_layers
        num_experts = args.num_experts
        num_experts_routed_to = args.moe_router_topk

    dense_mlp_params = (
        2 * args.hidden_size * (args.ffn_hidden_size * gated_linear_multiplier)
        # pre mlp layernorm
        + 2 * args.hidden_size
    )

    sparse_mlp_params = (
        2
        * args.hidden_size
        * (
            +(moe_ffn_hidden_size * num_experts_routed_to * gated_linear_multiplier)
            + (shared_expert_ffn_hidden_size * gated_linear_multiplier)
        )
        # gate
        + args.hidden_size * num_experts
        # pre mlp layernorm
        + 2 * args.hidden_size
    )

    # Part3: MTP ============================================================================
    if args.mtp_num_layers is not None:
        mtp_layer_is_moe = moe_layer_pattern[-1]
        mtp_num_moe_layers = mtp_layer_is_moe * args.mtp_num_layers
        mtp_num_dense_layers = (1 - mtp_layer_is_moe) * args.mtp_num_layers
    else:
        mtp_num_moe_layers = 0
        mtp_num_dense_layers = 0

    num_parameters_in_mtp_block = 0
    if args.mtp_num_layers is not None:
        num_parameters_in_mtp_block = (
            4 * args.hidden_size  # tow layernorm ops
            + 2 * args.hidden_size * args.hidden_size  # one linear
            + mtp_num_dense_layers * (attn_params + dense_mlp_params)
            + mtp_num_moe_layers * (attn_params + sparse_mlp_params)
            + 2 * args.hidden_size  # final norm
        )

    # PART4: TOTAL ===========================================================================
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    num_parameters_in_transformer_block = (
        (num_dense_layers * (attn_params + dense_mlp_params))
        + (num_moe_layers * (attn_params + sparse_mlp_params))
        + 2 * args.hidden_size  # final layernorm
    )
    num_total_parameters = (
        num_parameters_in_transformer_block
        + num_parameters_in_mtp_block
        + num_parameters_in_embedding_layers
    )
    if verbose:
        print(
            f"> Number of activated attn parameters in a transformer block in billions: "
            f"{attn_params / 10**9: .2f}"
        )
        print(
            f"> Number of activated dense mlp parameters in a transformer block in billions: "
            f"{dense_mlp_params / 10**9: .2f}"
        )
        print(
            f"> Number of activated sparse mlp parameters in a transformer block in billions: "
            f"{sparse_mlp_params / 10**9: .2f}"
        )
        print(
            f"> Number of activated parameters in transformer block in billions: "
            f"{num_parameters_in_transformer_block / 10**9: .2f}"
        )
        print(
            f"> Number of activated parameters in mtp transformer block in billions: "
            f"{num_parameters_in_mtp_block / 10**9: .2f}"
        )
        print(
            f"> Number of activated parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(
            f"> Total number of activated parameters in billions: {num_total_parameters / 10**9:.2f}"
        )


def compute_weight_and_optimizer_memory(args, verbose=False):
    # Part 1: Attention =======================================================================
    if args.multi_latent_attention:
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        if args.q_lora_rank is None:
            attn_params = args.hidden_size * args.num_attention_heads * q_head_dim
        else:
            attn_params = (
                args.hidden_size * args.q_lora_rank
                + args.q_lora_rank * args.num_attention_heads * q_head_dim
            )

        attn_params += args.hidden_size * (
            args.kv_lora_rank + args.qk_pos_emb_head_dim
        ) + args.kv_lora_rank * args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
        # out proj
        attn_params += args.v_head_dim * args.num_attention_heads * args.hidden_size
        # pre attn layernorm
        attn_params += 2 * args.hidden_size

        if args.qk_layernorm and args.q_lora_rank is None:
            attn_params += args.kv_lora_rank
        elif args.qk_layernorm:
            attn_params += args.kv_lora_rank + args.q_lora_rank
    else:
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        kv_projection_size = args.kv_channels * args.num_query_groups

        # qkv proj
        attn_params = args.hidden_size * (query_projection_size + 2 * kv_projection_size)
        # out proj
        attn_params += query_projection_size * args.hidden_size
        # pre attn layernorm
        attn_params += 2 * args.hidden_size

        if args.qk_layernorm:
            if not args.qk_layernorm_hidden_dim:
                attn_params += 2 * query_projection_size // args.num_attention_heads
            else:
                attn_params += query_projection_size
                attn_params += kv_projection_size

    # Part 2: MLP or MoE ====================================================================
    moe_ffn_hidden_size = (
        args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
    )
    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )
    # SwiGLU.
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    if args.num_experts is None:
        # Every Transformer MLP is dense.
        num_dense_layers = args.num_layers
        num_moe_layers = 0
        num_experts = 0
    else:
        # Calculate number of dense and MoE Transformer MLPs.
        if isinstance(args.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
            ]
        elif isinstance(args.moe_layer_freq, list):
            moe_layer_pattern = args.moe_layer_freq
        else:
            raise RuntimeError("Illegal --moe-layer-freq argument provided!")
        assert len(moe_layer_pattern) == args.num_layers
        num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
        num_dense_layers = args.num_layers - num_moe_layers
        num_experts = args.num_experts

    dense_mlp_params = (
        2 * args.hidden_size * (args.ffn_hidden_size * gated_linear_multiplier)
        # pre mlp layernorm
        + 2 * args.hidden_size
    )

    sparse_mlp_params = (
        2
        * args.hidden_size
        * (
            # MoE mlp
            +(moe_ffn_hidden_size * num_experts * gated_linear_multiplier)
            # Shared MoE mlp
            + (shared_expert_ffn_hidden_size * gated_linear_multiplier)
        )
        # gate
        + args.hidden_size * num_experts
        # pre mlp layernorm
        + 2 * args.hidden_size
    )

    # Part3: MTP ============================================================================
    if args.mtp_num_layers is not None:
        mtp_layer_is_moe = moe_layer_pattern[-1]
        mtp_num_moe_layers = mtp_layer_is_moe * args.mtp_num_layers
        mtp_num_dense_layers = (1 - mtp_layer_is_moe) * args.mtp_num_layers
    else:
        mtp_num_moe_layers = 0
        mtp_num_dense_layers = 0

    num_parameters_in_mtp_block = 0
    if args.mtp_num_layers is not None:
        num_parameters_in_mtp_block = (
            4 * args.hidden_size  # tow layernorm ops
            + 2 * args.hidden_size * args.hidden_size  # one linear
            + mtp_num_dense_layers * (attn_params + dense_mlp_params)
            + mtp_num_moe_layers * (attn_params + sparse_mlp_params)
            + 2 * args.hidden_size  # final norm
        )

    # PART4: TOTAL ===========================================================================
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    num_parameters_in_transformer_block = (
        (num_dense_layers * (attn_params + dense_mlp_params))
        + (num_moe_layers * (attn_params + sparse_mlp_params))
        + 2 * args.hidden_size  # final layernorm
    )
    num_total_parameters = (
        num_parameters_in_transformer_block
        + num_parameters_in_mtp_block
        + num_parameters_in_embedding_layers
    )
    if verbose:
        print(
            f"> Number of attn parameters in a transformer block in billions: "
            f"{attn_params / 10**9: .2f}"
        )
        print(
            f"> Number of dense mlp parameters in a transformer block in billions: "
            f"{dense_mlp_params / 10**9: .2f}"
        )
        print(
            f"> Number of sparse mlp parameters in a transformer block in billions: "
            f"{sparse_mlp_params / 10**9: .2f}"
        )
        print(
            f"> Number of parameters in transformer block in billions: "
            f"{num_parameters_in_transformer_block / 10**9: .2f}"
        )
        print(
            f"> Number of parameters in mtp transformer block in billions: "
            f"{num_parameters_in_mtp_block / 10**9: .2f}"
        )
        print(
            f"> Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"> Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # PART5: Distributed =====================================================================
    sparse_mlp_params_per_ep_rank_ddp = (
        2
        * args.hidden_size
        # MoE mlp
        * (
            moe_ffn_hidden_size
            * gated_linear_multiplier
            * (num_experts / args.expert_model_parallel_size)
        )
    )

    sparse_mlp_params_per_ep_rank_noddp = (
        2 * args.hidden_size
        # Shared MoE mlp
        * (shared_expert_ffn_hidden_size * gated_linear_multiplier)
        # gate
        + args.hidden_size * num_experts
        # pre mlp layernorm
        + 2 * args.hidden_size
    )

    # for hetero (TODO: suppot hetero train)
    expert_tensor_parallel_size = (
        args.expert_tensor_parallel_size
        if args.expert_tensor_parallel_size is not None
        else args.tensor_model_parallel_size
    )

    num_parameters_in_transformer_layers_per_tp_ep_rank_ddp = num_moe_layers * (
        sparse_mlp_params_per_ep_rank_ddp / expert_tensor_parallel_size
    )

    num_parameters_in_transformer_layers_per_tp_ep_rank_noddp = (
        num_dense_layers * (attn_params + dense_mlp_params) / args.tensor_model_parallel_size
        + num_moe_layers
        * (attn_params + sparse_mlp_params_per_ep_rank_noddp)
        / args.tensor_model_parallel_size
        + 2 * args.hidden_size  # final layernorm
    )

    num_parameters_in_mtp_block_per_tp_ep_rank_ddp = mtp_num_moe_layers * (
        sparse_mlp_params_per_ep_rank_ddp / expert_tensor_parallel_size
    )

    num_parameters_in_mtp_block_per_tp_ep_rank_noddp = mtp_num_moe_layers * (
        4 * args.hidden_size  # tow layernorm ops
        + 2 * args.hidden_size * args.hidden_size  # one linear
        + (attn_params + sparse_mlp_params_per_ep_rank_noddp) / args.tensor_model_parallel_size
        + 2 * args.hidden_size  # final layernorm
    )

    num_parameters_on_most_loaded_model_shard_ddp = (
        num_parameters_in_transformer_layers_per_tp_ep_rank_ddp / args.pipeline_model_parallel_size
        + num_parameters_in_mtp_block_per_tp_ep_rank_ddp
    )

    num_parameters_on_most_loaded_model_shard_noddp = (
        num_parameters_in_transformer_layers_per_tp_ep_rank_noddp
        / args.pipeline_model_parallel_size
        + embedding_size / args.tensor_model_parallel_size
        + num_parameters_in_mtp_block_per_tp_ep_rank_noddp
    )

    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard_noddp += (
            embedding_size / args.tensor_model_parallel_size
        )

    if verbose:
        num_parameters_on_most_loaded_model_shard = (
            num_parameters_on_most_loaded_model_shard_ddp
            + num_parameters_on_most_loaded_model_shard_noddp
        )
        print(
            f"> Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = (
            num_parameters_in_transformer_layers_per_tp_ep_rank_ddp
            + num_parameters_in_transformer_layers_per_tp_ep_rank_noddp
        ) / args.pipeline_model_parallel_size
        if verbose:
            print(
                f"> Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    if args.use_distributed_optimizer:
        expert_tensor_model_pipeline_parallel_size = (
            expert_tensor_parallel_size
            * args.expert_model_parallel_size
            * args.pipeline_model_parallel_size
        )
        expert_data_parallel_size = args.world_size // expert_tensor_model_pipeline_parallel_size
        weight_and_optimizer_memory = num_parameters_on_most_loaded_model_shard_ddp * (
            6 + 12 / expert_data_parallel_size
        ) + num_parameters_on_most_loaded_model_shard_noddp * (6 + 12 / args.data_parallel_size)
    else:
        weight_and_optimizer_memory = (
            num_parameters_on_most_loaded_model_shard_ddp
            + num_parameters_on_most_loaded_model_shard_noddp
        ) * 18

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # Megatron TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.
    # But in FlagScale, we implement it.

    # Memory footprint from transformer layer (self-attention and MLP).
    # In FlagScale, we provide the most detailed memory information, including each sub layer of transformer layer.
    # Pre-attn layernorm
    pre_attn_layernorm_activation_memory = (
        2 * args.seq_length * args.micro_batch_size * args.hidden_size
    )
    # Attention:
    if args.multi_latent_attention:
        # 1. Q, K, V matrix multiplies
        if args.q_lora_rank is None:
            QKV_activation_memory = (
                2 * args.seq_length * args.micro_batch_size * args.hidden_size
                + 4 * args.seq_length * args.micro_batch_size * args.kv_lora_rank
            )
        else:
            QKV_activation_memory = (
                2 * args.seq_length * args.micro_batch_size * args.hidden_size
                + 4 * args.seq_length * args.micro_batch_size * args.q_lora_rank
                + 4 * args.seq_length * args.micro_batch_size * args.kv_lora_rank
            )
        # 2. QKT matrix multiply
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        QKT_activation_memory = (
            4 * args.micro_batch_size * args.num_attention_heads * args.seq_length * q_head_dim
        )
        # 3. Softmax
        softmax_activation_memory = (
            2 * args.micro_batch_size * args.num_attention_heads * args.seq_length * args.seq_length
        )
        # 4. Softmax Dropout
        softmax_dropout_activation_memory = (
            args.micro_batch_size * args.num_attention_heads * args.seq_length * args.seq_length
        )
        # 5. Attention over V
        attention_over_V_activation_memory = (
            2 * args.micro_batch_size * args.num_attention_heads * args.seq_length * args.seq_length
            + 2
            * args.micro_batch_size
            * args.num_attention_heads
            * args.seq_length
            * args.v_head_dim
        )
        # 6. Linear
        linear_activation_memory = (
            2 * args.micro_batch_size * args.num_attention_heads * args.seq_length * args.v_head_dim
        )
        # 7. linear dropout
        linear_dropout_activation_memory = (
            args.seq_length * args.micro_batch_size * args.hidden_size
        )
    else:
        # 1. Q, K, V matrix multiplies
        QKV_activation_memory = 2 * args.seq_length * args.micro_batch_size * args.hidden_size
        # 2. QKT matrix multiply
        QKT_activation_memory = (
            2
            * args.micro_batch_size
            * args.num_attention_heads
            * args.seq_length
            * args.kv_channels
            + 2 * args.micro_batch_size * args.num_query_groups * args.kv_channels * args.seq_length
        )
        # 3. Softmax
        softmax_activation_memory = (
            2 * args.micro_batch_size * args.num_attention_heads * args.seq_length * args.seq_length
        )
        # 4. Softmax Dropout
        softmax_dropout_activation_memory = (
            args.micro_batch_size * args.num_attention_heads * args.seq_length * args.seq_length
        )
        # 5. Attention over V
        attention_over_V_activation_memory = (
            2 * args.micro_batch_size * args.num_attention_heads * args.seq_length * args.seq_length
            + 2 * args.micro_batch_size * args.num_query_groups * args.kv_channels * args.seq_length
        )
        # 6. Linear
        linear_activation_memory = (
            2
            * args.micro_batch_size
            * args.num_attention_heads
            * args.seq_length
            * args.kv_channels
        )
        # 7. linear dropout
        linear_dropout_activation_memory = (
            args.seq_length * args.micro_batch_size * args.hidden_size
        )

    # Split into two parts
    attention_parallel_by_tp_activation_memory = (
        QKT_activation_memory
        + softmax_activation_memory
        + softmax_dropout_activation_memory
        + attention_over_V_activation_memory
        + linear_activation_memory
    )
    attention_not_parallel_by_tp_activation_memory = (
        pre_attn_layernorm_activation_memory
        + QKV_activation_memory
        + linear_dropout_activation_memory
    )

    # FFN:
    # LayerNorm
    pre_mlp_layernorm_activation_memory = (
        2 * args.seq_length * args.micro_batch_size * args.hidden_size
    )
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    # for hetero (TODO: suppot hetero train)
    expert_tensor_parallel_size = (
        args.expert_tensor_parallel_size
        if args.expert_tensor_parallel_size is not None
        else args.tensor_model_parallel_size
    )
    # In FFN, we split memory into two parts, one that can paralleled by tensor parallellism and the other that can't
    if args.num_experts is not None:
        ffn_parallel_by_tp_activation_memory = (
            4
            * args.seq_length
            * args.micro_batch_size
            * args.ffn_hidden_size
            * gated_linear_multiplier
        )
        ffn_not_parallel_by_tp_activation_memory = (
            pre_mlp_layernorm_activation_memory
            + 3 * args.seq_length * args.micro_batch_size * args.hidden_size
        )

        sparse_ffn_parallel_by_tp_activation_memory = (
            4
            * args.seq_length
            * args.micro_batch_size
            * args.moe_ffn_hidden_size
            * gated_linear_multiplier
            * args.moe_router_topk
            / args.tensor_model_parallel_size
        )
        sparse_ffn_not_parallel_by_tp_activation_memory = (
            # gate (fp32)
            4
            * args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            / args.tensor_model_parallel_size
            + 2
            * args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * args.moe_router_topk
            / args.tensor_model_parallel_size
            * expert_tensor_parallel_size
        )
        if args.moe_shared_expert_intermediate_size is not None:
            shared_sparse_ffn_parallel_by_tp_activation_memory = (
                4
                * args.seq_length
                * args.micro_batch_size
                * args.moe_shared_expert_intermediate_size
                * gated_linear_multiplier
            )
            shared_sparse_ffn_not_parallel_by_tp_activation_memory = (
                3 * args.seq_length * args.micro_batch_size * args.hidden_size
            )
        else:
            shared_sparse_ffn_parallel_by_tp_activation_memory = 0
            shared_sparse_ffn_not_parallel_by_tp_activation_memory = 0
    else:
        ffn_parallel_by_tp_activation_memory = (
            4
            * args.seq_length
            * args.micro_batch_size
            * args.ffn_hidden_size
            * gated_linear_multiplier
        )
        ffn_not_parallel_by_tp_activation_memory = (
            pre_mlp_layernorm_activation_memory
            + 3 * args.seq_length * args.micro_batch_size * args.hidden_size
        )
        sparse_ffn_parallel_by_tp_activation_memory = 0
        sparse_ffn_not_parallel_by_tp_activation_memory = 0
        shared_sparse_ffn_parallel_by_tp_activation_memory = 0
        shared_sparse_ffn_not_parallel_by_tp_activation_memory = 0

    # TODO(zhaoyinglia): add MTP module activation memory

    # Memory of bass
    bass_activation_memory = (
        5 * args.micro_batch_size * args.num_attention_heads * args.seq_length * args.seq_length
    )

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.
    # Input to embedding
    embedding_activation_memory = 8 * args.seq_length * args.micro_batch_size
    dropout_embedding_activation_memory = args.seq_length * args.micro_batch_size * args.hidden_size
    # Last LayerNorm and inputs to output layer and CE loss.
    output_layer_and_loss_activation_memory = (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * 4
        * (1 + (args.padded_vocab_size / args.hidden_size))
    )

    # Multiply by interleaved PP memory factor.
    interleaved_schedule_memory_penalty = 1
    in_flight_microbatches = num_microbatches
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    activation_memory = None
    _NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))
    _NVTE_FUSED_ATTN = int(os.getenv("NVTE_FUSED_ATTN", "1"))
    if args.recompute_granularity == "selective" or _NVTE_FLASH_ATTN or _NVTE_FUSED_ATTN:
        attention_parallel_by_tp_activation_memory -= bass_activation_memory

    if args.sequence_parallel:
        perlayer_activation = (
            attention_parallel_by_tp_activation_memory
            + attention_not_parallel_by_tp_activation_memory
            + ffn_parallel_by_tp_activation_memory
            + ffn_not_parallel_by_tp_activation_memory
        ) / args.tensor_model_parallel_size
        sparse_perlayer_activation = (
            attention_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + attention_not_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + sparse_ffn_parallel_by_tp_activation_memory
            + sparse_ffn_not_parallel_by_tp_activation_memory
            + shared_sparse_ffn_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + shared_sparse_ffn_not_parallel_by_tp_activation_memory
            / args.tensor_model_parallel_size
        )
    else:
        perlayer_activation = (
            attention_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + ffn_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + attention_not_parallel_by_tp_activation_memory
            + ffn_not_parallel_by_tp_activation_memory
        )
        sparse_perlayer_activation = (
            attention_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + attention_not_parallel_by_tp_activation_memory
            + sparse_ffn_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + sparse_ffn_not_parallel_by_tp_activation_memory
            + shared_sparse_ffn_parallel_by_tp_activation_memory / args.tensor_model_parallel_size
            + shared_sparse_ffn_not_parallel_by_tp_activation_memory
        )

    if args.num_experts is None:
        # Every Transformer MLP is dense.
        num_dense_layers = args.num_layers
        num_moe_layers = 0
    else:
        # Calculate number of dense and MoE Transformer MLPs.
        if isinstance(args.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
            ]
        elif isinstance(args.moe_layer_freq, list):
            moe_layer_pattern = args.moe_layer_freq
        else:
            raise RuntimeError("Illegal --moe-layer-freq argument provided!")
        assert len(moe_layer_pattern) == args.num_layers
        num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
        num_dense_layers = args.num_layers - num_moe_layers

    if verbose:
        if num_dense_layers > 0:
            print(
                f"> Activation memory footprint per dense transformer layer: "
                f"{perlayer_activation / NUM_BYTES_IN_MEGABYTE:.1f} MB"
            )
        if num_moe_layers > 0:
            print(
                f"> Activation memory footprint per moe transformer layer: "
                f"{sparse_perlayer_activation / NUM_BYTES_IN_MEGABYTE:.1f} MB"
            )

    if args.recompute_method == "uniform" and args.recompute_granularity == "full":
        recompute_layers = args.recompute_num_layers
        if args.pipeline_model_parallel_size > 1:
            activation_memory = (
                (
                    QKV_activation_memory
                    * (args.num_layers / args.pipeline_model_parallel_size / recompute_layers)
                )
                * (interleaved_schedule_memory_penalty * in_flight_microbatches)
                + (embedding_activation_memory * in_flight_microbatches)
                + (dropout_embedding_activation_memory * in_flight_microbatches)
            )
        else:
            activation_memory = (
                QKV_activation_memory * (args.num_layers / recompute_layers)
                + embedding_activation_memory
                + dropout_embedding_activation_memory
                + output_layer_and_loss_activation_memory
            )

    elif args.recompute_method == "block" and args.recompute_granularity == "full":
        recompute_layers = args.recompute_num_layers
        if args.pipeline_model_parallel_size > 1:
            activation_memory = (
                (
                    QKV_activation_memory * recompute_layers
                    + perlayer_activation
                    * (args.num_layers / args.pipeline_model_parallel_size - recompute_layers)
                )
                * (interleaved_schedule_memory_penalty * in_flight_microbatches)
                + embedding_activation_memory * in_flight_microbatches
                + dropout_embedding_activation_memory * in_flight_microbatches
            )
        else:
            activation_memory = (
                QKV_activation_memory * recompute_layers
                + perlayer_activation * (args.num_layers - recompute_layers)
                + embedding_activation_memory
                + dropout_embedding_activation_memory
                + output_layer_and_loss_activation_memory
            )

    else:
        if args.pipeline_model_parallel_size > 1:
            activation_memory = (
                (
                    (
                        perlayer_activation * num_dense_layers
                        + sparse_perlayer_activation * num_moe_layers
                    )
                    / args.pipeline_model_parallel_size
                )
                * (interleaved_schedule_memory_penalty * in_flight_microbatches)
                + (
                    (embedding_activation_memory / args.tensor_model_parallel_size)
                    * args.pipeline_model_parallel_size
                )
                + (
                    (dropout_embedding_activation_memory / args.tensor_model_parallel_size)
                    * args.pipeline_model_parallel_size
                )
            )
        else:
            activation_memory = (
                (
                    perlayer_activation * num_dense_layers
                    + sparse_perlayer_activation * num_moe_layers
                )
                + embedding_activation_memory / args.tensor_model_parallel_size
                + dropout_embedding_activation_memory / args.tensor_model_parallel_size
                + output_layer_and_loss_activation_memory / args.tensor_model_parallel_size
            )

    assert activation_memory is not None
    return activation_memory / args.context_parallel_size


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )

    compute_activated_weight_number(args, verbose=verbose)

    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f">>> [FS] Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )

    return int(total_memory)
