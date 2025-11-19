import itertools
import logging
import types

NUM_BYTES_IN_MEGABYTE = 1024 * 1024
logger = logging.getLogger("FlagScale-AutoTuner")


# Base formula functions (from homogeneous model)
def _calculate_attn_params(args):
    """Calculates Attention block parameters based on args."""
    attn_params = 0
    if hasattr(args, 'multi_latent_attention') and args.multi_latent_attention:
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        if args.q_lora_rank is None:
            attn_params += args.hidden_size * args.num_attention_heads * q_head_dim
        else:
            attn_params += (
                args.hidden_size * args.q_lora_rank
                + args.q_lora_rank * args.num_attention_heads * q_head_dim
            )
        attn_params += args.hidden_size * (
            args.kv_lora_rank + args.qk_pos_emb_head_dim
        ) + args.kv_lora_rank * args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
        attn_params += args.v_head_dim * args.num_attention_heads * args.hidden_size
        if hasattr(args, 'qk_layernorm') and args.qk_layernorm:
            if args.q_lora_rank is None:
                attn_params += args.kv_lora_rank
            else:
                attn_params += args.kv_lora_rank + args.q_lora_rank
    else:
        num_query_groups = (
            args.num_query_groups
            if hasattr(args, 'num_query_groups') and args.num_query_groups is not None
            else args.num_attention_heads
        )
        kv_channels = (
            args.kv_channels
            if hasattr(args, 'kv_channels') and args.kv_channels is not None
            else (args.hidden_size // args.num_attention_heads)
        )
        query_projection_size = kv_channels * args.num_attention_heads
        kv_projection_size = kv_channels * num_query_groups
        attn_params += args.hidden_size * (query_projection_size + 2 * kv_projection_size)
        attn_params += query_projection_size * args.hidden_size
        if hasattr(args, 'qk_layernorm') and args.qk_layernorm:
            if hasattr(args, 'qk_layernorm_hidden_dim') and not args.qk_layernorm_hidden_dim:
                attn_params += 2 * query_projection_size // args.num_attention_heads
            else:
                attn_params += query_projection_size
                attn_params += kv_projection_size

    attn_params += 2 * args.hidden_size  # pre-attn layernorm
    return attn_params


def _calculate_mlp_params(args, is_expert=False):
    """Calculates MLP block parameters (Dense or Expert)."""
    gated_linear_multiplier = 3 / 2 if hasattr(args, 'swiglu') and args.swiglu else 1
    ffn_h = args.ffn_hidden_size
    hidden_s = args.hidden_size

    if is_expert:
        moe_ffn_h = (
            args.moe_ffn_hidden_size
            if hasattr(args, 'moe_ffn_hidden_size') and args.moe_ffn_hidden_size is not None
            else ffn_h
        )
        mlp_params = 2 * hidden_s * (moe_ffn_h * gated_linear_multiplier)
    else:
        mlp_params = 2 * hidden_s * (ffn_h * gated_linear_multiplier)

    mlp_params += 2 * args.hidden_size  # pre-mlp layernorm
    return mlp_params


def _calculate_moe_gate_params(args):
    """Calculates MoE gating network parameters."""
    if not hasattr(args, 'num_experts') or args.num_experts is None or args.num_experts <= 1:
        return 0
    return args.hidden_size * args.num_experts


def _calculate_embedding_params(args):
    """Calculates embedding layer parameters."""
    padded_vocab_size = (
        args.padded_vocab_size
        if hasattr(args, 'padded_vocab_size') and args.padded_vocab_size is not None
        else args.vocab_size
    )
    embedding_size = args.hidden_size * padded_vocab_size
    untie = (
        hasattr(args, 'untie_embeddings_and_output_weights')
        and args.untie_embeddings_and_output_weights
    )
    if untie:
        return 2 * embedding_size
    else:
        return embedding_size


def _calculate_attn_activation_components(args):
    """Calculates Attention activation components."""
    mbs = args.micro_batch_size if hasattr(args, 'micro_batch_size') else 1
    sl = args.seq_length if hasattr(args, 'seq_length') else 1024
    hs = args.hidden_size
    nh = args.num_attention_heads
    kvc = (
        args.kv_channels
        if hasattr(args, 'kv_channels') and args.kv_channels is not None
        else (hs // nh)
    )
    nqg = (
        args.num_query_groups
        if hasattr(args, 'num_query_groups') and args.num_query_groups is not None
        else nh
    )

    pre_attn_layernorm_mem = 2 * sl * mbs * hs

    if hasattr(args, 'multi_latent_attention') and args.multi_latent_attention:
        logger.warning(
            "Multi-latent attention activation calculation not fully implemented in hetero model yet."
        )
        QKV_mem = 2 * sl * mbs * hs  # Approx
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        QKT_mem = 4 * mbs * nh * sl * q_head_dim  # Approx
        softmax_mem = 2 * mbs * nh * sl * sl
        softmax_dropout_mem = mbs * nh * sl * sl
        attn_over_v_mem = 2 * mbs * nh * sl * sl + 2 * mbs * nh * sl * args.v_head_dim  # Approx
        linear_mem = 2 * mbs * nh * sl * args.v_head_dim  # Approx
        linear_dropout_mem = sl * mbs * hs
    else:
        QKV_mem = 2 * sl * mbs * hs
        QKT_mem = 2 * mbs * nh * sl * kvc + 2 * mbs * nqg * kvc * sl
        softmax_mem = 2 * mbs * nh * sl * sl
        softmax_dropout_mem = mbs * nh * sl * sl
        attn_over_v_mem = 2 * mbs * nh * sl * sl + 2 * mbs * nqg * kvc * sl
        linear_mem = 2 * mbs * nh * sl * kvc
        linear_dropout_mem = sl * mbs * hs

    attn_activation_tp_scaled = (
        QKT_mem + softmax_mem + softmax_dropout_mem + attn_over_v_mem + linear_mem
    )
    attn_activation_not_tp_scaled = pre_attn_layernorm_mem + QKV_mem + linear_dropout_mem
    bass_mem = 5 * mbs * nh * sl * sl

    return {
        "tp_scaled": attn_activation_tp_scaled,
        "not_tp_scaled": attn_activation_not_tp_scaled,
        "bass": bass_mem,
    }


def _calculate_mlp_activation_components(args, is_expert=False):
    """Calculates MLP activation components."""
    mbs = args.micro_batch_size if hasattr(args, 'micro_batch_size') else 1
    sl = args.seq_length if hasattr(args, 'seq_length') else 1024
    hs = args.hidden_size
    ffn_h = args.ffn_hidden_size

    pre_mlp_layernorm_mem = 2 * sl * mbs * hs
    gated_linear_multiplier = 3 / 2 if hasattr(args, 'swiglu') and args.swiglu else 1

    if is_expert:
        moe_ffn_h = (
            args.moe_ffn_hidden_size
            if hasattr(args, 'moe_ffn_hidden_size') and args.moe_ffn_hidden_size is not None
            else ffn_h
        )
        moe_topk = getattr(args, 'moe_router_topk', 1)
        if moe_topk is None:
            moe_topk = 1
        mlp_activation_tp_scaled = 4 * sl * mbs * moe_ffn_h * gated_linear_multiplier * moe_topk
        mlp_activation_not_tp_scaled = pre_mlp_layernorm_mem + 3 * sl * mbs * hs
    else:  # Dense MLP
        mlp_activation_tp_scaled = 4 * sl * mbs * ffn_h * gated_linear_multiplier
        mlp_activation_not_tp_scaled = pre_mlp_layernorm_mem + 3 * sl * mbs * hs

    return {"tp_scaled": mlp_activation_tp_scaled, "not_tp_scaled": mlp_activation_not_tp_scaled}


def _calculate_moe_gate_activation(args):
    """Calculates MoE gating activation."""
    if not hasattr(args, 'num_experts') or args.num_experts is None or args.num_experts <= 1:
        return 0
    mbs = args.micro_batch_size if hasattr(args, 'micro_batch_size') else 1
    sl = args.seq_length if hasattr(args, 'seq_length') else 1024
    hs = args.hidden_size
    num_experts = args.num_experts
    moe_topk = getattr(args, 'moe_router_topk', 1)
    if moe_topk is None:
        moe_topk = 1
    gate_activation = sl * mbs * hs + 4 * sl * mbs * num_experts + 2 * sl * mbs * moe_topk
    return gate_activation


def _calculate_embedding_activation(args):
    """Calculates embedding activation."""
    mbs = args.micro_batch_size if hasattr(args, 'micro_batch_size') else 1
    sl = args.seq_length if hasattr(args, 'seq_length') else 1024
    hs = args.hidden_size
    padded_vocab_size = (
        args.padded_vocab_size
        if hasattr(args, 'padded_vocab_size') and args.padded_vocab_size is not None
        else args.vocab_size
    )

    embedding_dropout_mem = sl * mbs * hs
    embedding_grad_buffer_approx = 4 * sl * mbs * padded_vocab_size
    return embedding_dropout_mem + embedding_grad_buffer_approx


def _calculate_output_layer_activation(args):
    """Calculates final LN and output activation."""
    mbs = args.micro_batch_size if hasattr(args, 'micro_batch_size') else 1
    sl = args.seq_length if hasattr(args, 'seq_length') else 1024
    hs = args.hidden_size
    padded_vocab_size = (
        args.padded_vocab_size
        if hasattr(args, 'padded_vocab_size') and args.padded_vocab_size is not None
        else args.vocab_size
    )
    final_ln_mem = 2 * sl * mbs * hs
    output_proj_mem = 4 * sl * mbs * padded_vocab_size
    return final_ln_mem + output_proj_mem


def _get_mesh_params_for_stage(stage_idx, hetero_meshes):
    """
    Finds the mesh parameters (TP, CP, EP, DP, Mesh Index) for a
    given pipeline stage index by walking the mesh list.
    """
    if not hetero_meshes:
        return None
    current_stage_offset = 0
    for mesh_idx, mesh in enumerate(hetero_meshes):
        # mesh format [TP, CP, EP, DP, PP_mesh]
        try:
            mesh_pp_size = mesh[4]
            if not isinstance(mesh_pp_size, int) or mesh_pp_size <= 0:
                logger.warning(f"Invalid PP size {mesh_pp_size} in mesh {mesh_idx}. Skipping mesh.")
                continue

            start_stage = current_stage_offset
            end_stage = current_stage_offset + mesh_pp_size

            if start_stage <= stage_idx < end_stage:
                tp, cp, ep, dp, _ = mesh
                # Basic validation
                if not all(isinstance(p, int) and p >= 1 for p in [tp, cp, ep, dp]):
                    logger.warning(
                        f"Invalid parallelism degree in mesh {mesh_idx}: {[tp, cp, ep, dp]}. Using defaults (1)."
                    )
                    tp, cp, ep, dp = [
                        max(1, p) if isinstance(p, int) else 1 for p in [tp, cp, ep, dp]
                    ]  # Fallback
                return {"tp": tp, "cp": cp, "ep": ep, "dp": dp, "mesh_idx": mesh_idx}

            current_stage_offset = end_stage
        except (IndexError, TypeError) as e:
            logger.error(f"Error processing mesh {mesh_idx} with value {mesh}: {e}. Skipping.")
            continue

    logger.error(f"Could not find mesh for stage {stage_idx} within valid mesh definitions.")
    return None


def _calculate_mesh_weight_opt_mem_hetero(
    mesh_args, config, is_first_mesh, is_last_mesh, global_num_layers, layers_on_this_mesh
):
    """
    Calculates static (weight + optimizer) memory for a *single mesh*.

    This function calculates the parameters for the layers *only on this mesh*,
    shards them by the mesh's TP, and applies the mesh's optimizer state multiplier.
    """
    args = mesh_args  # Use the mesh-specific args

    # 1. Calculate base parameter counts for components
    attn_params_base = _calculate_attn_params(args)
    dense_mlp_params_base = _calculate_mlp_params(args, is_expert=False)
    sparse_mlp_params_base = _calculate_mlp_params(args, is_expert=True)  # Params for ONE expert
    moe_gate_params = _calculate_moe_gate_params(args)
    embedding_params_total = _calculate_embedding_params(args)

    # Determine MoE structure based on global layers
    num_dense_layers_global = global_num_layers
    num_moe_layers_global = 0
    num_experts = 0
    if hasattr(args, 'num_experts') and args.num_experts is not None and args.num_experts > 0:
        num_experts = args.num_experts
        if hasattr(args, 'moe_layer_freq'):
            if isinstance(args.moe_layer_freq, int) and args.moe_layer_freq > 0:
                moe_layer_pattern = [
                    1 if (i % args.moe_layer_freq == 0) else 0 for i in range(global_num_layers)
                ]
            elif isinstance(args.moe_layer_freq, list):
                moe_layer_pattern = args.moe_layer_freq[:global_num_layers]
            else:
                moe_layer_pattern = []
            num_moe_layers_global = sum(moe_layer_pattern)
            num_dense_layers_global = global_num_layers - num_moe_layers_global
        else:
            logger.warning(
                "num_experts defined but moe_layer_freq missing. Assuming no MoE layers."
            )

    # 2. Calculate parameters based on ACTUAL layers on this mesh
    if global_num_layers > 0:
        prop_moe = num_moe_layers_global / global_num_layers
        prop_dense = num_dense_layers_global / global_num_layers
    else:
        prop_moe, prop_dense = 0, 1.0

    num_moe_layers_on_mesh = round(layers_on_this_mesh * prop_moe)
    num_dense_layers_on_mesh = layers_on_this_mesh - num_moe_layers_on_mesh

    # Calculate total transformer parameters *only for this mesh's layers*
    params_for_this_mesh_layers = num_dense_layers_on_mesh * (
        attn_params_base + dense_mlp_params_base
    ) + num_moe_layers_on_mesh * (
        attn_params_base + sparse_mlp_params_base * num_experts + moe_gate_params
    )
    if is_last_mesh:
        params_for_this_mesh_layers += 2 * args.hidden_size  # Final layernorm

    # Embedding params relevant to this mesh
    embedding_params_mesh = 0
    untie = (
        hasattr(args, 'untie_embeddings_and_output_weights')
        and args.untie_embeddings_and_output_weights
    )
    if is_first_mesh:
        embedding_params_mesh += embedding_params_total / (2 if untie else 1)
    if is_last_mesh and untie:
        embedding_params_mesh += embedding_params_total / 2

    # 3. Apply Sharding based on THIS mesh's TP
    tp_divisor = max(1, args.tensor_model_parallel_size)
    params_per_shard_transformer = params_for_this_mesh_layers / tp_divisor
    params_per_shard_embedding = embedding_params_mesh / tp_divisor
    total_params_per_shard = params_per_shard_transformer + params_per_shard_embedding

    # 4. Calculate Optimizer Memory Multiplier
    dp_divisor = max(1, args.data_parallel_size)
    use_do = hasattr(args, 'use_distributed_optimizer') and args.use_distributed_optimizer

    if use_do:
        optimizer_multiplier = 6 + 12 / dp_divisor  # ZeRO-style
    else:
        optimizer_multiplier = 18  # Standard AdamW

    # 5. Total
    weight_and_optimizer_memory = total_params_per_shard * optimizer_multiplier
    return weight_and_optimizer_memory  # Bytes


def hetero_compute_weight_and_optimizer_memory(base_args, strategy, config):
    """
    Computes the static (weight + optimizer) memory for *each mesh*
    in the heterogeneous configuration.

    Returns: A dictionary {mesh_idx: memory_bytes}
    """
    hetero_meshes = strategy.get("hetero_process_meshes", [])
    hetero_split = strategy.get("hetero_pipeline_layer_split", None)

    if not hetero_meshes or not isinstance(hetero_meshes, list):
        logger.warning("Invalid or missing hetero_process_meshes, calculating based on base_args.")
        if not hasattr(base_args, 'all_pipeline_model_parallel_size'):
            base_args.all_pipeline_model_parallel_size = 1
        mem = _calculate_mesh_weight_opt_mem_hetero(
            base_args, config, True, True, base_args.num_layers, base_args.num_layers
        )
        return {0: mem}

    mesh_memory_dict = {}
    num_meshes = len(hetero_meshes)
    global_pp_size = strategy.get("pipeline_model_parallel_size", 1)
    global_num_layers = config.train.model.num_layers

    # Fallback for layer split (if not provided, assume uniform)
    if (
        not hetero_split
        or not isinstance(hetero_split, list)
        or len(hetero_split) != global_pp_size
    ):
        logger.warning(
            f"Invalid or missing hetero_pipeline_layer_split. Assuming uniform distribution."
        )
        if global_pp_size > 0:
            layers_per_stage = global_num_layers // global_pp_size
            remainder = global_num_layers % global_pp_size
            hetero_split = [layers_per_stage + 1] * remainder + [layers_per_stage] * (
                global_pp_size - remainder
            )
        else:
            hetero_split = []

    if not hasattr(base_args, 'all_pipeline_model_parallel_size'):
        base_args.all_pipeline_model_parallel_size = global_pp_size

    current_stage_offset = 0  # Track stages as we iterate meshes
    for i, mesh in enumerate(hetero_meshes):
        mesh_args = types.SimpleNamespace(**vars(base_args))
        mesh_pp_size = 0

        try:
            tp, cp, ep, dp, pp_mesh_val = mesh
            mesh_args.tensor_model_parallel_size = max(1, tp) if isinstance(tp, int) else 1
            mesh_args.context_parallel_size = max(1, cp) if isinstance(cp, int) else 1
            mesh_args.expert_model_parallel_size = max(1, ep) if isinstance(ep, int) else 1
            mesh_args.data_parallel_size = max(1, dp) if isinstance(dp, int) else 1
            mesh_pp_size = max(0, pp_mesh_val) if isinstance(pp_mesh_val, int) else 0
        except (TypeError, ValueError, IndexError) as e:
            logger.error(f"Invalid mesh format for mesh {i}: {mesh}. Skipping. Error: {e}")
            continue

        # Calculate total layers this mesh is responsible for
        layers_on_this_mesh = 0
        for stage_k in range(current_stage_offset, current_stage_offset + mesh_pp_size):
            if stage_k < len(hetero_split):
                layers_on_this_mesh += hetero_split[stage_k]

        current_stage_offset += mesh_pp_size

        mesh_memory = _calculate_mesh_weight_opt_mem_hetero(
            mesh_args,
            config,
            is_first_mesh=(i == 0),
            is_last_mesh=(i == num_meshes - 1),
            global_num_layers=global_num_layers,
            layers_on_this_mesh=layers_on_this_mesh,
        )
        mesh_memory_dict[i] = mesh_memory

        logger.debug(
            f"  > Mesh {i} (TP={mesh_args.tensor_model_parallel_size}, ..., Layers={layers_on_this_mesh}): "
            f"Est Static Mem = {mesh_memory / NUM_BYTES_IN_MEGABYTE:.2f} MB"
        )

    return mesh_memory_dict


def _calculate_stage_activation_hetero(
    base_args,
    mesh_params,
    num_microbatches_global,
    layers_in_stage,
    total_layers,
    is_first_stage,
    is_last_stage,
    strategy,
    config,
):
    """
    Calculates activation for a specific stage FOR A SINGLE MICROBATCH (MBS=1).

    This is the core "unit" of activation calculation, which is then scaled
    by the 'hetero_compute_activation_memory' manager function.
    """
    stage_args = types.SimpleNamespace(**vars(base_args))
    stage_args.tensor_model_parallel_size = mesh_params['tp']
    stage_args.context_parallel_size = mesh_params['cp']
    stage_args.expert_model_parallel_size = mesh_params['ep']
    stage_args.data_parallel_size = mesh_params['dp']

    # Force calculation for MBS=1
    stage_args.micro_batch_size = 1

    # Get (MBS=1) activation components
    attn_comps = _calculate_attn_activation_components(stage_args)
    mlp_comps = _calculate_mlp_activation_components(stage_args, is_expert=False)
    expert_comps = _calculate_mlp_activation_components(stage_args, is_expert=True)
    moe_gate_act = _calculate_moe_gate_activation(stage_args)
    embedding_act = _calculate_embedding_activation(stage_args)
    output_layer_act = _calculate_output_layer_activation(stage_args)

    # Determine MoE layer proportion (approximate)
    num_dense_layers_in_stage = layers_in_stage
    num_moe_layers_in_stage = 0
    if (
        hasattr(stage_args, 'num_experts')
        and stage_args.num_experts is not None
        and stage_args.num_experts > 0
    ):
        # (Assuming MoE logic for proportioning is here)
        pass

    # Calculate per-layer activation applying stage TP/SP scaling
    tp_divisor = max(1, stage_args.tensor_model_parallel_size)
    sp_enabled = strategy.get('sequence_parallel', getattr(stage_args, 'sequence_parallel', False))
    sp_tp_divisor = tp_divisor if sp_enabled else 1
    sp_gate_divisor = sp_tp_divisor

    attn_act_per_layer = (
        attn_comps["tp_scaled"] / tp_divisor + attn_comps["not_tp_scaled"] / sp_tp_divisor
    )
    dense_mlp_act_per_layer = (
        mlp_comps["tp_scaled"] / tp_divisor + mlp_comps["not_tp_scaled"] / sp_tp_divisor
    )
    moe_mlp_act_per_layer = moe_gate_act / sp_gate_divisor + (
        expert_comps["tp_scaled"] / tp_divisor + expert_comps["not_tp_scaled"] / sp_tp_divisor
    )
    # Bass memory adjustment
    recompute_granularity_check = strategy.get(
        "recompute_granularity", getattr(stage_args, 'recompute_granularity', None)
    )
    apply_bass_adjustment = not (recompute_granularity_check == "selective")
    if apply_bass_adjustment:
        bass_adjustment = attn_comps["bass"] / (tp_divisor if not sp_enabled else 1)
        attn_act_per_layer -= bass_adjustment
        moe_mlp_act_per_layer -= bass_adjustment

    # Apply Recompute Logic (Simplified: using global settings from strategy)
    recompute_method = strategy.get("recompute_method")
    recompute_granularity = strategy.get("recompute_granularity")
    recompute_num_layers_global = strategy.get("recompute_num_layers")

    stage_activation_memory_no_embed_out = 0
    qkv_act_stage_proxy = attn_comps["not_tp_scaled"] / sp_tp_divisor  # Input activation proxy

    if (
        recompute_method == "uniform"
        and recompute_granularity == "full"
        and recompute_num_layers_global is not None
        and total_layers > 0
        and recompute_num_layers_global > 0
    ):
        num_recompute_groups = max(
            1, round(layers_in_stage * (total_layers // recompute_num_layers_global) / total_layers)
        )
        stage_activation_memory_no_embed_out = qkv_act_stage_proxy * num_recompute_groups

    elif (
        recompute_method == "block"
        and recompute_granularity == "full"
        and recompute_num_layers_global is not None
    ):
        recompute_layers_this_stage = min(recompute_num_layers_global, layers_in_stage)
        non_recomputed_layers = layers_in_stage - recompute_layers_this_stage
        moe_non_recomputed = (
            round(non_recomputed_layers * (num_moe_layers_in_stage / layers_in_stage))
            if layers_in_stage > 0
            else 0
        )
        dense_non_recomputed = non_recomputed_layers - moe_non_recomputed
        stage_activation_memory_no_embed_out = (
            qkv_act_stage_proxy * recompute_layers_this_stage
            + dense_mlp_act_per_layer * max(0, dense_non_recomputed)
            + moe_mlp_act_per_layer * max(0, moe_non_recomputed)
        )
    else:  # No recompute or selective
        stage_activation_memory_no_embed_out = (
            dense_mlp_act_per_layer * num_dense_layers_in_stage
            + moe_mlp_act_per_layer * num_moe_layers_in_stage
        )

    # Add Embedding/Output Layer Memory
    extra_activation_one_mb = 0
    if is_first_stage:
        extra_activation_one_mb += embedding_act / sp_tp_divisor
    if is_last_stage:
        extra_activation_one_mb += output_layer_act / sp_tp_divisor

    # Apply Context Parallelism scaling
    cp_divisor = max(1, stage_args.context_parallel_size)

    # Return (Layers activation) and (Extra activation) separately
    layers_activation_final = stage_activation_memory_no_embed_out / cp_divisor
    extra_activation_final = extra_activation_one_mb / cp_divisor

    return layers_activation_final, extra_activation_final


def hetero_compute_activation_memory(base_args, strategy, config):
    """
    Computes peak activation memory *per stage* for the entire pipeline.

    It calculates the (MBS=1) activation for each stage, then scales
    it by the strategy's microbatch size (MBS) and the pipeline
    parallelism bubble size (in_flight_microbatches).
    """
    hetero_meshes = strategy.get("hetero_process_meshes", [])
    hetero_split = strategy.get("hetero_pipeline_layer_split", None)
    global_pp_size = strategy.get("pipeline_model_parallel_size", 1)
    gbs = config.train.model.global_batch_size
    mbs = strategy.get("micro_batch_size", 1)  # Strategy's MBS
    total_layers = config.train.model.num_layers

    # --- Calculate Pipeline Penalty Factor ---
    dp_fallback = getattr(base_args, 'data_parallel_size', 1)
    current_mbs_global = max(1, mbs)

    num_microbatches_global = (
        gbs // (dp_fallback * current_mbs_global)
        if dp_fallback > 0 and current_mbs_global > 0
        else gbs
    )
    if num_microbatches_global == 0:
        num_microbatches_global = 1

    # in_flight_microbatches is the pipeline bubble size
    if global_pp_size > 1:
        in_flight_microbatches = min(num_microbatches_global, global_pp_size)  # 1F1B schedule
    else:
        in_flight_microbatches = 1

    logger.debug(
        f"> Hetero Activation: Strategy_MBS={current_mbs_global}, Global_PP={global_pp_size}, "
        f"Global_Mbs_Count={num_microbatches_global}, In-Flight_Mbs={in_flight_microbatches}"
    )
    # --- End Calculate Pipeline Penalty Factor ---

    use_hetero_path = (
        isinstance(hetero_meshes, list)
        and len(hetero_meshes) > 0
        and isinstance(hetero_split, list)
        and len(hetero_split) > 0
        and global_pp_size > 1
        and len(hetero_split) == global_pp_size
        and sum(hetero_split) == total_layers
    )

    stage_activation_list = []

    if use_hetero_path:
        logger.debug(f"> Calculating activation memory based on hetero split: {hetero_split}")

        for stage_idx, layers_in_stage in enumerate(hetero_split):
            if not isinstance(layers_in_stage, int) or layers_in_stage <= 0:
                logger.warning(
                    f"Invalid layer count {layers_in_stage} for stage {stage_idx}. Adding 0."
                )
                stage_activation_list.append(0)
                continue

            mesh_params = _get_mesh_params_for_stage(stage_idx, hetero_meshes)
            if mesh_params is None:
                logger.error(f"Could not find mesh for stage {stage_idx}. Adding 0.")
                stage_activation_list.append(0)
                continue

            try:
                # 1. Calculate activation for ONE microbatch (MBS=1)
                layers_activation_one_mb, extra_activation_one_mb = (
                    _calculate_stage_activation_hetero(
                        base_args,
                        mesh_params,
                        num_microbatches_global,
                        layers_in_stage,
                        total_layers,
                        is_first_stage=(stage_idx == 0),
                        is_last_stage=(stage_idx == global_pp_size - 1),
                        strategy=strategy,
                        config=config,
                    )
                )

                # 2. Scale layer activation by Strategy MBS * Pipeline Penalty
                pipelined_layer_activation = (
                    layers_activation_one_mb * current_mbs_global * in_flight_microbatches
                )

                # 3. Scale Extra activation by Strategy MBS
                extra_activation = extra_activation_one_mb * current_mbs_global

                # 4. Total
                stage_activation = pipelined_layer_activation + extra_activation

                logger.debug(
                    f"  > Stage {stage_idx} (Mesh {mesh_params['mesh_idx']}): "
                    f"Est Act (Layers_1MB * {current_mbs_global} * {in_flight_microbatches}) = {pipelined_layer_activation / NUM_BYTES_IN_MEGABYTE:.1f} MB + "
                    f"Extra (Extra_1MB * {current_mbs_global}) = {extra_activation / NUM_BYTES_IN_MEGABYTE:.1f} MB = "
                    f"Total = {stage_activation / NUM_BYTES_IN_MEGABYTE:.1f} MB"
                )
                stage_activation_list.append(stage_activation)

            except Exception as e:
                logger.error(
                    f"Error calculating activation for stage {stage_idx}: {e}", exc_info=True
                )
                stage_activation_list.append(float('inf'))

        activation_memory_final = stage_activation_list

    else:  # Fallback to homogeneous logic (using base_args)
        logger.debug("> Calculating activation memory assuming uniform distribution or PP=1.")

        layers_to_calculate = total_layers
        if layers_to_calculate > 0:
            dummy_mesh_params = {
                'tp': getattr(base_args, 'tensor_model_parallel_size', 1),
                'cp': getattr(base_args, 'context_parallel_size', 1),
                'ep': getattr(base_args, 'expert_model_parallel_size', 1),
                'dp': dp_fallback,
                'mesh_idx': 0,
            }
            try:
                layers_activation_one_mb, extra_activation_one_mb = (
                    _calculate_stage_activation_hetero(
                        base_args,
                        dummy_mesh_params,
                        num_microbatches_global,
                        layers_to_calculate,
                        total_layers,
                        True,
                        True,
                        strategy,
                        config,
                    )
                )
                pipelined_layer_activation = (
                    layers_activation_one_mb * current_mbs_global * in_flight_microbatches
                )
                extra_activation = extra_activation_one_mb * current_mbs_global
                activation_memory_final_value = pipelined_layer_activation + extra_activation
                activation_memory_final = [activation_memory_final_value]
            except Exception as e:
                logger.error(f"Error calculating fallback activation: {e}", exc_info=True)
                activation_memory_final = [float('inf')]
        else:
            activation_memory_final = [0]

    return activation_memory_final  # Return Peak BYTES list


# Main entry point for hetero memory calculation
def hetero_report_theoretical_memory(strategy, config, base_args):
    """
    Main function for heterogeneous memory calculation.

    Calculates the peak memory for each mesh by combining its static (weight/opt)
    memory with the peak activation memory of all stages running on it.

    Returns a list where each element is the peak memory (in MB) for that
    pipeline stage, broadcasted from its parent mesh's peak.
    """
    try:
        # 1. Get Weight/Opt memory PER MESH (Bytes)
        #    e.g., {0: 30e9, 1: 15e9}
        weight_opt_dict_bytes = hetero_compute_weight_and_optimizer_memory(
            base_args, strategy, config
        )

        # 2. Get Activation memory PER STAGE (Bytes)
        #    e.g., [40e9, 45e9, 10e9, 12e9]
        activation_list_bytes = hetero_compute_activation_memory(base_args, strategy, config)

        global_pp_size = strategy.get("pipeline_model_parallel_size", 1)
        hetero_meshes = strategy.get("hetero_process_meshes", [])

        # 3. Handle calculation failures
        if not isinstance(weight_opt_dict_bytes, dict) or not isinstance(
            activation_list_bytes, list
        ):
            logger.error(
                "Mismatch in memory model return types (expected dict and list). Cannot combine."
            )
            return float('inf')

        # 4. Handle PP=1 (fallback) case
        if global_pp_size == 1:
            if len(activation_list_bytes) == 1:
                mesh_0_mem_bytes = weight_opt_dict_bytes.get(0, 0)
                activation_mem_bytes = activation_list_bytes[0]

                if mesh_0_mem_bytes == float('inf') or activation_mem_bytes == float('inf'):
                    total_mb = float('inf')
                else:
                    total_mb = (mesh_0_mem_bytes + activation_mem_bytes) / NUM_BYTES_IN_MEGABYTE

                logger.info(
                    f">>> [FS] Hetero Theoretical (PP=1) Footprints: "
                    f"Weight/Opt={mesh_0_mem_bytes / NUM_BYTES_IN_MEGABYTE:.2f} MB, "
                    f"Activation={activation_mem_bytes / NUM_BYTES_IN_MEGABYTE:.2f} MB, "
                    f"Total={total_mb:.2f} MB\n"
                )
                return [float('inf') if total_mb == float('inf') else int(total_mb)]
            else:
                logger.error(
                    f"PP=1 but activation list length is {len(activation_list_bytes)}. Mismatch."
                )
                return float('inf')

        # 5. Handle PP > 1 (hetero) case
        if len(activation_list_bytes) != global_pp_size:
            logger.error(
                f"Activation list length ({len(activation_list_bytes)}) != global PP size ({global_pp_size}). Mismatch."
            )
            return float('inf')

        # 6. Build Mesh-to-Stage mapping
        mesh_to_stages = {}
        peak_memory_per_mesh = {}  # Stores peak MB (int) per mesh_idx

        if not hetero_meshes:
            logger.error("hetero_process_meshes is missing or empty.")
            return float('inf')

        for stage_idx in range(global_pp_size):
            mesh_params = _get_mesh_params_for_stage(stage_idx, hetero_meshes)
            if mesh_params is None:
                logger.error(f"Cannot find mesh for stage {stage_idx} during final reporting.")
                return float('inf')

            mesh_idx = mesh_params['mesh_idx']
            if mesh_idx not in mesh_to_stages:
                mesh_to_stages[mesh_idx] = []
            mesh_to_stages[mesh_idx].append(stage_idx)

        # 7. Calculate Peak Memory PER MESH
        logger.debug("\n" + "=" * 50)
        logger.debug(">>> [FS-DEBUG] Hetero Memory Model Breakdown (Per-Mesh):")
        logger.debug(f"  Strategy Split: {strategy.get('hetero_pipeline_layer_split')}")
        logger.debug(f"  Strategy Meshes: {strategy.get('hetero_process_meshes')}")
        logger.debug(f"  Mesh-to-Stage Map: {mesh_to_stages}")
        logger.debug(
            f"  Weight/Opt (per Mesh): { {k: f'{v / NUM_BYTES_IN_MEGABYTE:.2f} MB' for k, v in weight_opt_dict_bytes.items()} }"
        )
        logger.debug(
            f"  Activation (per Stage): { [f'{a / NUM_BYTES_IN_MEGABYTE:.2f} MB' for a in activation_list_bytes] }"
        )
        logger.debug("  --- Per-Mesh Peak Calculation ---")

        for mesh_idx, stage_indices in mesh_to_stages.items():
            static_mem_bytes = weight_opt_dict_bytes.get(mesh_idx, 0)

            # Find the highest activation memory among all stages running on this mesh
            peak_activation_on_mesh_bytes = 0.0
            if stage_indices:
                try:
                    valid_activations = [
                        activation_list_bytes[idx]
                        for idx in stage_indices
                        if activation_list_bytes[idx] != float('inf')
                    ]
                    if valid_activations:
                        peak_activation_on_mesh_bytes = max(valid_activations)
                    elif stage_indices and activation_list_bytes[stage_indices[0]] == float('inf'):
                        peak_activation_on_mesh_bytes = float('inf')  # All stages failed
                except Exception:
                    peak_activation_on_mesh_bytes = float('inf')

            if static_mem_bytes == float('inf') or peak_activation_on_mesh_bytes == float('inf'):
                total_mesh_mem_mb = float('inf')
            else:
                total_mesh_mem_mb = (
                    static_mem_bytes + peak_activation_on_mesh_bytes
                ) / NUM_BYTES_IN_MEGABYTE

            peak_memory_per_mesh[mesh_idx] = (
                float('inf') if total_mesh_mem_mb == float('inf') else int(total_mesh_mem_mb)
            )

            logger.debug(
                f"    > Mesh {mesh_idx} (Stages {stage_indices}): "
                f"Static={static_mem_bytes / NUM_BYTES_IN_MEGABYTE:.2f} MB + "
                f"Peak Activation (max of stages {stage_indices})={peak_activation_on_mesh_bytes / NUM_BYTES_IN_MEGABYTE:.2f} MB = "
                f"Total Peak={total_mesh_mem_mb:.2f} MB"
            )

        # 8. Report and Return
        logger.debug("  ---------------------------")
        final_mesh_peak_list = [
            peak_memory_per_mesh[i] for i in sorted(peak_memory_per_mesh.keys())
        ]
        logger.info(  # Keep this summary log as INFO
            f">>> [FS] Hetero Theoretical memory: Peak Memory per mesh (MB)={final_mesh_peak_list}"
        )
        logger.debug("=" * 50 + "\n")

        return final_mesh_peak_list

    except Exception as e:
        logger.error(
            f"Failed to calculate heterogeneous memory for strategy. Error: {e}", exc_info=True
        )
        return float('inf')
