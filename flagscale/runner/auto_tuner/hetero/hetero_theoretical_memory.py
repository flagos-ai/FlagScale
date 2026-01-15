import logging
import math
import os
import types

NUM_BYTES_IN_MEGABYTE = 1024 * 1024
logger = logging.getLogger("FlagScale-AutoTuner")

"""
Computes theoretical memory footprint for heterogeneous model training.
Adapts the logic from Megatron-LM's homogeneous memory model to support:
1. Heterogeneous hardware (Meshes with different TP/DP/CP/EP).
2. Heterogeneous layer distribution (Pipeline stages with different layer counts).
3. Mixed Dense/MoE architectures (Precise layer-wise parameter calculation).
"""


# Helper Functions: Global Pattern & Mesh Mapping
def _get_global_moe_pattern(args):
    """
    Generates a global pattern list [0, 0, 1, ...] indicating Dense (0) or MoE (1)
    for each layer in the model.
    """
    num_layers = args.num_layers
    if not hasattr(args, "num_experts") or args.num_experts is None or args.num_experts <= 1:
        return [0] * num_layers

    if hasattr(args, "moe_layer_freq"):
        if isinstance(args.moe_layer_freq, int) and args.moe_layer_freq > 0:
            return [1 if (i % args.moe_layer_freq == 0) else 0 for i in range(num_layers)]
        elif isinstance(args.moe_layer_freq, list):
            # Extend pattern if list is shorter than num_layers
            full_pattern = []
            while len(full_pattern) < num_layers:
                full_pattern.extend(args.moe_layer_freq)
            return full_pattern[:num_layers]

    return [0] * num_layers


def _get_mesh_params_for_stage(stage_idx, hetero_meshes):
    """
    Identifies the mesh configuration (TP, CP, EP, DP) for a specific pipeline stage.
    Returns a dictionary containing the parallelism degrees and the mesh index.
    """
    if not hetero_meshes:
        return None

    current_stage_offset = 0
    for mesh_idx, mesh in enumerate(hetero_meshes):
        try:
            # Mesh format: [TP, CP, EP, DP, PP_on_this_mesh]
            mesh_pp_size = mesh[4]
            start_stage = current_stage_offset
            end_stage = current_stage_offset + mesh_pp_size

            if start_stage <= stage_idx < end_stage:
                tp, cp, ep, dp, _ = mesh
                # Ensure valid integer values (min 1)
                tp, cp, ep, dp = [max(1, int(p)) for p in [tp, cp, ep, dp]]
                return {"tp": tp, "cp": cp, "ep": ep, "dp": dp, "mesh_idx": mesh_idx}

            current_stage_offset = end_stage
        except Exception:
            continue

    return None


# Activation Component Calculators
def _calculate_attn_activation_components(args):
    """Calculates Attention activation components (per microbatch)."""
    mbs = args.micro_batch_size if hasattr(args, "micro_batch_size") else 1
    sl = args.seq_length if hasattr(args, "seq_length") else 1024
    hs = args.hidden_size
    nh = args.num_attention_heads
    kvc = (
        args.kv_channels
        if hasattr(args, "kv_channels") and args.kv_channels is not None
        else (hs // nh)
    )
    nqg = (
        args.num_query_groups
        if hasattr(args, "num_query_groups") and args.num_query_groups is not None
        else nh
    )

    pre_attn_layernorm_mem = 2 * sl * mbs * hs

    if hasattr(args, "multi_latent_attention") and args.multi_latent_attention:
        # Approx for MLA
        QKV_mem = 2 * sl * mbs * hs
        q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
        QKT_mem = 4 * mbs * nh * sl * q_head_dim
        softmax_mem = 2 * mbs * nh * sl * sl
        softmax_dropout_mem = mbs * nh * sl * sl
        attn_over_v_mem = 2 * mbs * nh * sl * sl + 2 * mbs * nh * sl * args.v_head_dim
        linear_mem = 2 * mbs * nh * sl * args.v_head_dim
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
    mbs = args.micro_batch_size if hasattr(args, "micro_batch_size") else 1
    sl = args.seq_length if hasattr(args, "seq_length") else 1024
    hs = args.hidden_size
    ffn_h = args.ffn_hidden_size

    pre_mlp_layernorm_mem = 2 * sl * mbs * hs
    gated_linear_multiplier = 3 / 2 if hasattr(args, "swiglu") and args.swiglu else 1

    if is_expert:
        moe_ffn_h = (
            args.moe_ffn_hidden_size
            if hasattr(args, "moe_ffn_hidden_size") and args.moe_ffn_hidden_size is not None
            else ffn_h
        )
        moe_topk = getattr(args, "moe_router_topk", 1)
        if moe_topk is None:
            moe_topk = 1
        # TP scaled part
        mlp_activation_tp_scaled = 4 * sl * mbs * moe_ffn_h * gated_linear_multiplier * moe_topk
        mlp_activation_not_tp_scaled = pre_mlp_layernorm_mem + 3 * sl * mbs * hs
    else:  # Dense MLP
        mlp_activation_tp_scaled = 4 * sl * mbs * ffn_h * gated_linear_multiplier
        mlp_activation_not_tp_scaled = pre_mlp_layernorm_mem + 3 * sl * mbs * hs

    return {"tp_scaled": mlp_activation_tp_scaled, "not_tp_scaled": mlp_activation_not_tp_scaled}


def _calculate_moe_gate_activation(args):
    """Calculates MoE gate activation."""
    if not hasattr(args, "num_experts") or args.num_experts is None or args.num_experts <= 1:
        return 0
    mbs = args.micro_batch_size if hasattr(args, "micro_batch_size") else 1
    sl = args.seq_length if hasattr(args, "seq_length") else 1024
    hs = args.hidden_size
    num_experts = args.num_experts
    moe_topk = getattr(args, "moe_router_topk", 1)
    if moe_topk is None:
        moe_topk = 1
    gate_activation = sl * mbs * hs + 4 * sl * mbs * num_experts + 2 * sl * mbs * moe_topk
    return gate_activation


def _calculate_embedding_activation(args):
    """Calculates embedding activation."""
    mbs = args.micro_batch_size if hasattr(args, "micro_batch_size") else 1
    sl = args.seq_length if hasattr(args, "seq_length") else 1024
    hs = args.hidden_size
    padded_vocab_size = (
        args.padded_vocab_size
        if hasattr(args, "padded_vocab_size") and args.padded_vocab_size is not None
        else args.vocab_size
    )
    embedding_dropout_mem = sl * mbs * hs
    embedding_grad_buffer_approx = 4 * sl * mbs * padded_vocab_size
    return embedding_dropout_mem + embedding_grad_buffer_approx


def _calculate_output_layer_activation(args):
    """Calculates output layer activation."""
    mbs = args.micro_batch_size if hasattr(args, "micro_batch_size") else 1
    sl = args.seq_length if hasattr(args, "seq_length") else 1024
    hs = args.hidden_size
    padded_vocab_size = (
        args.padded_vocab_size
        if hasattr(args, "padded_vocab_size") and args.padded_vocab_size is not None
        else args.vocab_size
    )
    final_ln_mem = 2 * sl * mbs * hs
    output_proj_mem = 4 * sl * mbs * padded_vocab_size
    return final_ln_mem + output_proj_mem


# Static Memory (Weight + Optimizer) - Per Mesh
def hetero_compute_weight_and_optimizer_memory(base_args, strategy, config):
    """
    Computes the static memory (Weight + Optimizer States) for each mesh.

    Returns:
        Dict[int, float]: A dictionary mapping mesh_idx to memory in Bytes.
    """
    hetero_meshes = strategy.get("hetero_process_meshes", [])
    hetero_split = strategy.get("hetero_pipeline_layer_split", [])

    # Get global architecture patterns
    global_moe_pattern = _get_global_moe_pattern(base_args)

    mesh_memory_dict = {}
    num_meshes = len(hetero_meshes)

    current_global_layer_idx = 0
    current_stage_offset = 0

    for i, mesh in enumerate(hetero_meshes):
        # Create a local args object for this mesh to override parallelism
        mesh_args = types.SimpleNamespace(**vars(base_args))

        try:
            tp, cp, ep, dp, pp_mesh_val = mesh
            mesh_args.tensor_model_parallel_size = max(1, int(tp))
            mesh_args.context_parallel_size = max(1, int(cp))
            mesh_args.expert_model_parallel_size = max(1, int(ep))
            mesh_args.data_parallel_size = max(1, int(dp))
            mesh_pp_size = max(0, int(pp_mesh_val))
        except Exception:
            logger.warning(f"Skipping invalid mesh config at index {i}: {mesh}")
            continue

        # 1. Identify Layers on this Mesh ---
        # Accumulate all layers from all pipeline stages assigned to this mesh
        layer_indices_on_this_mesh = []
        for stage_k in range(current_stage_offset, current_stage_offset + mesh_pp_size):
            if stage_k < len(hetero_split):
                num_layers_in_stage = hetero_split[stage_k]
                indices = list(
                    range(current_global_layer_idx, current_global_layer_idx + num_layers_in_stage)
                )
                layer_indices_on_this_mesh.extend(indices)
                current_global_layer_idx += num_layers_in_stage

        current_stage_offset += mesh_pp_size

        # 2. Base Parameter Calculations ---
        # Attention
        attn_params = 0
        if hasattr(mesh_args, "multi_latent_attention") and mesh_args.multi_latent_attention:
            q_head_dim = mesh_args.qk_head_dim + mesh_args.qk_pos_emb_head_dim
            if mesh_args.q_lora_rank is None:
                attn_params += mesh_args.hidden_size * mesh_args.num_attention_heads * q_head_dim
            else:
                attn_params += (
                    mesh_args.hidden_size * mesh_args.q_lora_rank
                    + mesh_args.q_lora_rank * mesh_args.num_attention_heads * q_head_dim
                )
            attn_params += mesh_args.hidden_size * (
                mesh_args.kv_lora_rank + mesh_args.qk_pos_emb_head_dim
            ) + mesh_args.kv_lora_rank * mesh_args.num_attention_heads * (
                mesh_args.qk_head_dim + mesh_args.v_head_dim
            )
            attn_params += (
                mesh_args.v_head_dim * mesh_args.num_attention_heads * mesh_args.hidden_size
            )
            attn_params += 2 * mesh_args.hidden_size  # Pre-norm
            if hasattr(mesh_args, "qk_layernorm") and mesh_args.qk_layernorm:
                attn_params += mesh_args.kv_lora_rank + (
                    0 if mesh_args.q_lora_rank is None else mesh_args.q_lora_rank
                )
        else:
            # Standard GQA/MHA
            num_query_groups = (
                mesh_args.num_query_groups
                if hasattr(mesh_args, "num_query_groups")
                else mesh_args.num_attention_heads
            )
            kv_channels = (
                mesh_args.kv_channels
                if hasattr(mesh_args, "kv_channels")
                else (mesh_args.hidden_size // mesh_args.num_attention_heads)
            )
            query_projection_size = kv_channels * mesh_args.num_attention_heads
            kv_projection_size = kv_channels * num_query_groups
            attn_params += mesh_args.hidden_size * (
                query_projection_size + 2 * kv_projection_size
            )  # QKV
            attn_params += query_projection_size * mesh_args.hidden_size  # Out
            attn_params += 2 * mesh_args.hidden_size  # Pre-norm
            if hasattr(mesh_args, "qk_layernorm") and mesh_args.qk_layernorm:
                attn_params += query_projection_size + kv_projection_size

        # MLP (Dense & Sparse)
        gated_linear_multiplier = 3 / 2 if hasattr(mesh_args, "swiglu") and mesh_args.swiglu else 1
        ffn_h = mesh_args.ffn_hidden_size

        dense_mlp_params = (
            2 * mesh_args.hidden_size * (ffn_h * gated_linear_multiplier)
            + 2 * mesh_args.hidden_size
        )

        # MoE Params
        num_experts = getattr(mesh_args, "num_experts", 0) or 0
        moe_ffn_h = getattr(mesh_args, "moe_ffn_hidden_size", ffn_h) or ffn_h
        shared_expert_h = getattr(mesh_args, "moe_shared_expert_intermediate_size", 0) or 0

        sparse_mlp_params_all_experts = (
            2
            * mesh_args.hidden_size
            * (
                (moe_ffn_h * num_experts * gated_linear_multiplier)
                + (shared_expert_h * gated_linear_multiplier)
            )
            + mesh_args.hidden_size * num_experts  # Gate
            + 2 * mesh_args.hidden_size  # Norm
        )

        # 3. Count Actual Layers on Mesh ---
        num_moe_on_mesh = 0
        num_dense_on_mesh = 0

        for l_idx in layer_indices_on_this_mesh:
            if l_idx < len(global_moe_pattern) and global_moe_pattern[l_idx] == 1:
                num_moe_on_mesh += 1
            else:
                num_dense_on_mesh += 1

        # 4. Calculate Total Parameters on Mesh (Pre-Sharding) ---
        total_params_on_mesh = num_dense_on_mesh * (
            attn_params + dense_mlp_params
        ) + num_moe_on_mesh * (attn_params + sparse_mlp_params_all_experts)

        if i == num_meshes - 1:
            total_params_on_mesh += 2 * mesh_args.hidden_size

        # Embeddings
        embedding_size = mesh_args.hidden_size * mesh_args.padded_vocab_size
        untie = getattr(mesh_args, "untie_embeddings_and_output_weights", False)

        embedding_params_on_mesh = 0
        if i == 0:
            embedding_params_on_mesh += embedding_size * (2 if untie else 1)
        elif i == num_meshes - 1 and untie:
            embedding_params_on_mesh += embedding_size

        # 5. Apply Parallelism Sharding (TP & EP) ---
        sharded_dense_params = (
            num_dense_on_mesh * (attn_params + dense_mlp_params)
        ) / mesh_args.tensor_model_parallel_size
        moe_attn_sharded = (num_moe_on_mesh * attn_params) / mesh_args.tensor_model_parallel_size

        expert_tp_size = getattr(
            mesh_args, "expert_tensor_parallel_size", mesh_args.tensor_model_parallel_size
        )
        moe_mlp_sharded = (num_moe_on_mesh * sparse_mlp_params_all_experts) / (
            mesh_args.expert_model_parallel_size * expert_tp_size
        )

        sharded_layer_params = sharded_dense_params + moe_attn_sharded + moe_mlp_sharded
        sharded_embedding_params = embedding_params_on_mesh / mesh_args.tensor_model_parallel_size
        sharded_final_norm = (
            (2 * mesh_args.hidden_size / mesh_args.tensor_model_parallel_size)
            if i == num_meshes - 1
            else 0
        )

        total_sharded_params = sharded_layer_params + sharded_embedding_params + sharded_final_norm

        # 6. Optimizer States ---
        use_do = getattr(mesh_args, "use_distributed_optimizer", False)
        if use_do:
            optimizer_multiplier = 4 + (12 / mesh_args.data_parallel_size)
        else:
            optimizer_multiplier = 18

        mesh_memory_dict[i] = total_sharded_params * optimizer_multiplier

    return mesh_memory_dict


# Activation Memory - Per Stage
def hetero_compute_activation_memory(base_args, strategy, config):
    """
    Computes the peak activation memory for each pipeline stage.

    Returns:
        List[float]: A list where index i corresponds to the activation memory
                     (in Bytes) for pipeline stage i.
    """
    hetero_meshes = strategy.get("hetero_process_meshes", [])
    hetero_split = strategy.get("hetero_pipeline_layer_split", [])
    global_pp_size = strategy.get("pipeline_model_parallel_size", 1)
    gbs = config.train.model.global_batch_size
    mbs = strategy.get("micro_batch_size", 1)

    global_moe_pattern = _get_global_moe_pattern(base_args)

    # Pipeline Schedule Factors
    dp_fallback = getattr(base_args, "data_parallel_size", 1)
    num_microbatches_global = max(1, gbs // (dp_fallback * max(1, mbs))) if dp_fallback > 0 else 1

    if global_pp_size > 1:
        in_flight_microbatches = min(num_microbatches_global, global_pp_size)
        if getattr(base_args, "virtual_pipeline_model_parallel_size", None) is not None:
            vpp = base_args.virtual_pipeline_model_parallel_size
            penalty = 1 + (global_pp_size - 1) / (global_pp_size * vpp)
            in_flight_microbatches = math.ceil(penalty * global_pp_size)
    else:
        in_flight_microbatches = 1

    stage_activation_list = []
    current_global_layer_idx = 0

    for stage_idx, layers_in_stage in enumerate(hetero_split):
        mesh_params = _get_mesh_params_for_stage(stage_idx, hetero_meshes)
        if not mesh_params:
            stage_activation_list.append(float("inf"))
            current_global_layer_idx += layers_in_stage
            continue

        # Set up local args for calculation
        stage_args = types.SimpleNamespace(**vars(base_args))
        stage_args.tensor_model_parallel_size = mesh_params["tp"]
        stage_args.context_parallel_size = mesh_params["cp"]
        stage_args.expert_model_parallel_size = mesh_params["ep"]
        stage_args.data_parallel_size = mesh_params["dp"]
        stage_args.micro_batch_size = 1

        # Identify layer types in this stage
        num_moe_in_stage = 0
        num_dense_in_stage = 0
        for i in range(layers_in_stage):
            idx = current_global_layer_idx + i
            if idx < len(global_moe_pattern) and global_moe_pattern[idx] == 1:
                num_moe_in_stage += 1
            else:
                num_dense_in_stage += 1

        current_global_layer_idx += layers_in_stage

        # Calculate Activation Components (MBS=1)
        attn_comps = _calculate_attn_activation_components(stage_args)
        dense_mlp_comps = _calculate_mlp_activation_components(stage_args, is_expert=False)
        expert_mlp_comps = _calculate_mlp_activation_components(stage_args, is_expert=True)
        moe_gate_act = _calculate_moe_gate_activation(stage_args)
        embedding_act = _calculate_embedding_activation(stage_args)
        output_act = _calculate_output_layer_activation(stage_args)

        # Apply Parallelism Scaling
        tp = stage_args.tensor_model_parallel_size
        sp_enabled = strategy.get(
            "sequence_parallel", getattr(stage_args, "sequence_parallel", False)
        )
        sp_divisor = tp if sp_enabled else 1

        attn_act_per_layer = (attn_comps["tp_scaled"] / tp) + (
            attn_comps["not_tp_scaled"] / sp_divisor
        )
        dense_mlp_act_per_layer = (dense_mlp_comps["tp_scaled"] / tp) + (
            dense_mlp_comps["not_tp_scaled"] / sp_divisor
        )
        moe_mlp_act_per_layer = (
            (moe_gate_act / sp_divisor)
            + (expert_mlp_comps["tp_scaled"] / tp)
            + (expert_mlp_comps["not_tp_scaled"] / sp_divisor)
        )

        _NVTE_FLASH = int(os.getenv("NVTE_FLASH_ATTN", "1"))
        if strategy.get("recompute_granularity") == "selective" or _NVTE_FLASH:
            bass_mem = attn_comps["bass"] / sp_divisor
            attn_act_per_layer -= bass_mem
            moe_mlp_act_per_layer -= bass_mem

        # Recompute Logic
        recompute_method = strategy.get("recompute_method")
        recompute_granularity = strategy.get("recompute_granularity")
        recompute_num_layers = strategy.get("recompute_num_layers", 0)

        layers_activation_no_extra = 0
        input_act_proxy = attn_comps["not_tp_scaled"] / sp_divisor

        if recompute_method == "uniform" and recompute_granularity == "full":
            recomputed_count = (
                min(layers_in_stage, recompute_num_layers) if recompute_num_layers else 0
            )
            standard_count = layers_in_stage - recomputed_count

            # Recomputed layers cost
            layers_activation_no_extra += recomputed_count * input_act_proxy

            # Standard layers cost
            if standard_count > 0:
                ratio_moe = num_moe_in_stage / layers_in_stage if layers_in_stage > 0 else 0
                rem_moe = round(standard_count * ratio_moe)
                rem_dense = standard_count - rem_moe
                layers_activation_no_extra += (rem_dense * dense_mlp_act_per_layer) + (
                    rem_moe * moe_mlp_act_per_layer
                )

        else:
            layers_activation_no_extra = (num_dense_in_stage * dense_mlp_act_per_layer) + (
                num_moe_in_stage * moe_mlp_act_per_layer
            )

        # Add Extra Activations
        extra_activation = 0
        if stage_idx == 0:
            extra_activation += embedding_act / sp_divisor
        if stage_idx == global_pp_size - 1:
            extra_activation += output_act / sp_divisor

        cp_divisor = max(1, stage_args.context_parallel_size)
        final_stage_activation_1mb = (layers_activation_no_extra + extra_activation) / cp_divisor

        total_stage_activation = final_stage_activation_1mb * mbs * in_flight_microbatches
        stage_activation_list.append(total_stage_activation)

    return stage_activation_list


# Final Reporting
def hetero_report_theoretical_memory(strategy, config, base_args):
    """
    Aggregates Static and Activation memory to report peak memory usage per mesh.

    Returns:
        List[int]: A list of integers representing the peak memory in MB for each mesh.
                   Returns [inf, inf...] if calculation fails.
    """
    try:
        weight_opt_dict = hetero_compute_weight_and_optimizer_memory(base_args, strategy, config)
        activation_list = hetero_compute_activation_memory(base_args, strategy, config)

        hetero_meshes = strategy.get("hetero_process_meshes", [])
        final_peaks = []

        mesh_peak_map = {}

        for m_idx, static_mem in weight_opt_dict.items():
            mesh_peak_map[m_idx] = static_mem

        # Group activations by mesh
        mesh_to_activations = {}
        for stage_idx, act_bytes in enumerate(activation_list):
            m_params = _get_mesh_params_for_stage(stage_idx, hetero_meshes)
            if m_params:
                m_idx = m_params["mesh_idx"]
                if m_idx not in mesh_to_activations:
                    mesh_to_activations[m_idx] = []
                mesh_to_activations[m_idx].append(act_bytes)

        sorted_mesh_indices = sorted(mesh_peak_map.keys())
        for m_idx in sorted_mesh_indices:
            static = mesh_peak_map[m_idx]
            acts = mesh_to_activations.get(m_idx, [0])
            peak_act = max(acts) if acts else 0

            if static == float("inf") or peak_act == float("inf"):
                total = float("inf")
            else:
                total = (static + peak_act) / NUM_BYTES_IN_MEGABYTE

            final_peaks.append(int(total) if total != float("inf") else float("inf"))

        logger.info(f">>> [FS] Hetero Theoretical Peak Memory per Mesh (MB): {final_peaks}\n")
        return final_peaks

    except Exception as e:
        logger.exception(f"Failed to calculate hetero memory: {e}")
        return [float("inf")] * len(strategy.get("hetero_process_meshes", []))
