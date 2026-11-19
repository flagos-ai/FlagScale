# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Measured MCore FLOPs accounting for DSv4 packed document attention.

Keep the complete source function so future DSv4 options use the same counting
formula; FlagScale's other model families retain their existing implementation.
"""
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    is_linear_attention_variant,
)
from megatron.training.utils import is_hybrid_model


def _dsv4_hybrid_self_attention_flops(
    *,
    hidden_size,
    num_attention_heads,
    v_head_dim,
    q_lora_rank,
    o_groups,
    o_lora_rank,
    csa_window_size,
    seq_length,
    n_layers_r0,
    n_layers_r4,
    n_layers_r128,
    dsa_indexer_n_heads,
    dsa_indexer_head_dim,
    dsa_indexer_topk,
):
    """Per-iteration DSv4-hybrid self-attention FLOPs coefficients.

    DSv4 attention layers are MLA-based but replace full ``O(L^2)`` core
    attention with sparse attention (sliding window + compressed-KV), plus a
    main compressor and, on ``ratio==4`` layers, a learned indexer (DSA). This
    returns coefficients summed over the per-ratio attention-layer counts.

    Layer-type ratios:
      * ``r==0``   (Window): window-only attention; no compressor / indexer.
      * ``r==4``   (CSA):    window + learned-topk over compressed KV
        (compressor + indexer).
      * ``r==128`` (HCA):    window + all compressed KV (compressor only).

    Returns ``(token_linear, core)`` WITHOUT the fwd+bwd (x3) or FMA (x2)
    expansion factors -- the caller applies those. Multiply ``token_linear`` by
    the real (unpadded) token count and ``core`` by ``sum_i(L_i ** 2)``.
    """
    n_attn_layers = n_layers_r0 + n_layers_r4 + n_layers_r128

    # ---- MLA projections (token-linear, per attention layer) ----
    # DSv4 hybrid MLA: ``qk_head_dim + qk_pos_emb_head_dim == v_head_dim`` and the
    # joint KV is a single ``hidden -> v_head_dim`` projection; the output uses a
    # grouped low-rank ``(o_groups x o_lora_rank)`` projection (NOT the dense
    # ``num_heads * v_head_dim -> hidden`` projection of plain MLA).
    q_term = q_lora_rank * (hidden_size + num_attention_heads * v_head_dim + 1)
    kv_term = hidden_size * v_head_dim + v_head_dim
    o_term = num_attention_heads * v_head_dim * o_lora_rank + o_groups * o_lora_rank * hidden_size
    mla_proj_term = (q_term + kv_term + o_term) * n_attn_layers

    # ---- Sparse attention (replaces full core attention) ----
    # Split into token-linear parts (window attention, constant per token) and
    # L^2 parts (compressed-KV attention, scales with sequence length) so THD
    # packed sequences get correct ``seqlen_squared_sum`` scaling.
    # r=0: window-only, fixed per-token cost.
    sparse_attn_r0 = n_layers_r0 * num_attention_heads * csa_window_size * v_head_dim * 2
    # r=128: window (token-linear) + all compressed KV (L^2). Compressed
    # positions per token ~= L/(128*2) (causal /2), x2 for QK^T + softmax@V ->
    # L^2 coefficient ``num_heads * v_head_dim / 128``.
    sparse_attn_r128_window = n_layers_r128 * num_attention_heads * csa_window_size * v_head_dim * 2
    sparse_attn_r128_core = n_layers_r128 * num_attention_heads * v_head_dim / 128

    # ---- Main compressor (ratio > 0 layers) ----
    # Two projections per layer (wkv + wgate): ``hidden -> coff * v_head_dim``.
    # ratio == 4: coff = 2 (overlapping windows); ratio == 128: coff = 1.
    main_compressor_term = (
        n_layers_r4 * hidden_size * (2 * v_head_dim) * 2
        + n_layers_r128 * hidden_size * (1 * v_head_dim) * 2
    )

    # ---- r=4 layers: sparse attention + indexer ----
    if n_layers_r4 > 0:
        assert (
            dsa_indexer_n_heads is not None
        ), "dsa_indexer_n_heads must be set for dsv4_hybrid with ratio==4 layers."
        assert (
            dsa_indexer_head_dim is not None
        ), "dsa_indexer_head_dim must be set for dsv4_hybrid with ratio==4 layers."
        assert (
            dsa_indexer_topk is not None
        ), "dsa_indexer_topk must be set for dsv4_hybrid with ratio==4 layers."
        effective_topk_4 = min(dsa_indexer_topk, seq_length // 4)
        avg_comp_4 = effective_topk_4 * (1 - effective_topk_4 * 4 / (2 * seq_length))
        sparse_attn_r4 = (
            n_layers_r4 * num_attention_heads * (csa_window_size + avg_comp_4) * v_head_dim * 2
        )
        # Indexer token-linear: compressor (coff=2, wkv + wgate), Q proj, weights proj.
        indexer_token_term = (
            n_layers_r4 * hidden_size * (2 * dsa_indexer_head_dim) * 2
            + n_layers_r4 * q_lora_rank * dsa_indexer_n_heads * dsa_indexer_head_dim
            + n_layers_r4 * hidden_size * dsa_indexer_n_heads
        )
        # Indexer L^2: scoring each query against ~L/4 compressed positions.
        indexer_scoring_core = n_layers_r4 * dsa_indexer_n_heads * dsa_indexer_head_dim / 4
    else:
        sparse_attn_r4 = 0
        indexer_token_term = 0
        indexer_scoring_core = 0

    token_linear = (
        mla_proj_term
        + sparse_attn_r0
        + sparse_attn_r4
        + sparse_attn_r128_window
        + main_compressor_term
        + indexer_token_term
    )
    core = sparse_attn_r128_core + indexer_scoring_core
    return token_linear, core


def num_floating_point_operations(
    args,
    batch_size,
    seqlen_squared_sum_in_batch=None,
    total_real_tokens_in_batch=None,
):
    """Compute the number of floating-point operations for one global batch.

    Args:
        args: Megatron args (provides hidden size, num layers, seq length, ...).
        batch_size: Global batch size (samples per iteration). Only used to
            compute the default token counts for the BSHD path -- with THD
            both ``total_real_tokens_in_batch`` and ``seqlen_squared_sum_in_batch``
            are passed in explicitly and ``batch_size`` is irrelevant.
        seqlen_squared_sum_in_batch: ``sum_i(L_i ** 2)`` across all REAL
            (unpadded) sub-sequences in the global batch. Drives the
            core-attention L^2 FLOPs. For BSHD this equals
            ``batch_size * args.seq_length ** 2`` (the default when ``None``).
            For THD it is the actual ragged sum and is strictly less than the
            BSHD value, reflecting per-chunk causal masking AND the fact that
            padding tokens do not contribute to attention scores.
        total_real_tokens_in_batch: ``sum_i(L_i)``, the TOTAL REAL (unpadded)
            token count across the global batch. Drives all token-linear FLOPs
            (QKV+output projections, MLP, MoE, MTP norms/projs, logits). For
            BSHD this equals ``batch_size * args.seq_length`` (the default when
            ``None``). For THD it equals the real token count -- strictly less
            than ``batch_size * args.seq_length`` whenever the dataloader added
            CP-alignment padding or end-of-sequence padding, so neither kind of
            padding shows up in the reported FLOPs.
    """
    # Defaults: BSHD layout assumption (full causal mask, every sample length =
    # seq_length, no padding). For BSHD ``total_real_tokens = batch * s`` and
    # ``seqlen_squared_sum = batch * s^2`` recover the original closed-form.
    if seqlen_squared_sum_in_batch is None:
        seqlen_squared_sum_in_batch = batch_size * args.seq_length * args.seq_length
    if total_real_tokens_in_batch is None:
        total_real_tokens_in_batch = batch_size * args.seq_length

    def mlp_layer_flops(total_tokens, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * total_tokens * hidden_size**2

    def moe_layer_flops(
        total_tokens,
        hidden_size,
        moe_ffn_hidden_size,
        shared_expert_ffn_hidden_size,
        num_experts_routed_to,
        moe_latent_size=None,
        swiglu=False,
    ):
        """Calculate FLOPs for an MoE layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        if moe_latent_size is None:
            routed_flops = (
                4
                * total_tokens
                * hidden_size
                * moe_ffn_hidden_size
                * num_experts_routed_to
                * scale_factor
            )
        else:
            # Routed experts run on moe_latent_size.
            routed_flops = (
                4
                * total_tokens
                * moe_latent_size
                * moe_ffn_hidden_size
                * num_experts_routed_to
                * scale_factor
            )
            # Up proj and down proj.
            routed_flops += 4 * total_tokens * hidden_size * moe_latent_size
        shared_flops = 4 * total_tokens * hidden_size * shared_expert_ffn_hidden_size * scale_factor
        return routed_flops + shared_flops

    def attn_layer_flops(
        total_tokens,
        seqlen_squared_sum,
        hidden_size,
        num_heads,
        gqa=True,
        gqa_groups=8,
        kv_channels=None,
    ):
        """Calculate FLOPs for an attention layer.

        Two parts:
          - Token-linear (QKV + output projections): scales with ``total_tokens``
            (= ``sum_i(L_i)``). Padding tokens are excluded by passing the real
            (unpadded) token count.
          - Core attention (``QK^T`` and ``softmax(QK^T) V``): scales with
            ``seqlen_squared_sum`` (= ``sum_i(L_i ** 2)``).
        """
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups if gqa else num_heads
        # 4 * total_tokens * h * p * (h + h*(g/n)): QKV + output projections (fwd*3, with FMA*2).
        # 2 * sum(L^2) * h * p: core attention (causal mask -> /2 cancels with FMA *2).
        return (
            4 * total_tokens * hidden_size * p * (hidden_size + hidden_size * (g / num_heads))
            + 2 * seqlen_squared_sum * hidden_size * p
        )

    def mamba_layer_flops(
        total_tokens, hidden_size, state_dim=16, head_dim=64, num_groups=1, num_heads=128
    ):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        if num_heads:
            nheads = num_heads
        else:
            nheads = d_in // head_dim
        return (
            (
                2 * total_tokens * hidden_size * (2 * d_in + 2 * num_groups * state_dim + nheads)
            )  # in_proj
            + (7 * total_tokens * d_in * state_dim)  # scan
            + (2 * total_tokens * d_in * hidden_size)  # out_proj
        )

    def gdn_layer_flops(
        total_tokens,
        hidden_size,
        qk_head_dim=128,
        v_head_dim=128,
        num_qk_heads=16,
        num_v_heads=32,
        conv_kernel_dim=4,
    ):
        """Calculate FLOPs for a Gated Delta Net (GDN) layer."""
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads
        return (
            2
            * total_tokens
            * (
                # in_proj: hidden_size -> (2*qk_dim + 2*v_dim + 2*num_v_heads)
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                # conv1d
                + conv_kernel_dim * (2 * qk_dim + v_dim)
                # gated delta rule: KK^T, VK^T, S(a(I-bKK^T)), and SQ
                + num_v_heads * (v_head_dim**2) * 4
                # out_proj: v_dim -> hidden_size
                + hidden_size * v_dim
            )
        )

    def hybrid_flops(
        total_tokens,
        seqlen_squared_sum,
        hidden_size,
        num_attn_layers,
        num_mamba_layers,
        num_mlp_layers,
        num_moe_layers,
        num_gdn_layers=0,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=128,
        num_attn_heads=32,
        gqa=True,
        gqa_groups=8,
        kv_channels=None,
        mlp_expansion=4.0,
        swiglu=False,
        moe_latent_size=None,
        moe_ffn_hidden_size=2048,
        shared_expert_ffn_hidden_size=2048,
        num_experts_routed_to=1,
        gdn_qk_head_dim=128,
        gdn_v_head_dim=128,
        gdn_num_qk_heads=16,
        gdn_num_v_heads=32,
        gdn_conv_kernel_dim=4,
        vocab_size=256000,
        mtp_num_layers=0,
    ):
        """Calculate total FLOPs for the hybrid model."""
        flops_fwd = (
            num_attn_layers
            * attn_layer_flops(
                total_tokens,
                seqlen_squared_sum,
                hidden_size,
                num_attn_heads,
                gqa,
                gqa_groups,
                kv_channels,
            )
            + num_mlp_layers * mlp_layer_flops(total_tokens, hidden_size, mlp_expansion, swiglu)
            + num_mamba_layers
            * mamba_layer_flops(
                total_tokens,
                hidden_size,
                mamba_state_dim,
                mamba_head_dim,
                mamba_num_groups,
                mamba_num_heads,
            )
            + num_moe_layers
            * moe_layer_flops(
                total_tokens,
                hidden_size,
                moe_ffn_hidden_size,
                shared_expert_ffn_hidden_size,
                num_experts_routed_to,
                moe_latent_size,
                swiglu,
            )
            + num_gdn_layers
            * gdn_layer_flops(
                total_tokens,
                hidden_size,
                gdn_qk_head_dim,
                gdn_v_head_dim,
                gdn_num_qk_heads,
                gdn_num_v_heads,
                gdn_conv_kernel_dim,
            )
            + (
                2 * total_tokens * hidden_size * vocab_size * (1 + mtp_num_layers)
            )  # logits computation
        )
        return flops_fwd * 3

    def transformer_flops():
        """Calculate FLOPs for a standard Transformer model."""
        # TODO(helenn/dnarayanan): Refactor this to reuse the helper methods.
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # MoE.
        if args.num_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = args.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
            last_layer_is_moe = 0
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
            assert len(moe_layer_pattern) == args.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {args.num_layers}, "
                f"current moe layer pattern: {args.moe_layer_freq}"
            )
            num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
            num_dense_layers = args.num_layers - num_moe_layers
            num_experts_routed_to = args.moe_router_topk
            last_layer_is_moe = moe_layer_pattern[-1]

        if args.mtp_num_layers is not None:
            mtp_num_layers = args.mtp_num_layers
            num_moe_layers += last_layer_is_moe * mtp_num_layers
            num_dense_layers += (1 - last_layer_is_moe) * mtp_num_layers
            num_layers = args.num_layers + mtp_num_layers
        else:
            mtp_num_layers = 0
            num_layers = args.num_layers

        moe_ffn_hidden_size = (
            args.moe_ffn_hidden_size
            if args.moe_ffn_hidden_size is not None
            else args.ffn_hidden_size
        )
        moe_latent_size = args.moe_latent_size
        shared_expert_ffn_hidden_size = (
            0
            if args.moe_shared_expert_intermediate_size is None
            else args.moe_shared_expert_intermediate_size
        )

        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        forward_backward_expansion_factor = 3
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        fma_expansion_factor = 2
        # - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
        # - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
        ffn_expansion_factor = 3 if args.swiglu else 2

        # self_attn is split into a token-linear part (projections, multiplied by
        # ``batch_size * args.seq_length`` like all other token-linear work) and a
        # core-attention part (``QK^T`` and ``softmax(QK^T) V``) whose compute scales
        # with ``sum_i(L_i ** 2)`` instead of ``batch * seq^2``. With unpacked BSHD the
        # two are equal, so the BSHD result is unchanged. With THD/packed sequences
        # ``seqlen_squared_sum_in_batch`` is strictly smaller, reflecting the per-chunk
        # causal mask of packed attention.
        if args.experimental_attention_variant == "dsv4_hybrid" and not getattr(
            args, "csa_dense_mode", False
        ):
            ratios = args.csa_compress_ratios
            assert (
                ratios is not None and len(ratios) == num_layers
            ), "DSv4 FLOPs require one compression ratio per decoder/MTP layer"
            token_term, core_term = _dsv4_hybrid_self_attention_flops(
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                v_head_dim=args.v_head_dim,
                q_lora_rank=args.q_lora_rank,
                o_groups=args.o_groups,
                o_lora_rank=args.o_lora_rank,
                csa_window_size=args.csa_window_size,
                seq_length=args.seq_length,
                n_layers_r0=ratios.count(0),
                n_layers_r4=ratios.count(4),
                n_layers_r128=ratios.count(128),
                dsa_indexer_n_heads=args.dsa_indexer_n_heads,
                dsa_indexer_head_dim=args.dsa_indexer_head_dim,
                dsa_indexer_topk=args.dsa_indexer_topk,
            )
            # The helper sums heterogeneous layers. The common aggregation below
            # multiplies these per-layer averages by num_standard_attention_layers.
            expansion = forward_backward_expansion_factor * fma_expansion_factor
            standard_self_attn_term = expansion * token_term / num_layers
            standard_self_attn_core_term = expansion * core_term / num_layers
        elif args.multi_latent_attention:
            assert not args.group_query_attention
            """
            Basic arithmetic
            let B is batch size, s is seq_len, h is embedding dim,
            for one self_attnetion block (prenorm is not included)
            qkv projection:  6Bsh^2
            attn:            2Bs^2h
            attn over value: 2Bs^2h
            oproj:           2Bsh^2

            references
            https://arxiv.org/abs/2305.10403
            https://arxiv.org/abs/2205.05198
            """
            ## MLA
            if args.q_lora_rank is None:
                q_term = (
                    args.hidden_size
                    * args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                )
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size
                    + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    + 1
                )
            # Token-linear part of MLA self-attention (lora projs, kv proj, RoPE, output proj).
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## q lora + rope + q norm
                    q_term
                    ## kv lora + rope + kv norm
                    + args.kv_lora_rank
                    * (
                        args.hidden_size
                        + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                        + 1
                    )
                    + args.hidden_size * args.qk_pos_emb_head_dim
                    ## o proj
                    + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
                )
            )
            # Core-attention (L^2) part: ``QK^T`` and ``(softmax(QK^T)) V``. The
            # ``/2`` accounts for the causal mask and the ``*2`` cancels it via FMA.
            standard_self_attn_core_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim) / 2
                    + args.num_attention_heads * args.v_head_dim / 2
                )
            )

        else:
            ## MHA or GQA
            query_projection_size = args.kv_channels * args.num_attention_heads
            key_projection_size = args.kv_channels * args.num_query_groups
            value_projection_size = args.kv_channels * args.num_query_groups
            gate_projection_size = query_projection_size if args.attention_output_gate else 0
            # Token-linear part of self-attention (QKV projection + output projection).
            standard_self_attn_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * (
                    ## qkv proj
                    args.hidden_size
                    * (
                        query_projection_size
                        + key_projection_size
                        + value_projection_size
                        + gate_projection_size
                    )
                    ## out proj
                    + query_projection_size * args.hidden_size
                )
            )
            # Core-attention (L^2) part: ``QK^T`` and ``(softmax(QK^T)) V``.
            standard_self_attn_core_term = (
                forward_backward_expansion_factor
                * fma_expansion_factor
                * query_projection_size
                / 2  # causal mask (only half of the mask is non-zero)
                * 2  # QK^T and (QK^T)V
            )

        if is_linear_attention_variant(args.experimental_attention_variant):
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.linear_attention_freq, int):
                linear_attention_pattern = [
                    # [1,1,...,1,0,1,1,...,1,0,...]
                    0 if ((i + 1) % args.linear_attention_freq == 0) else 1
                    for i in range(num_layers)
                ]
            elif isinstance(args.linear_attention_freq, list):
                linear_attention_pattern = args.linear_attention_freq
                assert len(linear_attention_pattern) == num_layers, (
                    f"Invalid length of linear_attention_pattern: {len(linear_attention_pattern)}, "
                    f"expected {num_layers}, "
                    f"current linear attention pattern: {args.linear_attention_freq}"
                )
            elif args.linear_attention_freq is None:
                # This should be caught by config validation, but raise here as a safety check
                raise ValueError(
                    f"Linear attention type {args.experimental_attention_variant} is specified "
                    "but linear_attention_freq is None. "
                    "Please set linear_attention_freq to specify the LA/SDPA layer pattern."
                )
            else:
                raise ValueError(
                    f"Invalid linear_attention_freq: {type(args.linear_attention_freq)},"
                    f" {args.linear_attention_freq}"
                )
            num_linear_attention_layers = sum(linear_attention_pattern)
            num_standard_attention_layers = num_layers - num_linear_attention_layers

            if args.experimental_attention_variant == "gated_delta_net":
                # Calculate the FLOPs for the gated delta net attention.
                qk_head_dim = args.linear_key_head_dim
                v_head_dim = args.linear_value_head_dim
                num_qk_heads = args.linear_num_key_heads
                num_v_heads = args.linear_num_value_heads
                qk_dim = qk_head_dim * num_qk_heads
                v_dim = v_head_dim * num_v_heads
                linear_self_attn_term = (
                    forward_backward_expansion_factor
                    * fma_expansion_factor
                    * (
                        ## in proj
                        args.hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                        ## conv1d
                        + args.linear_conv_kernel_dim * (2 * qk_dim + v_dim)
                        ## gated delta rule
                        + num_v_heads * (v_head_dim**2) * 4  # KK^T, VK^T, S(a(I-bKK^T)), and SQ
                        ## out proj
                        + args.hidden_size * v_dim
                    )
                )
            else:
                raise ValueError(
                    "Invalid experimental_attention_variant: "
                    f"{args.experimental_attention_variant}"
                )
        else:
            num_linear_attention_layers = 0
            linear_self_attn_term = 0
            num_standard_attention_layers = num_layers

        # Token-linear self-attention work (projections + linear-attention variants).
        # Linear attention has no L^2 term, so it stays entirely token-linear.
        self_attn_term = (
            linear_self_attn_term * num_linear_attention_layers
            + standard_self_attn_term * num_standard_attention_layers
        )
        # Core attention (L^2) FLOPs per standard-attention layer.
        self_attn_core_term = standard_self_attn_core_term * num_standard_attention_layers

        # Token-linear FLOPs scale with the real (unpadded) token count.
        # For BSHD this falls back to ``batch_size * seq_length`` (no padding).
        total_floating_point_operations = (
            total_real_tokens_in_batch
            * (
                # MLP
                forward_backward_expansion_factor
                * fma_expansion_factor
                * args.hidden_size
                * (
                    # dense layer (deepseek v2, v3 style)
                    (args.ffn_hidden_size * ffn_expansion_factor) * num_dense_layers
                    # routed experts
                    + (
                        (moe_ffn_hidden_size * num_experts_routed_to * ffn_expansion_factor)
                        if moe_latent_size is None
                        else (
                            (
                                moe_ffn_hidden_size
                                * num_experts_routed_to
                                * ffn_expansion_factor
                                * moe_latent_size
                                / args.hidden_size
                            )  # Routed experts run on moe_latent_size.
                            + 2 * moe_latent_size  # Up proj and down proj.
                        )
                    )
                    * num_moe_layers
                    # Shared Experts.
                    + (shared_expert_ffn_hidden_size * ffn_expansion_factor) * num_moe_layers
                )
                # Self Attention (token-linear part).
                + self_attn_term
                # MTP norms and proj
                + forward_backward_expansion_factor
                * fma_expansion_factor
                * mtp_num_layers
                * (
                    # MTP eh norm + final nrom
                    3 * args.hidden_size
                    # MTH eh proj
                    + 2 * args.hidden_size * args.hidden_size
                )
                # Logit.
                + forward_backward_expansion_factor
                * fma_expansion_factor
                * args.hidden_size
                * args.padded_vocab_size
                * (mtp_num_layers + 1)  # MTP + final logit
            )
            # Self Attention (core L^2 part). For BSHD the default
            # ``seqlen_squared_sum_in_batch = batch_size * seq_length^2`` recovers the
            # original ``batch * seq^2 * Q_proj`` term; for THD this uses the actual
            # ragged ``sum(L_i^2)`` and so undercounts the BSHD figure correctly.
            + seqlen_squared_sum_in_batch * self_attn_core_term
        )
        return total_floating_point_operations

    # Main entrypoint for FLOPs calculation.
    if is_hybrid_model(args):
        # Calculate the number of each type of layer.
        from operator import itemgetter

        from megatron.core.models.hybrid.hybrid_layer_allocation import (
            Symbols,
            get_hybrid_layer_counts,
        )

        num_mamba_layers, num_gdn_layers, num_attn_layers, num_mlp_layers, num_moe_layers = (
            itemgetter(Symbols.MAMBA, Symbols.GDN, Symbols.ATTENTION, Symbols.MLP, Symbols.MOE)(
                get_hybrid_layer_counts(args.hybrid_layer_pattern)
            )
        )

        mtp_num_layers = args.mtp_num_layers
        if mtp_num_layers is None:
            mtp_num_layers = 0
        # Compute hybrid model FLOPs.
        return hybrid_flops(
            total_tokens=total_real_tokens_in_batch,
            seqlen_squared_sum=seqlen_squared_sum_in_batch,
            hidden_size=args.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            num_moe_layers=num_moe_layers,
            num_gdn_layers=num_gdn_layers,
            mamba_state_dim=args.mamba_state_dim,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            mamba_num_heads=args.mamba_num_heads,
            num_attn_heads=args.num_attention_heads,
            gqa=args.group_query_attention,
            gqa_groups=args.num_query_groups,
            kv_channels=args.kv_channels,
            mlp_expansion=args.ffn_hidden_size / args.hidden_size,
            swiglu=args.swiglu,
            moe_latent_size=args.moe_latent_size,
            moe_ffn_hidden_size=(
                args.moe_ffn_hidden_size
                if args.moe_ffn_hidden_size is not None
                else args.ffn_hidden_size
            ),
            shared_expert_ffn_hidden_size=(
                0
                if args.moe_shared_expert_intermediate_size is None
                else args.moe_shared_expert_intermediate_size
            ),
            num_experts_routed_to=args.moe_router_topk,
            gdn_qk_head_dim=args.linear_key_head_dim or 128,
            gdn_v_head_dim=args.linear_value_head_dim or 128,
            gdn_num_qk_heads=args.linear_num_key_heads or 16,
            gdn_num_v_heads=args.linear_num_value_heads or 32,
            gdn_conv_kernel_dim=args.linear_conv_kernel_dim or 4,
            vocab_size=args.padded_vocab_size,
            mtp_num_layers=mtp_num_layers,
        )
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()
