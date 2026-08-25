# Copyright (c) 2025, BAAI. All rights reserved.

"""MIMO generic parallelism configuration."""

from dataclasses import dataclass


@dataclass
class ModuleParallelismConfig:
    """Parallelism configuration for a single MIMO module.

    All sizes must multiply to the world size for colocated deployment.
    ``expert_model_parallel_size`` subdivides the data-parallel domain
    (``dp = ep * expert_dp``) and therefore does not appear in the product.
    Only the language module may set it > 1 (the vision module is dense).
    """

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    data_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1


def compute_vit_batch_factor(
    vision_data_parallel_size: int,
    vision_micro_batch_size: int,
    language_data_parallel_size: int,
    language_micro_batch_size: int,
    num_microbatches: int,
) -> int:
    """Return the vit_batch_factor relating vision and language batching.

    ``vision_micro_batch_size`` (vbs) is the ViT-side micro batch size: the
    number of samples each vision DP entity processes in one forward, the
    vision analog of the LM ``micro_batch_size`` (mbs).  ``vit_batch_factor``
    (vbf) is the derived relation ``(vision_dp * vbs) / (language_dp * mbs)`` —
    how many LLM microbatches one ViT macro forward serves.
    """
    vision_samples = vision_data_parallel_size * vision_micro_batch_size
    language_samples = language_data_parallel_size * language_micro_batch_size
    # Reject floor-division truncation that would silently shrink the
    # effective vit_batch_factor.
    assert vision_samples % language_samples == 0, (
        f"vision_dp * vision_micro_batch_size ({vision_samples}) must be a multiple of "
        f"language_dp * micro_batch_size ({language_samples}). "
        f"Adjust vision_micro_batch_size or micro_batch_size."
    )
    vit_batch_factor = vision_samples // language_samples
    assert num_microbatches % vit_batch_factor == 0, (
        f"num_microbatches ({num_microbatches}) must be a multiple of "
        f"vit_batch_factor ({vit_batch_factor}) for MIMO correctness. "
        f"Adjust global_batch_size, micro_batch_size, or vision_micro_batch_size."
    )
    return vit_batch_factor


def validate_mimo_config(
    args,
    vision_parallelism: ModuleParallelismConfig,
    language_parallelism: ModuleParallelismConfig,
    num_microbatches: int,
) -> int:
    """Validate all model-agnostic MIMO configuration constraints in one place.

    Training entries must call this once before constructing the MIMO model.

    Returns the validated ``vit_batch_factor``.
    """
    assert getattr(args, "rampup_batch_size", None) is None, (
        "Colocated MIMO does not support --rampup-batch-size: the scheduler "
        "requires num_microbatches to stay divisible by vit_batch_factor "
        "throughout training, which a batch-size ramp cannot guarantee."
    )
    assert not (args.overlap_param_gather or args.overlap_grad_reduce), (
        "MIMO per-module DDP overlap is not yet validated; "
        "disable --overlap-param-gather and --overlap-grad-reduce."
    )
    assert args.ckpt_format == "torch", (
        "ChainedOptimizer returns a list of optimizer state dicts, which only the "
        "legacy 'torch' checkpoint path handles; torch_dist support is future work."
    )
    for name, parallelism in (("vision", vision_parallelism), ("language", language_parallelism)):
        assert parallelism.context_parallel_size == 1, (
            f"Colocated MIMO currently requires CP=1, got {name} "
            f"cp={parallelism.context_parallel_size}."
        )
    # Language PP > 1 is supported with the vision module colocated on the
    # language first-stage ranks only.  Vision PP follows language PP purely
    # as a grid artifact (the vision DP groups then land on same-stage rank
    # sets); the vision module itself is never pipelined.
    assert vision_parallelism.pipeline_model_parallel_size == (
        language_parallelism.pipeline_model_parallel_size
    ), (
        f"Colocated MIMO requires vision PP == language PP (vision PP is a "
        f"grid artifact, the vision module is not pipelined), got vision "
        f"pp={vision_parallelism.pipeline_model_parallel_size}, language "
        f"pp={language_parallelism.pipeline_model_parallel_size}."
    )
    assert vision_parallelism.tensor_model_parallel_size == 1, (
        f"Colocated MIMO currently requires vision TP=1, got "
        f"vision tp={vision_parallelism.tensor_model_parallel_size}."
    )
    # Expert parallelism (EP) subdivides the language DP domain as
    # dp = ep * expert_dp and never applies to the dense vision module.
    # MoE token dispatch runs in the language forward, which every rank
    # enters for every microbatch, so the all-to-all collectives stay
    # aligned; expert parameter gradients reduce over the expert-DP
    # group (singleton when ep == dp).
    assert vision_parallelism.expert_model_parallel_size == 1, (
        f"Colocated MIMO requires vision EP=1 (vision module is dense), "
        f"got vision ep={vision_parallelism.expert_model_parallel_size}."
    )
    language_ep = language_parallelism.expert_model_parallel_size
    assert language_ep >= 1 and (language_parallelism.data_parallel_size % language_ep == 0), (
        f"language ep ({language_ep}) must divide language dp "
        f"({language_parallelism.data_parallel_size})."
    )
    # The process-group builder aliases the expert-TP group to the module TP
    # group (hetero_pg_utils), so a configured ETP different from the language
    # TP would be silently ignored.  Megatron defaults ETP to TP when unset,
    # so any mismatch here is an explicit user setting and must be rejected.
    language_etp = getattr(args, "expert_tensor_parallel_size", None)
    assert language_etp is None or language_etp == (
        language_parallelism.tensor_model_parallel_size
    ), (
        f"Colocated MIMO aliases the expert-TP group to the language TP group, "
        f"so expert_tensor_parallel_size ({language_etp}) must equal the language "
        f"tensor_model_parallel_size ({language_parallelism.tensor_model_parallel_size}); "
        f"leave it unset (Megatron then defaults it to TP) or set it equal."
    )

    vision_micro_batch_size = getattr(args, "vision_micro_batch_size", args.micro_batch_size)
    vit_batch_factor = compute_vit_batch_factor(
        vision_data_parallel_size=vision_parallelism.data_parallel_size,
        vision_micro_batch_size=vision_micro_batch_size,
        language_data_parallel_size=language_parallelism.data_parallel_size,
        language_micro_batch_size=args.micro_batch_size,
        num_microbatches=num_microbatches,
    )
    assert vit_batch_factor > 1, (
        f"MIMO requires vit_batch_factor > 1, got {vit_batch_factor} "
        f"(gbs={args.global_batch_size}, "
        f"language: tp={language_parallelism.tensor_model_parallel_size}, "
        f"dp={language_parallelism.data_parallel_size}, mbs={args.micro_batch_size}; "
        f"vision: tp={vision_parallelism.tensor_model_parallel_size}, "
        f"dp={vision_parallelism.data_parallel_size}, mbs={vision_micro_batch_size}). "
        f"The vbf=1 direct path duplicates ViT compute without macro batching; "
        f"consider disabling MIMO (use_mimo: false) for this configuration."
    )
    language_tp = language_parallelism.tensor_model_parallel_size
    assert vit_batch_factor % language_tp == 0, (
        f"vit_batch_factor ({vit_batch_factor}) must be a multiple of language TP "
        f"({language_tp}) for the intra-TP owner mapping. "
        f"Adjust vision_micro_batch_size or micro_batch_size."
    )
    return vit_batch_factor
