# Copyright (c) 2026, BAAI. All rights reserved.

"""Qwen3.5 non-colocated grid configuration contract (FlagScale-native).

Defines the training-time contract of the non-colocated grid path on
:mod:`flagscale.models.mimo.bridge.parallelism`: a fail-fast validator
(:func:`validate_qwen35_grid_config`; capability boundary in its docstring)
and a builder (:func:`build_qwen35_grid_config_from_args`) that turns the
repeatable ``--mimo-module-specs`` string into a finalized, validated
:class:`MIMOParallelismConfig`.  Imports only that module plus stdlib, so the
validator stays CPU-testable without torch or Megatron.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..parallelism import ModuleParallelismConfig

from ..parallelism import (
    LANGUAGE_MODULE_NAME,
    MIMOLayout,
    MIMOParallelismConfig,
    parse_module_parallelisms,
)

#: Images module name used in the grid specs (the MIMO modality key, matching
#: ``providers.qwen35.VISION_MODALITY_NAME``).
IMAGES_MODULE_NAME = "images"


def _require_dense(module_name: str, cfg: ModuleParallelismConfig) -> None:
    """Fail fast unless the module uses CP=1, EP=1, ETP=1."""
    if cfg.context_parallel_size > 1:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid: module '{module_name}' uses "
            f"context_parallel_size={cfg.context_parallel_size}; CP > 1 is not "
            "supported in this stage (fail-fast)."
        )
    if cfg.expert_model_parallel_size > 1:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid: module '{module_name}' uses "
            f"expert_model_parallel_size={cfg.expert_model_parallel_size}; "
            "EP > 1 is not supported in this stage (fail-fast)."
        )
    if cfg.expert_tensor_parallel_size > 1:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid: module '{module_name}' uses "
            f"expert_tensor_parallel_size={cfg.expert_tensor_parallel_size}; "
            "ETP > 1 is not supported in this stage (fail-fast)."
        )


def validate_qwen35_grid_config(
    config: MIMOParallelismConfig,
    world_size: int,
    *,
    images_module_name: str = IMAGES_MODULE_NAME,
    num_layers: int | None = None,
    num_mtp_layers: int | None = None,
) -> None:
    """Validate a MIMO parallelism config for the non-colocated Qwen3.5 grid path.

    Fail-fast predicate checks (in order):
    1. Exactly two modules: the images module and the language module.
    2. NON_COLOCATED layout whose module rank ranges tile ``[0, world_size)``
       exactly; ``finalize`` enforces the generic invariants (TP powers of two,
       pairwise-divisible DP sizes, dense modality modules).
    3. CP == 1, EP == 1 and ETP == 1 for every module.
    4. Vision PP == 1.
    5. Vision DP <= language DP (variable visual tokens require fan-out).
    6. ``num_layers`` >= language PP (one layer per pipeline stage;
       divisibility is NOT required — MCore's uneven pipeline allocation
       handles it).
    7. ``num_mtp_layers`` == 0 (the grid path has no MTP wiring).

    ``finalize`` is called internally, so a non-finalized config is accepted;
    ``num_layers`` / ``num_mtp_layers`` = ``None`` skips the respective check.
    Raises ``ValueError`` on any violated constraint.
    """
    if images_module_name not in config.module_parallelisms:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid requires a module named "
            f"'{images_module_name}' in the parallelism config; found "
            f"{config.module_names}. Use --mimo-module-specs, e.g. "
            f"'images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2'."
        )
    if LANGUAGE_MODULE_NAME not in config.module_parallelisms:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid requires the '{LANGUAGE_MODULE_NAME}' "
            f"module; found {config.module_names}."
        )
    if len(config.module_parallelisms) != 2:
        raise ValueError(
            "Non-colocated Qwen3.5 grid supports exactly the 'images' and "
            f"'language' modules, got {config.module_names}."
        )

    # Exact tiling + generic invariants (no gaps / no overlaps / full world,
    # TP powers of two, pairwise-divisible DP, dense modality modules).
    config.finalize(world_size)
    if config.layout is not MIMOLayout.NON_COLOCATED:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid requires a non-colocated layout, got "
            f"{config.layout.value}. The images and language modules must "
            "span disjoint rank ranges."
        )

    images = config.module_parallelisms[images_module_name]
    language = config.module_parallelisms[LANGUAGE_MODULE_NAME]

    _require_dense(images_module_name, images)
    _require_dense(LANGUAGE_MODULE_NAME, language)

    # The vision module is never pipelined in MIMO.
    if images.pipeline_model_parallel_size > 1:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid: vision (images) PP must be 1, got "
            f"PP={images.pipeline_model_parallel_size} (fail-fast: vision "
            "pipeline parallelism is not supported)."
        )

    # Variable visual tokens: the encoder DP must never exceed the language DP
    # (fan-out is supported by the MCore bridge, fan-in of variable per-sample
    # token counts is not).
    if images.data_parallel_size > language.data_parallel_size:
        raise ValueError(
            "Non-colocated Qwen3.5 grid requires vision DP <= language DP for "
            f"variable visual tokens, got vision DP={images.data_parallel_size}, "
            f"language DP={language.data_parallel_size} (fail-fast)."
        )

    # Only the trivial bound holds (at least one layer per stage); the uneven
    # split is MCore's, encoded by the grid provider (see
    # compute_qwen35_pipeline_layer_split).
    if num_layers is not None:
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}.")
        pp = language.pipeline_model_parallel_size
        if num_layers < pp:
            raise ValueError(
                "Non-colocated Qwen3.5 grid: language num_layers must be at "
                f"least the language pipeline_model_parallel_size: "
                f"num_layers={num_layers}, PP={pp}. The uneven pipeline split "
                "(first stage base+remainder, last stage base) needs at least "
                "one layer per stage. Increase --num-layers or reduce the "
                "language PP (fail-fast)."
            )

    # MTP: the MCore MimoModel grid path has no MTP wiring.
    if num_mtp_layers is not None:
        if (
            isinstance(num_mtp_layers, bool)
            or not isinstance(num_mtp_layers, int)
            or num_mtp_layers < 0
        ):
            raise ValueError(
                f"num_mtp_layers must be a non-negative integer, got {num_mtp_layers!r}."
            )
        if num_mtp_layers > 0:
            raise ValueError(
                "Non-colocated Qwen3.5 grid: MTP (num_mtp_layers > 0) is not "
                "supported in this stage - the MCore MimoModel grid path has "
                "no MTP wiring. Set mtp_num_layers=0 (fail-fast)."
            )


def build_qwen35_grid_config_from_args(
    module_specs: str,
    world_size: int,
    *,
    num_layers: int | None = None,
    num_mtp_layers: int | None = None,
) -> MIMOParallelismConfig:
    """Build and validate the non-colocated grid config from ``--mimo-module-specs``.

    ``module_specs`` is a repeatable spec string, e.g.
    ``"images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2"``.  Raises
    ``ValueError`` on parse or validation failure; see
    :func:`validate_qwen35_grid_config` for the checked contract.
    """
    module_parallelisms = parse_module_parallelisms(module_specs)
    config = MIMOParallelismConfig(
        module_parallelisms=module_parallelisms, layout=MIMOLayout.NON_COLOCATED
    )
    validate_qwen35_grid_config(
        config,
        world_size,
        num_layers=num_layers,
        num_mtp_layers=num_mtp_layers,
    )
    return config


def describe_qwen35_grid_modules(config: MIMOParallelismConfig) -> str:
    """Human-readable per-module summary of a validated grid config."""
    parts = []
    for name, parallelism in config.module_parallelisms.items():
        parts.append(
            f"{name} tp={parallelism.tensor_model_parallel_size} "
            f"pp={parallelism.pipeline_model_parallel_size} "
            f"dp={parallelism.data_parallel_size} "
            f"ranks [{parallelism.rank_offset}, {parallelism.rank_end})"
        )
    return "; ".join(parts)


def compute_qwen35_pipeline_layer_split(
    num_layers: int,
    pipeline_model_parallel_size: int,
) -> list[int]:
    """Per-stage language layer counts for an (possibly uneven) PP split.

    Mirrors the allocation the grid provider encodes via
    ``num_layers_in_first/last_pipeline_stage``: ``base = num_layers // pp``;
    the first stage gets ``base + remainder``, every other stage ``base``.
    Example: 32 layers with PP3 -> [12, 10, 10]; with PP6 -> [7, 5, 5, 5, 5, 5].
    """
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}.")
    if (
        isinstance(pipeline_model_parallel_size, bool)
        or not isinstance(pipeline_model_parallel_size, int)
        or pipeline_model_parallel_size < 1
    ):
        raise ValueError(
            "pipeline_model_parallel_size must be a positive integer, got "
            f"{pipeline_model_parallel_size!r}."
        )
    if num_layers < pipeline_model_parallel_size:
        raise ValueError(
            f"num_layers ({num_layers}) must be at least pipeline_model_parallel_size "
            f"({pipeline_model_parallel_size}); every PP stage needs at least one layer."
        )
    base, remainder = divmod(num_layers, pipeline_model_parallel_size)
    split = [base] * pipeline_model_parallel_size
    split[0] += remainder
    return split


def compute_qwen35_grid_sequence_parallel(
    config: MIMOParallelismConfig,
    requested_sequence_parallel: bool,
    *,
    sp_capable_modules: Iterable[str] | None = None,
) -> dict[str, bool]:
    """Per-module sequence-parallel flags for the grid path.

    Grid mode forces the global parallel state to TP=1 and global
    ``args.sequence_parallel`` to False (``ModelParallelConfig`` rejects SP
    without TP), so the requested value (``args.mimo_sequence_parallel``) is
    applied per module: ``module_sp = requested && module_tp > 1 &&
    module_sp_capable``.  ``sp_capable_modules`` defaults to empty — an explicit
    opt-in for a future capable implementation.

    The language module is not SP-capable: the grid forward does not shard
    embeddings (``QwenVLLanguageModelEmbedding`` asserts no scatter-to-SP) and
    the mRoPE freqs stay full-length, so with SP the first column-parallel qkv
    all-gathers the full sequence (2x tokens per rank) against full-length
    freqs — a ``(2S, ...)`` query times ``(S, ...)`` freqs: silent shape
    corruption.

    The vision module is likewise not SP-capable: its patch/position embeddings
    and packed-seq attention/rotary operate on the full token dimension, so SP
    causes the same all-gather mismatch.  Tensor parallelism is unaffected; the
    vision projection config hardcodes ``sequence_parallel = False``
    (``get_vision_projection_config``).

    Returns ``{module_name: sequence_parallel}`` for every module.
    """
    if sp_capable_modules is None:
        sp_capable_modules = set()
    else:
        sp_capable_modules = set(sp_capable_modules)
    return {
        name: (
            bool(requested_sequence_parallel)
            and parallelism.tensor_model_parallel_size > 1
            and name in sp_capable_modules
        )
        for name, parallelism in config.module_parallelisms.items()
    }


def qwen35_grid_data_contract(
    config: MIMOParallelismConfig,
    *,
    micro_batch_size: int,
    global_batch_size: int,
    num_microbatches: int,
) -> dict[str, int]:
    """Validate batch-size divisibility against every module DP (fail-fast).

    Every data-loading rank samples the same global micro-batch and per-module
    DP slicing happens in the forward step, so both batch sizes must be
    divisible by every module's DP and ``num_microbatches * micro_batch_size``
    must equal ``global_batch_size`` (the calculator runs with
    data_parallel_size == 1).  Returns ``{module_name: dp_size}``.
    """
    if num_microbatches * micro_batch_size != global_batch_size:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid batch contract: {num_microbatches} "
            f"microbatches * {micro_batch_size} = "
            f"{num_microbatches * micro_batch_size} != global_batch_size "
            f"({global_batch_size}). In grid mode the sampler is unsharded "
            "(data_parallel_size == 1), so num_microbatches = "
            "global_batch_size / micro_batch_size must hold exactly."
        )
    per_module_dp: dict[str, int] = {}
    for name, parallelism in config.module_parallelisms.items():
        dp_size = parallelism.data_parallel_size
        per_module_dp[name] = dp_size
        for batch_size, label in (
            (micro_batch_size, "micro_batch_size"),
            (global_batch_size, "global_batch_size"),
        ):
            if batch_size % dp_size != 0:
                raise ValueError(
                    f"Non-colocated Qwen3.5 grid batch contract: {label} "
                    f"({batch_size}) is not divisible by module '{name}' DP "
                    f"({dp_size}). Per-module DP slicing of the global "
                    "micro-batch requires divisibility (fail-fast)."
                )
    return per_module_dp
