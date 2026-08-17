# Copyright (c) 2026, BAAI. All rights reserved.

"""Qwen3.5 non-colocated grid configuration contract (FlagScale-native).

This module defines the *training-time* contract of the non-colocated grid
path on top of the dependency-free layout layer in
:mod:`flagscale.models.mimo.bridge.parallelism`:

- A fail-fast validator (:func:`validate_qwen35_grid_config`) that rejects
  anything outside this stage's capability boundary with an explicit error.
- A builder (:func:`build_qwen35_grid_config_from_args`) that turns the
  repeatable ``--mimo-module-specs`` string into a finalized, validated
  :class:`MIMOParallelismConfig`.

Capability boundary of this stage — any layout satisfying ALL of the
following predicates is accepted; the validator checks mechanism support,
not a list of previously tested combinations:

1. Exactly two modules: the images (vision) module and the language
   module (the Qwen3.5 grid adapter's scope this stage).
2. The generic invariants of :meth:`MIMOParallelismConfig.finalize`:
   NON_COLOCATED exact tiling (no gaps / no overlaps / full-world
   coverage), TP sizes are powers of two, DP sizes pairwise divisible,
   modality-side EP=ETP=1.
3. Dense scope: both modules use CP=1, EP=1, ETP=1.
4. The vision module is not pipelined (PP=1).
5. Vision DP <= language DP (variable visual tokens fan out from the
   encoder to the language DP replicas; fan-in of variable per-sample
   visual token counts is not supported by the MCore bridge).
6. ``num_layers`` >= language PP (at least one layer per pipeline stage;
   non-divisible counts are handled by MCore's uneven pipeline
   allocation, which the grid provider encodes via
   ``num_layers_in_first/last_pipeline_stage``).
7. ``num_mtp_layers`` == 0 (the grid path has no MTP wiring).

Sequence parallelism is conservatively disabled for BOTH modules in this
grid path (:func:`compute_qwen35_grid_sequence_parallel`): the grid language
forward cannot shard the embeddings (``QwenVLLanguageModelEmbedding``
asserts no scatter-to-SP) and the mRoPE freqs stay full-length, so an
SP-enabled TP2 language module would all-gather the full sequence at the
first column-parallel qkv (dim0 2x tokens) against full-length freqs - the
4096-vs-2048 shape corruption.  The vision encoder has the same packed-seq
limitation.  Requested global SP therefore resolves to per-module False for
every accepted layout; tensor parallelism itself is unaffected.

The module keeps the pure-stdlib dependency rule of
``mimo_parallelism_config``: it imports only that module plus stdlib, so the
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
#: ``qwen35_grid_mimo_model.VISION_MODALITY_NAME``).
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
    2. The layout resolves to :class:`MIMOLayout.NON_COLOCATED` and the
       module rank ranges tile ``[0, world_size)`` exactly (no
       gaps/overlaps, full-world coverage).  ``finalize`` also enforces the
       generic invariants: TP powers of two, pairwise-divisible DP sizes,
       dense modality modules.
    3. CP == 1, EP == 1 and ETP == 1 for every module (dense stage scope).
    4. The vision module is not pipelined (PP == 1).
    5. Vision DP <= language DP (variable visual tokens require fan-out).
    6. ``num_layers`` (when given) is at least the language PP (one layer per
       pipeline stage).  Divisibility is NOT required: MCore's uneven pipeline
       allocation (set by the grid provider) gives the first stage
       ``base + remainder`` layers and the last stage ``base`` (32 layers ->
       PP3 12/10/10, PP6 7/5/5/5/5/5).
    7. ``num_mtp_layers`` (when given) is 0: the grid path has no MTP wiring.

    Args:
        config: The MIMO parallelism config (``finalize`` is called
            internally, so a non-finalized config is accepted too).
        world_size: Distributed world size.
        images_module_name: Name of the modality module in the config.
        num_layers: Language module layer count (``config.num_layers``);
            ``None`` skips the PP-divisibility check.
        num_mtp_layers: Language MTP layer count (``args.mtp_num_layers``);
            ``None`` skips the MTP check.

    Raises:
        ValueError: On any violated constraint, with an explicit message.
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

    # Dense stage scope: CP / EP / ETP must be 1 everywhere.
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

    # Language layer-count bound.  MCore's uneven pipeline allocation
    # (``num_layers_in_first_pipeline_stage`` / ``num_layers_in_last_pipeline_stage``)
    # gives the first stage ``base + remainder`` layers and the last stage
    # ``base`` (32 layers -> PP3 12/10/10, PP6 7/5/5/5/5/5); the grid provider
    # sets those fields.  Only the trivial bound holds: at least one layer per
    # pipeline stage (``pp <= num_layers``).
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

    # MTP: the grid path has no MTP wiring (the MCore MimoModel is built from
    # the language spec directly; no MultiTokenPredictionBlock support).
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

    Args:
        module_specs: Repeatable spec string, e.g.
            ``"images=tp=2,dp=1; language=tp=1,pp=1,dp=6,rank_offset=2"``.
        world_size: Distributed world size.
        num_layers: Language module layer count; threaded into the validator
            for the PP-divisibility check (see :func:`validate_qwen35_grid_config`).
        num_mtp_layers: Language MTP layer count; threaded into the validator
            for the MTP fail-fast check.

    Returns:
        The finalized, validated config (see :func:`validate_qwen35_grid_config`).

    Raises:
        ValueError: On parse or validation failure (fail-fast).
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

    Mirrors the allocation the grid provider encodes into the language
    transformer config (``num_layers_in_first_pipeline_stage`` /
    ``num_layers_in_last_pipeline_stage``): ``base = num_layers // pp``,
    ``remainder = num_layers % pp``; the first stage gets ``base + remainder``
    layers and every other stage (including the last) gets ``base``.  The
    middle stages split evenly by construction, which is exactly what MCore's
    uneven-pipeline machinery requires.

    Examples: 32 layers with PP3 -> [12, 10, 10]; with PP6 -> [7, 5, 5, 5, 5, 5].

    Args:
        num_layers: Total language decoder layer count.
        pipeline_model_parallel_size: Language pipeline depth.

    Returns:
        Per-stage layer counts (one entry per PP stage, first stage first).
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

    Grid mode forces the *global* parallel state to TP=1 and the global
    ``args.sequence_parallel`` to False (``ModelParallelConfig`` rejects
    sequence parallelism without tensor parallelism), so the user's requested
    value - preserved in the grid-specific ``args.mimo_sequence_parallel`` -
    must be applied per module:

        ``module_sequence_parallel = requested && module_tp > 1 && module_sp_capable``

    ``sp_capable_modules`` names the modules whose implementation supports
    sequence parallelism.  The default is **no module**: in this grid path
    both implementations are SP-incapable, so any requested global SP
    resolves to per-module False for every accepted layout (language TP2
    included) while tensor parallelism itself is untouched.

    The language module is not SP-capable in the non-colocated grid for two
    coupled reasons:

    - The grid language forward does not shard the embeddings: the Qwen3.5
      language module's ``QwenVLLanguageModelEmbedding`` asserts no
      scatter-to-SP (``scatter_to_sequence_parallel == False``), so every TP
      rank feeds the FULL sequence into the transformer stack.
    - The mRoPE freqs stay full-length: ``Qwen35LanguageRotaryEmbedding``
      builds ``emb`` of shape ``(S, bs, 1, 2*dim)`` from the full-length
      ``position_ids`` ``(3, bs, S)``.

    With SP enabled on a TP2 language module, the first column-parallel qkv
    layer would all-gather along dim 0 - the full sequence on every rank
    (2x tokens with TP2, e.g. 4096 for S=2048) - while the mRoPE freqs still
    describe S positions: ``apply_rotary_pos_emb_absolute`` then multiplies
    a ``(2S, ...)`` query against ``(S, ...)`` freqs (silent broadcast
    failure / shape corruption).  SP is therefore disabled for the language
    module too until the grid forward shards embeddings and slices freqs per
    SP rank; ``sp_capable_modules`` remains as an explicit opt-in escape
    hatch for a future capable implementation.

    The vision module is likewise not SP-capable: the Qwen3-VL vision
    encoder's patch and position embeddings produce full-sequence
    activations and its packed-seq attention / rotary operate on the full
    token dimension (per-frame ``cu_seqlens`` summing to the total token
    count).  Enabling SP makes the first column-parallel qkv layer
    all-gather the full sequence along dim 0 (2x tokens with TP2) while the
    packed seq params still describe the full dimension - silent shape
    corruption (query dim0 6720 vs cu_seqlens sum 3360 in the reported VTP2
    failure).  The vision module therefore keeps tensor parallelism but must
    never run with sequence parallelism; the vision projection config
    hardcodes ``sequence_parallel = False`` by design
    (``get_vision_projection_config``).

    Returns:
        ``{module_name: sequence_parallel}`` for every module in the config.
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

    In the non-colocated grid path every data-loading rank samples the same
    global micro-batch and per-module DP slicing happens in the forward step,
    so ``micro_batch_size`` and ``global_batch_size`` must be divisible by
    every module's DP and ``num_microbatches * micro_batch_size`` must equal
    ``global_batch_size`` (the num-microbatches calculator runs with
    data_parallel_size == 1 in grid mode).

    Returns:
        ``{module_name: dp_size}`` for the config's modules.
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
