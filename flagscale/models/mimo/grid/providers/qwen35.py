# Copyright (c) 2026, BAAI. All rights reserved.
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

"""Grid-based Qwen3.5 MIMO model provider."""

from typing import Any

import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.config.role import (
    MIMO_LANGUAGE_MODULE_KEY,
    ModuleLayout,
)
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel
from flagscale.models.megatron.qwen35.language_model import Qwen35LanguageModule
from flagscale.models.megatron.qwen35.layer_specs import get_qwen35_language_model_spec
from flagscale.models.megatron.qwen35.rope import get_rope_index
from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig
from flagscale.models.mimo.grid.contracts import (
    SBH_DIM_MAPPING,
    GridCommunicatorContract,
    register_grid_communicator_contract,
)
from flagscale.models.mimo.grid.data import (
    ModuleDataRole,
    drop_modality_inputs,
    slice_batch_for_module_dp,
)
from flagscale.models.mimo.grid.infra import build_mimo_infra, set_per_module_random_seed
from flagscale.models.mimo.grid.parallelism import (
    LANGUAGE_MODULE_NAME,
    MIMOLayout,
    MIMOParallelismConfig,
    ModuleParallelismConfig,
    compute_pipeline_layer_split,
    describe_grid_modules,
    parse_module_parallelisms,
    resolve_module_sequence_parallel,
    validate_grid_batch_divisibility,
)
from flagscale.models.mimo.grid.runtime import validate_no_stub_ranks
from flagscale.models.mimo.grid.training import (
    GridTrainingState,
    build_grid_multimodule_communicator,
    finalize_grid_training_state,
    register_grid_batch_preparer,
)

# ---------------------------------------------------------------------------
# Component names and capability constants.
# ---------------------------------------------------------------------------

VISION_MODALITY_NAME = "images"

VISION_ENCODER_NAME = "qwen3_vit"

#: Modules whose implementation supports sequence parallelism in the grid
#: path.  Empty: the language module cannot use SP because the grid forward
#: does not shard the embeddings (``QwenVLLanguageModelEmbedding`` asserts no
#: scatter-to-SP) and the mRoPE freqs stay full-length, so with SP the first
#: column-parallel qkv all-gathers the full sequence (2x tokens per rank)
#: against full-length freqs — a ``(2S, ...)`` query times ``(S, ...)`` freqs:
#: silent shape corruption.  The Qwen3-VL vision encoder is likewise not
#: SP-capable: its patch/position embeddings and packed-seq attention/rotary
#: operate on the full token dimension, so SP causes the same all-gather
#: mismatch.  Tensor parallelism is unaffected; the vision projection config
#: hardcodes ``sequence_parallel = False`` (``get_vision_projection_config``).
QWEN35_SP_CAPABLE_MODULES: frozenset[str] = frozenset()

#: Grid communicator contract for the Qwen3.5 images -> language topology
#: (images is a source module, language a sink).  The patch-packed vision
#: encoder emits flat ``[tokens, H]`` tensors (ndim 2, fan-in/out on dim 0);
#: the language module emits SBH hidden states (ndim 3).
QWEN35_GRID_COMMUNICATOR_CONTRACT = GridCommunicatorContract(
    topology={VISION_MODALITY_NAME: [LANGUAGE_MODULE_NAME], LANGUAGE_MODULE_NAME: []},
    dim_mapping=SBH_DIM_MAPPING,
    module_output_ndim={VISION_MODALITY_NAME: 2, LANGUAGE_MODULE_NAME: 3},
)


def _require_dense(module_name: str, cfg: ModuleParallelismConfig) -> None:
    """Fail fast on expert/context parallelism the grid path cannot handle.

    Modality-module density (EP = ETP = 1) is already enforced generically by
    ``MIMOParallelismConfig.finalize``; this adds CP for every module and
    expert parallelism on the language module.
    """
    if cfg.context_parallel_size > 1:
        raise ValueError(
            f"Non-colocated Qwen3.5 grid: module '{module_name}' uses "
            f"context_parallel_size={cfg.context_parallel_size}; CP > 1 is not "
            "supported in this stage (fail-fast)."
        )
    if module_name != LANGUAGE_MODULE_NAME:
        return
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


def _validate_config(
    config: MIMOParallelismConfig,
    world_size: int,
    *,
    images_module_name: str = VISION_MODALITY_NAME,
    num_layers: int | None = None,
    num_mtp_layers: int | None = None,
) -> None:
    """Validate a MIMO parallelism config for the non-colocated Qwen3.5 grid path.

    Fail-fast predicate checks (in order):
    1. Exactly two modules: the images module and the language module.
    2. NON_COLOCATED layout whose module rank ranges tile ``[0, world_size)``
       exactly; ``finalize`` enforces the generic invariants (TP powers of two,
       pairwise-divisible DP sizes, dense modality modules).
    3. CP == 1 for every module; EP == 1 and ETP == 1 for the language module.
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
    # split is MCore's, computed by
    # parallelism.compute_pipeline_layer_split.
    if num_layers is not None:
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}.")
        pp = language.pipeline_model_parallel_size
        if num_layers < pp:
            raise ValueError(
                "Non-colocated Qwen3.5 grid: language num_layers must be at "
                "least the language pipeline_model_parallel_size: "
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
    ``ValueError`` on parse or validation failure; see :func:`_validate_config`
    for the checked contract.
    """
    module_parallelisms = parse_module_parallelisms(module_specs)
    config = MIMOParallelismConfig(
        module_parallelisms=module_parallelisms, layout=MIMOLayout.NON_COLOCATED
    )
    _validate_config(
        config,
        world_size,
        num_layers=num_layers,
        num_mtp_layers=num_mtp_layers,
    )
    return config


class Qwen35VisionSubmodules(VisionModalitySubmodules):
    """Qwen3.5 vision modality submodule for ``MimoModel`` (single encoder
    ``qwen3_vit`` under the ``"images"`` component).

    The ViT's internal multimodal projector already maps the merged patch
    embeddings to the language hidden size, so no ``input_projections`` are
    needed.  :meth:`encode` stashes the Qwen3-VL deepstack auxiliary feature
    lists on ``last_deepstack_features`` (captured, not dropped) for the owning
    model to inject into the language forward (colocated) or reject at build
    time (non-colocated: the bridge has no auxiliary channel).
    """

    def __init__(self, *args, **kwargs) -> None:
        # Config is threaded by the spec builder so per-module DDP wrapping
        # (setup_grid_mimo_ddp) can read module.config on the images submodule.
        self.config = kwargs.pop("config", None)
        super().__init__(*args, **kwargs)
        self.last_deepstack_features: list | None = None

    def encode(self, encoders_data_batch: dict) -> list:
        if not encoders_data_batch:
            return []

        embeddings = []
        deepstack_features: list = []
        for name, encoder in self.encoders.items():
            if name not in encoders_data_batch:
                raise ValueError(f"No inputs found for encoder '{name}'")

            encoder_inputs = encoders_data_batch[name]
            encoder_outputs = encoder(**encoder_inputs)
            # Qwen3VisionModel returns (embeddings, deepstack_feature_lists).
            if (
                isinstance(encoder_outputs, tuple)
                and encoder_outputs
                and torch.is_tensor(encoder_outputs[0])
            ):
                if len(encoder_outputs) > 1 and isinstance(encoder_outputs[1], list):
                    deepstack_features = encoder_outputs[1]
                encoder_outputs = encoder_outputs[0]

            if encoder_outputs.ndim == 3:
                encoder_outputs = encoder_outputs.reshape(-1, encoder_outputs.size(-1))
            elif encoder_outputs.ndim != 2:
                raise ValueError(
                    f"Encoder '{name}' output shape {encoder_outputs.shape} is not supported. "
                    f"Expected 3D (b,s,h) or 2D (b*s,h) tensor, got {encoder_outputs.ndim}D"
                )

            embeddings.append(encoder_outputs)

        self.last_deepstack_features = deepstack_features or None
        return embeddings


def compute_grid_visual_split_sizes(
    grid_thw: torch.Tensor | None,
    output_size: int,
    spatial_merge_size: int,
) -> list[int] | None:
    """Per-sample visual-token counts for bridge fan-out, derived from ``grid_thw``.

    Each ``grid_thw`` row is one image ``(t, h, w)`` in patch units; after
    spatial merging an image contributes ``t * h * w / spatial_merge_size**2``
    tokens, emitted in image order.  Assumes one image per language sample
    (``image_thw_grids`` row i <-> sample i) — the per-image counts ARE the
    per-sample counts the bridge metadata contract requires.

    Returns ``None`` when the counts are uniform (the bridge's uniform
    ``tensor_split`` fallback is then exact) or when there is no visual data.
    Raises ``ValueError`` when the counts do not reconcile with the encoder
    output size or a grid is not divisible by the merge unit: silently falling
    back to a uniform split would corrupt variable-resolution fan-out.
    """
    if grid_thw is None or not torch.is_tensor(grid_thw) or grid_thw.numel() == 0:
        return None
    if (
        isinstance(spatial_merge_size, bool)
        or not isinstance(spatial_merge_size, int)
        or (spatial_merge_size < 1)
    ):
        raise ValueError(
            f"spatial_merge_size must be a positive integer, got {spatial_merge_size!r}."
        )
    merge_unit = spatial_merge_size * spatial_merge_size
    sizes = []
    for patches in grid_thw.prod(dim=-1).tolist():
        if patches % merge_unit != 0:
            raise ValueError(
                "Qwen3.5 grid: image grid with t*h*w patches "
                f"({patches}) is not divisible by the spatial merge unit "
                f"({merge_unit}, spatial_merge_size={spatial_merge_size}); "
                "the per-image visual token counts cannot be derived from "
                "grid_thw (fail-fast)."
            )
        sizes.append(patches // merge_unit)
    if sum(sizes) != int(output_size):
        raise ValueError(
            "Qwen3.5 grid: per-sample visual token counts derived from "
            f"grid_thw sum to {sum(sizes)} but the encoder output has "
            f"{int(output_size)} tokens; the bridge fan-out metadata would "
            "misalign with the encoder output. Refusing to silently fall back "
            "to a uniform split (fail-fast)."
        )
    if len(set(sizes)) <= 1:
        # Uniform counts: the bridge's uniform tensor_split fallback produces the
        # same chunks (micro_batch is divisible by every module's DP).
        return None
    return sizes


class Qwen35GridMIMOModel(MimoModel):
    """Qwen3.5 MIMO model on ``MimoModel``: adds mRoPE position-index computation
    and deepstack visual-feature injection; module construction, rank-role
    handling and forward dispatch live in the parent.

    ``forward`` accepts the colocated batch masks (``image_input_mask`` /
    ``video_input_mask`` / ``video_start_index``) and derives the deepstack
    ``visual_pos_masks`` from them.
    """

    def __init__(
        self,
        mimo_config: MimoModelConfig,
        cp_group=None,
        tp_group=None,
    ) -> None:
        super().__init__(mimo_config, cp_group=cp_group, tp_group=tp_group)

        # Non-colocated: the MCore bridge carries one tensor per modality and
        # cannot transport the deepstack auxiliary feature lists.
        if self.role.mode is ModuleLayout.NON_COLOCATED:
            vision_config = None
            images_spec = mimo_config.modality_submodules_spec.get(VISION_MODALITY_NAME)
            if images_spec is not None and images_spec.submodules:
                encoder_spec = images_spec.submodules.get("encoders", {}).get(VISION_ENCODER_NAME)
                if encoder_spec is not None and encoder_spec.params is not None:
                    vision_config = encoder_spec.params.get("transformer_config")
            if vision_config is not None and getattr(
                vision_config, "deepstack_visual_indexes", None
            ):
                raise ValueError(
                    "Qwen3.5 non-colocated grid: the vision encoder has "
                    f"deepstack_visual_indexes="
                    f"{vision_config.deepstack_visual_indexes}, but the MIMO "
                    "bridge has a single tensor channel per modality and cannot "
                    "transport the deepstack auxiliary features to the language "
                    "ranks. Disable deepstack (deepstack_visual_indexes=[]) for "
                    "non-colocated training (fail-fast)."
                )

        # Per-forward batch masks (set by forward(); consumed by
        # _forward_all_modules for the deepstack visual position masks).
        self._image_input_mask = None
        self._video_input_mask = None
        self._video_start_index = 0

        # Per-forward encoder grid metadata (set by forward(); consumed by
        # _attach_modality_split_sizes for the bridge fan-out split sizes).
        self._current_grid_thw: torch.Tensor | None = None

    def get_rope_index(
        self,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ) -> tuple[Any, Any]:
        """Compute mRoPE position indices for Qwen3.5 (see ``qwen35.rope``)."""
        return get_rope_index(
            spatial_merge_size=self.config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        modality_inputs: dict[str, dict[str, Any]] | None = None,
        packing_kwargs: dict | None = None,
        image_input_mask: torch.Tensor | None = None,
        video_input_mask: torch.Tensor | None = None,
        video_start_index: int = 0,
    ):
        """Qwen3.5 forward: the extra masks feed the deepstack
        ``visual_pos_masks`` (no-op when deepstack is disabled).
        """
        self._image_input_mask = image_input_mask
        self._video_input_mask = video_input_mask
        self._video_start_index = int(video_start_index or 0)

        # Split sizes derive from grid_thw, not special-token counts (see
        # compute_grid_visual_split_sizes).
        self._current_grid_thw = None
        if modality_inputs is not None:
            images_inputs = modality_inputs.get(VISION_MODALITY_NAME)
            if images_inputs is not None:
                encoder_inputs = images_inputs.get(VISION_ENCODER_NAME)
                if encoder_inputs is not None:
                    self._current_grid_thw = encoder_inputs.get("grid_thw")

        if (
            self.role.mode is ModuleLayout.NON_COLOCATED
            and self.role.has_language_module
            and video_input_mask is not None
            and bool(video_input_mask.any().item())
        ):
            raise NotImplementedError(
                "Qwen3.5 non-colocated grid: video inputs are not supported in "
                "this stage; the images modality covers image data only "
                "(fail-fast)."
            )

        return super().forward(
            input_ids,
            position_ids,
            attention_mask,
            loss_mask,
            labels,
            modality_inputs,
            packing_kwargs,
        )

    def _deepstack_visual_pos_masks(self, num_visual_embeds: int | None) -> torch.Tensor | None:
        """Deepstack ``visual_pos_masks`` from the stashed batch masks
        (mirrors the colocated ``Qwen35ColocatedMIMOModel`` split logic).
        """
        image_input_mask = self._image_input_mask
        video_input_mask = self._video_input_mask
        if image_input_mask is None and video_input_mask is None:
            return None
        if image_input_mask is not None:
            image_input_mask = image_input_mask.T
        if video_input_mask is not None:
            video_input_mask = video_input_mask.T
        video_start_index = self._video_start_index
        if num_visual_embeds is None:
            return None
        if video_start_index == 0:
            return video_input_mask
        if video_start_index == num_visual_embeds:
            return image_input_mask
        if 0 < video_start_index < num_visual_embeds:
            return torch.logical_or(image_input_mask, video_input_mask)
        raise ValueError(
            f"Expect video token start index in range [0, {num_visual_embeds}], "
            f"but got {video_start_index}"
        )

    def _attach_modality_split_sizes(
        self,
        output: torch.Tensor,
        input_ids: torch.Tensor | None,
        encoder_name: str,
    ) -> None:
        """Attach per-sample fan-out split sizes derived from ``grid_thw``
        (not special-token counts); see :func:`compute_grid_visual_split_sizes`.
        """
        if (
            encoder_name == VISION_MODALITY_NAME
            and self._current_grid_thw is not None
            and output.ndim == 2
        ):
            split_sizes = compute_grid_visual_split_sizes(
                self._current_grid_thw,
                int(output.size(0)),
                self.config.spatial_merge_size,
            )
            if split_sizes is not None:
                # Runtime mirror of the config-time guard in _validate_config.
                if (
                    self.role.mode is ModuleLayout.NON_COLOCATED
                    and self.mimo_config.module_to_grid_map
                ):
                    encoder_grid = self.mimo_config.module_to_grid_map[encoder_name]
                    language_grid = self.mimo_config.module_to_grid_map[MIMO_LANGUAGE_MODULE_KEY]
                    if hasattr(encoder_grid, "shape") and "dp" in encoder_grid.dim_names:
                        encoder_dp = encoder_grid.shape[encoder_grid.dim_names.index("dp")]
                        language_dp = language_grid.shape[language_grid.dim_names.index("dp")]
                        if encoder_dp > language_dp:
                            raise ValueError(
                                "Qwen3.5 grid: bridge fan-out with non-uniform "
                                "per-sample visual token counts requires encoder "
                                f"DP <= language DP (got encoder DP={encoder_dp}, "
                                f"language DP={language_dp}); fan-in of variable "
                                "visual token counts is not supported (fail-fast)."
                            )
                output._mimo_bridge_split_sizes = split_sizes
                return
        super()._attach_modality_split_sizes(output, input_ids, encoder_name)

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze the locally present modules (``Qwen35Model.freeze`` semantics).

        The grid model is rank-selective: each flag applies only when the module
        is present on this rank (encoder-only / language-only ranks).
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        images_submodule = None
        if (
            self.modality_submodules is not None
            and VISION_MODALITY_NAME in self.modality_submodules
        ):
            images_submodule = self.modality_submodules[VISION_MODALITY_NAME]
        if freeze_vision_model and images_submodule is not None:
            modules.append(images_submodule)
        if freeze_vision_projection and images_submodule is not None:
            encoders = getattr(images_submodule, "module", images_submodule).encoders
            encoder = encoders[VISION_ENCODER_NAME] if VISION_ENCODER_NAME in encoders else None
            if encoder is not None:
                projection = getattr(getattr(encoder, "module", encoder), "projection", None)
                if projection is not None:
                    modules.append(projection)

        for module in modules:
            for param in getattr(module, "module", module).parameters():
                param.requires_grad = False

    def _forward_all_modules(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        loss_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        modality_inputs: dict[str, dict[str, Any]] | None,
        packing_kwargs: dict | None = None,
    ):
        """``MimoModel._forward_all_modules`` plus deepstack visual-feature
        injection into the language forward (colocated semantics).
        """
        packed_seq_params = None
        if packing_kwargs is not None:
            for key in packing_kwargs:
                if "cu_seqlens" in key and packing_kwargs[key] is not None:
                    packing_kwargs[key] = packing_kwargs[key].to(dtype=torch.int32)
            packed_seq_params = PackedSeqParams(**packing_kwargs)
            packed_seq_params.qkv_format = "thd"

        modality_embeddings = {}
        deepstack_feature_lists: list | None = None
        for modality_name, submodule in self.modality_submodules.items():
            if (
                modality_inputs
                and modality_name in modality_inputs
                and modality_inputs[modality_name] is not None
            ):
                embeddings = submodule.forward(encoder_inputs=modality_inputs[modality_name])
                if embeddings is not None:
                    modality_embeddings[modality_name] = embeddings
                # The submodule may be DDP-wrapped; unwrap for the type check.
                inner_submodule = getattr(submodule, "module", submodule)
                if isinstance(inner_submodule, Qwen35VisionSubmodules) and getattr(
                    inner_submodule, "last_deepstack_features", None
                ):
                    deepstack_feature_lists = inner_submodule.last_deepstack_features

        if self.colocated_comms:
            modality_embeddings = self._apply_colocated_comms(modality_embeddings)

        text_embeddings = self.get_text_embeddings(input_ids, position_ids, self.special_token_ids)
        modality_embeddings["text"] = text_embeddings

        combined_embeddings = self.align_embeddings_by_token_positions(
            modality_embeddings=modality_embeddings,
            input_ids=input_ids,
            special_token_ids=self.special_token_ids,
        )

        if self.partition_adapter is not None:
            combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
            combined_embeddings, labels, loss_mask, _, packed_seq_params = (
                self.partition_adapter.shard(
                    embeddings=combined_embeddings,
                    labels=labels,
                    loss_mask=loss_mask,
                    attention_mask=attention_mask,
                    packed_seq_params=packed_seq_params,
                )
            )
            if combined_embeddings is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

        # 4. Qwen3.5 deepstack: inject the auxiliary visual features.
        visual_pos_masks = None
        if deepstack_feature_lists is not None:
            primary = modality_embeddings.get(VISION_MODALITY_NAME)
            num_visual_embeds = primary.size(0) if primary is not None else None
            visual_pos_masks = self._deepstack_visual_pos_masks(num_visual_embeds)

        lm_output = self.language_model(
            # decoder_input replaces the embedding lookup (input_ids unused);
            # position_ids is still consumed by mRoPE.
            input_ids=None,
            position_ids=position_ids,
            decoder_input=combined_embeddings,
            labels=labels,
            attention_mask=None,
            packed_seq_params=packed_seq_params,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_feature_lists,
        )

        return lm_output, loss_mask


# ---------------------------------------------------------------------------
# Spec builders.
# ---------------------------------------------------------------------------


def build_qwen35_language_model_spec(
    config: Qwen35TransformerConfig,
    transformer_layer_spec: ModuleSpec,
    vocab_size: int,
    max_sequence_length: int,
    *,
    parallel_output: bool = True,
    position_embedding_type: str = "mrope",
    rotary_percent: float = 0.25,
    pre_process: bool = True,
    post_process: bool = True,
    rotary_base: int = 10000000,
    fp16_lm_cross_entropy: bool = False,
    share_embeddings_and_output_weights: bool = False,
    mtp_block_spec: ModuleSpec | None = None,
    vp_stage: int | None = None,
    pg_collection=None,
) -> ModuleSpec:
    """Build the ``MimoModelConfig`` language module spec for Qwen3.5;
    ``pg_collection`` is nullable (``MimoModel.sharded_state_dict`` injects
    ``dp_cp_group`` from it when present, global parallel-state fallback
    otherwise).
    """
    return ModuleSpec(
        module=Qwen35LanguageModule,
        params={
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": max_sequence_length,
            "parallel_output": parallel_output,
            "position_embedding_type": position_embedding_type,
            "rotary_percent": rotary_percent,
            "pre_process": pre_process,
            "post_process": post_process,
            "rotary_base": rotary_base,
            "fp16_lm_cross_entropy": fp16_lm_cross_entropy,
            "share_embeddings_and_output_weights": share_embeddings_and_output_weights,
            "rope_scaling": False,
            "mtp_block_spec": mtp_block_spec,
            "vp_stage": vp_stage,
            "pg_collection": pg_collection,
        },
    )


def build_qwen35_images_submodule_spec(
    transformer_config: TransformerConfig,
    transformer_layer_spec: ModuleSpec,
    projection_config: TransformerConfig,
    projection_layer_spec: ModuleSpec,
    *,
    projection_type: str = "mlp",
    pg_collection=None,
) -> ModuleSpec:
    """Build the ``MimoModelConfig`` images modality submodule spec for Qwen3.5.

    No ``input_projections`` (the encoder's internal projection already emits
    language-hidden-size embeddings); ``pg_collection`` is nullable and stored
    on the built submodule for checkpoint metadata injection.
    """
    encoder_spec = ModuleSpec(
        module=Qwen3VisionModel,
        params={
            "transformer_config": transformer_config,
            "transformer_layer_spec": transformer_layer_spec,
            "projection_config": projection_config,
            "projection_layer_spec": projection_layer_spec,
            "projection_type": projection_type,
            "pre_process": True,
            "post_process": True,
            # Vision TP-sharded layers must use the vision TP group, not the
            # global parallel state (which describes the language module).
            "pg_collection": pg_collection,
        },
    )
    return ModuleSpec(
        module=Qwen35VisionSubmodules,
        params={
            "config": transformer_config,
            "pg_collection": pg_collection,
        },
        submodules={"encoders": {VISION_ENCODER_NAME: encoder_spec}},
    )


# ---------------------------------------------------------------------------
# Config and provider.
# ---------------------------------------------------------------------------


def build_qwen35_mimo_config(
    language_transformer_config: Qwen35TransformerConfig,
    language_transformer_layer_spec: ModuleSpec,
    language_vocab_size: int,
    language_max_sequence_length: int,
    vision_transformer_config: TransformerConfig,
    vision_transformer_layer_spec: ModuleSpec,
    vision_projection_config: TransformerConfig,
    vision_projection_layer_spec: ModuleSpec,
    *,
    vision_projection_type: str = "mlp",
    parallel_output: bool = True,
    language_position_embedding_type: str = "mrope",
    language_rotary_percent: float = 0.25,
    pre_process: bool = True,
    post_process: bool = True,
    language_rotary_base: int = 10000000,
    fp16_lm_cross_entropy: bool = False,
    language_share_embeddings_and_output_weights: bool = False,
    mtp_block_spec: ModuleSpec | None = None,
    vp_stage: int | None = None,
    special_token_ids: dict[str, int] | None = None,
    module_to_grid_map: dict[str, HyperCommGrid] | None = None,
    pg_collection=None,
    images_pg_collection: Any | None = None,
    kv_format: str = "sbhd",
) -> MimoModelConfig:
    """Assemble the ``MimoModelConfig`` for the Qwen3.5 grid-based MIMO model.

    Args:
        special_token_ids: Per-modality special token ids for embedding
            alignment; ``None`` defaults to the language config's ``image_token_id``.
        module_to_grid_map: Prebuilt ``HyperCommGrid`` per component; ``None``
            selects the colocated layout (global parallel state, COLOCATED role).
        pg_collection: Nullable ``ProcessGroupCollection`` for the language
            module (threaded into the language spec only).
        images_pg_collection: Nullable ``ProcessGroupCollection`` for the images
            module (threaded into the images submodule spec only).  Kept separate
            from ``pg_collection`` so that on an encoder-only rank the images
            submodule gets the *vision* collection, not the (absent) language one.
        kv_format: Key-value cache format ("sbhd" or "thd").

    Grid-map keys are validated by ``MimoModelConfig.__post_init__``.
    """
    if special_token_ids is None:
        special_token_ids = {VISION_MODALITY_NAME: language_transformer_config.image_token_id}

    language_model_spec = build_qwen35_language_model_spec(
        config=language_transformer_config,
        transformer_layer_spec=language_transformer_layer_spec,
        vocab_size=language_vocab_size,
        max_sequence_length=language_max_sequence_length,
        parallel_output=parallel_output,
        position_embedding_type=language_position_embedding_type,
        rotary_percent=language_rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        rotary_base=language_rotary_base,
        fp16_lm_cross_entropy=fp16_lm_cross_entropy,
        share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
    images_submodule_spec = build_qwen35_images_submodule_spec(
        transformer_config=vision_transformer_config,
        transformer_layer_spec=vision_transformer_layer_spec,
        projection_config=vision_projection_config,
        projection_layer_spec=vision_projection_layer_spec,
        projection_type=vision_projection_type,
        pg_collection=images_pg_collection,
    )
    return MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={VISION_MODALITY_NAME: images_submodule_spec},
        special_token_ids=special_token_ids,
        module_to_grid_map=module_to_grid_map,
        kv_format=kv_format,
    )


def qwen35_grid_mimo_model_provider(
    language_transformer_config: Qwen35TransformerConfig,
    language_transformer_layer_spec: ModuleSpec,
    language_vocab_size: int,
    language_max_sequence_length: int,
    vision_transformer_config: TransformerConfig,
    vision_transformer_layer_spec: ModuleSpec,
    vision_projection_config: TransformerConfig,
    vision_projection_layer_spec: ModuleSpec,
    *,
    vision_projection_type: str = "mlp",
    parallel_output: bool = True,
    language_position_embedding_type: str = "mrope",
    language_rotary_percent: float = 0.25,
    pre_process: bool = True,
    post_process: bool = True,
    language_rotary_base: int = 10000000,
    fp16_lm_cross_entropy: bool = False,
    language_share_embeddings_and_output_weights: bool = False,
    mtp_block_spec: ModuleSpec | None = None,
    vp_stage: int | None = None,
    special_token_ids: dict[str, int] | None = None,
    module_to_grid_map: dict[str, HyperCommGrid] | None = None,
    pg_collection=None,
    images_pg_collection: Any | None = None,
    mimo_infra: Any | None = None,
    mimo_grid_state: Any | None = None,
    kv_format: str = "sbhd",
) -> Qwen35GridMIMOModel:
    """Build the Qwen3.5 grid-based MIMO model (see :func:`build_qwen35_mimo_config`).

    ``mimo_infra`` (duck-typed ``grid.infra.MIMOInfra``: ``module_to_grid_map``
    and ``module_to_pg_collection``) supplies the grid map and the per-module
    ``pg_collection`` values; the explicit ``module_to_grid_map`` /
    ``pg_collection`` / ``images_pg_collection`` arguments must then be left
    unset.  Without it, one ``pg_collection`` is threaded into both specs (the
    colocated single-collection call pattern).  ``cp_group`` / ``tp_group``
    derive from the language ``pg_collection``; ``mimo_grid_state`` is attached
    as ``model.mimo_grid_state`` for the training loop.
    """
    if mimo_infra is not None:
        if (
            module_to_grid_map is not None
            or pg_collection is not None
            or images_pg_collection is not None
        ):
            raise ValueError(
                "qwen35_grid_mimo_model_provider: when mimo_infra is given, "
                "module_to_grid_map, pg_collection and images_pg_collection "
                "must be None (the infra provides them)."
            )
        module_to_grid_map = mimo_infra.module_to_grid_map
        pg_collection = mimo_infra.module_to_pg_collection.get(LANGUAGE_MODULE_NAME)
        images_pg_collection = mimo_infra.module_to_pg_collection.get(VISION_MODALITY_NAME)
    elif images_pg_collection is None:
        images_pg_collection = pg_collection

    mimo_config = build_qwen35_mimo_config(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=language_vocab_size,
        language_max_sequence_length=language_max_sequence_length,
        vision_transformer_config=vision_transformer_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        vision_projection_type=vision_projection_type,
        parallel_output=parallel_output,
        language_position_embedding_type=language_position_embedding_type,
        language_rotary_percent=language_rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        language_rotary_base=language_rotary_base,
        fp16_lm_cross_entropy=fp16_lm_cross_entropy,
        language_share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        special_token_ids=special_token_ids,
        module_to_grid_map=module_to_grid_map,
        pg_collection=pg_collection,
        images_pg_collection=images_pg_collection,
        kv_format=kv_format,
    )

    cp_group = pg_collection.cp if pg_collection is not None else None
    tp_group = pg_collection.tp if pg_collection is not None else None
    model = Qwen35GridMIMOModel(mimo_config, cp_group=cp_group, tp_group=tp_group)
    if mimo_grid_state is not None:
        model.mimo_grid_state = mimo_grid_state
    return model


# ---------------------------------------------------------------------------
# Module-role batch preparation (registered grid batch preparer).
# ---------------------------------------------------------------------------


def _build_language_forward_kwargs(
    batch: dict[str, Any],
    *,
    dp_rank: int,
    dp_size: int,
    pp_rank: int,
    pp_size: int,
) -> dict[str, Any]:
    """Assemble the exact ``Qwen35GridMIMOModel.forward`` kwargs for a language rank.

    Language-only ranks consume encoder outputs from the MIMO bridge, so raw
    modality inputs are dropped BEFORE the module-local DP slice: they are
    patch-packed (``imgs`` dim 0 is the total patch count across the batch's
    images, not the sample count), so the generic sample-DP slicer must never
    see them.  Only sample-aligned keys (``tokens``, ``labels``, ``loss_mask``, ``position_ids``, the input
    masks, ...) are sliced.

    The returned dict holds exactly the model's accepted kwargs:
    ``input_ids`` only on the first PP stage (embedding lives there),
    ``labels`` / ``loss_mask`` only on the last (loss lives there),
    ``modality_inputs`` / ``packing_kwargs`` always ``None`` in grid mode,
    and the image/video masks plus ``video_start_index`` for the deepstack /
    video fail-fast handling.  Keys the model does NOT accept (``imgs``,
    ``image_thw_grids``, ...) are never emitted, even nulled - the grid
    forward step splats the dict into the model call.
    """
    data_batch = drop_modality_inputs(batch)
    data_batch = slice_batch_for_module_dp(data_batch, dp_rank, dp_size)
    role = ModuleDataRole(module_name=MIMO_LANGUAGE_MODULE_KEY, pp_rank=pp_rank, pp_size=pp_size)
    image_input_mask = data_batch.get("image_input_mask")
    video_start_index = (
        int(image_input_mask.sum().item()) if torch.is_tensor(image_input_mask) else 0
    )
    return {
        "input_ids": data_batch.get("tokens") if role.is_first_stage else None,
        "position_ids": data_batch.get("position_ids"),
        "attention_mask": data_batch.get("attention_mask"),
        "loss_mask": data_batch.get("loss_mask") if role.is_last_stage else None,
        "labels": data_batch.get("labels") if role.is_last_stage else None,
        "modality_inputs": None,
        "packing_kwargs": None,
        "image_input_mask": image_input_mask,
        "video_input_mask": data_batch.get("video_input_mask"),
        "video_start_index": video_start_index,
    }


def _build_vision_forward_kwargs(
    batch: dict[str, Any],
    *,
    dp_rank: int,
    dp_size: int,
) -> dict[str, Any]:
    """Assemble the exact ``Qwen35GridMIMOModel.forward`` kwargs for a vision rank.

    Raw modality tensors are patch-packed - ``imgs`` / ``videos`` dim 0 is
    the TOTAL patch count across the batch's images and the grid rows are
    the images - so they cannot be sample-sliced; they are packed into the
    ``{hidden_states, grid_thw}`` dict form that ``slice_batch_for_module_dp``
    routes to the joint per-image slicer, while every other (sample-aligned)
    key is sliced by sample as usual.  Videos are not supported by the grid
    path yet - fail fast instead of producing a silent embedding-count
    mismatch.

    Returns exactly the keys ``Qwen35GridMIMOModel.forward`` accepts, with
    ``modality_inputs`` carrying the DP-sliced ``vision_data`` / ``grid_thw``
    for the ``qwen3_vit`` encoder.
    """
    image_data = batch.get("imgs")
    video_data = batch.get("videos")
    image_grid = batch.get("image_thw_grids")
    video_grid = batch.get("video_thw_grids")
    data_tensors = [t for t in (image_data, video_data) if torch.is_tensor(t)]
    grid_tensors = [t for t in (image_grid, video_grid) if torch.is_tensor(t)]
    if not data_tensors or not grid_tensors:
        raise ValueError(
            "Qwen3.5 non-colocated grid requires tensor-valued visual data and "
            "grid metadata; use empty tensors for text-only batches."
        )
    vision_data = torch.cat(data_tensors, dim=0)
    vision_grid = torch.cat(grid_tensors, dim=0)
    video_mask = batch.get("video_input_mask")
    if video_mask is not None and bool(video_mask.any().item()):
        raise NotImplementedError(
            "Qwen3.5 non-colocated grid: video inputs are not supported in "
            "this stage; the images modality covers image data only "
            "(fail-fast)."
        )

    image_mask = batch.get("image_input_mask")
    if torch.is_tensor(image_mask):
        samples_with_images = int(image_mask.any(dim=-1).sum().item())
        num_images = int(image_grid.size(0)) if torch.is_tensor(image_grid) else 0
        if samples_with_images != num_images:
            raise ValueError(
                "Qwen3.5 non-colocated grid currently requires at most one image per "
                "sample; image grid rows must match the number of samples containing "
                f"image tokens (images={num_images}, samples={samples_with_images})."
            )

    # Remove the raw modality keys (they are patch-packed and must not be
    # touched by the sample-DP slicer) and add the packed dict form in their
    # place; the slicer routes it to the joint per-image slicing.
    sliceable = {
        key: value
        for key, value in batch.items()
        if key not in ("imgs", "videos", "image_thw_grids", "video_thw_grids")
    }
    sliceable["vision_packed"] = {"hidden_states": vision_data, "grid_thw": vision_grid}
    sliced = slice_batch_for_module_dp(sliceable, dp_rank, dp_size)
    packed = sliced.pop("vision_packed")

    modality_inputs = (
        {
            VISION_MODALITY_NAME: {
                VISION_ENCODER_NAME: {
                    "vision_data": packed["hidden_states"],
                    "grid_thw": packed["grid_thw"],
                }
            }
        }
        if packed["grid_thw"] is not None and packed["grid_thw"].numel() > 0
        else None
    )
    return {
        "input_ids": sliced.get("tokens"),
        "position_ids": None,
        "attention_mask": None,
        "loss_mask": None,
        "labels": None,
        "modality_inputs": modality_inputs,
    }


def prepare_qwen35_grid_batch(
    batch: dict[str, Any], grid_state: "GridTrainingState"
) -> dict[str, Any]:
    """Prepare the global micro-batch for this rank's grid module role.

    Every data-loading rank samples the *same* global micro-batch
    (``args.data_parallel_size == 1`` in grid mode, broadcast over the world
    TP group).  This function then:

    1. drops the raw modality inputs on language-only ranks (they consume
       encoder outputs from the MIMO bridge) and assembles the exact kwargs
       accepted by the module forward (no leftover batch keys),
    2. contiguously slices the batch for the module-local DP shard,
    3. nulls out fields the module does not consume (input_ids on non-first
       language PP stages; labels/loss_mask on non-last stages),
    4. assembles ``modality_inputs`` for vision ranks.
    """
    module_name = grid_state.active_module_name
    grid = grid_state.infra.module_to_grid_map[module_name]
    dp_size = grid.shape[grid.dim_names.index("dp")]
    pg_collection = grid_state.local_pg_collection
    dp_rank = torch.distributed.get_group_rank(pg_collection.dp, torch.distributed.get_rank())
    pp_size = pg_collection.pp.size()
    pp_rank = torch.distributed.get_group_rank(pg_collection.pp, torch.distributed.get_rank())
    role = ModuleDataRole(module_name=module_name, pp_rank=pp_rank, pp_size=pp_size)

    if role.is_language:
        # Language-only ranks (non-colocated) get encoder outputs from the
        # bridge; the kwargs builders emit exactly what the model forward
        # accepts (extra keys would make ``model(**data_batch)`` raise).
        return _build_language_forward_kwargs(
            batch,
            dp_rank=dp_rank,
            dp_size=dp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
        )
    return _build_vision_forward_kwargs(batch, dp_rank=dp_rank, dp_size=dp_size)


# ---------------------------------------------------------------------------
# Construction orchestration.
# ---------------------------------------------------------------------------


def build_qwen35_grid_mimo_model(
    args,
    *,
    language_transformer_config,
    language_transformer_layer_spec,
    vision_transformer_config,
    vision_transformer_layer_spec,
    vision_projection_config,
    vision_projection_layer_spec,
    mtp_block_spec=None,
):
    """Build the non-colocated grid Qwen3.5 MIMO model (MCore MimoModel path).

    Every rank participates in exactly one module; the language module config
    mirrors the global args (TP/PP below are the language's).  Order matters:
    config build + validation, infra/process groups, module-local config,
    uneven PP layer split, per-module sequence parallel, per-module RNG,
    language PP-rank spec rebuild, then the provider and training-state
    finalization.
    """
    # Keep package imports independent of the full megatron.training stack.
    from megatron.training.utils import print_rank_0

    if getattr(args, "mtp_num_layers", None):
        raise ValueError(
            "Qwen3.5 non-colocated grid: MTP (mtp_num_layers > 0) is not "
            "supported in this stage - the MCore MimoModel grid path has "
            "no MTP wiring. Set mtp_num_layers=0 (fail-fast)."
        )
    world_size = dist.get_world_size()
    mimo_config = build_qwen35_grid_config_from_args(
        args.mimo_module_specs,
        world_size,
        num_layers=language_transformer_config.num_layers,
        num_mtp_layers=getattr(args, "mtp_num_layers", None),
    )

    # Batch contract: the sampler is unsharded (data_parallel_size == 1)
    # and module-local DP slicing happens in the forward step.
    num_microbatches = get_num_microbatches()
    per_module_dp = validate_grid_batch_divisibility(
        mimo_config,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        num_microbatches=num_microbatches,
    )
    print_rank_0(
        f"Non-colocated grid MIMO: {describe_grid_modules(mimo_config)}; module DP: {per_module_dp}"
    )

    # Grid + nullable process groups.  Collective on every world rank.
    infra = build_mimo_infra(mimo_config.module_parallelisms)
    validate_no_stub_ranks(infra.module_to_grid_map, world_size)

    config = language_transformer_config
    language_parallelism = mimo_config.get_parallelism(LANGUAGE_MODULE_NAME)
    vision_parallelism = mimo_config.get_parallelism(VISION_MODALITY_NAME)

    # The language transformer config must describe the *language* module
    # (per-stage layer slicing via pipeline_model_parallel_size), while the
    # global parallel state is initialized with TP=1/PP=1 in grid mode.
    config.tensor_model_parallel_size = language_parallelism.tensor_model_parallel_size
    config.pipeline_model_parallel_size = language_parallelism.pipeline_model_parallel_size
    config.data_parallel_size = language_parallelism.data_parallel_size
    config.context_parallel_size = 1
    config.expert_model_parallel_size = 1

    # Uneven pipeline layer allocation via MCore's explicit first/last stage
    # counts (e.g. 32 layers -> PP3 12/10/10); even splits leave the fields
    # None (default MCore even split).
    split = compute_pipeline_layer_split(config.num_layers, config.pipeline_model_parallel_size)
    if len(set(split)) > 1:
        config.num_layers_in_first_pipeline_stage = split[0]
        config.num_layers_in_last_pipeline_stage = split[-1]

    # The vision encoder TP-shards by its own module TP (config field +
    # per-module pg_collection), not by the global parallel state; vision is
    # never pipelined (the ViT asserts post_process).
    vision_config = vision_transformer_config
    vision_config.tensor_model_parallel_size = vision_parallelism.tensor_model_parallel_size
    vision_config.context_parallel_size = 1
    vision_config.expert_model_parallel_size = 1
    vision_config.pipeline_model_parallel_size = 1
    vision_projection_config.tensor_model_parallel_size = (
        vision_parallelism.tensor_model_parallel_size
    )
    vision_projection_config.context_parallel_size = 1
    vision_projection_config.expert_model_parallel_size = 1

    # Sequence parallelism is a per-module property in grid mode (the global
    # state is TP=1).  ``QWEN35_SP_CAPABLE_MODULES`` is empty: neither module
    # is SP-capable in this path (no embedding sharding / full-length mRoPE
    # and packed-seq freqs), so requested SP resolves to per-module False for
    # every accepted layout; TP itself is unaffected.
    requested_sp = bool(getattr(args, "mimo_sequence_parallel", False))
    per_module_sp = resolve_module_sequence_parallel(
        mimo_config, requested_sp, sp_capable_modules=QWEN35_SP_CAPABLE_MODULES
    )
    config.sequence_parallel = per_module_sp[LANGUAGE_MODULE_NAME]
    vision_config.sequence_parallel = per_module_sp[VISION_MODALITY_NAME]
    if requested_sp and not any(per_module_sp.values()):
        print_rank_0(
            "Non-colocated grid MIMO: requested sequence parallelism is "
            "disabled for BOTH modules (the grid path cannot shard the "
            "embeddings and the mRoPE/packed-seq freqs stay full-length). "
            "Both modules keep tensor parallelism with SP disabled (fail-safe)."
        )

    # Per-module RNG: the standard path seeds by the *global* TP/PP ranks
    # (all zero in grid mode), which would initialize TP-sharded module
    # weights differently across a module's TP group.
    set_per_module_random_seed(args, infra)

    # Per-rank language PP stage flags: the language module is built per
    # stage (embedding on the first stage, output layer on the last).  A
    # ``None`` language pg collection means this rank is outside the
    # language grid (encoder-only rank: no language module is built,
    # role-driven) - never call ``get_group_rank`` on a nullable collection.
    language_grid = infra.module_to_grid_map.get(LANGUAGE_MODULE_NAME)
    language_pg = infra.module_to_pg_collection.get(LANGUAGE_MODULE_NAME)
    language_pp_rank = 0
    if language_pg is not None:
        pre_process = is_pp_first_stage(language_pg.pp)
        post_process = is_pp_last_stage(language_pg.pp)
        if language_grid is not None and language_grid.is_current_rank_in_grid():
            language_pp_rank = dist.get_group_rank(language_pg.pp, dist.get_rank())
    else:
        pre_process = post_process = True

    # The language layer spec must describe *this* rank's PP stage.  It was
    # built from the pre-mutation config (global TP=1/PP=1), so both stages
    # would slice all ``config.num_layers`` layer specs and save overlapping
    # checkpoint keys.  Rebuild with the language module's explicit PP rank
    # (offset/count from the language config's pipeline fields); the built
    # layers keep global numbering, so checkpoint keys stay non-overlapping.
    language_layer_spec = get_qwen35_language_model_spec(config, pp_rank=language_pp_rank)

    model = qwen35_grid_mimo_model_provider(
        language_transformer_config=config,
        language_transformer_layer_spec=language_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        vision_projection_type="mlp",
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        language_share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        mtp_block_spec=mtp_block_spec,
        mimo_infra=infra,
    )

    # Training-lifecycle state: schedule PGs / communicator + the local
    # module's collection for logging/checkpoint reductions.
    grid_state = GridTrainingState(
        infra=infra,
        parallelism_config=mimo_config,
        world_size=world_size,
    )
    finalize_grid_training_state(grid_state)
    model.mimo_grid_state = grid_state
    model.pg_collection = grid_state.local_pg_collection
    build_grid_multimodule_communicator(grid_state, model)
    print_rank_0(
        f"Rank {dist.get_rank()}: grid module "
        f"'{grid_state.active_module_name}' ("
        f"tp={dist.get_world_size(grid_state.local_pg_collection.tp)}, "
        f"dp={dist.get_world_size(grid_state.local_pg_collection.dp)}, "
        f"pp={dist.get_world_size(grid_state.local_pg_collection.pp)})"
    )
    return model


# Import-time registrations: build_grid_multimodule_communicator resolves the
# communicator contract, and prepare_grid_batch the batch preparer, from the
# grid registries — importing this provider wires the grid path.
register_grid_communicator_contract("qwen35", QWEN35_GRID_COMMUNICATOR_CONTRACT)
register_grid_batch_preparer("qwen35", prepare_qwen35_grid_batch)
