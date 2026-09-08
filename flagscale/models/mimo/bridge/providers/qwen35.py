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

"""Grid-based Qwen3.5 MIMO model provider on ``megatron.core.models.mimo.MimoModel``.

Composes the FlagScale Qwen3.5 / Qwen3-VL modules (language:
``Qwen35LanguageModule``, hybrid GDN + attention with mRoPE; images:
``Qwen3VisionModel`` under ``Qwen35VisionSubmodules``, component ``"images"``,
encoder ``"qwen3_vit"``) into the grid-aware ``MimoModel``.  The rank role
(COLOCATED vs NON_COLOCATED, driving selective encoder-only / language-only
construction) is derived by ``MimoModel`` from ``module_to_grid_map``.
Deepstack auxiliary features are captured on the vision submodule and either
injected into the language forward (colocated) or rejected at build time
(non-colocated: the bridge has a single tensor channel per modality).  This
module must not import ``flagscale.models.mimo`` (the colocated scheduler
package) or ``megatron.bridge``; the colocated ``Qwen35MIMOModel`` is left
untouched.
"""

from typing import Any

import torch

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.config.role import (
    MIMO_LANGUAGE_MODULE_KEY,
    ModuleLayout,
)
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel
from flagscale.models.megatron.qwen35.language_model import Qwen35LanguageModule
from flagscale.models.megatron.qwen35.rope import get_rope_index
from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig
from flagscale.models.mimo.bridge.contracts import (
    SBH_DIM_MAPPING,
    GridCommunicatorContract,
    register_grid_communicator_contract,
)

# ---------------------------------------------------------------------------
# Component names.
# ---------------------------------------------------------------------------

VISION_MODALITY_NAME = "images"

VISION_ENCODER_NAME = "qwen3_vit"

#: MIMO language component name (fixed by MIMO: "language").
LANGUAGE_MODULE_NAME = MIMO_LANGUAGE_MODULE_KEY

#: Grid communicator contract for the Qwen3.5 images -> language topology
#: (images is a source module, language a sink).  The patch-packed vision
#: encoder emits flat ``[tokens, H]`` tensors (ndim 2, fan-in/out on dim 0);
#: the language module emits SBH hidden states (ndim 3).
QWEN35_GRID_COMMUNICATOR_CONTRACT = GridCommunicatorContract(
    topology={VISION_MODALITY_NAME: [LANGUAGE_MODULE_NAME], LANGUAGE_MODULE_NAME: []},
    dim_mapping=SBH_DIM_MAPPING,
    module_output_ndim={VISION_MODALITY_NAME: 2, LANGUAGE_MODULE_NAME: 3},
)


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
        (mirrors the colocated ``Qwen35MIMOModel`` split logic).
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
                # Runtime mirror of the config-time guard in
                # validate_qwen35_grid_config.
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

    ``mimo_infra`` (duck-typed ``bridge.infra.MIMOInfra``: ``module_to_grid_map``
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


# Import-time registration: build_grid_multimodule_communicator resolves the
# contract from bridge.contracts, so importing this provider wires the grid path.
register_grid_communicator_contract("qwen35", QWEN35_GRID_COMMUNICATOR_CONTRACT)
