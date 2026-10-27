# Copyright (c) 2025, BAAI. All rights reserved.

"""Unified Qwen3.5 MIMO model entry: thin ``mimo_layout`` dispatcher.

:func:`build_qwen35_mimo_model` is the single construction entry for both
MIMO layouts.  It only dispatches — colocated to
``flagscale.models.mimo.colocated.providers.qwen35`` (which owns
``Qwen35ColocatedMIMOModel``), grid to
``flagscale.models.mimo.grid.providers.qwen35`` (which owns the full grid
construction); unknown layouts fail fast.  Imports stay lazy so a MIMO run
never pays the other layout's (MCore MIMO stack) import; the providers never
import this module.
"""


def build_qwen35_mimo_model(
    args,
    *,
    mimo_layout: str,
    language_transformer_config,
    language_transformer_layer_spec,
    vision_transformer_config,
    vision_transformer_layer_spec,
    vision_projection_config,
    vision_projection_layer_spec,
    mtp_block_spec,
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
):
    """Build the Qwen3.5 MIMO model for ``mimo_layout`` (unified entry).

    ``grid`` builds the non-colocated MCore MimoModel (rank-selective role);
    ``colocated`` delegates to the colocated provider.  The role flags
    (``pre_process``/``post_process``/``add_encoder``/``add_decoder``) apply
    to ``colocated`` only: ``grid`` ignores them and derives the role from
    the language module's PP group.  Unknown layouts fail fast (argparse
    ``choices`` normally guarantees the value set).
    """
    if mimo_layout == "grid":
        from flagscale.models.mimo.grid.providers.qwen35 import (
            build_qwen35_grid_mimo_model,
        )

        return build_qwen35_grid_mimo_model(
            args,
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            mtp_block_spec=mtp_block_spec,
        )
    if mimo_layout == "colocated":
        from flagscale.models.mimo.colocated.providers.qwen35 import (
            build_qwen35_colocated_mimo_model,
        )

        return build_qwen35_colocated_mimo_model(
            args,
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=vision_transformer_config,
            vision_transformer_layer_spec=vision_transformer_layer_spec,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_layer_spec,
            mtp_block_spec=mtp_block_spec,
            pre_process=pre_process,
            post_process=post_process,
            add_encoder=add_encoder,
            add_decoder=add_decoder,
        )
    raise ValueError(
        f"Unsupported --mimo-layout {mimo_layout!r}: expected 'colocated' or 'grid'"
    )
