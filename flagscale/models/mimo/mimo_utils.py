# Copyright (c) 2025, BAAI. All rights reserved.

"""Stateless pure-function helpers for colocated MIMO.

Placement rule: only stateless tensor/layout utilities and duck-typed
dispatch over the training loop's model list live here — no collectives, no
scheduling state, no Megatron dependencies.
"""

from typing import Any

import torch


def release_mimo_training_state(model) -> None:
    """Release scheduler-held training state on every MIMO chunk.

    Called by the training loop before an exit checkpoint save so the save
    has maximum GPU headroom.  Duck-typed: non-MIMO chunks simply lack
    ``release_training_state``.  Only valid at end of training — the release
    asserts no macro batch has outstanding gradients.
    """
    for model_chunk in model:
        if hasattr(model_chunk, "release_training_state"):
            model_chunk.release_training_state()


def drop_mimo_completed_macros(model) -> None:
    """Drop exhausted, gradient-free MIMO macro batches before a periodic save.

    Safe while training continues: macros with outstanding gradients are
    never dropped, and the next ``advance()`` re-establishes serving state.
    Duck-typed like ``release_mimo_training_state``.
    """
    for model_chunk in model:
        if hasattr(model_chunk, "drop_completed_macros"):
            model_chunk.drop_completed_macros()


def compute_microbatch_token_counts(
    grid_thw_list: list[torch.Tensor | None],
    merge_unit: int = 1,
) -> list[int]:
    """Return the number of visual tokens for each microbatch.

    Args:
        grid_thw_list: List of ``[N, 3]`` tensors, one per microbatch.  Each
            row is ``(t, h, w)`` for one image/video.  ``None`` entries are
            treated as microbatches with no visual data.
        merge_unit: Optional spatial/temporal merge unit used by the vision
            encoder (e.g. Qwen3-VL ``spatial_merge_unit``).  The raw token
            count is divided by this value before splitting.

    Returns:
        Token counts per microbatch as Python ints.
    """
    counts = []
    for grid_thw in grid_thw_list:
        if grid_thw is None or grid_thw.numel() == 0:
            counts.append(0)
            continue
        num_tokens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
        counts.append(int(num_tokens // merge_unit))
    return counts


def split_visual_embeds(
    main_embeds: torch.Tensor,
    aux_features: list[torch.Tensor] | None,
    token_counts: list[int],
    dim: int = 0,
) -> list[dict[str, Any]]:
    """Split a macro visual tensor into per-microbatch chunks.

    Args:
        main_embeds: Concatenated visual embeddings.
        aux_features: Optional list of auxiliary (deepstack-like) tensors with
            the same token layout as ``main_embeds``.
        token_counts: Token count for each microbatch.
        dim: Dimension along which tokens are concatenated.

    Returns:
        List of dicts with the canonical layout ``{"main", "aux"}``.
    """
    if main_embeds is None:
        return []

    embed_splits = torch.split(main_embeds, token_counts, dim=dim)
    aux_splits = None
    if aux_features is not None:
        aux_splits = [torch.split(f, token_counts, dim=dim) for f in aux_features]

    outputs = []
    for i in range(len(token_counts)):
        out: dict[str, Any] = {"main": embed_splits[i].contiguous()}
        if aux_splits is not None:
            out["aux"] = [d[i].contiguous() for d in aux_splits]
        else:
            out["aux"] = None
        outputs.append(out)
    return outputs


def concatenate_visual_grads(
    gradients: list[dict[str, Any] | None],
    key: str,
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate per-microbatch gradients back into a macro gradient.

    ``key`` names the gradient slot (canonical keys are ``"main"`` and
    ``"aux_{i}"``).  ``None`` entries (microbatches without visual data) are
    skipped.
    """
    grads = [g[key] for g in gradients if g is not None and key in g]
    if not grads:
        raise ValueError(f"no gradients found for key '{key}'")
    return torch.cat(grads, dim=dim).contiguous()
