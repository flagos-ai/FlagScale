# Copyright (c) 2025, BAAI. All rights reserved.

"""Generic colocated MIMO model wrapper.

``ColocatedMIMOModel`` owns all model-agnostic orchestration for colocated
MIMO training: the microbatch scheduler lifecycle, intra-TP microbatch
slicing, macro-batch output exchange, and the delayed ViT backward skeleton.
Model adapters (e.g. Qwen3.5) subclass it, build the two modules, and
implement the small hook surface below; everything else is generic.

Vision output entries use the canonical layout ``{"main": Tensor | None,
"aux": list[Tensor] | None}``: ``main`` is the embedding injected into the
language model, ``aux`` holds optional deepstack-like auxiliary features.
"""

from typing import Any

import torch
import torch.distributed as dist

from megatron.core.transformer import MegatronModule

from .mimo_bridge import exchange_macro_outputs, get_my_microbatch_range
from .mimo_scheduler import MIMOMicrobatchScheduler
from .mimo_utils import concatenate_visual_grads, split_visual_embeds
from .parallel_state_ctx import switch_parallel_state


class ColocatedMIMOModel(MegatronModule):
    """Generic colocated MIMO wrapper with heterogeneous module parallelism.

    The subclass builds ``self.vision_model`` / ``self.language_model`` after
    ``super().__init__()`` (each under its own parallel context) and implements
    the ``_*`` hook methods.  All scheduling, slicing, exchange, and backward
    orchestration lives here.
    """

    def __init__(
        self,
        config,
        pg_collections: dict[str, object],
        vit_batch_factor: int,
        use_fp32_grad_cache: bool = False,
    ) -> None:
        super().__init__(config=config)

        self.pg_collections = pg_collections
        self.vision_pg = pg_collections["vision"]
        self.language_pg = pg_collections["language"]

        # Assigned by the subclass after __init__ (built under each module's
        # own parallel context).
        self.vision_model = None
        self.language_model = None

        # Set by ``freeze``: a frozen ViT forwards under no_grad and skips the
        # delayed backward entirely.
        self._vision_frozen = False

        # Validated once at the training entry by validate_mimo_config (vbf > 1).
        self.vit_batch_factor = vit_batch_factor
        self.scheduler = MIMOMicrobatchScheduler(
            vit_batch_factor=self.vit_batch_factor,
            vision_forward_fn=self._vision_forward_fn,
            vision_backward_fn=self._vision_backward_fn,
            use_fp32_grad_cache=use_fp32_grad_cache,
        )

    # ------------------------------------------------------------------
    # Model adapter interface: the only model-specific surface.
    # ------------------------------------------------------------------
    def _count_vision_tokens(self, batches: list[dict[str, Any]]) -> list[int] | None:
        """Return per-microbatch visual token counts over the FULL macro batch.

        ``None`` means no microbatch carries visual data (early-out).  Counts
        must already account for the vision encoder's merge unit.
        """
        raise NotImplementedError

    def _drop_vision_data(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Drop heavy vision inputs from a microbatch this rank does not own.

        Only called for microbatches outside this rank's slice of the macro
        batch (see ``get_my_microbatch_range``).  Must preserve everything
        needed by ``_count_vision_tokens`` (grid metadata) and by the language
        model.  Default is a no-op.
        """
        return batch

    def _extract_vision_inputs(self, my_batches: list[dict[str, Any]]):
        """Concat this rank's microbatch slice into ``_run_vision`` inputs.

        ``None`` means this slice carries no visual data.
        """
        raise NotImplementedError

    def _run_vision(self, vision_inputs):
        """Run one ViT forward over this rank's slice.

        Called under the vision parallel context.  Returns
        ``(main_embeds [T, H], aux_features | None)``; may return
        ``(None, None)`` when the slice has no tokens.
        """
        raise NotImplementedError

    def _num_aux_features(self) -> int:
        """Number of aux tensors per output entry (0 when unsupported).

        Must be correct even when this rank's slice is empty — receive-buffer
        allocation for non-owned microbatches depends on it.
        """
        return 0

    def _embed_hidden_size(self) -> int:
        """Hidden size of the injected embeddings (receive-buffer allocation)."""
        return self.config.hidden_size

    def _vision_projection_module(self):
        """Vision projection module for ``freeze`` (None when not separable)."""
        return None

    # ------------------------------------------------------------------
    # Scheduler orchestration (generic).
    # ------------------------------------------------------------------
    def next_microbatch(self, data_iterator, get_batch_fn):
        """Return the next LLM microbatch and its vision output from the scheduler.

        Assembles a new ViT macro batch when the current one is exhausted.  All
        scheduler orchestration lives behind this method so that
        ``forward_step`` never touches the scheduler directly.
        """
        if self.scheduler.need_new_macro_batch():
            get_batch_fn = self._dedup_get_batch(get_batch_fn)
            self.scheduler.prepare_macro_batch(get_batch_fn, data_iterator, self)
        _, batch, vision_output = self.scheduler.advance()
        return batch, vision_output

    def _dedup_get_batch(self, get_batch_fn):
        """Wrap ``get_batch_fn`` to drop vision inputs this rank does not own.

        The TP-group broadcast in ``get_batch`` delivers every microbatch's
        images to all TP peers, but only the owner rank computes on them (see
        ``get_my_microbatch_range``).  Ranks without a vision module (language
        PP stages beyond the first) never compute on any of them.  Dropping
        the non-owned copies at pull time keeps them from being held for the
        lifetime of the macro batch.
        """
        lo, hi = get_my_microbatch_range(self.language_pg, self.vit_batch_factor)
        pull_idx = 0

        def get_batch_dedup(data_iterator, model):
            nonlocal pull_idx
            batch = get_batch_fn(data_iterator, model)
            if self.vision_model is None or not lo <= pull_idx < hi:
                batch = self._drop_vision_data(batch)
            pull_idx += 1
            return batch

        return get_batch_dedup

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules."""
        self._vision_frozen = freeze_vision_model and self.vision_model is not None
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        projection = self._vision_projection_module()
        if freeze_vision_projection and projection is not None:
            modules.append(projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Scheduler callbacks: run ViT on a macro batch and back-propagate later.
    # ------------------------------------------------------------------
    def _vision_forward_fn(self, batches: list[dict[str, Any]], macro) -> list[dict[str, Any]]:
        """Run one ViT forward over this rank's slice of the macro batch.

        ``batches`` is a list of dictionaries returned by the training loop's
        ``get_batch``.  This rank extracts and runs only its own microbatch
        slice (see ``get_my_microbatch_range``), splits the output back into
        per-microbatch chunks, and exchanges chunks inside the language TP
        group so every rank holds the full macro batch's outputs (its own
        slice computed, the rest received from their owner ranks).  The ViT
        outputs to back-propagate later are stashed on ``macro.ctx``.
        """
        if self.vision_model is None:
            return [{"main": None, "aux": None} for _ in batches]

        vision_tp_size = dist.get_world_size(self.vision_pg.tp)
        assert vision_tp_size == 1, (
            f"MIMO vision backward currently supports vision TP=1 only; got {vision_tp_size}."
        )

        token_counts = self._count_vision_tokens(batches)
        if token_counts is None:
            return [{"main": None, "aux": None} for _ in batches]

        lo, hi = get_my_microbatch_range(self.language_pg, len(batches))
        my_inputs = self._extract_vision_inputs(batches[lo:hi])
        if my_inputs is not None:
            with switch_parallel_state(self.vision_pg):
                if self._vision_frozen:
                    # A frozen ViT is a pure feature extractor: build no graph.
                    with torch.no_grad():
                        macro_main, macro_aux = self._run_vision(my_inputs)
                else:
                    macro_main, macro_aux = self._run_vision(my_inputs)
        else:
            macro_main, macro_aux = None, None

        macro.ctx = (macro_main, macro_aux)

        # Entries outside this rank's slice are empty receive buffers that
        # the exchange below fills from their owner ranks.
        hidden_size = self._embed_hidden_size()
        dtype = macro_main.dtype if macro_main is not None else self.config.params_dtype
        device = torch.cuda.current_device()
        aux_levels = len(macro_aux) if macro_aux is not None else self._num_aux_features()

        def _empty_entry(n_tokens: int) -> dict[str, Any]:
            return {
                "main": torch.empty(n_tokens, hidden_size, dtype=dtype, device=device),
                "aux": [
                    torch.empty(n_tokens, hidden_size, dtype=dtype, device=device)
                    for _ in range(aux_levels)
                ]
                or None,
            }

        if macro_main is not None:
            my_outputs = split_visual_embeds(macro_main, macro_aux, token_counts[lo:hi], dim=0)
        else:
            # This rank's slice has no visual data; zero-token entries keep the
            # exchange below collective-consistent with the other slices.
            my_outputs = [_empty_entry(0) for _ in range(lo, hi)]

        entries = []
        my_idx = 0
        for i in range(len(batches)):
            if lo <= i < hi:
                entries.append(my_outputs[my_idx])
                my_idx += 1
            else:
                entries.append(_empty_entry(token_counts[i]))

        # A frozen ViT skips the requires_grad marking: no hooks are
        # registered, so the exhausted macro batch is dropped silently and the
        # delayed ViT backward never runs.
        exchange_macro_outputs(
            entries,
            self.language_pg,
            self.vit_batch_factor,
            mark_requires_grad=not self._vision_frozen,
        )
        return entries

    def _vision_backward_fn(self, macro) -> None:
        """Run ViT backward after all microbatch gradients are collected.

        Each rank's captured gradient for a microbatch is already the complete
        dL/d(main): the LM's TP boundary collectives (all-reduce, or all-gather
        for sequence parallel) aggregate input gradients before they reach the
        embedding injection point, so every language TP peer holds an identical
        copy.  No reduce/broadcast is needed — each rank simply backwards
        the gradients of its own microbatch slice through its own ViT forward,
        and vision DDP then averages parameter gradients over disjoint slices
        (equivalent to averaging over the full global batch).
        """
        if self.vision_model is None:
            return

        gradients = macro.gradients
        macro_main, macro_aux = macro.ctx
        if macro_main is None or not macro_main.requires_grad:
            # No graph through the ViT outputs (frozen ViT): release the macro
            # so an un-backwarded batch does not pin the ViT graph.
            macro.ctx = None
            return

        assert dist.get_world_size(self.vision_pg.tp) == 1, (
            "MIMO vision backward currently supports vision TP=1 only; got "
            f"{dist.get_world_size(self.vision_pg.tp)}."
        )

        lo, hi = get_my_microbatch_range(self.language_pg, len(gradients))
        my_gradients = gradients[lo:hi]

        def _assemble_slice_grad(key: str) -> torch.Tensor:
            # Microbatches without visual data contribute ``None`` gradient
            # dicts; skip them so the concatenated gradient matches this
            # rank's slice of the macro output.
            if not any(g is not None and key in g for g in my_gradients):
                return None
            return concatenate_visual_grads(my_gradients, key=key, dim=0)

        main_grad = _assemble_slice_grad("main")
        if main_grad is None:
            # No gradients for this slice; release the graph so the macro does
            # not pin the ViT activations.
            macro.ctx = None
            return

        # Cast gradients back to the ViT output dtype (no-op when the fp32 grad cache is off).
        target_dtype = macro_main.dtype

        def _to_vit_dtype(g):
            return g.to(target_dtype) if g is not None else None

        grads = [_to_vit_dtype(main_grad)]
        if macro_aux is not None:
            for i in range(len(macro_aux)):
                grads.append(_to_vit_dtype(_assemble_slice_grad(f"aux_{i}")))

        targets = [macro_main] + (macro_aux or [])
        with switch_parallel_state(self.vision_pg):
            torch.autograd.backward(targets, grads)

        macro.ctx = None

    # ------------------------------------------------------------------
    # Forward helpers for the subclass (generic).
    # ------------------------------------------------------------------
    def _register_vision_output_hooks(self, vision_output: dict[str, Any]):
        """Detach a served vision output and register gradient hooks.

        Returns ``(main, aux)`` tensors safe to feed into the language model;
        their gradients are captured by the scheduler for the delayed ViT
        backward instead of propagating into the ViT graph.
        """
        main = self.scheduler.register_visual_grad_hook(vision_output["main"], "main")
        aux = vision_output["aux"]
        if aux is not None:
            aux = [
                self.scheduler.register_visual_grad_hook(f, f"aux_{i}") for i, f in enumerate(aux)
            ]
        return main, aux

    # ------------------------------------------------------------------
    # Exit-time cleanup.
    # ------------------------------------------------------------------
    def release_training_state(self) -> None:
        """Release scheduler-held training state ahead of an exit checkpoint save.

        Delegates to the scheduler, which asserts every macro batch is fully
        consumed and gradient-free before dropping it.
        """
        self.scheduler.release()

    def drop_completed_macros(self) -> None:
        """Drop fully consumed, gradient-free macro batches (pre-save cleanup)."""
        self.scheduler.drop_completed_macros()
