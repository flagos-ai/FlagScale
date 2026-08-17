# Copyright (c) 2025, BAAI. All rights reserved.

"""Microbatch scheduler for colocated MIMO deployment.

When the vision module has a larger effective batch size than the language
module, one ViT forward can feed multiple LLM microbatches.  This scheduler hides
the macro-batch buffering and gradient routing from the outer Megatron loop.

Design notes
------------
* The scheduler is model-agnostic: it stores opaque batch dictionaries and
  relies on the owning model to provide ``vision_forward_fn`` and
  ``vision_backward_fn`` callbacks.
* Split visual tensors are detached before being injected into the language
  model, keeping the ViT computation graph alive for the delayed backward.
* Gradients accumulated on the detached splits are concatenated and handed
  back to ``vision_backward_fn`` once every microbatch in the macro batch has
  produced a gradient.
* Several macro batches may be live at once under a pipeline schedule: the
  pipeline interleaves forward and backward of different microbatches, so a
  newer macro batch can be prepared while an older one still has gradients
  outstanding.  Each macro batch therefore owns its gradient state and its
  model-side context (``ctx``), and gradient hooks route back to the macro
  batch they were registered from.  Macro batches complete in FIFO order
  (microbatch backward order matches forward order in the supported
  schedules).
"""

from collections import deque
from collections.abc import Callable
from typing import Any

import torch


class _MacroBatch:
    """Per-macro-batch state (kept alive until its gradients complete)."""

    __slots__ = ("batches", "vision_outputs", "gradients", "expected_keys", "next_idx", "ctx")

    def __init__(self, batches, vision_outputs):
        self.batches = batches
        self.vision_outputs = vision_outputs
        self.gradients: list[dict[str, Any] | None] = [None] * len(batches)
        self.expected_keys: list[set] = [set() for _ in range(len(batches))]
        self.next_idx: int = 0
        # Model-side context (e.g. the macro ViT outputs to back-propagate).
        self.ctx: Any = None

    def exhausted(self) -> bool:
        return self.next_idx >= len(self.batches)


class MIMOMicrobatchScheduler:
    """Manage ViT macro batches and route gradients back to ViT.

    Args:
        vit_batch_factor: Number of LLM microbatches served by one ViT forward.
        vision_forward_fn: Callable ``(batches, macro) -> list[dict]`` running
            one ViT forward over the macro batch; may stash model-side state
            on ``macro.ctx``.  Each returned output is an opaque dictionary.
        vision_backward_fn: Callable ``(macro) -> None`` running ViT backward
            for that macro batch (its gradients are in ``macro.gradients``).
            Invoked automatically once every microbatch in the macro batch has
            produced a gradient.
    """

    def __init__(
        self,
        vit_batch_factor: int,
        vision_forward_fn: Callable[[list[dict[str, Any]], _MacroBatch], list[dict[str, Any]]],
        vision_backward_fn: Callable[[_MacroBatch], None],
        use_fp32_grad_cache: bool = False,
    ) -> None:
        if vit_batch_factor < 1:
            raise ValueError(f"vit_batch_factor must be >= 1, got {vit_batch_factor}")
        self.vit_batch_factor = vit_batch_factor
        self.vision_forward_fn = vision_forward_fn
        self.vision_backward_fn = vision_backward_fn
        self.use_fp32_grad_cache = use_fp32_grad_cache

        self._macros: deque[_MacroBatch] = deque()
        self._serving: _MacroBatch | None = None

    # ------------------------------------------------------------------
    # Macro-batch lifecycle.
    # ------------------------------------------------------------------
    def need_new_macro_batch(self) -> bool:
        """Return True when a new ViT macro batch must be assembled."""
        return not any(not m.exhausted() for m in self._macros)

    def prepare_macro_batch(
        self,
        get_batch_fn: Callable,
        data_iterator: Any,
        model: Any,
    ) -> None:
        """Collect ``vit_batch_factor`` microbatches and run ViT forward.

        The caller (usually ``forward_step``) is responsible for providing a
        ``get_batch_fn`` that yields one LLM microbatch per call.  Older macro
        batches with outstanding gradients stay alive alongside the new one.
        """
        assert self.need_new_macro_batch(), "scheduler still has unconsumed microbatches"

        batches: list[dict[str, Any]] = []
        for _ in range(self.vit_batch_factor):
            batches.append(get_batch_fn(data_iterator, model))

        macro = _MacroBatch(batches, None)
        outputs = self.vision_forward_fn(batches, macro)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(batches):
            raise RuntimeError(
                f"vision_forward_fn must return one output dict per microbatch, "
                f"got {len(outputs)} outputs for {len(batches)} batches"
            )
        macro.vision_outputs = list(outputs)
        self._macros.append(macro)

    def drop_completed_macros(self) -> None:
        """Drop exhausted macro batches that registered no gradient hooks.

        Mirrors the leading-drop in ``advance``; callers may invoke it between
        iterations (e.g. before a periodic checkpoint save) to release the
        trailing no-hook macro's buffers early.  Macros with outstanding
        gradients are never dropped.
        """
        while (
            self._macros and self._macros[0].exhausted() and not any(self._macros[0].expected_keys)
        ):
            macro = self._macros.popleft()
            # The serving pointer must not outlive its macro (same invariant
            # as in ``_collect_grad``).
            if self._serving is macro:
                self._serving = None
            # Break the ViT graph reference held by un-backwarded macros.
            macro.ctx = None

    def advance(self) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Return the next microbatch index, batch dict, and vision output dict."""
        # Drop exhausted macro batches that will never produce gradients
        # (nothing was registered on them, e.g. ranks without a vision module).
        self.drop_completed_macros()

        if self.need_new_macro_batch():
            raise RuntimeError(
                "scheduler has no active macro batch; call prepare_macro_batch first"
            )
        macro = next(m for m in self._macros if not m.exhausted())
        idx = macro.next_idx
        macro.next_idx += 1
        self._serving = macro
        return idx, macro.batches[idx], macro.vision_outputs[idx]

    # ------------------------------------------------------------------
    # Gradient collection.
    # ------------------------------------------------------------------
    def register_visual_grad_hook(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """Detach ``tensor`` and register a hook that collects its gradient.

        ``key`` names the gradient slot (canonical keys are ``"main"`` and
        ``"aux_{i}"``, assigned by the model wrapper).  The returned tensor is
        safe to feed into the language model: its gradient will be captured by
        the scheduler instead of propagating back into the ViT graph.
        """
        if tensor is None:
            return None
        macro = self._serving
        assert macro is not None, (
            "register_visual_grad_hook called before scheduler.advance(); no active microbatch"
        )
        idx = macro.next_idx - 1
        assert idx < len(macro.expected_keys), (
            f"microbatch index {idx} out of range [0, {len(macro.expected_keys)})"
        )
        assert key not in macro.expected_keys[idx], (
            f"gradient hook for key '{key}' already registered for microbatch {idx}"
        )
        tensor = tensor.detach().requires_grad_(tensor.requires_grad)
        if tensor.requires_grad:
            macro.expected_keys[idx].add(key)
            tensor.register_hook(lambda grad: self._collect_grad(grad, macro, idx, key))
        return tensor

    def _collect_grad(self, grad: torch.Tensor, macro: _MacroBatch, idx: int, key: str) -> None:
        """Store one gradient slice and trigger ViT backward when complete."""
        if macro.gradients[idx] is None:
            macro.gradients[idx] = {}
        # Accumulate gradients in fp32 when requested to reduce bf16 round-off.
        if self.use_fp32_grad_cache and grad is not None:
            grad = grad.float()
        macro.gradients[idx][key] = grad

        if self._macro_ready(macro):
            self.vision_backward_fn(macro)
            # Macro batches complete in FIFO order.
            assert macro is self._macros[0], "macro batch completed out of order"
            self._macros.popleft()
            # The serving pointer must not outlive its macro: it would pin the
            # batch tensors and gradients until the next advance.  It is re-set
            # by the next advance() before any hook registration.
            if self._serving is macro:
                self._serving = None

    def _macro_ready(self, macro: _MacroBatch) -> bool:
        """Check whether every registered key for every served microbatch is present.

        ViT backward must not run until the whole macro batch has been forwarded,
        otherwise unserved microbatches look "ready" because no keys have been
        registered for them yet.
        """
        if not macro.exhausted():
            return False
        for i, expected in enumerate(macro.expected_keys):
            if not expected:
                # No tensor was registered for this microbatch; treat as ready.
                continue
            grads = macro.gradients[i]
            if grads is None or set(grads.keys()) != expected:
                return False
        return True

    # ------------------------------------------------------------------
    # Exit-time cleanup.
    # ------------------------------------------------------------------
    def release(self) -> None:
        """Drop scheduler-held training state ahead of an exit checkpoint save.

        The serving pointer and any residual macro batch keep multi-GiB GPU
        buffers (batch tensors, ViT outputs, gradients) alive through the
        save.  At exit time every macro batch is fully consumed and
        gradient-free, so this normally only drops the serving pointer; the
        assertions below guard against ever dropping a macro that still has
        unconsumed microbatches or outstanding gradients.
        """
        for macro in self._macros:
            assert macro.exhausted(), (
                "cannot release training state while a macro batch still has "
                "unconsumed microbatches (gradients would be lost)"
            )
            assert not any(macro.expected_keys), (
                "cannot release training state while gradients are still outstanding"
            )
            # Break the ViT graph reference held by un-backwarded macros.
            macro.ctx = None
        self._macros.clear()
        self._serving = None
