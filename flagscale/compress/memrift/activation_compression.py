import functools
import weakref
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from flagscale.compress.memrift.async_compressor import AsyncCompressor, PlaceHolderToken
from flagscale.compress.memrift.megatron_dynamic_loader import (
    WeightPlaceholder,
    _lookup_weight_group,
)

# Diagnostics: counts of _pack invocations. Mutable lists used as box for closure access.
_pack_diag_total = [0]
_pack_diag_placeholder = [0]
_pack_diag_other = [0]


def _should_compress_activation(t: torch.Tensor, skip_storage_ptrs: set[int] | None = None) -> bool:
    if not t.is_cuda or t.numel() == 0:
        return False
    if t.dtype not in (torch.float32, torch.bfloat16):
        return False
    if t.is_leaf or (not t.requires_grad):
        return False
    if skip_storage_ptrs:
        try:
            if t.untyped_storage().data_ptr() in skip_storage_ptrs:
                return False
        except Exception:
            pass
    return True


def _unpack(tok: Any, compressor: AsyncCompressor):
    if isinstance(tok, WeightPlaceholder):
        group = tok.group_ref()
        if group is None or group.target_module is None:
            raise RuntimeError("WeightPlaceholder: MergedWeightGroup was GC'd before backward")
        # On-demand materialize at the exact moment TE's fused backward consumes
        # the weight (zero race; self-heals if a prior release freed it).
        from flagscale.compress.memrift.megatron_dynamic_loader import (
            unpack_weight_for_backward,
        )

        weight = unpack_weight_for_backward(group)
        if weight.numel() == 0:
            raise RuntimeError(
                f"WeightPlaceholder: weight could not be materialized at backward "
                f"time for {group.megatron_target} (layer {group.layer_idx})"
            )
        if tuple(weight.shape) != tok.shape or tuple(weight.stride()) != tok.stride:
            weight = weight.as_strided(tok.shape, tok.stride, 0)
        return weight
    if isinstance(tok, PlaceHolderToken):
        tok_act_async = bool(getattr(tok, "act_async", False))
        if tok_act_async and compressor.enable_async:
            if tok.decomped_data is not None and tok.CtoD_copy_evt is None:
                return tok.decomped_data
            # Fallback: if bwd_pre_hook did not schedule async decode yet, do it here.
            if tok.decomped_data is None and getattr(tok, "future", None) is not None:
                compressor.decompress_async(tok, tok.future)
                tok.future = None
            tok.ready_evt.wait()
            if tok.error is not None:
                raise RuntimeError("MemRift activation decompression failed") from tok.error
            if tok.decomped_data is None:
                raise RuntimeError("MemRift activation decompression completed without a tensor")
            tok.decomped_data.record_stream(torch.cuda.current_stream())
            tok._clear_after_recover()
        else:
            if tok.decomped_data is None:
                compressor.decompress_sync(tok)
                tok.decomped_data.record_stream(torch.cuda.current_stream())
        return tok.decomped_data
    return tok


class DecoderLayerWrapper(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        compressor: AsyncCompressor,
        use_async: bool,
        release_after_unpack: bool = True,
        skip_storage_ptrs: set[int] | None = None,
        do_empty: bool = False,
        compress_activations: bool = True,
    ):
        super().__init__()
        self.layer = layer
        self.comp = compressor
        self.use_async = use_async
        self.release_after_unpack = release_after_unpack
        self.skip_storage_ptrs = skip_storage_ptrs or set()
        self.tokens = []
        self.futures = []
        self.do_empty = do_empty
        self.compress_activations = compress_activations

        self.register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        # register_state_dict_post_hook annotates the callable; a functools.partial
        # is used because Python bound-method objects do not allow that attribute.
        self._memrift_state_dict_post_hook = functools.partial(self._state_dict_post_hook)
        self.register_state_dict_post_hook(self._memrift_state_dict_post_hook)
        self.register_full_backward_pre_hook(self._bwd_pre_hook)
        self.register_full_backward_hook(self._bwd_hook)

    def _state_dict_post_hook(
        self,
        _module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict | None,
    ) -> None:
        """Expose the wrapped layer under its original checkpoint key names."""
        layer_prefix = prefix + "layer."
        for key in list(state_dict.keys()):
            if not key.startswith(layer_prefix):
                continue
            original_key = prefix + key[len(layer_prefix) :]
            if original_key in state_dict:
                raise RuntimeError(f"MemRift checkpoint key collision: {key!r} -> {original_key!r}")
            state_dict[original_key] = state_dict.pop(key)

    def _load_state_dict_pre_hook(
        self,
        _module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict | None,
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _errors: list[Any],
    ):
        """Map checkpoint keys for the original layer into this wrapper's layer."""
        layer_prefix = prefix + "layer."
        streamed_weight_suffixes = (
            "self_attention.linear_qkv.weight",
            "self_attention.linear_proj.weight",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc2.weight",
        )
        for key in list(state_dict.keys()):
            if not key.startswith(prefix) or key.startswith(layer_prefix):
                continue
            suffix = key[len(prefix) :]
            if suffix.endswith(streamed_weight_suffixes):
                continue
            new_key = layer_prefix + suffix
            if new_key not in state_dict:
                state_dict[new_key] = state_dict[key]

    def sharded_state_dict(self, *args, **kwargs):
        """Delegate distributed-checkpoint layout to the original Megatron layer."""
        sharded_state_dict = getattr(self.layer, "sharded_state_dict", None)
        if sharded_state_dict is None:
            raise RuntimeError(
                f"Wrapped layer {type(self.layer).__name__} does not provide "
                "sharded_state_dict required by the selected checkpoint format"
            )
        # Keep the prefix passed for the wrapper itself; adding ``layer.`` here
        # would make MemRift checkpoints incompatible with ordinary Megatron.
        return sharded_state_dict(*args, **kwargs)

    def forward(self, *inp, **kw):
        self.tokens.clear()
        self.futures.clear()
        seen = {}

        def _pack(t):
            _pack_diag_total[0] += 1
            # Intercept materialized weights so autograd saves a tiny placeholder
            # instead of the full bf16 tensor — avoids pinning ~15 GB of decoder
            # weights in the saved-tensors of frozen-base LoRA training.
            if isinstance(t, torch.Tensor) and not t.requires_grad and t.is_cuda and t.numel() > 0:
                try:
                    group = _lookup_weight_group(t.data_ptr())
                except Exception:
                    group = None
                if group is not None:
                    _pack_diag_placeholder[0] += 1
                    return WeightPlaceholder(group, t.shape, tuple(t.stride()))
            _pack_diag_other[0] += 1

            if not self.compress_activations:
                return t

            if not _should_compress_activation(t, self.skip_storage_ptrs):
                return t

            # Include shape+stride in key: two views of the same storage with
            # different shapes share the same data_ptr and nbytes but must get
            # separate tokens, otherwise the wrong shape is returned on unpack.
            key = (t.data_ptr(), t.shape, tuple(t.stride()), t.storage_offset())
            if key in seen:
                tok_ref, t_ref = seen[key]
                if t_ref() is not None:
                    tok = tok_ref()
                    if tok is not None:
                        return tok

            tok = PlaceHolderToken(t.dtype, t.shape, tuple(t.stride()), t.storage_offset())
            tok.act_async = bool(self.use_async)
            seen[key] = (weakref.ref(tok), weakref.ref(t))
            if self.use_async and self.comp.enable_async:
                fut = self.comp.kickoff_async(tok, t)
                self.futures.append(fut)
                tok.fut_id = len(self.futures) - 1
                tok.future = fut
            else:
                self.comp.kickoff_sync(tok, t)
            self.tokens.append(weakref.ref(tok))
            return tok

        unpack_fn = functools.partial(_unpack, compressor=self.comp)
        with torch.autograd.graph.saved_tensors_hooks(_pack, unpack_fn):
            out = self.layer(*inp, **kw)

        if self.do_empty:
            torch.cuda.empty_cache()
        return out

    def _bwd_pre_hook(self, _mod, _grad_in):
        if not (self.use_async and self.comp.enable_async):
            return
        for tok_ptr in self.tokens[::-1]:
            tok = tok_ptr()
            if tok is None:
                continue
            fut = self.futures[tok.fut_id]
            self.comp.decompress_async(tok, fut)
            self.futures[tok.fut_id] = None
            tok.future = None

    def _bwd_hook(self, _mod, _gin, _gout):
        for tok_ptr in self.tokens:
            tok = tok_ptr()
            if tok is not None:
                del tok
        self.tokens.clear()
        self.futures.clear()


@contextmanager
def activation_compression_context(
    compressor: AsyncCompressor | None = None,
    use_async: bool = False,
    release_after_unpack: bool = True,
    zstd_level: int = 18,
    skip_storage_ptrs: set[int] | None = None,
):
    if compressor is None:
        compressor = AsyncCompressor(
            compress_workers=2,
            decode_workers=2,
            concurrency_limit=4,
            zstd_level=zstd_level,
            enable_async=False,
        )

    seen = {}

    def pack(t):
        if not t.requires_grad and t.is_cuda and t.numel() > 0:
            try:
                group = _lookup_weight_group(t.data_ptr())
            except Exception:
                group = None
            if group is not None:
                return WeightPlaceholder(group, t.shape, tuple(t.stride()))

        if not _should_compress_activation(t, skip_storage_ptrs):
            return t

        key = (t.data_ptr(), t.shape, tuple(t.stride()), t.storage_offset())
        if key in seen:
            tok_ref, t_ref = seen[key]
            if t_ref() is not None:
                tok = tok_ref()
                if tok is not None:
                    return tok

        tok = PlaceHolderToken(t.dtype, t.shape, tuple(t.stride()), t.storage_offset())
        tok.act_async = bool(use_async)
        seen[key] = (weakref.ref(tok), weakref.ref(t))
        if use_async and compressor.enable_async:
            fut = compressor.kickoff_async(tok, t)
            tok.fut_id = 0
            tok.future = fut
        else:
            compressor.kickoff_sync(tok, t)
        return tok

    unpack = functools.partial(_unpack, compressor=compressor)
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield compressor
