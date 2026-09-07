"""
Megatron Dynamic Weight Loader for MemRift.

Handles:
- HF to Megatron weight name mapping (TP=1 only in v1)
- Merged weight assembly (qkv -> linear_qkv, gate+up -> linear_fc1)
- cp -> (target_module, target_attr) binding
- Forward/backward hooks for dynamic load/release
- Original weight release to save GPU memory
- TE compatibility (pre-decompress first K layers)

Note: This version only supports TP=1 (tensor_model_parallel_size=1).
"""

import concurrent.futures as fut
import json
import os
import struct
import threading
import time
import weakref
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


# Thread-local decompressor
_tls = threading.local()
_PTR2GROUP: dict[int, "MergedWeightGroup"] = {}


class WeightPlaceholder:
    """Lightweight placeholder for autograd-saved materialized weights.

    Holds a weakref to the MergedWeightGroup. The actual bf16 tensor is NOT held,
    allowing _clear_param to free it after forward. During backward, the
    corresponding layer's bwd_pre re-materializes the weight, and _unpack
    retrieves the fresh bf16 via group.target_module.weight.data.
    """

    __slots__ = ("group_ref", "shape", "stride")

    def __init__(self, group: "MergedWeightGroup", shape, stride):
        self.group_ref = weakref.ref(group)
        self.shape = tuple(shape)
        self.stride = tuple(stride)


def _lookup_weight_group(data_ptr: int) -> Optional["MergedWeightGroup"]:
    return _PTR2GROUP.get(int(data_ptr))


# Prefer stream/event-based dependency by default to maximize overlap.
# Set MEMRIFT_DEEP_ASYNC=0 to fall back to host-side synchronize behavior.
_DEEP_ASYNC = os.environ.get("MEMRIFT_DEEP_ASYNC", "1") == "1"


def _trace(msg: str):
    if os.environ.get("MEMRIFT_TRACE", "0") != "1":
        return
    print(f"[MemRiftTrace][weights] {msg}", flush=True)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _get_dctx():
    if not hasattr(_tls, "dctx"):
        _tls.dctx = zstd.ZstdDecompressor()
    return _tls.dctx


def _c_contiguous_strides(shape):
    """Calculate C-contiguous strides for a shape."""
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 2, -1, -1):
        running *= shape[i + 1]
        strides[i] = running
    return tuple(strides)


class CompressedParam(nn.Parameter):
    """
    A Parameter subclass that holds compressed weight data.

    The actual bf16/fp32 data is materialized on-demand during forward/backward,
    then released after use to save GPU memory.

    Key attributes for param binding:
        target_module: The nn.Module whose weight this cp represents
        target_attr: The attribute name ('weight' or 'bias')
        hf_name: Original HF parameter name
        megatron_target: Megatron target (e.g., 'self_attention.linear_qkv')
        merge_key: For merged weights, the component key (e.g., 'q', 'k', 'v')
    """

    def __new__(cls, orig_shape, sm_cpu, exp_mv, dtype, device):
        # Create a dummy empty tensor on the target CUDA device
        dummy = torch.empty(0, dtype=dtype, device=device)
        return super().__new__(cls, dummy, requires_grad=False)

    def __init__(self, orig_shape, sm_cpu, exp_mv, dtype, device):
        super().__init__()
        self.orig_shape = tuple(orig_shape)
        self._sm_gpu = sm_cpu.to(device)  # sign matrix lives on GPU permanently
        self.exp_mv = exp_mv
        self._dtype = dtype
        self._device = device  # target CUDA device
        self._bf16 = None
        self._ready_event = threading.Event()
        self._CtoD_evt = None
        self._exp_host = None

        # Param binding (set by loader)
        self.target_module: nn.Module | None = None
        self.target_attr: str = "weight"
        self.hf_name: str = ""
        self.megatron_target: str = ""
        self.merge_key: str | None = None  # 'q'/'k'/'v' or 'gate'/'up'
        self.layer_idx: int = -1

        # Async prefetch
        self._prefetch_future = None

    def materialize(self, sync: bool = True):
        """Decompress and materialize the full tensor."""
        if self._bf16 is not None:
            return self._bf16

        # Consume async prefetch if present (demo-style: wait for async, never decompress on main thread).
        # When prefetch was submitted, we always wait for it so decompress stays in thread pool and GPU
        # can stay busy with merge on h2d_stream while main thread blocks in result().
        if self._prefetch_future is not None:
            pref = self._prefetch_future
            self._prefetch_future = None
            try:
                if not pref.done():
                    _trace(
                        f"materialize: prefetch not ready (layer={self.layer_idx}), wait for async (demo-style)"
                    )
                pref_out = pref.result()
                if isinstance(pref_out, tuple) and len(pref_out) == 3:
                    # (bf16, evt, cpu_exp): thread exited without GPU sync.
                    # cpu_exp must stay alive until the merge DMA completes.
                    self._bf16, self._CtoD_evt, self._exp_host = pref_out
                elif isinstance(pref_out, tuple) and len(pref_out) == 2:
                    self._bf16, self._CtoD_evt = pref_out
                else:
                    self._bf16 = pref_out
                self._ready_event.set()
                _trace(f"materialize: consumed prefetched tensor (layer={self.layer_idx})")
                if sync and self._CtoD_evt is not None and not _DEEP_ASYNC:
                    # Non-deep-async: CPU-side sync; safe to free cpu_exp after this.
                    self._CtoD_evt.synchronize()
                    if self._exp_host is not None:
                        del self._exp_host
                        self._exp_host = None
                # _DEEP_ASYNC path: _exp_host freed in cp.release() after use.
                return self._bf16
            except fut.CancelledError:
                pass  # fall through to sync path below
            except Exception as e:
                _trace(f"materialize: prefetch failed (layer={self.layer_idx}): {type(e).__name__}")
                # fall through to sync path

        # Import CUDA extension
        try:
            from flagscale.compress.memrift import ops as fs_sp

            if not fs_sp.is_available():
                raise RuntimeError(
                    "CUDA extension float_split_stride_pin not available. "
                    "Please build it first: cd flagscale/compress/memrift/csrc && pip install -e ."
                )
        except ImportError as e:
            raise RuntimeError(
                f"CUDA extension not importable: {e}. "
                "MemRift requires the CUDA extension for weight decompression."
            )

        numel = int(np.prod(self.orig_shape))

        # Decompress exponent to pinned memory
        self._exp_host = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
        dctx = _get_dctx()
        with dctx.stream_reader(memoryview(self.exp_mv)) as reader:
            view = memoryview(self._exp_host.numpy())
            nread = reader.readinto(view)
            assert nread == numel, f"decompress size mismatch: {nread} vs {numel}"

        # Merge on GPU: sm is already resident on GPU
        strides = _c_contiguous_strides(self.orig_shape)
        stream = torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self._bf16 = fs_sp.merge(
                self._exp_host,
                self._sm_gpu,
                list(self.orig_shape),
                list(strides),
                0,
                self._dtype,
                stream.cuda_stream,
            )
        ev = stream.record_event()
        self._CtoD_evt = ev
        self._ready_event.set()

        if sync and not _DEEP_ASYNC:
            ev.synchronize()

        return self._bf16

    def wait_ready(self):
        """Wait for materialization to complete."""
        self._ready_event.wait()
        if self._CtoD_evt and not _DEEP_ASYNC:
            self._CtoD_evt.synchronize()

    def release(self):
        """Release the materialized tensor to free GPU memory."""
        # Drain a dangling prefetch future so its pinned/GPU buffers are not orphaned.
        # Non-blocking: cancel if not started; free the result only if already done.
        fut_obj = self._prefetch_future
        if fut_obj is not None:
            self._prefetch_future = None
            try:
                if not fut_obj.cancel() and fut_obj.done():
                    res = fut_obj.result()
                    if isinstance(res, tuple):
                        bf16 = res[0]
                        evt = res[1] if len(res) >= 2 else None
                        # Ensure the h2d_stream merge kernel has finished writing into
                        # bf16 before returning it to the pool, else it could be
                        # recycled mid-write. res (and its cpu_exp) stays referenced
                        # until the sync completes.
                        if evt is not None:
                            evt.synchronize()
                    else:
                        bf16 = res
                    if bf16 is not None:
                        from flagscale.compress.memrift import ops as fs_sp

                        if fs_sp.is_available():
                            fs_sp.release_cuda(bf16)
            except Exception:
                pass

        if self._bf16 is None:
            return

        try:
            from flagscale.compress.memrift import ops as fs_sp

            if fs_sp.is_available():
                fs_sp.release_cuda(self._bf16)
        except:
            pass

        self._bf16 = None
        self._ready_event.clear()
        self._CtoD_evt = None

        if self._exp_host is not None:
            del self._exp_host
            self._exp_host = None

    def release_compressed(self):
        """Release compressed data (when no longer needed)."""
        self.exp_mv = None
        self._sm_gpu = None


@dataclass
class MergedWeightGroup:
    """
    Group of CompressedParams that need to be merged into one Megatron weight.

    For TP=1:
    - qkv: grouped Megatron order, e.g. q_head(s), k, v per query group
    - fc1: concat(gate, up) along dim 0
    """

    megatron_target: str  # e.g., 'self_attention.linear_qkv'
    layer_idx: int
    components: dict[str, CompressedParam] = field(default_factory=dict)
    target_module: nn.Module | None = None
    target_attr: str = "weight"
    target_shape: tuple[int, ...] | None = None
    num_attention_heads: int | None = None
    num_query_groups: int | None = None
    hidden_size: int | None = None
    kv_channels: int | None = None
    group_query_attention: bool = False

    def is_complete(self) -> bool:
        """Check if all components are present."""
        if "linear_qkv" in self.megatron_target:
            return all(k in self.components for k in ["q", "k", "v"])
        elif "linear_fc1" in self.megatron_target:
            return all(k in self.components for k in ["gate", "up"])
        return True

    def get_merged_shape(self) -> tuple[int, ...]:
        """Get the shape of the merged weight (TP=1)."""
        if "linear_qkv" in self.megatron_target:
            # qkv: concat along dim 0
            q_shape = self.components["q"].orig_shape
            k_shape = self.components["k"].orig_shape
            v_shape = self.components["v"].orig_shape
            return (q_shape[0] + k_shape[0] + v_shape[0], q_shape[1])
        elif "linear_fc1" in self.megatron_target:
            # fc1: concat along dim 0
            gate_shape = self.components["gate"].orig_shape
            up_shape = self.components["up"].orig_shape
            return (gate_shape[0] + up_shape[0], gate_shape[1])
        else:
            # Single weight
            cp = next(iter(self.components.values()))
        return cp.orig_shape


def get_group_by_tensor_ptr(ptr: int) -> MergedWeightGroup | None:
    """Lookup merged weight group by currently materialized tensor ptr."""
    return _PTR2GROUP.get(int(ptr))


def _expand_qkv_to_target_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    target_rows: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand K/V rows when the Megatron target is dense QKV but HF stores GQA K/V."""
    if target_rows is None:
        return q, k, v

    merged_rows = int(q.shape[0] + k.shape[0] + v.shape[0])
    if merged_rows == target_rows:
        return q, k, v

    q_rows = int(q.shape[0])
    kv_rows_total = target_rows - q_rows
    if kv_rows_total <= 0 or kv_rows_total % 2 != 0:
        raise RuntimeError(
            f"QKV row mismatch cannot be adapted: target_rows={target_rows}, "
            f"q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
        )

    expect_k_rows = kv_rows_total // 2
    expect_v_rows = kv_rows_total // 2
    rk_ok = expect_k_rows % int(k.shape[0]) == 0
    rv_ok = expect_v_rows % int(v.shape[0]) == 0
    if not (rk_ok and rv_ok):
        raise RuntimeError(
            f"QKV row mismatch cannot be adapted: target_rows={target_rows}, "
            f"q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}"
        )

    rk = expect_k_rows // int(k.shape[0])
    rv = expect_v_rows // int(v.shape[0])
    if rk > 1:
        k = k.repeat_interleave(rk, dim=0)
    if rv > 1:
        v = v.repeat_interleave(rv, dim=0)
    return q, k, v


def _materialize_qkv_megatron_order(
    group: MergedWeightGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Assemble HF q/k/v tensors into Megatron-Core linear_qkv row order."""
    hidden_size = int(group.hidden_size or q.shape[1])
    num_attention_heads = int(group.num_attention_heads or 0)
    if num_attention_heads <= 0:
        raise RuntimeError(
            "MemRift QKV materialization needs num_attention_heads to match Megatron order"
        )

    kv_channels = int(group.kv_channels or (hidden_size // num_attention_heads))
    if kv_channels <= 0:
        raise RuntimeError(
            f"Invalid kv_channels={kv_channels} for hidden_size={hidden_size}, "
            f"num_attention_heads={num_attention_heads}"
        )

    configured_query_groups = int(group.num_query_groups or num_attention_heads)
    effective_gqa = bool(group.group_query_attention) or (
        configured_query_groups != num_attention_heads
    )
    num_query_groups = configured_query_groups if effective_gqa else num_attention_heads

    target_rows = None
    if group.target_shape is not None and len(group.target_shape) >= 1:
        target_rows = int(group.target_shape[0])
    q, k, v = _expand_qkv_to_target_rows(q, k, v, target_rows)

    try:
        q = q.reshape((num_query_groups, -1, kv_channels, hidden_size))
        k = k.reshape((num_query_groups, -1, kv_channels, hidden_size))
        v = v.reshape((num_query_groups, -1, kv_channels, hidden_size))
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to reshape QKV into Megatron grouped order: "
            f"ng={num_query_groups}, nh={num_attention_heads}, "
            f"kv_channels={kv_channels}, hidden_size={hidden_size}, "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        ) from exc

    return torch.cat([q, k, v], dim=1).reshape((-1, hidden_size))


def _materialize_group_tensor(group: MergedWeightGroup, sync: bool = True) -> torch.Tensor:
    """Materialize and assemble one merged weight group.

    Fast path: if the target param already holds a valid (non-empty) weight, return it
    directly without re-decompressing.  This prevents redundant work for layers that were
    pre-materialized by prefetch_initial_layers(): their param.data is filled, but
    cp._bf16 was already cleared after torch.cat, so a naive call to cp.materialize()
    would trigger a full re-decompress even though the weight is already on GPU.

    After torch.cat, the individual component tensors (cp._bf16) are released
    immediately so that only the merged result stays on GPU.  This avoids a 2x
    memory peak that would otherwise occur (components + merged copy).
    cp.release() is idempotent so _clear_param can still call it safely later.
    """
    # Fast path: weight already resident in param.data – skip decompression.
    if group.target_module is not None:
        _param = getattr(group.target_module, group.target_attr, None)
        if _param is not None and _param.data.numel() > 0:
            return _param.data

    if "linear_qkv" in group.megatron_target:
        q_cp = group.components.get("q")
        k_cp = group.components.get("k")
        v_cp = group.components.get("v")
        if q_cp is None or k_cp is None or v_cp is None:
            raise RuntimeError(f"Incomplete QKV group: {list(group.components.keys())}")
        q = q_cp.materialize(sync=sync)
        k = k_cp.materialize(sync=sync)
        v = v_cp.materialize(sync=sync)
        if sync and not _DEEP_ASYNC:
            q_cp.wait_ready()
            k_cp.wait_ready()
            v_cp.wait_ready()
        else:
            # DEEP_ASYNC / event-based: make compute stream wait for h2d_stream
            # before torch.cat so the merge DMA is visible on the compute stream.
            cur = torch.cuda.current_stream()
            for _cp in (q_cp, k_cp, v_cp):
                if _cp._CtoD_evt is not None:
                    cur.wait_event(_cp._CtoD_evt)
        merged = _materialize_qkv_megatron_order(group, q, k, v)
        # Release component tensors immediately to avoid 2x peak memory.
        q_cp.release()
        k_cp.release()
        v_cp.release()
        return merged

    if "linear_fc1" in group.megatron_target:
        gate_cp = group.components.get("gate")
        up_cp = group.components.get("up")
        if gate_cp is None or up_cp is None:
            raise RuntimeError(f"Incomplete FC1 group: {list(group.components.keys())}")
        gate = gate_cp.materialize(sync=sync)
        up = up_cp.materialize(sync=sync)
        if sync and not _DEEP_ASYNC:
            gate_cp.wait_ready()
            up_cp.wait_ready()
        else:
            cur = torch.cuda.current_stream()
            for _cp in (gate_cp, up_cp):
                if _cp._CtoD_evt is not None:
                    cur.wait_event(_cp._CtoD_evt)
        merged = torch.cat([gate, up], dim=0)
        gate_cp.release()
        up_cp.release()
        return merged

    cp = group.components.get("single")
    if cp is None:
        cp = next(iter(group.components.values()))
    weight = cp.materialize(sync=sync)
    if sync and not _DEEP_ASYNC:
        cp.wait_ready()
    return weight


def ensure_group_param_materialized(group: MergedWeightGroup) -> torch.Tensor | None:
    """Ensure target param data for group is present and return it."""
    if group.target_module is None:
        return None
    param = getattr(group.target_module, group.target_attr, None)
    if param is None:
        return None
    if param.data.numel() == 0:
        weight = _materialize_group_tensor(group, sync=True)
        with torch.no_grad():
            if weight.device == param.device and weight.dtype == param.dtype:
                param.data = weight
            else:
                param.data = weight.to(device=param.device, dtype=param.dtype)
        _PTR2GROUP[int(param.data_ptr())] = group
    return param.data


# Counts on-demand re-materializations triggered from _unpack during backward.
_BWD_REMATERIALIZE_COUNT = [0]

# layer_idx -> list[MergedWeightGroup] currently materialized during backward.
_BWD_MATERIALIZED: dict[int, list["MergedWeightGroup"]] = {}


def _read_bwd_release_lag() -> int:
    try:
        return max(1, int(os.environ.get("MEMRIFT_BWD_RELEASE_LAG", "1")))
    except Exception:
        return 1


# Number of most-recent backward layers to keep resident (>=1). Read once at import.
_BWD_RELEASE_LAG: int = _read_bwd_release_lag()


def _clear_param_group(group: "MergedWeightGroup") -> None:
    """Free a group's materialized weight (free-function form of loader._clear_param)."""
    if group.target_module is not None:
        param = getattr(group.target_module, group.target_attr, None)
        if param is not None and param.data.numel() > 0:
            old_ptr = int(param.data_ptr())
            with torch.no_grad():
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            _PTR2GROUP.pop(old_ptr, None)
    for cp in group.components.values():
        cp.release()


def _release_layers_above(threshold: int) -> None:
    """Release all tracked groups whose layer_idx > threshold (already backward'd)."""
    for L in [k for k in list(_BWD_MATERIALIZED) if k > threshold]:
        for g in _BWD_MATERIALIZED.pop(L, []):
            _clear_param_group(g)


def reset_bwd_tracking() -> int:
    """Clear backward tracking state; return and reset the re-materialize count."""
    _BWD_MATERIALIZED.clear()
    n = _BWD_REMATERIALIZE_COUNT[0]
    _BWD_REMATERIALIZE_COUNT[0] = 0
    return n


def unpack_weight_for_backward(group: "MergedWeightGroup") -> torch.Tensor:
    """Materialize a group's weight on demand from saved_tensors_hooks._unpack.

    Runs INSIDE the TE fused autograd Function's backward, at the exact moment
    the weight is consumed (zero race; self-heals if a prior release freed it).
    After materializing, releases already-processed layers (idx > L + lag - 1) to
    bound resident weights to ~`lag` layers during backward.

    Returns the materialized weight tensor (param.data), or an empty tensor if
    materialization failed (the caller in _unpack then raises).
    """
    if group.target_module is None:
        return torch.empty(0)
    param = getattr(group.target_module, group.target_attr, None)
    if param is not None and param.data.numel() > 0:
        weight = param.data
    else:
        if param is None:
            return torch.empty(0)
        _BWD_REMATERIALIZE_COUNT[0] += 1
        weight = ensure_group_param_materialized(group)
        if weight is None:
            return torch.empty(0)

    L = group.layer_idx
    if L >= 0:
        bucket = _BWD_MATERIALIZED.setdefault(L, [])
        if group not in bucket:
            bucket.append(group)
        _release_layers_above(L + (_BWD_RELEASE_LAG - 1))
    return weight


class MegatronDynamicLoader:
    """
    Dynamic weight loader for single-GPU Megatron training.

    Key features:
    1. HF -> Megatron name mapping
    2. Merged weight assembly (qkv, fc1)
    3. cp -> param binding and write-back
    4. Original weight release
    5. TE-compatible pre-decompression
    6. Forward/backward hooks

    Usage:
        loader = MegatronDynamicLoader(
            model=model,
            comp_dir="/path/to/compressed",
            device=torch.device("cuda:0"),
        )
        loader.load_weights()
        loader.build_param_mapping()
        loader.release_original_weights()
        loader.install_hooks()
        loader.prefetch_initial_layers()  # For TE compatibility
    """

    # HF to Megatron name mapping
    HF_TO_MEGATRON = {
        "q_proj": ("self_attention.linear_qkv", "q"),
        "k_proj": ("self_attention.linear_qkv", "k"),
        "v_proj": ("self_attention.linear_qkv", "v"),
        "o_proj": ("self_attention.linear_proj", None),
        "gate_proj": ("mlp.linear_fc1", "gate"),
        "up_proj": ("mlp.linear_fc1", "up"),
        "down_proj": ("mlp.linear_fc2", None),
    }

    # Megatron target weights
    MEGATRON_WEIGHT_TARGETS = [
        "self_attention.linear_qkv",
        "self_attention.linear_proj",
        "mlp.linear_fc1",
        "mlp.linear_fc2",
    ]

    # HF global (non-layer) param name → candidate Megatron paths (model-relative).
    # Tried in order; first match wins.  Handles both wrapped (language_model.*) and
    # bare Megatron GPT model structures.
    NON_LAYER_HF_TO_MEGATRON_HINTS: dict[str, list[str]] = {
        "model.embed_tokens.weight": [
            "embedding.word_embeddings.weight",
            "language_model.embedding.word_embeddings.weight",
        ],
        "model.norm.weight": [
            "decoder.final_layernorm.weight",
            "language_model.decoder.final_layernorm.weight",
        ],
        "lm_head.weight": [
            "output_layer.weight",
            "language_model.output_layer.weight",
        ],
    }

    # Per-layer HF norm suffix → candidate Megatron paths (layer-relative).
    # TE (Transformer Engine) fuses layer norms into the adjacent linear module;
    # the non-TE fallback paths are also listed.
    LAYER_NORM_SUFFIX_TO_MEGATRON: dict[str, list[str]] = {
        "input_layernorm.weight": [
            "self_attention.linear_qkv.layer_norm_weight",  # TE fused
            "input_layernorm.weight",  # non-TE
        ],
        "post_attention_layernorm.weight": [
            "mlp.linear_fc1.layer_norm_weight",  # TE fused
            "post_attention_layernorm.weight",  # non-TE
        ],
    }

    def __init__(
        self,
        model: nn.Module,
        comp_dir: str,
        device: torch.device,
        prefetch_layers: int = 1,
        print_debug: bool = False,
        allowed_targets: set[str] | None = None,
        num_attention_heads: int | None = None,
        num_query_groups: int | None = None,
        hidden_size: int | None = None,
        kv_channels: int | None = None,
        group_query_attention: bool = False,
    ):
        """
        Initialize the loader.

        Args:
            model: Megatron model (unwrapped, should be the decoder/GPTModel)
            comp_dir: Path to a single-GPU compressed weights directory.
            device: Target CUDA device
            prefetch_layers: Number of layers to prefetch ahead
            print_debug: Print debug messages
            allowed_targets: Optional Megatron target filter.
            num_attention_heads/num_query_groups/hidden_size/kv_channels:
                Model shape metadata needed to materialize HF Q/K/V tensors in
                Megatron-Core linear_qkv order.
        """
        self.model = model
        self.comp_dir = comp_dir
        self.device = device
        self.prefetch_layers = prefetch_layers
        self.print_debug = print_debug
        self.allowed_targets = allowed_targets
        self.num_attention_heads = num_attention_heads
        self.num_query_groups = num_query_groups
        self.hidden_size = hidden_size
        self.kv_channels = kv_channels
        self.group_query_attention = group_query_attention

        _idx_path = os.path.join(comp_dir, "index.json")

        # Relax hook-side hard sync to reduce main-thread stalls.
        strict_sync = os.environ.get("MEMRIFT_WEIGHT_SYNC", "0") == "1"
        if os.environ.get("MEMRIFT_WEIGHT_RELAX_SYNC", "") == "1":
            strict_sync = False
        self._hook_materialize_sync = strict_sync

        # Load index from the effective (rank-specific) directory
        with open(_idx_path) as f:
            self.index = json.load(f)

        if print_debug:
            print(f"[MemRift] Init: single_gpu mode=HF dir={comp_dir}")

        # Data structures
        self.num_layers = 0
        self.layer_names: list[str] = []

        # layer_idx -> megatron_target -> MergedWeightGroup
        self.merged_groups: dict[int, dict[str, MergedWeightGroup]] = defaultdict(dict)

        # layer_name -> list of MergedWeightGroup (for hooks)
        self.layer2groups: dict[str, list[MergedWeightGroup]] = defaultdict(list)

        # All CompressedParams (for memory tracking)
        self.all_cps: list[CompressedParam] = []

        # Non-layer params (embed, lm_head, etc.)
        self.non_layer_cps: list[CompressedParam] = []

        # Track original weights that were released
        self.released_params: set[int] = set()

        # Parameter identities whose values are supplied by comp_dir. Streamed
        # layer weights are registered during build_param_mapping(); permanent
        # embedding/norm/output weights are registered while materializing them.
        self.compressed_source_param_ids: set[int] = set()
        self.materialized_non_layer_names: set[str] = set()

        # Async compressor reference
        self.async_compressor = None
        self._hook_warn_ms = _env_float("MEMRIFT_HOOK_WARN_MS", 800.0)

        # Backward empty_cache counter (aligned with memrift_demo)
        self._bwd_counter = 0
        self._bwd_empty_step = int(_env_float("MEMRIFT_BWD_EMPTY_STEP", 5))

    def _count_pending_prefetch(self, groups: list[MergedWeightGroup]) -> int:
        pending = 0
        for group in groups:
            for cp in group.components.values():
                fut_obj = cp._prefetch_future
                if fut_obj is not None and not fut_obj.done():
                    pending += 1
        return pending

    def _get_layer_idx(self, hf_name: str) -> int | None:
        """Extract layer index from HF parameter name."""
        parts = hf_name.split(".")
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def _map_hf_to_megatron(self, hf_name: str) -> tuple[str | None, str | None]:
        """
        Map HF parameter name to Megatron target and merge key.

        Returns:
            (megatron_target, merge_key) or (None, None) if not mapped
        """
        if not hf_name.endswith(".weight"):
            return None, None
        for hf_suffix, (mg_target, merge_key) in self.HF_TO_MEGATRON.items():
            if f".{hf_suffix}." in hf_name or hf_name.endswith(f".{hf_suffix}.weight"):
                return mg_target, merge_key
        return None, None

    def load_weights(self):
        """
        Load compressed weights from disk and organize into MergedWeightGroups.

        The compressed directory uses HuggingFace-format names:
        "model.layers.{i}.self_attn.q_proj.weight". QKV and FC1 components are
        loaded separately and merged at materialize time.
        """
        if self.print_debug:
            print(f"[MemRift] Loading weights from {self.comp_dir}")

        if not self.index:
            raise ValueError(f"MemRift index is empty: {self.comp_dir}/index.json")

        names = [entry.get("name") for entry in self.index]
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicate_names:
            raise ValueError(
                "MemRift index contains duplicate parameter names: "
                + ", ".join(duplicate_names[:10])
            )

        unsupported = [
            entry.get("name", "<unnamed>")
            for entry in self.index
            if entry.get("scheme") != "split_zstd"
        ]
        if unsupported:
            raise ValueError(
                "MemRift uses the compressed directory as the sole frozen-weight source; "
                "all entries must use split_zstd. Unsupported entries: "
                + ", ".join(unsupported[:10])
            )
        unsupported_dtypes = [
            entry.get("name", "<unnamed>")
            for entry in self.index
            if entry.get("dtype") not in {"bfloat16", "float32"}
        ]
        if unsupported_dtypes:
            raise ValueError(
                "MemRift split_zstd entries must use bfloat16 or float32. Unsupported entries: "
                + ", ".join(unsupported_dtypes[:10])
            )

        # First pass: count layers
        for entry in self.index:
            layer_idx = self._get_layer_idx(entry["name"])
            if layer_idx is not None:
                self.num_layers = max(self.num_layers, layer_idx + 1)

        self.layer_names = [f"decoder.layers.{i}" for i in range(self.num_layers)]

        if self.print_debug:
            print(f"[MemRift] HF mode: {self.num_layers} layers")

        for entry in self.index:
            hf_name = entry["name"]
            layer_idx = self._get_layer_idx(hf_name)
            megatron_target, merge_key = self._map_hf_to_megatron(hf_name)

            file_path = os.path.join(self.comp_dir, entry["file"])
            cp = self._read_compressed_file(file_path, entry)
            cp.hf_name = hf_name
            cp.megatron_target = megatron_target or ""
            cp.merge_key = merge_key
            cp.layer_idx = layer_idx if layer_idx is not None else -1

            self.all_cps.append(cp)

            if layer_idx is not None and megatron_target:
                if megatron_target not in self.merged_groups[layer_idx]:
                    self.merged_groups[layer_idx][megatron_target] = MergedWeightGroup(
                        megatron_target=megatron_target,
                        layer_idx=layer_idx,
                        num_attention_heads=self.num_attention_heads,
                        num_query_groups=self.num_query_groups,
                        hidden_size=self.hidden_size,
                        kv_channels=self.kv_channels,
                        group_query_attention=self.group_query_attention,
                    )
                group = self.merged_groups[layer_idx][megatron_target]
                group.components[merge_key if merge_key else "single"] = cp
                if self.print_debug:
                    print(
                        f"[MemRift] HF {hf_name} → layer {layer_idx} / "
                        f"{megatron_target} / {merge_key or 'single'}"
                    )
            else:
                self.non_layer_cps.append(cp)

        # Validate
        for layer_idx, groups in self.merged_groups.items():
            for target, group in groups.items():
                if not group.is_complete():
                    missing = (
                        [k for k in ["q", "k", "v"] if k not in group.components]
                        if "linear_qkv" in target
                        else [k for k in ["gate", "up"] if k not in group.components]
                    )
                    raise ValueError(
                        f"Incomplete MemRift group at layer {layer_idx} / {target}; "
                        f"missing compressed components: {missing}"
                    )

        if self.print_debug:
            print(f"[MemRift] Loaded {len(self.all_cps)} compressed params")

    def _read_compressed_file(self, file_path: str, entry: dict) -> "CompressedParam":
        """Read one split_zstd file and return an unbound CompressedParam."""
        with open(file_path, "rb") as f:
            numel = struct.unpack("<Q", f.read(8))[0]
            sm_size = numel * (1 if entry["dtype"] == "bfloat16" else 3)
            sm_bytes = np.frombuffer(f.read(sm_size), dtype=np.uint8)
            exp_bytes = f.read()
        sm_gpu = torch.tensor(sm_bytes, dtype=torch.uint8, device=self.device)
        dtype = torch.bfloat16 if entry["dtype"] == "bfloat16" else torch.float32
        return CompressedParam(entry["shape"], sm_gpu, exp_bytes, dtype, self.device)

    def build_param_mapping(self):
        """
        Build mapping from MergedWeightGroup to actual model parameters.

        Layer indices are single-GPU HF indices and map directly to decoder_layers.
        """
        if self.print_debug:
            print("[MemRift] Building param mapping...")

        # Find decoder layers
        decoder_layers = None
        for name, module in self.model.named_modules():
            if hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
                decoder_layers = module.layers
                if self.print_debug:
                    print(
                        f"[MemRift] Found decoder layers at: {name}.layers "
                        f"({len(decoder_layers)} layers)"
                    )
                break

        if decoder_layers is None:
            if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
                decoder_layers = self.model.decoder.layers
            elif hasattr(self.model, "language_model") and hasattr(
                self.model.language_model, "decoder"
            ):
                decoder_layers = self.model.language_model.decoder.layers
            elif hasattr(self.model, "module"):
                unwrapped = self.model.module
                self.model = unwrapped
                return self.build_param_mapping()

        if decoder_layers is None:
            raise RuntimeError("MemRift could not find decoder layers for compressed weights")

        # Map each group to its target module
        for layer_idx, groups in self.merged_groups.items():
            if layer_idx >= len(decoder_layers):
                raise ValueError(
                    f"Compressed layer index {layer_idx} exceeds model layer count "
                    f"{len(decoder_layers)}"
                )

            layer = decoder_layers[layer_idx]
            layer_name = self.layer_names[layer_idx]

            for target, group in groups.items():
                if self.allowed_targets is not None and target not in self.allowed_targets:
                    raise ValueError(
                        "A streamed-weight target filter is incompatible with using the "
                        f"compressed directory as the sole source: layer {layer_idx} / {target}"
                    )
                # Navigate to target module
                target_module = layer
                parts = target.split(".")
                for part in parts[:-1]:
                    if hasattr(target_module, part):
                        target_module = getattr(target_module, part)
                    else:
                        target_module = None
                        break

                if target_module is not None:
                    final_attr = parts[-1]
                    if hasattr(target_module, final_attr):
                        linear_module = getattr(target_module, final_attr)
                        # Resolve PEFT wrappers to an inner module that owns `.weight`.
                        bind_module = linear_module
                        if not (hasattr(bind_module, "weight") and bind_module.weight is not None):
                            for unwrap_attr in ("to_wrap", "base_layer", "module"):
                                inner = getattr(bind_module, unwrap_attr, None)
                                if isinstance(inner, nn.Module) and hasattr(inner, "weight"):
                                    if inner.weight is not None:
                                        bind_module = inner
                                        break
                        if not (hasattr(bind_module, "weight") and bind_module.weight is not None):
                            raise RuntimeError(
                                f"MemRift target has no weight: layer {layer_idx} / {target} "
                                f"({type(linear_module).__name__})"
                            )
                        group.target_module = bind_module
                        group.target_attr = "weight"
                        param = bind_module.weight
                        if param.requires_grad:
                            raise ValueError(
                                "MemRift weight streaming requires a frozen base parameter, "
                                f"but layer {layer_idx} / {target} requires gradients"
                            )
                        group.target_shape = tuple(param.shape)
                        if "linear_qkv" not in target:
                            compressed_shape = tuple(group.get_merged_shape())
                            if compressed_shape != group.target_shape:
                                raise ValueError(
                                    f"Compressed weight shape mismatch at layer {layer_idx} / "
                                    f"{target}: compressed={compressed_shape}, "
                                    f"model={group.target_shape}"
                                )
                        self.compressed_source_param_ids.add(id(param))

                        # Also set on component cps for easier access
                        for cp in group.components.values():
                            cp.target_module = bind_module
                            cp.target_attr = "weight"

                        if self.print_debug:
                            print(
                                f"[MemRift] Mapped layer {layer_idx} / {target} -> {type(linear_module).__name__}"
                            )
                    else:
                        raise RuntimeError(
                            f"MemRift target {final_attr!r} was not found in "
                            f"{type(target_module).__name__} at layer {layer_idx}"
                        )
                else:
                    raise RuntimeError(
                        f"MemRift could not navigate to {target!r} in layer {layer_idx}"
                    )

                self.layer2groups[layer_name].append(group)

        total_mapped = sum(len(groups) for groups in self.layer2groups.values())
        total_compressed = sum(len(groups) for groups in self.merged_groups.values())
        if total_mapped != total_compressed:
            raise RuntimeError(
                f"MemRift mapped {total_mapped}/{total_compressed} streamed weight groups"
            )
        if self.print_debug:
            print(f"[MemRift] Mapped {total_mapped} weight groups")

    def release_original_weights(self):
        """
        Release original weight tensors to free GPU memory.

        Replaces each mapped parameter's .data with torch.empty(0, ...) so the
        allocator can reclaim memory. Must be called after build_param_mapping().
        After this, TE would see (0,) shape on first forward, so call
        prefetch_initial_layers() after install_hooks() to pre-fill first K layers.
        """
        if self.print_debug:
            mem_before = torch.cuda.memory_allocated(self.device)
            print(f"[MemRift] Memory before release: {mem_before / 1024**2:.1f} MB")

        released_count = 0
        for layer_name, groups in self.layer2groups.items():
            for group in groups:
                if group.target_module is not None:
                    param = getattr(group.target_module, group.target_attr, None)
                    if param is not None and id(param) not in self.released_params:
                        dtype = param.dtype
                        dev = param.device
                        with torch.no_grad():
                            empty = torch.empty(0, dtype=dtype, device=dev)
                            if hasattr(param, "data"):
                                param.data = empty
                        self.released_params.add(id(param))
                        released_count += 1

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        # Non-layer params (embed, lm_head, norm) are never dynamically loaded via
        # hooks — their weights stay in param.data permanently.  Free their compressed
        # form (_sm_gpu + exp_mv) since it will never be used for decompression.
        for cp in self.non_layer_cps:
            cp.release_compressed()

        if self.print_debug:
            mem_after = torch.cuda.memory_allocated(self.device)
            print(f"[MemRift] Released {released_count} original weights")
            print(f"[MemRift] Memory after release: {mem_after / 1024**2:.1f} MB")
            print(f"[MemRift] Memory saved: {(mem_before - mem_after) / 1024**2:.1f} MB")

    def _materialize_group(self, group: MergedWeightGroup, sync: bool = True):
        """
        Materialize all components of a group and assemble merged weight.

        For TP=1, merged weights are:
        - qkv: concat(q, k, v) along dim 0
        - fc1: concat(gate, up) along dim 0
        """
        return _materialize_group_tensor(group, sync=sync)

    def _set_param(self, group: MergedWeightGroup, weight: torch.Tensor):
        """
        Write materialized weight back to the model parameter.

        Forward/backward use the parameter's .data; replacing param.data ensures
        the next forward/backward on this layer uses the decompressed weight.

        Avoids .to() when device/dtype already match to prevent an extra copy.
        """
        if group.target_module is None:
            return
        param = getattr(group.target_module, group.target_attr, None)
        if param is None:
            if self.print_debug:
                print(f"[MemRift] set_param skip: no param at {group.target_attr}")
            return
        with torch.no_grad():
            if weight.is_cuda:
                # Tell the caching allocator this tensor is now used on the compute
                # stream so it won't be reclaimed for h2d_stream allocations while
                # forward/backward kernels are still reading it.  Mirrors demo's
                # set_param(): self._bf16.record_stream(current_stream).
                weight.record_stream(torch.cuda.current_stream(weight.device))
            if weight.device == param.device and weight.dtype == param.dtype:
                param.data = weight
            else:
                param.data = weight.to(device=param.device, dtype=param.dtype)
        # Avoid host blocking: always attach producer events to current stream.
        # This is a no-op when tensors are already ready, and preserves correctness
        # for relaxed-sync mode without forcing synchronize() on host.
        cur_stream = torch.cuda.current_stream(param.device)
        for cp in group.components.values():
            if cp._CtoD_evt is not None:
                cur_stream.wait_event(cp._CtoD_evt)
        _PTR2GROUP[int(param.data_ptr())] = group
        if self.print_debug:
            print(
                f"[MemRift] set_param: layer {group.layer_idx} / {group.megatron_target} shape={weight.shape}"
            )

    def _clear_param(self, group: MergedWeightGroup):
        """Clear parameter data to free memory."""
        if group.target_module is not None:
            param = getattr(group.target_module, group.target_attr, None)
            if param is not None:
                old_ptr = int(param.data_ptr()) if param.data.numel() > 0 else -1
                dtype = param.dtype
                device = param.device
                with torch.no_grad():
                    param.data = torch.empty(0, dtype=dtype, device=device)
                if old_ptr >= 0:
                    _PTR2GROUP.pop(old_ptr, None)
                if self.print_debug:
                    print(
                        f"[MemRift] clear_param: layer {group.layer_idx} / {group.megatron_target}"
                    )

        # Also release component cps
        for cp in group.components.values():
            cp.release()

    def _materialize_and_set_layer(self, layer_name: str):
        """Materialize all groups in a layer and set params."""
        for group in self.layer2groups.get(layer_name, []):
            weight = self._materialize_group(group, sync=True)
            self._set_param(group, weight)

    def _release_layer(self, layer_name: str):
        """Release all groups in a layer."""
        for group in self.layer2groups.get(layer_name, []):
            self._clear_param(group)

    def release_all_layers(self):
        """Force-release every layer's materialized weights and cp._bf16.

        Used as post-backward cleanup: per-linear bwd_pre fallback hooks can
        re-materialize weights AFTER layer bwd_post clears them, leaving 32
        layers pinned at end of iteration. Calling this after the autograd
        graph is released frees them all.
        """
        for layer_name in self.layer_names:
            self._release_layer(layer_name)
        n_remat = reset_bwd_tracking()
        if self.print_debug:
            print(f"[MemRift] bwd re-materialize count this iter: {n_remat}", flush=True)

    def materialize_all_layers(self) -> None:
        """Temporarily make every streamed parameter full-sized for checkpoint I/O."""
        try:
            for layer_name in self.layer_names:
                self._materialize_and_set_layer(layer_name)
            torch.cuda.synchronize(self.device)
            for groups in self.merged_groups.values():
                for group in groups.values():
                    if group.target_module is None or group.target_shape is None:
                        raise RuntimeError(
                            f"Checkpoint materialization has an unbound group: "
                            f"layer {group.layer_idx} / {group.megatron_target}"
                        )
                    param = getattr(group.target_module, group.target_attr, None)
                    if param is None or tuple(param.shape) != tuple(group.target_shape):
                        actual_shape = None if param is None else tuple(param.shape)
                        raise RuntimeError(
                            f"Checkpoint materialization shape mismatch at layer "
                            f"{group.layer_idx} / {group.megatron_target}: "
                            f"expected={group.target_shape}, actual={actual_shape}"
                        )
        except Exception:
            # A failed checkpoint preparation must not leave an arbitrary subset of
            # the model materialized.
            self.release_all_layers()
            raise

    # ── Non-layer weight helpers ─────────────────────────────────────────────

    def _navigate(self, root: nn.Module, path: str):
        """Navigate to a sub-module or parameter by dot-separated path."""
        obj = root
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    @staticmethod
    def _navigate_wrapped(root: nn.Module, path: str):
        """Resolve a path through PEFT/module wrappers.

        LoRA replaces a Transformer Engine linear module with ``LoRALinear`` and
        stores the frozen base module under ``to_wrap``.  Other integrations use
        ``base_layer`` or ``module`` for the same pattern.  Keep the direct path
        first, then traverse those wrapper attributes only when a path component
        is not present on the current object.
        """
        parts = tuple(part for part in path.split(".") if part)
        queue: deque[tuple[Any, str, int]] = deque([(root, "", 0)])
        visited: set[tuple[int, int]] = set()
        unwrap_attrs = ("to_wrap", "base_layer", "module")

        while queue:
            obj, resolved_path, part_idx = queue.popleft()
            state = (id(obj), part_idx)
            if state in visited:
                continue
            visited.add(state)

            if part_idx == len(parts):
                return obj, resolved_path.lstrip(".")

            part = parts[part_idx]
            direct = getattr(obj, part, None)
            if direct is not None:
                queue.append((direct, f"{resolved_path}.{part}", part_idx + 1))

            if isinstance(obj, nn.Module):
                for unwrap_attr in unwrap_attrs:
                    inner = getattr(obj, unwrap_attr, None)
                    if isinstance(inner, nn.Module):
                        queue.append((inner, f"{resolved_path}.{unwrap_attr}", part_idx))

        return None, None

    def _get_decoder_layers_for_norms(self) -> nn.ModuleList | None:
        """Return the decoder ModuleList (used for per-layer norm writes)."""
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            return self.model.decoder.layers
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "decoder"):
            return self.model.language_model.decoder.layers
        for _name, m in self.model.named_modules():
            if hasattr(m, "layers") and isinstance(m.layers, nn.ModuleList):
                return m.layers
        return None

    def materialize_non_layer_weights(self) -> int:
        """
        Decompress and write all non-layer weights into the model.

        This includes:
        1. Global params: embed_tokens, final norm, lm_head
        2. Per-layer norms: input_layernorm, post_attention_layernorm
           (both TE-fused and non-TE paths are tried)

        Call this **before** release_original_weights() so that the model has
        correct weights for inference — the compressed directory is the sole
        weight source when no Megatron checkpoint is loaded via --load.

        Returns: number of parameters successfully written.
        """
        written = 0
        missing: list[str] = []
        decoder_layers = self._get_decoder_layers_for_norms()

        def write_weight(cp: CompressedParam, target: torch.Tensor, target_path: str) -> None:
            nonlocal written
            if target.requires_grad:
                raise ValueError(
                    f"MemRift compressed base parameter {target_path!r} requires gradients"
                )
            weight = cp.materialize(sync=True)
            if tuple(weight.shape) != tuple(target.shape):
                raise ValueError(
                    f"Compressed weight shape mismatch for {cp.hf_name!r} -> {target_path!r}: "
                    f"compressed={tuple(weight.shape)}, model={tuple(target.shape)}"
                )

            # Permanent non-layer weights must not alias CompressedParam's pooled
            # materialization buffer, which is returned to the CUDA pool below.
            restored = weight.to(device=target.device, dtype=target.dtype)
            if restored.data_ptr() == weight.data_ptr():
                restored = restored.clone()
            with torch.no_grad():
                target.data = restored

            self.compressed_source_param_ids.add(id(target))
            self.materialized_non_layer_names.add(cp.hf_name)
            cp.release()
            written += 1

        for cp in self.non_layer_cps:
            hf_name = cp.hf_name
            written_this = False

            # 1) Global non-layer mapping (embed_tokens, final norm, lm_head)
            hints = self.NON_LAYER_HF_TO_MEGATRON_HINTS.get(hf_name)
            if hints:
                for hint_path in hints:
                    target = self._navigate(self.model, hint_path)
                    if target is not None and isinstance(target, torch.Tensor):
                        write_weight(cp, target, hint_path)
                        written_this = True
                        _trace(f"non-layer: {hf_name} → {hint_path}")
                        break

            # 2) Per-layer norm mapping
            if not written_this and decoder_layers is not None:
                layer_idx = cp.layer_idx
                if 0 <= layer_idx < len(decoder_layers):
                    layer = decoder_layers[layer_idx]
                    # Activation/weight placeholder support wraps each decoder
                    # layer after initial source materialization. Checkpoint resume
                    # must write norms into the original layer inside that wrapper.
                    inner_layer = getattr(layer, "layer", None)
                    if isinstance(inner_layer, nn.Module):
                        layer = inner_layer
                    for norm_suffix, mg_paths in self.LAYER_NORM_SUFFIX_TO_MEGATRON.items():
                        if not hf_name.endswith(norm_suffix):
                            continue
                        for mg_path in mg_paths:
                            target, resolved_path = self._navigate_wrapped(layer, mg_path)
                            if not isinstance(target, torch.Tensor):
                                continue
                            target_path = f"decoder.layers.{layer_idx}.{resolved_path}"
                            write_weight(cp, target, target_path)
                            written_this = True
                            _trace(f"layer norm: {hf_name} → {target_path}")
                            break
                        if written_this:
                            break

            if not written_this:
                missing.append(hf_name)

        if missing:
            raise ValueError(
                "The compressed directory is the sole frozen-weight source, but these "
                "compressed parameters could not be mapped into the Megatron model: "
                + ", ".join(missing[:20])
            )

        if self.print_debug:
            print(
                f"[MemRift] materialize_non_layer_weights: "
                f"wrote {written}/{len(self.non_layer_cps)} params"
            )
        return written

    def validate_compressed_source_coverage(self) -> None:
        """Require every frozen model parameter to be supplied by ``comp_dir``."""
        unsourced = [
            name
            for name, param in self.model.named_parameters()
            if not param.requires_grad and id(param) not in self.compressed_source_param_ids
        ]
        if unsourced:
            raise ValueError(
                "The compressed directory is configured as the sole frozen-weight source, "
                "but these frozen model parameters have no compressed mapping: "
                + ", ".join(unsourced[:20])
            )

        expected_entries = len(self.all_cps)
        consumed_entries = sum(
            len(group.components)
            for groups in self.merged_groups.values()
            for group in groups.values()
        ) + len(self.materialized_non_layer_names)
        if consumed_entries != expected_entries:
            raise RuntimeError(
                f"MemRift consumed {consumed_entries}/{expected_entries} compressed entries"
            )

    def prefetch_initial_layers(self):
        """
        Pre-materialize first K layers and write back to param.data (TE compatibility).

        After release_original_weights(), all param.data are empty; TE would fail
        on first forward due to shape checks. This fills param.data for the first
        prefetch_layers+1 layers so the first forward sees valid shapes.
        """
        # Keep the first layer materialized for TE shape checks and honor the
        # configured look-ahead for the remaining initial window.
        k = min(len(self.layer_names), max(1, int(self.prefetch_layers) + 1))
        if self.print_debug:
            print(f"[MemRift] Pre-materializing first {k} layer(s) (param.data write-back for TE)")

        for i in range(k):
            if i < len(self.layer_names):
                layer_name = self.layer_names[i]
                self._materialize_and_set_layer(layer_name)

        torch.cuda.synchronize(self.device)

        # Verify first K layers have non-empty params
        ok = 0
        for i in range(k):
            if i >= len(self.layer_names):
                break
            for group in self.layer2groups.get(self.layer_names[i], []):
                if group.target_module is not None:
                    param = getattr(group.target_module, group.target_attr, None)
                    if param is not None and param.data.numel() > 0:
                        ok += 1
        if self.print_debug:
            mem = torch.cuda.memory_allocated(self.device)
            print(f"[MemRift] Prefetch done: {ok} params filled, memory {mem / 1024**2:.1f} MB")

    def install_hooks(self, async_compressor=None):
        """
        Install forward/backward hooks for dynamic weight loading.

        Hooks:
        - forward_pre: materialize current layer (consuming prefetch future if any),
                       write to param via _set_param; then prefetch next layer (store future on cp).
        - forward_post: release current layer (except last)
        - backward_pre: materialize current layer, prefetch prev
        - backward_post: release current layer

        Async prefetch: materialize_async() future is stored on cp._prefetch_future.
        When the next layer runs, cp.materialize() consumes the future and returns
        the tensor; _materialize_group merges and _set_param writes to model param,
        so the prefetched result does land on the real parameter.

        Args:
            async_compressor: Optional AsyncCompressor for async prefetch
        """
        self.async_compressor = async_compressor

        # Find layer modules
        name2layer = {}
        for name, module in self.model.named_modules():
            name2layer[name] = module

        # Also try to find by decoder.layers pattern
        decoder_layers = None
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "layers"):
            decoder_layers = self.model.decoder.layers
        elif hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "decoder"
        ):
            decoder_layers = self.model.language_model.decoder.layers

        if decoder_layers is not None:
            for i, layer in enumerate(decoder_layers):
                name2layer[f"decoder.layers.{i}"] = layer

        if self.print_debug:
            found = [n for n in self.layer_names if n in name2layer]
            print(f"[MemRift] Found {len(found)}/{len(self.layer_names)} layer modules for hooks")
            print(f"[MemRift] Hook materialize sync={self._hook_materialize_sync}")

        # Forward hooks
        for i in range(len(self.layer_names)):
            cur = self.layer_names[i]
            span = max(0, int(self.prefetch_layers))
            nxt_names = self.layer_names[i + 1 : min(len(self.layer_names), i + 1 + span)]

            if cur not in name2layer:
                continue

            layer_module = name2layer[cur]
            cur_groups = self.layer2groups.get(cur, [])
            nxt_groups_list = [self.layer2groups.get(nm, []) for nm in nxt_names]

            # Capture variables
            def make_fwd_pre(cur_groups, nxt_groups_list, cur_name, async_comp):
                def _hook(mod, inp):
                    t0 = time.perf_counter()
                    _trace(f"fwd_pre: enter layer={cur_name}, groups={len(cur_groups)}")
                    pending_cur = self._count_pending_prefetch(cur_groups)
                    if pending_cur > 0:
                        _trace(f"fwd_pre: layer={cur_name} pending_prefetch_cur={pending_cur}")

                    # 1) Submit prefetch for NEXT K layers FIRST so CPU decompression
                    #    can overlap with current layer materialize + forward compute.
                    #    K is controlled by prefetch_layers.
                    if async_comp and nxt_groups_list:
                        for nxt_groups in nxt_groups_list:
                            for group in nxt_groups:
                                for cp in group.components.values():
                                    if cp._bf16 is None and cp._prefetch_future is None:
                                        cp._prefetch_future = async_comp.materialize_async(
                                            cp.exp_mv, cp._sm_gpu, cp.orig_shape, cp._dtype
                                        )

                    # 2) Materialize current layer and write back to model param.
                    #    If the layer was pre-fetched, cp.materialize() consumes the future.
                    #    Fast path in _materialize_group_tensor: if param.data is already
                    #    valid (from prefetch_initial_layers) it returns it directly.
                    for group in cur_groups:
                        tg = time.perf_counter()
                        weight = self._materialize_group(group, sync=self._hook_materialize_sync)
                        self._set_param(group, weight)
                        group_ms = (time.perf_counter() - tg) * 1000.0
                        if group_ms > self._hook_warn_ms:
                            _trace(
                                f"WARN fwd_pre: slow group materialize {group_ms:.1f} ms "
                                f"(layer={cur_name}, target={group.megatron_target})"
                            )

                    hook_ms = (time.perf_counter() - t0) * 1000.0
                    if hook_ms > self._hook_warn_ms:
                        _trace(f"WARN fwd_pre: slow hook {hook_ms:.1f} ms (layer={cur_name})")
                    _trace(f"fwd_pre: done layer={cur_name}")

                return _hook

            def make_fwd_post(cur_groups):
                def _hook(mod, inp, out):
                    # Always release; backward_pre re-materializes on demand
                    for group in cur_groups:
                        self._clear_param(group)

                return _hook

            layer_module.register_forward_pre_hook(
                make_fwd_pre(cur_groups, nxt_groups_list, cur, async_compressor)
            )
            layer_module.register_forward_hook(make_fwd_post(cur_groups))

        # Backward hooks (reverse order)
        for i in range(len(self.layer_names) - 1, -1, -1):
            cur = self.layer_names[i]
            span = max(0, int(self.prefetch_layers))
            prv_names = self.layer_names[max(0, i - span) : i][::-1]

            if cur not in name2layer:
                continue

            layer_module = name2layer[cur]
            cur_groups = self.layer2groups.get(cur, [])
            prv_groups_list = [self.layer2groups.get(nm, []) for nm in prv_names]

            def make_bwd_pre(cur_groups, prv_groups_list, cur_name, async_comp):
                def _hook(mod, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        print(f"[HOOK_ORDER] bwd_pre  layer={cur_name}", flush=True)
                    _trace(f"bwd_pre: enter layer={cur_name}, groups={len(cur_groups)}")
                    pending_cur = self._count_pending_prefetch(cur_groups)
                    if pending_cur > 0:
                        _trace(f"bwd_pre: layer={cur_name} pending_prefetch_cur={pending_cur}")

                    # 1) Submit prefetch for PREVIOUS K layers FIRST so CPU decompression
                    #    can overlap with current layer materialize + backward compute.
                    #    K is controlled by prefetch_layers.
                    if async_comp and prv_groups_list:
                        for prv_groups in prv_groups_list:
                            for group in prv_groups:
                                for cp in group.components.values():
                                    if cp._bf16 is None and cp._prefetch_future is None:
                                        cp._prefetch_future = async_comp.materialize_async(
                                            cp.exp_mv, cp._sm_gpu, cp.orig_shape, cp._dtype
                                        )

                    # 2) Do NOT materialize here. Weights are materialized on-demand by
                    #    saved_tensors_hooks._unpack at the exact moment TE's fused
                    #    backward consumes them (see unpack_weight_for_backward).

                    _trace(f"bwd_pre: done layer={cur_name}")

                return _hook

            def make_bwd_post(cur_name=cur):
                def _hook(mod, grad_in, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        print(f"[HOOK_ORDER] bwd_post layer={cur_name}", flush=True)
                    # Weight release is driven by _unpack's release-lag, not here.
                    self._bwd_counter += 1
                    if self._bwd_counter % self._bwd_empty_step == 0:
                        torch.cuda.empty_cache()

                return _hook

            layer_module.register_full_backward_pre_hook(
                make_bwd_pre(cur_groups, prv_groups_list, cur, async_compressor)
            )
            layer_module.register_full_backward_hook(make_bwd_post())

            # Per-linear hooks bound to a single group: ensures the materialized
            # weight is released as soon as THAT linear's backward truly finishes,
            # not when the layer module's full_backward_hook fires (which can
            # precede RowParallelLinear's deferred all-reduce backward Function).
            # Module->group: deduplicate by id(target_module) since multiple
            # weight groups might map to the same linear module (rare but safe).
            # TE fuses fc1+fc2 (and self-attn qkv+proj) into single autograd
            # Functions (e.g. _LayerNormMLP). Backward of these Functions reads
            # BOTH weights simultaneously. So when ANY RowParallel linear in the
            # layer fires lin_bwd_pre, we must materialize ALL the layer's groups,
            # and the corresponding lin_bwd_post clears them all. ColumnParallel
            # linears never independently fire their hooks (no standalone autograd
            # Node), so registering on them is a harmless no-op.
            # NOTE: disabled by default; only active when MEMRIFT_DISABLE_LINEAR_BWD_PRE=0.
            def make_linear_bwd_pre_all(cur_groups, cur_name=cur):
                def _hook(mod, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        empty = True
                        try:
                            w = getattr(mod, "weight", None)
                            if w is not None and w.data.numel() > 0:
                                empty = False
                        except Exception:
                            pass
                        print(
                            f"[HOOK_ORDER] lin_bwd_pre layer={cur_name} mod={type(mod).__name__} empty={empty}",
                            flush=True,
                        )
                    try:
                        w = getattr(mod, "weight", None)
                        if w is not None and w.data.numel() > 0:
                            return
                    except Exception:
                        pass
                    for group in cur_groups:
                        weight = self._materialize_group(group, sync=self._hook_materialize_sync)
                        self._set_param(group, weight)

                return _hook

            def make_linear_bwd_post_all(cur_groups, cur_name=cur):
                def _hook(mod, grad_in, grad_out):
                    if os.environ.get("MEMRIFT_HOOK_ORDER", "0") == "1":
                        print(
                            f"[HOOK_ORDER] lin_bwd_post layer={cur_name} mod={type(mod).__name__}",
                            flush=True,
                        )
                    for group in cur_groups:
                        self._clear_param(group)

                return _hook

            if os.environ.get("MEMRIFT_DISABLE_LINEAR_BWD_PRE", "1") != "1":
                mod2group = {}
                for group in cur_groups:
                    tm = getattr(group, "target_module", None)
                    if tm is None:
                        continue
                    mod2group.setdefault(id(tm), (tm, group))
                lpre = make_linear_bwd_pre_all(cur_groups)
                lpost = make_linear_bwd_post_all(cur_groups)
                for tm, _g in mod2group.values():
                    tm.register_full_backward_pre_hook(lpre)
                    tm.register_full_backward_hook(lpost)

        if self.print_debug:
            print(f"[MemRift] Installed hooks for {len(self.layer_names)} layers")

    def reset(self):
        if self.async_compressor is not None:
            self.async_compressor.reset()
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.empty_cache()

    def get_memory_stats(self) -> dict[str, float]:
        """Get memory statistics for monitoring."""
        sm_gpu_bytes = sum(cp._sm_gpu.numel() for cp in self.all_cps if cp._sm_gpu is not None)
        exp_bytes = sum(len(cp.exp_mv) for cp in self.all_cps if cp.exp_mv is not None)

        return {
            "sm_gpu_bytes": sm_gpu_bytes,
            "sm_gpu_mb": sm_gpu_bytes / 1024**2,
            "exp_cpu_bytes": exp_bytes,
            "exp_cpu_mb": exp_bytes / 1024**2,
            "num_layers": self.num_layers,
            "num_weight_groups": sum(len(g) for g in self.layer2groups.values()),
            "cuda_allocated_mb": torch.cuda.memory_allocated(self.device) / 1024**2,
        }
