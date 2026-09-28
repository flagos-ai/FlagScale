import concurrent.futures as fut
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


# Thread-local storage for compressor/decompressor contexts
_tls = threading.local()


def get_compression_ctx(level: int = 18):
    """Get thread-local zstd compression context."""
    if not hasattr(_tls, "cctx") or _tls.cctx_level != level:
        _tls.cctx = zstd.ZstdCompressor(level=level, write_checksum=False)
        _tls.cctx_level = level
    return _tls.cctx


def get_decompression_ctx():
    """Get thread-local zstd decompression context."""
    if not hasattr(_tls, "dctx"):
        _tls.dctx = zstd.ZstdDecompressor()
    return _tls.dctx


@dataclass
class PlaceHolderToken:
    """
    Placeholder for compressed activation tensor.

    Used with torch.autograd.graph.saved_tensors_hooks to replace
    large activation tensors with compressed representations.
    """

    dtype: torch.dtype
    shape: torch.Size
    stride: tuple
    offset: int

    # Runtime fields (filled during compression/decompression)
    sm_bits: torch.Tensor | None = None
    comped_cpu_exp: bytes | None = None
    numel: int = 0
    decomped_data: torch.Tensor | None = None
    # Set by the asynchronous decode worker when decompression or CUDA merge
    # fails.  The consumer checks this after ready_evt is signalled so worker
    # failures cannot turn into an unbounded wait in autograd backward.
    error: BaseException | None = None
    ready_evt: threading.Event = field(default_factory=threading.Event)
    CtoD_copy_evt: torch.cuda.Event | None = None
    fut_id: int = -1

    def _clear_after_recover(self):
        """Clean up after decompression is complete."""
        if hasattr(self, "fut_id"):
            del self.fut_id
        self.error = None
        self.ready_evt.clear()
        if hasattr(self, "CtoD_copy_evt") and self.CtoD_copy_evt is not None:
            del self.CtoD_copy_evt
            self.CtoD_copy_evt = None


class AsyncCompressor:
    """
    Async compression/decompression manager.

    Manages:
    - Thread pools for CPU-side compression/decompression
    - CUDA streams for D2H/H2D transfers
    - Semaphore for concurrency control

    Usage:
        compressor = AsyncCompressor(
            compress_workers=8,
            decode_workers=4,
            concurrency_limit=4,
            zstd_level=18
        )

        # Async compression
        fut = compressor.kickoff_async(token, tensor)

        # Async decompression
        compressor.decompress_async(token, fut)
    """

    def __init__(
        self,
        compress_workers: int = 8,
        decode_workers: int = 4,
        concurrency_limit: int = 4,
        zstd_level: int = 18,
        enable_async: bool = True,
    ):
        """
        Initialize AsyncCompressor.

        Args:
            compress_workers: Number of compression thread pool workers
            decode_workers: Number of decompression thread pool workers
            concurrency_limit: Max concurrent operations (semaphore limit)
            zstd_level: Zstd compression level (1-22)
            enable_async: If False, use synchronous operations
        """
        if not ZSTD_AVAILABLE:
            raise RuntimeError("zstandard not available. Install with: pip install zstandard")

        self.zstd_level = zstd_level
        # concurrency_limit 允许环境变量覆盖:默认沿用调用方传入值(通常4,保守,控解压显存峰值);
        # 真实场景显存吃紧时保持小值,显存富余时可调大 MEMRIFT_DECODE_CONCURRENCY 提高解压并发。
        concurrency_limit = int(os.getenv("MEMRIFT_DECODE_CONCURRENCY", str(concurrency_limit)))
        self.act_split_path = os.getenv("MEMRIFT_ACT_SPLIT_PATH", "copy").strip().lower()
        if self.act_split_path not in {"mapped", "copy"}:
            raise ValueError(
                f"MEMRIFT_ACT_SPLIT_PATH must be 'mapped' or 'copy', got {self.act_split_path!r}"
            )
        self.act_split_profile = os.getenv("MEMRIFT_ACT_SPLIT_PROFILE", "0") == "1"
        self.enable_async = enable_async
        # 保存并发配置,供 _build()/reset() 重建时复用(避免写死,且 concurrency_limit 可配置)
        self._compress_workers = compress_workers
        self._decode_workers = decode_workers
        self._concurrency_limit = concurrency_limit

        if enable_async:
            self.compress_pool = fut.ThreadPoolExecutor(compress_workers)
            self.decode_pool = fut.ThreadPoolExecutor(decode_workers)
            self.decomp_semaphore = threading.Semaphore(value=concurrency_limit)
            self._h2d_lock = threading.Lock()
        else:
            self.cctx = zstd.ZstdCompressor(level=zstd_level, write_checksum=False)
            self.dctx = zstd.ZstdDecompressor()

        # CUDA streams for data transfer
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

        # Try to import CUDA extension
        try:
            from flagscale.compress.memrift import ops as fs_sp

            self._fs_sp = fs_sp if fs_sp.is_available() else None
        except ImportError:
            self._fs_sp = None

    def _build(self):
        """Rebuild pools (used after reset)."""
        if self.enable_async:
            self.compress_pool = fut.ThreadPoolExecutor(self._compress_workers)
            self.decode_pool = fut.ThreadPoolExecutor(self._decode_workers)
            # 重建信号量(原 _build 漏建,reset 后解压并发控制会失效)
            self.decomp_semaphore = threading.Semaphore(value=self._concurrency_limit)
            self._h2d_lock = threading.Lock()
        self.d2h_stream = torch.cuda.Stream()
        self.h2d_stream = torch.cuda.Stream()

    def reset(self):
        """Reset pools and streams (call between training rounds)."""
        if self.enable_async:
            self.compress_pool.shutdown(wait=True)
            self.decode_pool.shutdown(wait=True)
            del self.compress_pool
            del self.decode_pool

        self._build()
        torch.cuda.reset_peak_memory_stats()

    def shutdown(self):
        """Shutdown all pools."""
        if self.enable_async:
            self.compress_pool.shutdown(wait=True)
            self.decode_pool.shutdown(wait=True)

    # -------------------------------------------------------------------------
    #  Synchronous compression/decompression
    # -------------------------------------------------------------------------

    def kickoff_sync(self, tok: PlaceHolderToken, t: torch.Tensor):
        """Synchronous compression (blocking)."""
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for sync compression")

        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.d2h_stream):
            cpu_exp, sm_bits = self._fs_sp.split(t, self.d2h_stream.cuda_stream)
            evt = self.d2h_stream.record_event()
        evt.synchronize()

        tok.sm_bits = sm_bits
        arr = cpu_exp.numpy()
        cctx = (
            get_compression_ctx(self.zstd_level) if not getattr(self, "cctx", None) else self.cctx
        )
        comped_bytes = cctx.compress(arr)
        tok.comped_cpu_exp = comped_bytes
        tok.numel = arr.size

    def decompress_sync(self, tok: PlaceHolderToken):
        """Synchronous decompression (blocking)."""
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for sync decompression")

        dctx = get_decompression_ctx() if not getattr(self, "dctx", None) else self.dctx
        # Decompress exponent
        cpu_exp = torch.empty(tok.numel, dtype=torch.uint8, pin_memory=True)
        with dctx.stream_reader(memoryview(tok.comped_cpu_exp)) as reader:
            view = memoryview(cpu_exp.numpy())
            nread = reader.readinto(view)
            assert nread == tok.numel, "decompress size mismatch"

        # Merge on GPU
        stream = self.h2d_stream
        with torch.cuda.stream(stream):
            rst = self._fs_sp.merge(
                cpu_exp,
                tok.sm_bits,
                list(tok.shape),
                list(tok.stride),
                tok.offset,
                tok.dtype,
                stream.cuda_stream,
            )
        evt = stream.record_event()
        evt.synchronize()

        tok.decomped_data = rst

    # -------------------------------------------------------------------------
    #  Asynchronous compression/decompression
    # -------------------------------------------------------------------------

    def _split_for_activation_async(self, t: torch.Tensor):
        submit_start = time.perf_counter()
        start_evt = torch.cuda.Event(enable_timing=True) if self.act_split_profile else None
        end_evt = torch.cuda.Event(enable_timing=True) if self.act_split_profile else None

        if start_evt is not None:
            start_evt.record(self.d2h_stream)

        if self.act_split_path == "copy":
            if not hasattr(self._fs_sp, "split_copy"):
                raise RuntimeError(
                    "MEMRIFT_ACT_SPLIT_PATH=copy requires float_split_stride_pin.split_copy"
                )
            cpu_exp, sm_bits, staging_exp = self._fs_sp.split_copy(t, self.d2h_stream.cuda_stream)
        else:
            cpu_exp, sm_bits = self._fs_sp.split(t, self.d2h_stream.cuda_stream)
            staging_exp = None

        if end_evt is not None:
            end_evt.record(self.d2h_stream)

        submit_ms = (time.perf_counter() - submit_start) * 1000.0
        return cpu_exp, sm_bits, staging_exp, start_evt, end_evt, submit_ms

    def kickoff_async(self, tok: PlaceHolderToken, t: torch.Tensor):
        """
        Start async compression.

        Returns a Future that resolves to (compressed_bytes, numel).
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for async compression")

        self.d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.d2h_stream):
            cpu_exp, sm_bits, staging_exp, start_evt, end_evt, submit_ms = (
                self._split_for_activation_async(t)
            )
            evt = self.d2h_stream.record_event()
            t.record_stream(self.d2h_stream)

        tok.sm_bits = sm_bits

        def _encode(cpu_exp, evt, staging_exp, start_evt, end_evt, submit_ms, shape, dtype, numel):
            arr = None
            try:
                wait_start = time.perf_counter()
                evt.synchronize()
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
                cuda_event_ms = (
                    start_evt.elapsed_time(end_evt)
                    if start_evt is not None and end_evt is not None
                    else 0.0
                )

                numpy_start = time.perf_counter()
                arr = cpu_exp.numpy()
                numpy_ms = (time.perf_counter() - numpy_start) * 1000.0

                cctx = get_compression_ctx(self.zstd_level)
                zstd_start = time.perf_counter()
                comped_bytes = cctx.compress(arr)
                zstd_ms = (time.perf_counter() - zstd_start) * 1000.0

                if self.act_split_profile:
                    print(
                        "[MemRiftSplitProfile] "
                        f"path={self.act_split_path} "
                        f"shape={shape} "
                        f"dtype={dtype} "
                        f"numel={numel} "
                        f"submit={submit_ms:.3f}ms "
                        f"wait={wait_ms:.3f}ms "
                        f"cuda_event={cuda_event_ms:.3f}ms "
                        f"numpy={numpy_ms:.3f}ms "
                        f"zstd={zstd_ms:.3f}ms"
                    )

                return comped_bytes, arr.size
            finally:
                del cpu_exp
                del staging_exp
                if arr is not None:
                    del arr

        return self.compress_pool.submit(
            _encode,
            cpu_exp,
            evt,
            staging_exp,
            start_evt,
            end_evt,
            submit_ms,
            t.shape,
            t.dtype,
            t.numel(),
        )

    def decompress_async(self, tok: PlaceHolderToken, future: fut.Future):
        """
        Start async decompression.

        The token's ready_evt will be set when decompression is complete.
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available for async decompression")

        fs_sp = self._fs_sp
        h2d_stream = self.h2d_stream
        semaphore = self.decomp_semaphore
        h2d_lock = self._h2d_lock

        # A token may be reused after a successful recovery.  Clear stale
        # state before scheduling the next decode attempt.
        tok.error = None
        tok.decomped_data = None
        tok.CtoD_copy_evt = None
        tok.ready_evt.clear()

        def _decode(tok, future):
            semaphore.acquire()
            comped_bytes, numel = None, None
            cpu_exp = None
            try:
                try:
                    comped_bytes, numel = future.result()

                    cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                    dctx = get_decompression_ctx()
                    with dctx.stream_reader(memoryview(comped_bytes)) as reader:
                        view = memoryview(cpu_exp.numpy())
                        nread = reader.readinto(view)
                        assert nread == numel, "decompress size mismatch"

                    with h2d_lock:
                        with torch.cuda.stream(h2d_stream):
                            rst = fs_sp.merge(
                                cpu_exp,
                                tok.sm_bits,
                                list(tok.shape),
                                list(tok.stride),
                                tok.offset,
                                tok.dtype,
                                h2d_stream.cuda_stream,
                            )
                            tok.sm_bits.record_stream(h2d_stream)
                        evt = h2d_stream.record_event()
                    evt.synchronize()
                    tok.CtoD_copy_evt = evt
                    tok.decomped_data = rst
                except BaseException as exc:
                    # The consumer waits on ready_evt rather than the worker
                    # Future, so retain the original exception explicitly.
                    tok.error = exc
            finally:
                try:
                    del future
                    if comped_bytes is not None:
                        del comped_bytes
                    if tok.sm_bits is not None:
                        try:
                            fs_sp.release_cuda(tok.sm_bits)
                        except BaseException as exc:
                            if tok.error is None:
                                tok.error = exc
                    if cpu_exp is not None:
                        del cpu_exp
                finally:
                    semaphore.release()
                    # Always wake the consumer, including all failure paths.
                    tok.ready_evt.set()

        self.decode_pool.submit(_decode, tok, future)

    def materialize_async(
        self,
        exp_mv: bytes,
        sm_gpu: torch.Tensor,
        orig_shape: tuple,
        dtype: torch.dtype,
        callback: Callable[[torch.Tensor], None] | None = None,
    ):
        """
        Async weight materialization (decompression + merge).

        Args:
            exp_mv: Compressed exponent bytes
            sm_gpu: Sign+mantissa tensor already on GPU (transferred by caller before submission)
            orig_shape: Original tensor shape
            dtype: Target dtype
            callback: Optional callback with decompressed tensor
        """
        if self._fs_sp is None:
            raise RuntimeError("CUDA extension not available")

        fs_sp = self._fs_sp
        h2d_stream = self.h2d_stream
        semaphore = self.decomp_semaphore

        def _c_contiguous_strides(shape):
            strides = [1] * len(shape)
            running = 1
            for i in range(len(shape) - 2, -1, -1):
                running *= shape[i + 1]
                strides[i] = running
            return tuple(strides)

        def _materialize():
            semaphore.acquire()
            cpu_exp = None
            _sem_released = False
            _success = False
            try:
                numel = int(np.prod(orig_shape))

                # Decompress exponent into pinned CPU buffer
                cpu_exp = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
                dctx = get_decompression_ctx()
                with dctx.stream_reader(memoryview(exp_mv)) as reader:
                    view = memoryview(cpu_exp.numpy())
                    nread = reader.readinto(view)
                    assert nread == numel, "decompress size mismatch"

                # Submit merge kernel on dedicated h2d_stream (async).
                # sm_gpu is already on GPU (transferred by caller on main thread).
                strides = _c_contiguous_strides(orig_shape)
                with torch.cuda.stream(h2d_stream):
                    bf16 = fs_sp.merge(
                        cpu_exp,
                        sm_gpu,
                        list(orig_shape),
                        list(strides),
                        0,
                        dtype,
                        h2d_stream.cuda_stream,
                    )
                    sm_gpu.record_stream(h2d_stream)  # keep sm_gpu alive until stream completes
                evt = h2d_stream.record_event()

                # Release semaphore immediately after submitting the GPU kernel
                # so other workers can begin CPU decompression right away.
                semaphore.release()
                _sem_released = True
                _success = True

                if callback:
                    # Callback path: synchronize in thread for backward compat.
                    evt.synchronize()
                    del cpu_exp
                    _success = False  # prevent double-free in finally
                    callback(bf16)
                    return bf16

                # Non-callback (prefetch) path: exit thread immediately without
                # waiting for GPU.  Return a 3-tuple so the consumer can:
                #   - synchronize evt on the GPU stream (stream.wait_event)  OR
                #   - synchronize on CPU (evt.synchronize) before using bf16.
                # cpu_exp is included in the tuple to keep it alive until the
                # merge DMA completes; the consumer frees it after synchronization.
                return (bf16, evt, cpu_exp)

            finally:
                if not _sem_released:
                    semaphore.release()
                # Only free cpu_exp on failure; on success it travels in the tuple.
                if not _success and cpu_exp is not None:
                    del cpu_exp

        return self.decode_pool.submit(_materialize)
