"""
float_split_stride_pin: CUDA extension for bf16/fp32 float split/merge operations.

Provides:
- split(tensor, stream_ptr): Split tensor into (exp, sm) uint8 components
- merge(exp, sm, size, stride, offset, dtype, stream_ptr): Merge back to original tensor
- acquire_pin(numel, dtype): Acquire pinned memory from pool
- release_pin(tensor): Release pinned memory back to pool
- release_cuda(tensor): Release CUDA memory back to pool
"""

import os
import sys
import warnings
from importlib import import_module
from types import ModuleType

# Try multiple import methods
_ext = None
_AVAILABLE = False
_REQUIRED_SYMBOLS = (
    "split",
    "split_copy",
    "merge",
    "acquire_pin",
    "release_pin",
    "release_cuda",
)


def _is_valid_extension(module):
    return isinstance(module, ModuleType) and all(
        hasattr(module, symbol) for symbol in _REQUIRED_SYMBOLS
    )


try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
        import_module("torch")
except Exception:
    pass

# Method 1: Try direct import (when package is installed)
try:
    _ext = import_module("float_split_stride_pin._ext")
    _AVAILABLE = _is_valid_extension(_ext)
except ImportError:
    pass

# Method 2: Try relative import from package directory
if not _AVAILABLE:
    try:
        _ext = import_module(f"{__name__}._ext")
        _AVAILABLE = _is_valid_extension(_ext)
    except ImportError:
        pass

# Method 3: Try adding package directory to path
if not _AVAILABLE:
    try:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        import _ext as _ext_module

        _ext = _ext_module
        _AVAILABLE = _is_valid_extension(_ext)
    except ImportError:
        pass


def is_available():
    """Check if the CUDA extension is available."""
    return _AVAILABLE and _ext is not None


def split(t, stream_ptr):
    """
    Split a bf16/fp32 tensor into exponent and sign-mantissa components.

    Args:
        t: Input tensor (bf16 or fp32, on CUDA)
        stream_ptr: CUDA stream pointer (from stream.cuda_stream)

    Returns:
        (exp, sm): Tuple of uint8 tensors
            - exp: Exponent bytes (CPU pinned memory)
            - sm: Sign+mantissa bytes (CUDA)
    """
    if not _AVAILABLE:
        raise RuntimeError("float_split_stride_pin CUDA extension not available")
    return _ext.split(t, stream_ptr)


def split_copy(t, stream_ptr):
    """
    Split a bf16/fp32 CUDA tensor using GPU exponent staging followed by
    cudaMemcpyAsync into pinned CPU memory.

    Args:
        t: Input tensor, bf16 or fp32, on CUDA.
        stream_ptr: CUDA stream pointer from stream.cuda_stream.

    Returns:
        (exp, sm, exp_gpu): Tuple of tensors.
            - exp: Exponent bytes in pinned CPU memory.
            - sm: Sign+mantissa bytes on CUDA.
            - exp_gpu: Temporary CUDA exponent buffer. Keep this alive until the
              stream event after split_copy has completed.
    """
    if not _AVAILABLE:
        raise RuntimeError("float_split_stride_pin CUDA extension not available")
    return _ext.split_copy(t, stream_ptr)


def merge(exp, sm, size, stride, offset, dtype, stream_ptr):
    """
    Merge exponent and sign-mantissa components back to original tensor.

    Args:
        exp: Exponent uint8 tensor (pinned CPU or CUDA)
        sm: Sign-mantissa uint8 tensor (CUDA)
        size: Output tensor size (list/tuple)
        stride: Output tensor stride (list/tuple)
        offset: Storage offset
        dtype: Output dtype (torch.bfloat16 or torch.float32)
        stream_ptr: CUDA stream pointer

    Returns:
        Reconstructed tensor
    """
    if not _AVAILABLE:
        raise RuntimeError("float_split_stride_pin CUDA extension not available")
    return _ext.merge(exp, sm, size, stride, offset, dtype, stream_ptr)


def acquire_pin(numel, dtype):
    """
    Acquire a pinned memory tensor from the pool.

    Args:
        numel: Number of elements
        dtype: Tensor dtype

    Returns:
        Pinned memory tensor
    """
    if not _AVAILABLE:
        raise RuntimeError("float_split_stride_pin CUDA extension not available")
    return _ext.acquire_pin(numel, dtype)


def release_pin(t):
    """
    Release a pinned memory tensor back to the pool.

    Args:
        t: Tensor to release
    """
    if not _AVAILABLE:
        raise RuntimeError("float_split_stride_pin CUDA extension not available")
    return _ext.release_pin(t)


def release_cuda(t):
    """
    Release a CUDA tensor back to the pool.

    Args:
        t: Tensor to release
    """
    if not _AVAILABLE:
        raise RuntimeError("float_split_stride_pin CUDA extension not available")
    return _ext.release_cuda(t)
