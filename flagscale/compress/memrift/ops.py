"""Python access to the optional MemRift split/merge CUDA kernels.

The MemRift-scoped extension is preferred.  The legacy package is retained as
a runtime fallback for environments that already built the old extension.
"""

from importlib import import_module
from types import ModuleType

_REQUIRED_SYMBOLS = (
    "split",
    "split_copy",
    "merge",
    "acquire_pin",
    "release_pin",
    "release_cuda",
)


def _valid(module: ModuleType | None) -> bool:
    return module is not None and all(hasattr(module, name) for name in _REQUIRED_SYMBOLS)


_ext: ModuleType | None = None
for _name in (
    "flagscale.compress.memrift._float_split_stride_pin",
    "float_split_stride_pin._ext",
    "flagscale.compress.float_split_stride_pin.float_split_stride_pin._ext",
):
    try:
        _candidate = import_module(_name)
    except ImportError:
        continue
    if _valid(_candidate):
        _ext = _candidate
        break


def is_available() -> bool:
    """Return whether a compatible CUDA extension was imported."""
    return _ext is not None


def _require() -> ModuleType:
    if _ext is None:
        raise RuntimeError(
            "MemRift CUDA kernels are unavailable; build flagscale/compress/memrift/csrc first"
        )
    return _ext


def split(*args, **kwargs):
    return _require().split(*args, **kwargs)


def split_copy(*args, **kwargs):
    return _require().split_copy(*args, **kwargs)


def merge(*args, **kwargs):
    return _require().merge(*args, **kwargs)


def acquire_pin(*args, **kwargs):
    return _require().acquire_pin(*args, **kwargs)


def release_pin(*args, **kwargs):
    return _require().release_pin(*args, **kwargs)


def release_cuda(*args, **kwargs):
    return _require().release_cuda(*args, **kwargs)
