"""Python access to the optional MemRift split/merge CUDA kernels."""

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
_load_error: BaseException | None = None
_loaded = False


def _load() -> ModuleType | None:
    """Load the extension after package initialization has completed."""
    global _ext, _load_error, _loaded
    if _loaded:
        return _ext
    _loaded = True
    try:
        candidate = import_module("flagscale.compress.memrift._float_split_stride_pin")
    except (ImportError, OSError) as exc:
        _load_error = exc
        return None
    if _valid(candidate):
        _ext = candidate
    else:
        _load_error = RuntimeError("MemRift CUDA extension is missing one or more required symbols")
    return _ext


def is_available() -> bool:
    """Return whether a compatible CUDA extension was imported."""
    return _load() is not None


def _require() -> ModuleType:
    ext = _load()
    if ext is None:
        raise RuntimeError(
            "MemRift CUDA kernels are unavailable; build flagscale/compress/memrift/csrc first"
        ) from _load_error
    return ext


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
