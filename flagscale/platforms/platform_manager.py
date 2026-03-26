import os

from .platform_register import PLATFORMS

_current_platform = None


def get_platform():
    """Get the current platform instance.

    Detection priority:
    1. Environment variable FS_PLATFORM (e.g. "cuda", "npu", "musa")
    2. Auto-detect: cuda > npu > musa
    """
    global _current_platform
    if _current_platform is not None:
        return _current_platform

    # 1. Check environment variable override
    env_platform = os.environ.get("FS_PLATFORM", "").lower()
    if env_platform:
        if env_platform not in PLATFORMS:
            raise ValueError(
                f"FS_PLATFORM='{env_platform}' is not available. "
                f"Registered platforms: {list(PLATFORMS.keys())}"
            )
        _current_platform = PLATFORMS[env_platform]
        return _current_platform

    # 2. Auto-detect in priority order
    for name in ("cuda", "npu", "musa"):
        if name in PLATFORMS:
            _current_platform = PLATFORMS[name]
            return _current_platform

    raise RuntimeError(
        "No available platform detected. "
        "Ensure CUDA, NPU (torch_npu), or MUSA (torch_musa) is installed."
    )


def set_platform(platform_obj):
    """Manually set the current platform."""
    global _current_platform
    _current_platform = platform_obj
