# Copyright (c) 2026, BAAI. All rights reserved.

PLATFORMS = {}


def register_platforms() -> None:
    """Register all available platforms."""

    from .platform_cuda import PlatformCUDA

    platform_cuda = PlatformCUDA()
    if platform_cuda.is_available():
        PLATFORMS["cuda"] = platform_cuda

    from .platform_npu import PlatformNPU

    platform_npu = PlatformNPU()
    if platform_npu.is_available():
        PLATFORMS["npu"] = platform_npu

    from .platform_musa import PlatformMUSA

    platform_musa = PlatformMUSA()
    if platform_musa.is_available():
        PLATFORMS["musa"] = platform_musa
