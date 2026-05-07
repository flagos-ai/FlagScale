from .platform_base import PlatformBase
from .platform_manager import get_platform, set_platform
from .platform_register import register_platforms

register_platforms()

__all__ = ["PlatformBase", "get_platform", "set_platform"]
