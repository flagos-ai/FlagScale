"""FlagScale Agent Module

Modules for collaboration, tool matching, and memory management.
"""

# Export collaboration module
from .collaboration import Collaborator

# Export memory module
from .memory import MemoryManager, MemoryModuleConfig, create_memory_toolkit, register_memory_tools

# Export tool matching module
from .tool_match import ToolMatcher, ToolRegistry

__all__ = [
    # Collaboration
    "Collaborator",
    # Tool matching
    "ToolRegistry",
    "ToolMatcher",
    # Memory management
    "MemoryManager",
    "register_memory_tools",
    "create_memory_toolkit",
    "MemoryModuleConfig",
]
