"""FlagScale Agent Memory Module

Memory management module providing short-term and long-term memory capabilities.
"""

from .base import LongTermMemoryBase, MemoryBase, Msg, StateModule, TextBlock, ToolResponse
from .long_term_memory import Mem0LongTermMemory
from .memory_config import (
    EmbeddingConfig,
    LLMConfig,
    LongTermMemoryConfig,
    MemoryModuleConfig,
    ModelFactory,
    ShortTermMemoryConfig,
    create_config_file,
    create_default_config,
    get_config_from_env,
    load_config,
    load_config_from_file,
)
from .memory_manager import MemoryManager
from .memory_tools import MemoryToolkit, create_memory_toolkit, register_memory_tools
from .short_term_memory import InMemoryMemory

__all__ = [
    "MemoryManager",
    "register_memory_tools",
    "create_memory_toolkit",
    "MemoryBase",
    "LongTermMemoryBase",
    "StateModule",
    "Msg",
    "TextBlock",
    "ToolResponse",
    "InMemoryMemory",
    "Mem0LongTermMemory",
    "MemoryToolkit",
    "MemoryModuleConfig",
    "ShortTermMemoryConfig",
    "LongTermMemoryConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "create_default_config",
    "load_config_from_file",
    "load_config",
    "create_config_file",
    "get_config_from_env",
    "ModelFactory",
]

__version__ = "1.0.0"
