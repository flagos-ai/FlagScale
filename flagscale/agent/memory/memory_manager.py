"""Memory manager for unified management of short-term and long-term memory."""

import logging

from typing import Any, Optional

from .long_term_memory import Mem0LongTermMemory
from .memory_config import (
    LongTermMemoryConfig,
    MemoryModuleConfig,
    ModelFactory,
    ShortTermMemoryConfig,
)
from .short_term_memory import InMemoryMemory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory manager for unified management of short-term and long-term memory.

    This class provides a simplified interface to manage Agent's memory system,
    adapted for FlagScale usage scenarios.
    """

    def __init__(
        self,
        config: Optional[MemoryModuleConfig] = None,
        short_term_memory: Optional[InMemoryMemory] = None,
        long_term_memory: Optional[Mem0LongTermMemory] = None,
        **kwargs: Any,
    ):
        """Initialize memory manager.

        Args:
            config: Memory module configuration (optional)
            short_term_memory: Short-term memory instance (optional, used directly if provided)
            long_term_memory: Long-term memory instance (optional, used directly if provided)
            **kwargs: Additional configuration parameters for creating memory instances
        """
        self.config = config
        # Internal status for diagnostics
        self._status = {
            "short_term": {"enabled": False, "reason": None},
            "long_term": {
                "enabled": False,
                "reason": None,
                "errors": [],
                "llm_created": False,
                "embedding_created": False,
            },
        }

        # Initialize short-term memory
        if short_term_memory is not None:
            self.short_term_memory = short_term_memory
            self._status["short_term"]["enabled"] = True
        elif config and config.enable_short_term:
            self.short_term_memory = InMemoryMemory(max_size=config.short_term.max_size)
            self._status["short_term"]["enabled"] = True
        elif kwargs.get("enable_short_term", True):
            max_size = kwargs.get("max_size", 1000)
            self.short_term_memory = InMemoryMemory(max_size=max_size)
            self._status["short_term"]["enabled"] = True
        else:
            self.short_term_memory = None
            self._status["short_term"]["reason"] = "disabled_by_config"

        # Initialize long-term memory
        if long_term_memory is not None:
            self.long_term_memory = long_term_memory
            self._status["long_term"]["enabled"] = True
        elif config and config.enable_long_term:
            # Try to get models from config or kwargs
            llm_model = kwargs.get("llm_model")
            embedding_model = kwargs.get("embedding_model")

            # If models not provided, try to create from config
            if llm_model is None and config.long_term.llm:
                try:
                    llm_model = ModelFactory.create_llm_model(config.long_term.llm)
                    if llm_model:
                        logger.info("Successfully created LLM model from config")
                        self._status["long_term"]["llm_created"] = True
                except Exception as e:
                    logger.warning(f"Failed to create LLM model from config: {e}")
                    self._status["long_term"]["errors"].append(str(e))
                    self._status["long_term"]["reason"] = "llm_model_creation_failed"

            if embedding_model is None and config.long_term.embedding:
                try:
                    embedding_model = ModelFactory.create_embedding_model(
                        config.long_term.embedding
                    )
                    if embedding_model:
                        logger.info("Successfully created Embedding model from config")
                        self._status["long_term"]["embedding_created"] = True
                except Exception as e:
                    logger.warning(f"Failed to create Embedding model from config: {e}")
                    self._status["long_term"]["errors"].append(str(e))
                    # keep first reason if already set by llm failure
                    if not self._status["long_term"]["reason"]:
                        self._status["long_term"]["reason"] = "embedding_model_creation_failed"

            if llm_model is None or embedding_model is None:
                logger.warning(
                    "Long-term memory requires llm_model and embedding_model. "
                    "Not provided and cannot be created from config, long-term memory will be unavailable"
                )
                self.long_term_memory = None
                missing = []
                if llm_model is None:
                    missing.append("llm_model")
                if embedding_model is None:
                    missing.append("embedding_model")
                if not self._status["long_term"]["reason"]:
                    self._status["long_term"]["reason"] = f"missing_models: {', '.join(missing)}"
            else:
                # Get embedding dimensions from config if available
                embedding_dimensions = None
                if config.long_term.embedding:
                    embedding_dimensions = config.long_term.embedding.dimensions

                self.long_term_memory = Mem0LongTermMemory(
                    agent_name=config.long_term.agent_name,
                    user_name=config.long_term.user_name,
                    run_name=config.long_term.run_name,
                    llm_model=llm_model,
                    embedding_model=embedding_model,
                    embedding_dimensions=embedding_dimensions,
                    path=config.long_term.vector_store_path,
                    vector_store_type=config.long_term.vector_store_type,
                    collection_name=config.long_term.collection_name,
                    on_disk=config.long_term.on_disk,
                    default_memory_type=config.long_term.default_memory_type,
                )
                self._status["long_term"]["enabled"] = True
        elif kwargs.get("enable_long_term", False):
            llm_model = kwargs.get("llm_model")
            embedding_model = kwargs.get("embedding_model")

            if llm_model is None or embedding_model is None:
                logger.warning("Long-term memory requires llm_model and embedding_model")
                self.long_term_memory = None
                missing = []
                if llm_model is None:
                    missing.append("llm_model")
                if embedding_model is None:
                    missing.append("embedding_model")
                self._status["long_term"]["reason"] = f"missing_models: {', '.join(missing)}"
            else:
                self.long_term_memory = Mem0LongTermMemory(
                    agent_name=kwargs.get("agent_name"),
                    user_name=kwargs.get("user_name"),
                    run_name=kwargs.get("run_name"),
                    llm_model=llm_model,
                    embedding_model=embedding_model,
                    embedding_dimensions=kwargs.get("embedding_dimensions"),
                    path=kwargs.get("vector_store_path", "./qdrant_storage"),
                    vector_store_type=kwargs.get("vector_store_type", "qdrant"),
                    collection_name=kwargs.get("collection_name", "memory_collection"),
                    on_disk=kwargs.get("on_disk", True),
                    default_memory_type=kwargs.get("default_memory_type"),
                )
                self._status["long_term"]["enabled"] = True
        else:
            self.long_term_memory = None
            self._status["long_term"]["reason"] = "disabled_by_config"

        logger.info(
            f"MemoryManager initialized - "
            f"Short-term memory: {'enabled' if self.short_term_memory else 'disabled'}, "
            f"Long-term memory: {'enabled' if self.long_term_memory else 'disabled'}"
        )

    @classmethod
    def from_config(
        cls,
        config: MemoryModuleConfig,
        llm_model: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        auto_create_models: bool = True,
    ) -> "MemoryManager":
        """Create memory manager from configuration.

        Args:
            config: Memory module configuration
            llm_model: LLM model (optional, will try to create from config if not provided and auto_create_models=True)
            embedding_model: Embedding model (optional, will try to create from config if not provided and auto_create_models=True)
            auto_create_models: Whether to automatically create models from config (default True)

        Returns:
            MemoryManager: Memory manager instance
        """
        # If auto-create enabled and models not provided, try to create from config
        if auto_create_models:
            if llm_model is None and config.long_term.llm:
                try:
                    llm_model = ModelFactory.create_llm_model(config.long_term.llm)
                except Exception as e:
                    logger.warning(f"Failed to create LLM model from config: {e}")

            if embedding_model is None and config.long_term.embedding:
                try:
                    embedding_model = ModelFactory.create_embedding_model(
                        config.long_term.embedding
                    )
                except Exception as e:
                    logger.warning(f"Failed to create Embedding model from config: {e}")

        return cls(config=config, llm_model=llm_model, embedding_model=embedding_model)

    def get_short_term_memory(self) -> Optional[InMemoryMemory]:
        """Get short-term memory instance."""
        return self.short_term_memory

    def get_long_term_memory(self) -> Optional[Mem0LongTermMemory]:
        """Get long-term memory instance."""
        return self.long_term_memory

    def has_short_term(self) -> bool:
        """Check if short-term memory is enabled."""
        return self.short_term_memory is not None

    def has_long_term(self) -> bool:
        """Check if long-term memory is enabled."""
        return self.long_term_memory is not None

    def get_status(self) -> dict:
        """Return diagnostics status for memory manager.
        Includes enable flags and reasons/errors when long-term memory is unavailable.
        """
        return self._status
