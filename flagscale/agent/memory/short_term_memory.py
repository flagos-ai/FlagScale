"""Short-term memory implementation with fast in-memory access."""

import logging

from typing import Any, Iterable, List, Union

from .base import MemoryBase, Msg

logger = logging.getLogger(__name__)


class InMemoryMemory(MemoryBase):
    """In-memory short-term memory implementation.

    This class references AgentScope's InMemoryMemory implementation, providing:
    - Fast message storage and access
    - Index-based deletion operations
    - Duplicate control
    - State persistence support
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize short-term memory.

        Args:
            max_size (int): Maximum number of messages to store, default 1000
        """
        super().__init__()
        self.content: List[Msg] = []
        self.max_size = max_size
        logger.info(f"InMemoryMemory initialized, max capacity: {max_size}")

    def state_dict(self) -> dict:
        """Convert current memory to dictionary format.

        Returns:
            dict: Dictionary containing memory content and configuration
        """
        return {"content": [msg.to_dict() for msg in self.content], "max_size": self.max_size}

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load memory state from dictionary.

        Args:
            state_dict (dict): State dictionary
            strict (bool): If True, raise error when keys are missing
        """
        self.content = []
        for data in state_dict.get("content", []):
            # Remove possible type field (for compatibility)
            data.pop("type", None)
            self.content.append(Msg.from_dict(data))

        self.max_size = state_dict.get("max_size", 1000)
        logger.info(f"Loaded {len(self.content)} messages from state dictionary")

    async def size(self) -> int:
        """Get memory size.

        Returns:
            int: Current number of stored messages
        """
        return len(self.content)

    async def retrieve(self, query: str = "", limit: int = 10) -> List[Msg]:
        """Retrieve messages from memory.

        Args:
            query (str): Retrieval query (simple text matching)
            limit (int): Maximum number of results to return

        Returns:
            List[Msg]: List of matching messages
        """
        if not query:
            # If no query, return recent messages
            return self.content[-limit:] if limit > 0 else self.content

        # Simple text matching retrieval
        results = []
        query_lower = query.lower()

        for msg in self.content:
            content_str = msg.content
            if isinstance(msg.content, list):
                # Handle structured content
                content_str = " ".join(
                    str(block.get("text", "")) if isinstance(block, dict) else str(block)
                    for block in msg.content
                )

            if query_lower in str(content_str).lower():
                results.append(msg)

        # Return recent matching results
        return results[-limit:] if limit > 0 else results

    async def delete(self, index: Union[Iterable, int]) -> None:
        """Delete messages at specified indices.

        Args:
            index: Index or list of indices to delete

        Raises:
            IndexError: If index does not exist
        """
        if isinstance(index, int):
            index = [index]

        # Check for invalid indices
        invalid_index = [i for i in index if i < 0 or i >= len(self.content)]

        if invalid_index:
            raise IndexError(f"Index {invalid_index} does not exist.")

        # Delete messages at specified indices
        self.content = [msg for idx, msg in enumerate(self.content) if idx not in index]
        logger.debug(f"Deleted {len(list(index))} messages")

    async def add(
        self, memories: Union[List[Msg], Msg, None], allow_duplicates: bool = False
    ) -> None:
        """Add messages to memory.

        Args:
            memories: Messages or list of messages to add
            allow_duplicates (bool): Whether to allow adding duplicate messages (same id)

        Raises:
            TypeError: If message type is incorrect
        """
        if memories is None:
            return

        if isinstance(memories, Msg):
            memories = [memories]

        if not isinstance(memories, list):
            raise TypeError(
                f"memories should be a list of Msg or a single Msg, " f"but got {type(memories)}."
            )

        for msg in memories:
            if not isinstance(msg, Msg):
                raise TypeError(
                    f"memories should be a list of Msg or a single Msg, " f"but got {type(msg)}."
                )

        # Deduplication
        if not allow_duplicates:
            existing_ids = {msg.id for msg in self.content}
            memories = [msg for msg in memories if msg.id not in existing_ids]

        self.content.extend(memories)

        # If exceeds max capacity, remove oldest messages
        while len(self.content) > self.max_size:
            removed = self.content.pop(0)
            logger.debug(f"Memory full, removing oldest message: {removed.id}")

        logger.debug(f"Added {len(memories)} messages to short-term memory")

    async def get_memory(self, recent: int = 0) -> List[Msg]:
        """Get memory content.

        Args:
            recent (int): If greater than 0, only return the most recent N messages

        Returns:
            List[Msg]: List of messages
        """
        if recent > 0:
            return self.content[-recent:]
        return self.content.copy()

    async def clear(self) -> None:
        """Clear memory content."""
        count = len(self.content)
        self.content = []
        logger.info(f"Cleared short-term memory, deleted {count} messages")

    async def get_recent_messages(self, limit: int = 20) -> List[Msg]:
        """Get recent messages.

        Args:
            limit (int): Number of messages to return

        Returns:
            List[Msg]: List of recent messages
        """
        return self.content[-limit:] if limit > 0 else self.content

    async def search(self, query: str, limit: int = 10) -> List[Msg]:
        """Search messages (similar to retrieve, for compatibility).

        Args:
            query (str): Search query
            limit (int): Maximum number of results to return

        Returns:
            List[Msg]: List of matching messages
        """
        return await self.retrieve(query, limit)

    async def update(self, message_id: str, content: str = None, metadata: dict = None) -> bool:
        """Update message with specified ID.

        Args:
            message_id (str): Message ID
            content (str): New content (optional)
            metadata (dict): New metadata (optional)

        Returns:
            bool: Whether update was successful
        """
        for msg in self.content:
            if msg.id == message_id:
                if content is not None:
                    msg.content = content
                if metadata is not None:
                    msg.metadata = metadata
                logger.debug(f"Updated message: {message_id}")
                return True

        logger.warning(f"Message not found: {message_id}")
        return False

    async def get_by_id(self, message_id: str) -> Union[Msg, None]:
        """Get message by ID.

        Args:
            message_id (str): Message ID

        Returns:
            Union[Msg, None]: Message object or None
        """
        for msg in self.content:
            if msg.id == message_id:
                return msg
        return None
