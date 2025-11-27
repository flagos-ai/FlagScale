"""Base class definitions and message data structures for memory system."""

import uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class Msg:
    """Message class compatible with AgentScope's Msg structure.

    This is the core data structure in the memory system for storing and passing messages.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for the message"""

    role: str = "assistant"
    """Message role, can be 'user', 'assistant', or 'system'"""

    content: Union[str, List[Dict[str, Any]]] = ""
    """Message content, can be a string or a list of structured content blocks"""

    name: Optional[str] = None
    """Message sender name (optional)"""

    metadata: Optional[Dict[str, Any]] = None
    """Message metadata (optional)"""

    timestamp: datetime = field(default_factory=datetime.now)
    """Message timestamp"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary representation of the message
        """
        data = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

        if self.name is not None:
            data["name"] = self.name

        if self.metadata is not None:
            data["metadata"] = self.metadata

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Msg":
        """Create message instance from dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing message data

        Returns:
            Msg: Message instance
        """
        # Handle timestamp
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            name=data.get("name"),
            metadata=data.get("metadata"),
            timestamp=timestamp,
        )

    def __str__(self) -> str:
        """Return string representation of the message."""
        content_str = self.content
        if isinstance(self.content, list):
            # If structured content, extract text parts
            text_parts = [
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in self.content
            ]
            content_str = " ".join(text_parts)

        return f"Msg(role={self.role}, content={content_str[:50]}...)"


@dataclass
class TextBlock:
    """Text content block."""

    text: str = ""

    def __init__(self, text: str = ""):
        """Initialize text block.

        Args:
            text (str): Text content
        """
        self.text = text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBlock":
        """Create instance from dictionary."""
        return cls(text=data.get("text", ""))


@dataclass
class ToolResponse:
    """Tool response class for encapsulating tool function return results."""

    content: Union[str, List[TextBlock], List[Dict[str, Any]]] = field(default_factory=list)

    def __init__(self, content: Union[str, List[TextBlock], List[Dict[str, Any]]] = None):
        """Initialize tool response.

        Args:
            content: Response content, can be a string, list of TextBlocks, or list of dictionaries
        """
        if content is None:
            content = []

        if isinstance(content, str):
            content = [TextBlock(text=content)]
        elif isinstance(content, list):
            # Convert dicts to TextBlocks
            converted = []
            for item in content:
                if isinstance(item, TextBlock):
                    converted.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        converted.append(TextBlock(text=item.get("text", "")))
                    else:
                        # Keep other types of blocks unchanged
                        converted.append(item)
                else:
                    # Convert to text block
                    converted.append(TextBlock(text=str(item)))
            content = converted

        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        if isinstance(self.content, list):
            return {
                "content": [
                    item.to_dict() if isinstance(item, TextBlock) else item for item in self.content
                ]
            }
        return {"content": self.content}

    def get_text(self) -> str:
        """Get plain text content of the response.

        Returns:
            str: Plain text content
        """
        if isinstance(self.content, str):
            return self.content

        if isinstance(self.content, list):
            text_parts = []
            for item in self.content:
                if isinstance(item, TextBlock):
                    text_parts.append(item.text)
                elif isinstance(item, dict):
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            return "\n".join(text_parts)

        return str(self.content)


class StateModule(ABC):
    """State module base class providing state persistence functionality."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Get module state dictionary.

        Returns:
            dict: State dictionary
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load module state from state dictionary.

        Args:
            state_dict (dict): State dictionary
            strict (bool): If True, raise error when keys are missing in state dictionary
        """
        pass


class MemoryBase(StateModule):
    """Base class for memory system, defining common interface for all memory types.

    This base class references AgentScope's MemoryBase design to ensure architectural consistency.
    """

    @abstractmethod
    async def add(self, *args: Any, **kwargs: Any) -> None:
        """Add items to memory.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    @abstractmethod
    async def delete(self, *args: Any, **kwargs: Any) -> None:
        """Delete items from memory.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        pass

    @abstractmethod
    async def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        """Retrieve items from memory.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Any: Retrieved results
        """
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get memory size.

        Returns:
            int: Number of items in memory
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear memory content."""
        pass

    @abstractmethod
    async def get_memory(self, *args: Any, **kwargs: Any) -> List[Msg]:
        """Get memory content.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            List[Msg]: List of messages
        """
        pass


class LongTermMemoryBase(StateModule):
    """Long-term memory base class, defining dedicated interface for long-term memory.

    Long-term memory is a temporal memory management system that supports:
    - record/retrieve: Developer interface for actively managing memory in code
    - record_to_memory/retrieve_from_memory: Tool interface for agent autonomous calls
    """

    @abstractmethod
    async def record(self, msgs: Union[Msg, List[Msg], None], **kwargs: Any) -> None:
        """Developer interface: Record messages to long-term memory.

        This method is called by developers in code, e.g., automatically record after each conversation.

        Args:
            msgs: Messages or list of messages to record
            **kwargs: Additional keyword arguments
        """
        pass

    @abstractmethod
    async def retrieve(self, msg: Union[Msg, List[Msg], None], **kwargs: Any) -> str:
        """Developer interface: Retrieve information from long-term memory.

        This method is called by developers in code, e.g., retrieve related memories before each response.

        Args:
            msg: Messages or list of messages used for retrieval
            **kwargs: Additional keyword arguments

        Returns:
            str: Retrieved memory content
        """
        pass

    @abstractmethod
    async def record_to_memory(self, thinking: str, content: List[str], **kwargs: Any) -> Any:
        """Tool interface: Record important information to long-term memory.

        This method is wrapped as a tool function for agent autonomous calls.
        Agent can actively decide when to record information.

        Args:
            thinking (str): Agent's thinking process and reasoning
            content (List[str]): List of content to record
            **kwargs: Additional keyword arguments

        Returns:
            Tool response object
        """
        pass

    @abstractmethod
    async def retrieve_from_memory(self, keywords: List[str], **kwargs: Any) -> Any:
        """Tool interface: Retrieve information from long-term memory.

        This method is wrapped as a tool function for agent autonomous calls.
        Agent can actively decide when to retrieve memories.

        Args:
            keywords (List[str]): List of retrieval keywords
            **kwargs: Additional keyword arguments

        Returns:
            Tool response object
        """
        pass
