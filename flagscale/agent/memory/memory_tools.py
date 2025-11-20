"""Memory tool functions for agent calls."""

import logging

from typing import Any, Dict, List, Optional

from .base import TextBlock, ToolResponse
from .long_term_memory import Mem0LongTermMemory
from .short_term_memory import InMemoryMemory

logger = logging.getLogger(__name__)


class MemoryToolkit:
    """Memory toolkit that wraps memory operations as tool functions.

    This class provides a set of tool functions that can be registered to agent's toolkit,
    allowing agents to autonomously decide when to use the memory system.
    """

    def __init__(
        self, short_term_memory: InMemoryMemory = None, long_term_memory: Mem0LongTermMemory = None
    ):
        """Initialize memory toolkit.

        Args:
            short_term_memory: Short-term memory instance
            long_term_memory: Long-term memory instance
        """
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        logger.info("MemoryToolkit initialized")

    async def record_to_long_term_memory(
        self, thinking: str, content: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        """Tool function: Record important information to long-term memory.

        Use this function to record important information that needs to be saved long-term,
        which can be retrieved and used in the future.
        Use cases:
        - User's important preferences and habits
        - Key facts and knowledge
        - Tasks and goals that need to be remembered long-term

        Args:
            thinking (str): Your thinking and reasoning about the content to record,
                            explaining why this information is important
            content (List[str]): List of content to record,
                                each item should be specific and clear information
            **kwargs: Additional metadata, such as importance, category, etc.

        Returns:
            Dict[str, Any]: Dictionary containing operation results
                - success (bool): Whether the operation was successful
                - message (str): Result message
                - content (str): Summary of recorded content

        Example:
            >>> result = await record_to_long_term_memory(
            ...     thinking="User explicitly stated they like coffee in the morning",
            ...     content=["User drinks black coffee at 7 AM every morning", "No sugar or milk"],
            ...     importance="high",
            ...     category="preferences"
            ... )
        """
        if self.long_term_memory is None:
            return {"success": False, "message": "Long-term memory not initialized", "content": ""}

        try:
            # Call long-term memory tool interface
            tool_response = await self.long_term_memory.record_to_memory(
                thinking=thinking, content=content, **kwargs
            )

            # Extract response text
            result_text = tool_response.get_text()

            return {
                "success": True,
                "message": "Successfully recorded to long-term memory",
                "content": result_text,
            }

        except Exception as e:
            logger.error(f"Failed to record to long-term memory: {e}")
            return {"success": False, "message": f"Recording failed: {str(e)}", "content": ""}

    async def retrieve_from_long_term_memory(
        self, keywords: List[str], limit: int = 5, **kwargs: Any
    ) -> Dict[str, Any]:
        """Tool function: Retrieve information from long-term memory.

        Use this function to retrieve related information from long-term memory.
        The system will use semantic search to find memories most relevant to the keywords.

        Args:
            keywords (List[str]): List of retrieval keywords,
                                 should be specific and clear words, such as names, places, events, etc.
            limit (int): Maximum number of results to return, default 5
            **kwargs: Additional retrieval parameters

        Returns:
            Dict[str, Any]: Dictionary containing retrieval results
                - success (bool): Whether the operation was successful
                - message (str): Result message
                - results (List[str]): List of retrieved memories
                - count (int): Number of results

        Example:
            >>> result = await retrieve_from_long_term_memory(
            ...     keywords=["coffee", "breakfast habits"],
            ...     limit=3
            ... )
        """
        if self.long_term_memory is None:
            return {
                "success": False,
                "message": "Long-term memory not initialized",
                "results": [],
                "count": 0,
            }

        try:
            # Call long-term memory tool interface
            tool_response = await self.long_term_memory.retrieve_from_memory(
                keywords=keywords, limit=limit, **kwargs
            )

            # Extract response text
            result_text = tool_response.get_text()

            # Split results
            results = [r.strip() for r in result_text.split("\n") if r.strip()]

            return {
                "success": True,
                "message": f"Found {len(results)} related memories",
                "results": results,
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"Failed to retrieve from long-term memory: {e}")
            return {
                "success": False,
                "message": f"Retrieval failed: {str(e)}",
                "results": [],
                "count": 0,
            }

    async def search_short_term_memory(
        self, query: str, limit: int = 10, **kwargs: Any
    ) -> Dict[str, Any]:
        """Tool function: Search short-term memory.

        Search for information in the current conversation's short-term memory.
        Suitable for finding recent conversation content.

        Args:
            query (str): Search query
            limit (int): Maximum number of results to return, default 10
            **kwargs: Additional search parameters

        Returns:
            Dict[str, Any]: Dictionary containing search results
                - success (bool): Whether the operation was successful
                - message (str): Result message
                - results (List[Dict]): List of found messages
                - count (int): Number of results
        """
        if self.short_term_memory is None:
            return {
                "success": False,
                "message": "Short-term memory not initialized",
                "results": [],
                "count": 0,
            }

        try:
            # Search short-term memory
            messages = await self.short_term_memory.search(query=query, limit=limit)

            # Convert to dictionary format
            results = [
                {
                    "role": msg.role,
                    "content": str(msg.content),
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in messages
            ]

            return {
                "success": True,
                "message": f"Found {len(results)} matching messages",
                "results": results,
                "count": len(results),
            }

        except Exception as e:
            logger.error(f"Failed to search short-term memory: {e}")
            return {
                "success": False,
                "message": f"Search failed: {str(e)}",
                "results": [],
                "count": 0,
            }

    async def get_recent_conversation(self, limit: int = 10, **kwargs: Any) -> Dict[str, Any]:
        """Tool function: Get recent conversation records.

        Get recent conversation history for reviewing previous exchanges.

        Args:
            limit (int): Number of messages to return, default 10
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Dictionary containing conversation records
                - success (bool): Whether the operation was successful
                - message (str): Result message
                - conversation (List[Dict]): List of conversation messages
                - count (int): Number of messages
        """
        if self.short_term_memory is None:
            return {
                "success": False,
                "message": "Short-term memory not initialized",
                "conversation": [],
                "count": 0,
            }

        try:
            # Get recent messages
            messages = await self.short_term_memory.get_recent_messages(limit=limit)

            # Convert to conversation format
            conversation = [
                {
                    "role": msg.role,
                    "content": str(msg.content),
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in messages
            ]

            return {
                "success": True,
                "message": f"Retrieved {len(conversation)} recent conversations",
                "conversation": conversation,
                "count": len(conversation),
            }

        except Exception as e:
            logger.error(f"Failed to get recent conversation: {e}")
            return {
                "success": False,
                "message": f"Retrieval failed: {str(e)}",
                "conversation": [],
                "count": 0,
            }


def create_memory_tool_functions(toolkit: MemoryToolkit) -> Dict[str, callable]:
    """Create memory tool function dictionary.

    This function returns a dictionary containing all tool functions that can be registered to agent toolkit.

    Args:
        toolkit (MemoryToolkit): Memory toolkit instance

    Returns:
        Dict[str, callable]: Tool function dictionary, keys are function names, values are function objects

    Example:
        >>> toolkit = MemoryToolkit(short_term, long_term)
        >>> tools = create_memory_tool_functions(toolkit)
        >>> # Register to agent
        >>> for name, func in tools.items():
        >>>     agent.toolkit.register_tool_function(func, name=name)
    """
    return {
        "record_to_long_term_memory": toolkit.record_to_long_term_memory,
        "retrieve_from_long_term_memory": toolkit.retrieve_from_long_term_memory,
        "search_short_term_memory": toolkit.search_short_term_memory,
        "get_recent_conversation": toolkit.get_recent_conversation,
    }


# Tool Registration Functions


def register_memory_tools(
    tool_registry: Any, memory_manager: Any, category: str = "memory"
) -> List[str]:
    """Register memory functionality to ToolRegistry.

    Args:
        tool_registry: ToolRegistry instance
        memory_manager: MemoryManager instance
        category: Tool category, default "memory"

    Returns:
        List[str]: List of registered tool names

    Example:
        >>> from flagscale.agent.tool_match import ToolRegistry
        >>> from flagscale.agent.memory import MemoryManager
        >>>
        >>> registry = ToolRegistry()
        >>> memory = MemoryManager(...)
        >>>
        >>> registered_tools = register_memory_tools(registry, memory)
        >>> print(f"Registered {len(registered_tools)} memory tools")
    """
    # Import here to avoid circular import
    from .memory_manager import MemoryManager

    registered_tools = []

    # Create MemoryToolkit
    toolkit = MemoryToolkit(
        short_term_memory=memory_manager.get_short_term_memory(),
        long_term_memory=memory_manager.get_long_term_memory(),
    )

    # Define tool list
    tools = []

    # Record to long-term memory
    if memory_manager.has_long_term():
        tools.append(
            {
                "function": {
                    "name": "record_to_long_term_memory",
                    "description": (
                        "Record important information to long-term memory. Use this tool to record "
                        "information that needs to be saved long-term, such as user preferences, "
                        "key facts, important tasks, etc."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thinking": {
                                "type": "string",
                                "description": "Thinking and reasoning about the content to record, explaining why this information is important",
                            },
                            "content": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of content to record, each item should be specific and clear information",
                            },
                        },
                        "required": ["content"],
                    },
                },
                "func": toolkit.record_to_long_term_memory,
            }
        )

    # Retrieve from long-term memory
    if memory_manager.has_long_term():
        tools.append(
            {
                "function": {
                    "name": "retrieve_from_long_term_memory",
                    "description": (
                        "Retrieve related information from long-term memory. Uses semantic search "
                        "to find memories most relevant to the keywords."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of retrieval keywords, should be specific and clear words",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5,
                            },
                        },
                        "required": ["keywords"],
                    },
                },
                "func": toolkit.retrieve_from_long_term_memory,
            }
        )

    # Search short-term memory
    if memory_manager.has_short_term():
        tools.append(
            {
                "function": {
                    "name": "search_short_term_memory",
                    "description": (
                        "Search for information in the current conversation's short-term memory. "
                        "Suitable for finding recent conversation content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
                "func": toolkit.search_short_term_memory,
            }
        )

    # Get recent conversation
    if memory_manager.has_short_term():
        tools.append(
            {
                "function": {
                    "name": "get_recent_conversation",
                    "description": (
                        "Get recent conversation records. Used to review previous exchange content."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of messages to return",
                                "default": 10,
                            }
                        },
                    },
                },
                "func": toolkit.get_recent_conversation,
            }
        )

    # Register tools to ToolRegistry
    for tool in tools:
        try:
            # Ensure tool dictionary contains function reference
            tool_dict = {
                "function": tool["function"],
                "func": tool["func"],  # Store actual function object
            }
            tool_registry.register_tool(tool_dict, category=category)
            registered_tools.append(tool["function"]["name"])
            logger.debug(f"Registered memory tool: {tool['function']['name']}")
        except Exception as e:
            logger.error(f"Failed to register tool {tool['function']['name']}: {e}")

    logger.info(
        f"Registered {len(registered_tools)} memory tools to ToolRegistry "
        f"(category: {category})"
    )

    return registered_tools


def create_memory_toolkit(memory_manager: Any) -> MemoryToolkit:
    """Create MemoryToolkit instance.

    Args:
        memory_manager: MemoryManager instance

    Returns:
        MemoryToolkit: Memory toolkit instance
    """
    return MemoryToolkit(
        short_term_memory=memory_manager.get_short_term_memory(),
        long_term_memory=memory_manager.get_long_term_memory(),
    )
