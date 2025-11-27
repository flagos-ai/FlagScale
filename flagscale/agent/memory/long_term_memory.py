"""Long-term memory implementation using mem0 library for vector retrieval and semantic search."""

import asyncio
import json
import logging

from importlib import metadata
from typing import Any, Dict, List, Literal, Optional, Union

from packaging import version

from .base import LongTermMemoryBase, Msg, TextBlock, ToolResponse

logger = logging.getLogger(__name__)

# Try to import mem0
try:
    import mem0

    from mem0.configs.base import MemoryConfig
    from mem0.configs.embeddings.base import BaseEmbedderConfig
    from mem0.configs.llms.base import BaseLlmConfig
    from mem0.embeddings.base import EmbeddingBase
    from mem0.llms.base import LLMBase
    from mem0.vector_stores.configs import VectorStoreConfig

    MEM0_AVAILABLE = True
except ImportError as e:
    MEM0_AVAILABLE = False
    logger.warning("mem0 library not installed, long-term memory will be unavailable")


class CustomLLMAdapter(LLMBase):
    """Custom LLM adapter for integrating custom LLM models into mem0.

    This adapter references AgentScope's AgentScopeLLM implementation,
    can adapt any LLM model that implements the standard interface.
    """

    def __init__(self, config: BaseLlmConfig = None):
        """Initialize LLM adapter.

        Args:
            config: LLM configuration object, should contain 'model' parameter

        Raises:
            ValueError: If required configuration parameters are missing
        """
        super().__init__(config)

        if self.config.model is None:
            raise ValueError("`model` parameter is required")

        self.llm_model = self.config.model
        logger.info(f"CustomLLMAdapter initialized: {type(self.llm_model).__name__}")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Any = None,
        tools: List[Dict] = None,
        tool_choice: str = "auto",
    ) -> str:
        """Generate response using custom LLM model.

        Args:
            messages: List of messages, each containing 'role' and 'content'
            response_format: Response format (not used in this adapter)
            tools: List of tools (not used in this adapter)
            tool_choice: Tool choice method (not used in this adapter)

        Returns:
            str: Generated response text

        Raises:
            RuntimeError: If response generation fails
        """
        try:
            # Check if llm_model has __call__ method
            if not callable(self.llm_model):
                raise ValueError("LLM model must be callable")

            # Async models are not supported in this sync method to avoid event loop conflicts.
            if asyncio.iscoroutinefunction(self.llm_model):
                raise TypeError(
                    "Async LLM models are not supported by this adapter. "
                    "Please provide a synchronous callable."
                )

            # Sync call
            response = self.llm_model(messages, tools=tools)

            # Extract text response
            return self._extract_text_from_response(response)

        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            raise RuntimeError(f"Error generating response with custom LLM model: {str(e)}") from e

    async def _async_generate(
        self, messages: List[Dict[str, str]], tools: List[Dict] = None
    ) -> Any:
        """Helper method for async response generation."""
        return await self.llm_model(messages, tools=tools)

    def _extract_text_from_response(self, response: Any) -> str:
        """Extract text content from response.

        Args:
            response: LLM model response object

        Returns:
            str: Extracted text content
        """
        # If response is already a string, return directly
        if isinstance(response, str):
            return response

        # If response has content attribute
        if hasattr(response, "content"):
            content = response.content

            # content is a string
            if isinstance(content, str):
                return content

            # content is a list (may contain multiple blocks)
            if isinstance(content, list):
                text_parts = []
                thinking_parts = []
                tool_parts = []

                for block in content:
                    # Handle dict-type blocks
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        if block_type == "text":
                            text_parts.append(block.get("text", ""))
                        elif block_type == "thinking":
                            thinking_parts.append(f"[Thinking: {block.get('thinking', '')}]")
                        elif block_type == "tool_use":
                            tool_name = block.get("name")
                            tool_input = block.get("input", {})
                            tool_parts.append(f"[Tool: {tool_name} - {str(tool_input)}]")
                    # Handle object-type blocks
                    elif hasattr(block, "type"):
                        if block.type == "text" and hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif block.type == "thinking" and hasattr(block, "thinking"):
                            thinking_parts.append(f"[Thinking: {block.thinking}]")

                # Combine all parts
                all_parts = thinking_parts + text_parts + tool_parts
                if all_parts:
                    return "\n".join(all_parts)

        # If response has text attribute
        if hasattr(response, "text"):
            return response.text

        # If response has message attribute
        if hasattr(response, "message") and hasattr(response.message, "content"):
            return response.message.content

        # Finally, try to convert to string
        return str(response)


class CustomEmbeddingAdapter(EmbeddingBase):
    """Custom Embedding adapter for integrating custom Embedding models into mem0.

    This adapter references AgentScope's AgentScopeEmbedding implementation,
    can adapt any Embedding model that implements the standard interface.
    """

    def __init__(self, config: BaseEmbedderConfig = None):
        """Initialize Embedding adapter.

        Args:
            config: Embedding configuration object, should contain 'model' parameter

        Raises:
            ValueError: If required configuration parameters are missing
        """
        super().__init__(config)

        if self.config.model is None:
            raise ValueError("`model` parameter is required")

        self.embedding_model = self.config.model
        logger.info(f"CustomEmbeddingAdapter initialized: {type(self.embedding_model).__name__}")

    def embed(
        self, text: Union[str, List[str]], memory_action: Literal["add", "search", "update"] = None
    ) -> List[float]:
        """Generate embeddings using custom Embedding model.

        Args:
            text: Text or list of texts to embed
            memory_action: Memory action type (not used in this adapter)

        Returns:
            List[float]: Embedding vector

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Convert to list format
            text_list = [text] if isinstance(text, str) else text

            # Check if embedding_model has __call__ method
            if not callable(self.embedding_model):
                raise ValueError("Embedding model must be callable")

            # Async models are not supported in this sync method to avoid event loop conflicts.
            if asyncio.iscoroutinefunction(self.embedding_model):
                raise TypeError(
                    "Async embedding models are not supported by this adapter. "
                    "Please provide a synchronous callable."
                )

            # Sync call
            response = self.embedding_model(text_list)

            # Extract embedding vector
            return self._extract_embedding_from_response(response)

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(
                f"Error generating embeddings with custom Embedding model: {str(e)}"
            ) from e

    async def _async_embed(self, text_list: List[str]) -> Any:
        """Helper method for async embedding generation."""
        return await self.embedding_model(text_list)

    def _extract_embedding_from_response(self, response: Any) -> List[float]:
        """Extract embedding vector from response.

        Args:
            response: Embedding model response object

        Returns:
            List[float]: Embedding vector

        Raises:
            ValueError: If embedding vector cannot be extracted
        """
        # If response is already a list, return directly
        if isinstance(response, list):
            # Check if it's an embedding vector (list of numbers)
            if response and isinstance(response[0], (int, float)):
                return response
            # If it's a list of objects, try to extract the first one
            if hasattr(response[0], "embedding"):
                return response[0].embedding

        # If response has embeddings attribute (list)
        if hasattr(response, "embeddings"):
            embeddings = response.embeddings
            if isinstance(embeddings, list) and embeddings:
                # Get first embedding
                first_embedding = embeddings[0]

                # If it's a vector
                if isinstance(first_embedding, list):
                    return first_embedding

                # If it's an object
                if hasattr(first_embedding, "embedding"):
                    return first_embedding.embedding

                # If it's directly a vector
                if isinstance(first_embedding, (int, float)):
                    return embeddings

        # If response has embedding attribute
        if hasattr(response, "embedding"):
            return response.embedding

        # If response has data attribute
        if hasattr(response, "data"):
            data = response.data
            if isinstance(data, list) and data:
                if hasattr(data[0], "embedding"):
                    return data[0].embedding

        raise ValueError(
            f"Cannot extract embedding vector from response. Response type: {type(response)}"
        )


def register_custom_adapters_to_mem0():
    """Register custom adapters to mem0 factory.

    This function needs to be called before using mem0 to register custom LLM and Embedding adapters.
    """
    if not MEM0_AVAILABLE:
        logger.error("mem0 library not available, cannot register adapters")
        return False

    try:
        from mem0.utils.factory import EmbedderFactory, LlmFactory

        # Check mem0 version
        current_version = metadata.version("mem0ai")
        is_mem0_version_low = version.parse(current_version) <= version.parse("0.1.115")

        # Register Embedding adapter
        EmbedderFactory.provider_to_class["custom"] = f"{__name__}.CustomEmbeddingAdapter"

        # Register LLM adapter (use different format based on version)
        if is_mem0_version_low:
            # mem0 version <= 0.1.115
            LlmFactory.provider_to_class["custom"] = f"{__name__}.CustomLLMAdapter"
        else:
            # mem0 version > 0.1.115
            LlmFactory.provider_to_class["custom"] = (f"{__name__}.CustomLLMAdapter", BaseLlmConfig)

        logger.info(f"Custom adapters registered to mem0 (version: {current_version})")
        return True

    except Exception as e:
        logger.error(f"Failed to register custom adapters: {e}")
        return False


class Mem0LongTermMemory(LongTermMemoryBase):
    """Long-term memory implementation using mem0 library.

    This class references AgentScope's Mem0LongTermMemory implementation, providing:
    - Vectorized storage and semantic retrieval
    - Persistent storage (Qdrant)
    - Tool interface for agent calls
    - Developer interface for code management

    Key features:
    - record/retrieve: Used by developers in code
    - record_to_memory/retrieve_from_memory: Wrapped as tools for agent calls
    """

    def __init__(
        self,
        agent_name: str = None,
        user_name: str = None,
        run_name: str = None,
        llm_model: Any = None,
        embedding_model: Any = None,
        vector_store_config: VectorStoreConfig = None,
        mem0_config: MemoryConfig = None,
        default_memory_type: str = None,
        embedding_dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize long-term memory.

        Args:
            agent_name (str): Agent name (at least one of agent_name, user_name, run_name must be provided)
            user_name (str): User name
            run_name (str): Run/session name
            llm_model: LLM model instance
            embedding_model: Embedding model instance
            vector_store_config: Vector store configuration
            mem0_config: mem0 configuration (if provided, will override other configs)
            default_memory_type (str): Default memory type
            **kwargs: Additional configuration parameters

        Raises:
            ImportError: If mem0 library is not installed
            ValueError: If required parameters are missing
        """
        super().__init__()

        if not MEM0_AVAILABLE:
            raise ImportError(
                "Please install mem0 library: pip install mem0ai\n"
                "Long-term memory requires mem0 library support"
            )

        # Validate that at least one identifier is provided
        if agent_name is None and user_name is None and run_name is None:
            raise ValueError("At least one of agent_name, user_name, or run_name must be provided")

        # Store identifiers
        self.agent_id = agent_name
        self.user_id = user_name
        self.run_id = run_name

        # Store embedding dimensions for config
        self._embedding_dimensions = embedding_dimensions or kwargs.get("embedding_dimensions")

        # Initialize mem0 configuration
        self._init_mem0_config(
            llm_model=llm_model,
            embedding_model=embedding_model,
            vector_store_config=vector_store_config,
            mem0_config=mem0_config,
            embedding_dimensions=self._embedding_dimensions,
            **kwargs,
        )

        # Store default memory type
        self.default_memory_type = default_memory_type

        logger.info(
            f"Mem0LongTermMemory initialized "
            f"(agent: {agent_name}, user: {user_name}, run: {run_name})"
        )

    def _init_mem0_config(
        self,
        llm_model: Any,
        embedding_model: Any,
        vector_store_config: VectorStoreConfig,
        mem0_config: MemoryConfig,
        embedding_dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize mem0 configuration."""
        # Register custom adapters
        register_custom_adapters_to_mem0()

        # Dynamically create config classes
        _LlmConfig, _EmbedderConfig = self._create_config_classes()

        # Prepare embedder config (dimensions should be set in vector_store, not here)
        embedder_config_dict = {"model": embedding_model}

        if mem0_config is not None:
            # Use provided mem0 config, but allow overrides
            if llm_model is not None:
                mem0_config.llm = _LlmConfig(provider="custom", config={"model": llm_model})

            if embedding_model is not None:
                mem0_config.embedder = _EmbedderConfig(
                    provider="custom", config=embedder_config_dict
                )

            if vector_store_config is not None:
                mem0_config.vector_store = vector_store_config

        else:
            # Create new mem0 configuration
            if llm_model is None or embedding_model is None:
                raise ValueError(
                    "If mem0_config is not provided, llm_model and embedding_model must be provided"
                )

            mem0_config = mem0.configs.base.MemoryConfig(
                llm=_LlmConfig(provider="custom", config={"model": llm_model}),
                embedder=_EmbedderConfig(provider="custom", config=embedder_config_dict),
            )

            # Set vector store
            if vector_store_config is not None:
                mem0_config.vector_store = vector_store_config
            else:
                # Build VectorStoreConfig from kwargs
                on_disk = kwargs.get("on_disk", True)
                collection_name = kwargs.get("collection_name", "memory_collection")
                # Accept both `path` and `vector_store_path` for compatibility
                path = kwargs.get("path") or kwargs.get("vector_store_path") or "./qdrant_storage"
                provider = kwargs.get("vector_store_type", "qdrant")

                # Include dimensions in vector store config if provided
                vector_store_config_dict = {
                    "on_disk": on_disk,
                    "collection_name": collection_name,
                    "path": path,
                }
                if embedding_dimensions is not None:
                    vector_store_config_dict["embedding_model_dims"] = embedding_dimensions

                mem0_config.vector_store = mem0.vector_stores.configs.VectorStoreConfig(
                    provider=provider, config=vector_store_config_dict
                )

        # Initialize AsyncMemory
        self.long_term_memory = mem0.AsyncMemory(mem0_config)
        logger.info("mem0 AsyncMemory initialized")

    def _create_config_classes(self):
        """Create custom configuration classes."""
        from mem0.embeddings.configs import EmbedderConfig
        from mem0.llms.configs import LlmConfig
        from pydantic import field_validator

        class _CustomLlmConfig(LlmConfig):
            """Custom LLM configuration class."""

            @field_validator("config")
            @classmethod
            def validate_config(cls, v: Any, values: Any) -> Any:
                from mem0.utils.factory import LlmFactory

                provider = values.data.get("provider")
                if provider in LlmFactory.provider_to_class:
                    return v
                raise ValueError(f"Unsupported LLM provider: {provider}")

        class _CustomEmbedderConfig(EmbedderConfig):
            """Custom Embedder configuration class."""

            @field_validator("config")
            @classmethod
            def validate_config(cls, v: Any, values: Any) -> Any:
                from mem0.utils.factory import EmbedderFactory

                provider = values.data.get("provider")
                if provider in EmbedderFactory.provider_to_class:
                    return v
                raise ValueError(f"Unsupported Embedder provider: {provider}")

        return _CustomLlmConfig, _CustomEmbedderConfig

    async def record(
        self,
        msgs: Union[Msg, List[Msg], None],
        memory_type: str = None,
        infer: bool = True,
        **kwargs: Any,
    ) -> None:
        """Developer interface: Record messages to long-term memory.

        Args:
            msgs: Messages or list of messages to record
            memory_type (str): Memory type
            infer (bool): Whether to infer memory content
            **kwargs: Additional parameters
        """
        if msgs is None:
            return

        if isinstance(msgs, Msg):
            msgs = [msgs]

        # Filter None
        msg_list = [m for m in msgs if m is not None]
        if not all(isinstance(m, Msg) for m in msg_list):
            raise TypeError("Input must be Msg object or list of Msg objects")

        # Convert to mem0 format
        messages = [
            {
                "role": "assistant",
                "content": "\n".join([str(m.content) for m in msg_list]),
                "name": "assistant",
            }
        ]

        await self._mem0_record(messages, memory_type=memory_type, infer=infer, **kwargs)

    async def retrieve(
        self, msg: Union[Msg, List[Msg], None], limit: int = 5, **kwargs: Any
    ) -> str:
        """Developer interface: Retrieve information from long-term memory.

        Args:
            msg: Messages or list of messages used for retrieval
            limit (int): Maximum number of results to return
            **kwargs: Additional parameters

        Returns:
            str: Retrieved memory content
        """
        if isinstance(msg, Msg):
            msg = [msg]

        if not isinstance(msg, list) or not all(isinstance(m, Msg) for m in msg):
            raise TypeError("Input must be Msg object or list of Msg objects")

        # Convert to query string
        msg_strs = [json.dumps(m.to_dict()["content"], ensure_ascii=False) for m in msg]

        results = []
        for query in msg_strs:
            result = await self.long_term_memory.search(
                query=query,
                agent_id=self.agent_id,
                user_id=self.user_id,
                run_id=self.run_id,
                limit=limit,
            )
            if result and "results" in result:
                results.extend([item["memory"] for item in result["results"]])

        return "\n".join(results)

    async def record_to_memory(
        self, thinking: str, content: List[str], **kwargs: Any
    ) -> ToolResponse:
        """Tool interface: Record important information to long-term memory.

        This method is wrapped as a tool function for agent calls.

        Args:
            thinking (str): Agent's thinking process
            content (List[str]): List of content to record
            **kwargs: Additional parameters

        Returns:
            ToolResponse: Tool response object
        """
        try:
            # Merge thinking process and content
            if thinking:
                full_content = [thinking] + content
            else:
                full_content = content

            # Call mem0 record
            results = await self._mem0_record(
                [{"role": "assistant", "content": "\n".join(full_content), "name": "assistant"}],
                **kwargs,
            )

            return ToolResponse(
                content=[
                    TextBlock(
                        text=f"Successfully recorded content to long-term memory. Result: {results}"
                    )
                ]
            )

        except Exception as e:
            logger.error(f"Failed to record to memory: {e}")
            return ToolResponse(content=[TextBlock(text=f"Failed to record memory: {str(e)}")])

    async def retrieve_from_memory(
        self, keywords: List[str], limit: int = 5, **kwargs: Any
    ) -> ToolResponse:
        """Tool interface: Retrieve information from long-term memory.

        This method is wrapped as a tool function for agent calls.

        Args:
            keywords (List[str]): List of retrieval keywords
            limit (int): Maximum number of results to return
            **kwargs: Additional parameters

        Returns:
            ToolResponse: Tool response object
        """
        try:
            results = []
            for keyword in keywords:
                result = await self.long_term_memory.search(
                    query=keyword,
                    agent_id=self.agent_id,
                    user_id=self.user_id,
                    run_id=self.run_id,
                    limit=limit,
                )
                if result and "results" in result:
                    results.extend([item["memory"] for item in result["results"]])

            if results:
                return ToolResponse(content=[TextBlock(text="\n".join(results))])
            else:
                return ToolResponse(content=[TextBlock(text="No related memories found")])

        except Exception as e:
            logger.error(f"Failed to retrieve from memory: {e}")
            return ToolResponse(content=[TextBlock(text=f"Failed to retrieve memory: {str(e)}")])

    async def _mem0_record(
        self,
        messages: Union[str, List[Dict]],
        memory_type: str = None,
        infer: bool = True,
        **kwargs: Any,
    ) -> Dict:
        """Internal method: Record content using mem0.

        Args:
            messages: Messages to record
            memory_type (str): Memory type
            infer (bool): Whether to infer memory
            **kwargs: Additional parameters

        Returns:
            Dict: mem0 return result
        """
        results = await self.long_term_memory.add(
            messages=messages,
            agent_id=self.agent_id,
            user_id=self.user_id,
            run_id=self.run_id,
            memory_type=(memory_type if memory_type is not None else self.default_memory_type),
            infer=infer,
            **kwargs,
        )
        logger.debug(f"mem0 record result: {results}")
        return results

    def state_dict(self) -> Dict:
        """Get state dictionary.

        Returns:
            Dict: State dictionary
        """
        return {
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "run_id": self.run_id,
            "default_memory_type": self.default_memory_type,
        }

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        """Load from state dictionary.

        Args:
            state_dict (Dict): State dictionary
            strict (bool): Strict mode
        """
        self.agent_id = state_dict.get("agent_id")
        self.user_id = state_dict.get("user_id")
        self.run_id = state_dict.get("run_id")
        self.default_memory_type = state_dict.get("default_memory_type")
        logger.info("Loaded long-term memory configuration from state dictionary")
