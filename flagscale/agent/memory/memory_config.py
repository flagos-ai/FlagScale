"""Memory module configuration management."""

import logging
import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ShortTermMemoryConfig:
    """Short-term memory configuration."""

    max_size: int = 1000
    """Maximum capacity of short-term memory"""

    auto_cleanup: bool = True
    """Whether to automatically clean up old messages"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ShortTermMemoryConfig":
        """Create configuration from dictionary."""
        return cls(
            max_size=config_dict.get("max_size", 1000),
            auto_cleanup=config_dict.get("auto_cleanup", True),
        )


@dataclass
class LLMConfig:
    """LLM model configuration."""

    provider: Optional[str] = None
    """Provider type: openai, huggingface, custom, null"""

    model: Optional[str] = None
    """Model name or path"""

    api_base: Optional[str] = None
    """API base URL (for external APIs)"""

    api_key: Optional[str] = None
    """API key (supports environment variables, format: ${ENV_VAR_NAME})"""

    temperature: float = 0.7
    """Temperature parameter"""

    max_tokens: int = 2000
    """Maximum number of tokens"""

    # Other custom parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    """Additional custom parameters"""

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["LLMConfig"]:
        """Create configuration from dictionary.

        Supports multiple configuration formats (in priority order):
        1. Direct configuration (recommended): {"provider": "openai", "model": "gpt-3.5-turbo", ...}
        2. Simplified configuration: {"provider": "openai", "model": "gpt-3.5-turbo"}  # others use defaults
        3. Preset configuration: {"preset": "openai-gpt35"}  # backward compatible, not recommended
        4. String shorthand: {"openai": "gpt-3.5-turbo"}  # backward compatible
        """
        if config_dict is None:
            return None

        # Prioritize direct configuration (most readable)
        if "provider" in config_dict or "model" in config_dict:
            return cls(
                provider=config_dict.get("provider"),
                model=config_dict.get("model"),
                api_base=config_dict.get("api_base"),
                api_key=config_dict.get("api_key"),
                temperature=config_dict.get("temperature", 0.7),
                max_tokens=config_dict.get("max_tokens", 2000),
                extra_params=config_dict.get("extra_params", {}),
            )

        # Handle preset configuration (backward compatible)
        if "preset" in config_dict:
            preset = config_dict["preset"]
            return cls._from_preset(preset, config_dict)

        # Handle string shorthand format (backward compatible): {"openai": "gpt-3.5-turbo"}
        if isinstance(config_dict, dict) and len(config_dict) == 1:
            key, value = next(iter(config_dict.items()))
            if isinstance(value, str) and key in ["openai", "huggingface", "custom"]:
                return cls(provider=key, model=value, temperature=0.7, max_tokens=2000)
            else:
                logger.warning(f"Unrecognized LLM shorthand config: {config_dict}")

        logger.warning(f"Unrecognized LLM config format: {config_dict}")
        return None

    @classmethod
    def _from_preset(cls, preset: str, overrides: Dict[str, Any] = None) -> "LLMConfig":
        """Create configuration from preset."""
        presets = {
            "openai-gpt35": {"provider": "openai", "model": "gpt-3.5-turbo"},
            "openai-gpt4": {"provider": "openai", "model": "gpt-4"},
            "openai-gpt4o": {"provider": "openai", "model": "gpt-4o"},
            "huggingface-llama": {
                "provider": "huggingface",
                "model": "meta-llama/Llama-2-7b-chat-hf",
            },
        }

        if preset not in presets:
            logger.warning(f"Unknown preset configuration: {preset}, using default configuration")
            preset_config = {}
        else:
            preset_config = presets[preset].copy()

        if overrides:
            preset_config.update({k: v for k, v in overrides.items() if k != "preset"})

        return cls.from_dict(preset_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.provider is not None:
            result["provider"] = self.provider
        if self.model is not None:
            result["model"] = self.model
        if self.api_base is not None:
            result["api_base"] = self.api_base
        if self.api_key is not None:
            result["api_key"] = self.api_key
        if self.temperature != 0.7:
            result["temperature"] = self.temperature
        if self.max_tokens != 2000:
            result["max_tokens"] = self.max_tokens
        if self.extra_params:
            result["extra_params"] = self.extra_params
        return result


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    provider: Optional[str] = None
    """Provider type: openai, huggingface, custom, null"""

    model: Optional[str] = None
    """Model name or path"""

    api_base: Optional[str] = None
    """API base URL (for external APIs)"""

    api_key: Optional[str] = None
    """API key (supports environment variables, format: ${ENV_VAR_NAME})"""

    dimensions: Optional[int] = None
    """Vector dimensions"""

    # Other custom parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    """Additional custom parameters"""

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional["EmbeddingConfig"]:
        """Create configuration from dictionary.

        Supports multiple configuration formats (in priority order):
        1. Direct configuration (recommended): {"provider": "openai", "model": "text-embedding-3-small", ...}
        2. Simplified configuration: {"provider": "openai", "model": "text-embedding-3-small"}  # others use defaults
        3. Preset configuration: {"preset": "openai-small"}  # backward compatible, not recommended
        4. String shorthand: {"openai": "text-embedding-3-small"}  # backward compatible
        """
        if config_dict is None:
            return None

        # Prioritize direct configuration (most readable)
        if "provider" in config_dict or "model" in config_dict:
            return cls(
                provider=config_dict.get("provider"),
                model=config_dict.get("model"),
                api_base=config_dict.get("api_base"),
                api_key=config_dict.get("api_key"),
                dimensions=config_dict.get("dimensions"),
                extra_params=config_dict.get("extra_params", {}),
            )

        # Handle preset configuration (backward compatible)
        if "preset" in config_dict:
            preset = config_dict["preset"]
            return cls._from_preset(preset, config_dict)

        # Handle string shorthand format (backward compatible): {"openai": "text-embedding-3-small"}
        if isinstance(config_dict, dict) and len(config_dict) == 1:
            key, value = next(iter(config_dict.items()))
            if isinstance(value, str) and key in ["openai", "huggingface", "custom"]:
                return cls(provider=key, model=value)
            else:
                logger.warning(f"Unrecognized Embedding shorthand config: {config_dict}")

        logger.warning(f"Unrecognized Embedding config format: {config_dict}")
        return None

    @classmethod
    def _from_preset(cls, preset: str, overrides: Dict[str, Any] = None) -> "EmbeddingConfig":
        """Create configuration from preset."""
        presets = {
            "openai-small": {"provider": "openai", "model": "text-embedding-3-small"},
            "openai-large": {"provider": "openai", "model": "text-embedding-3-large"},
            "huggingface-minilm": {
                "provider": "huggingface",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        }

        if preset not in presets:
            logger.warning(f"Unknown preset configuration: {preset}, using default configuration")
            preset_config = {}
        else:
            preset_config = presets[preset].copy()

        if overrides:
            preset_config.update({k: v for k, v in overrides.items() if k != "preset"})

        return cls.from_dict(preset_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        if self.provider is not None:
            result["provider"] = self.provider
        if self.model is not None:
            result["model"] = self.model
        if self.api_base is not None:
            result["api_base"] = self.api_base
        if self.api_key is not None:
            result["api_key"] = self.api_key
        if self.dimensions is not None:
            result["dimensions"] = self.dimensions
        if self.extra_params:
            result["extra_params"] = self.extra_params
        return result


@dataclass
class LongTermMemoryConfig:
    """Long-term memory configuration."""

    provider: str = "mem0"
    """Long-term memory provider"""

    agent_name: Optional[str] = None
    """Agent name"""

    user_name: Optional[str] = None
    """User name"""

    run_name: Optional[str] = None
    """Run session name"""

    vector_store_type: str = "qdrant"
    """Vector store type"""

    vector_store_path: str = "./qdrant_storage"
    """Vector store path"""

    collection_name: str = "memory_collection"
    """Collection name"""

    on_disk: bool = True
    """Whether to use disk storage"""

    default_memory_type: Optional[str] = None
    """Default memory type"""

    llm: Optional[LLMConfig] = None
    """LLM model configuration"""

    embedding: Optional[EmbeddingConfig] = None
    """Embedding model configuration"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LongTermMemoryConfig":
        """Create configuration from dictionary."""
        llm_dict = config_dict.get("llm")
        embedding_dict = config_dict.get("embedding")

        return cls(
            provider=config_dict.get("provider", "mem0"),
            agent_name=config_dict.get("agent_name"),
            user_name=config_dict.get("user_name"),
            run_name=config_dict.get("run_name"),
            vector_store_type=config_dict.get("vector_store_type", "qdrant"),
            vector_store_path=config_dict.get("vector_store_path", "./qdrant_storage"),
            collection_name=config_dict.get("collection_name", "memory_collection"),
            on_disk=config_dict.get("on_disk", True),
            default_memory_type=config_dict.get("default_memory_type"),
            llm=LLMConfig.from_dict(llm_dict),
            embedding=EmbeddingConfig.from_dict(embedding_dict),
        )


@dataclass
class MemoryModuleConfig:
    """Memory module overall configuration."""

    short_term: ShortTermMemoryConfig = field(default_factory=ShortTermMemoryConfig)
    """Short-term memory configuration"""

    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)
    """Long-term memory configuration"""

    enable_short_term: bool = True
    """Whether to enable short-term memory"""

    enable_long_term: bool = True
    """Whether to enable long-term memory"""

    auto_record: bool = False
    """Whether to automatically record conversations to long-term memory"""

    auto_record_threshold: int = 10
    """Threshold for number of messages to auto-record"""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MemoryModuleConfig":
        """Create configuration from dictionary."""
        short_term_dict = config_dict.get("short_term", {})
        long_term_dict = config_dict.get("long_term", {})

        return cls(
            short_term=ShortTermMemoryConfig.from_dict(short_term_dict),
            long_term=LongTermMemoryConfig.from_dict(long_term_dict),
            enable_short_term=config_dict.get("enable_short_term", True),
            enable_long_term=config_dict.get("enable_long_term", True),
            auto_record=config_dict.get("auto_record", False),
            auto_record_threshold=config_dict.get("auto_record_threshold", 10),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MemoryModuleConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path (str): YAML configuration file path

        Returns:
            MemoryModuleConfig: Configuration object

        Raises:
            FileNotFoundError: If configuration file does not exist
            yaml.YAMLError: If configuration file format is invalid
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file does not exist: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Get memory section configuration
        memory_config = config_dict.get("memory", config_dict)

        logger.info(f"Loaded memory configuration from {yaml_path}")
        return cls.from_dict(memory_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            "short_term": {
                "max_size": self.short_term.max_size,
                "auto_cleanup": self.short_term.auto_cleanup,
            },
            "long_term": {
                "provider": self.long_term.provider,
                "agent_name": self.long_term.agent_name,
                "user_name": self.long_term.user_name,
                "run_name": self.long_term.run_name,
                "vector_store_type": self.long_term.vector_store_type,
                "vector_store_path": self.long_term.vector_store_path,
                "collection_name": self.long_term.collection_name,
                "on_disk": self.long_term.on_disk,
                "default_memory_type": self.long_term.default_memory_type,
                "llm": self.long_term.llm.to_dict() if self.long_term.llm else None,
                "embedding": (
                    self.long_term.embedding.to_dict() if self.long_term.embedding else None
                ),
            },
            "enable_short_term": self.enable_short_term,
            "enable_long_term": self.enable_long_term,
            "auto_record": self.auto_record,
            "auto_record_threshold": self.auto_record_threshold,
        }

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path (str): YAML configuration file path
        """
        config_dict = {"memory": self.to_dict()}

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"Configuration saved to {yaml_path}")

    def validate(self) -> bool:
        """Validate configuration validity.

        Returns:
            bool: Whether configuration is valid
        """
        # Validate that long-term memory has at least one identifier
        if self.enable_long_term:
            if not any(
                [self.long_term.agent_name, self.long_term.user_name, self.long_term.run_name]
            ):
                logger.error(
                    "Long-term memory configuration error: must provide at least one of agent_name, user_name, or run_name"
                )
                return False

        # Validate short-term memory capacity
        if self.short_term.max_size <= 0:
            logger.error("Short-term memory configuration error: max_size must be greater than 0")
            return False

        logger.info("Configuration validation passed")
        return True


def create_default_config(
    agent_name: str = "default_agent", save_path: str = None
) -> MemoryModuleConfig:
    """Create default configuration.

    Args:
        agent_name (str): Agent name
        save_path (str): If provided, save configuration to this path

    Returns:
        MemoryModuleConfig: Default configuration object
    """
    config = MemoryModuleConfig(
        short_term=ShortTermMemoryConfig(max_size=1000, auto_cleanup=True),
        long_term=LongTermMemoryConfig(
            provider="mem0",
            agent_name=agent_name,
            vector_store_type="qdrant",
            vector_store_path="./qdrant_storage",
            collection_name="memory_collection",
            on_disk=True,
        ),
        enable_short_term=True,
        enable_long_term=True,
        auto_record=False,
        auto_record_threshold=10,
    )

    if save_path:
        config.to_yaml(save_path)

    logger.info(f"Created default configuration (agent: {agent_name})")
    return config


# Configuration Loading Functions


def load_config_from_file(
    config_path: str, create_if_not_exists: bool = False, default_agent_name: str = "default_agent"
) -> MemoryModuleConfig:
    """Load memory module configuration from configuration file.

    Args:
        config_path: Configuration file path (supports YAML format)
        create_if_not_exists: Whether to create default configuration if file does not exist
        default_agent_name: Agent name to use when creating default configuration

    Returns:
        MemoryModuleConfig: Configuration object

    Raises:
        FileNotFoundError: If configuration file does not exist and create_if_not_exists=False
        ValueError: If configuration file format is invalid
    """
    config_path = Path(config_path).expanduser().resolve()

    # If file does not exist
    if not config_path.exists():
        if create_if_not_exists:
            logger.info(f"Configuration file does not exist, creating default: {config_path}")
            config = create_default_config(
                agent_name=default_agent_name, save_path=str(config_path)
            )
            return config
        else:
            raise FileNotFoundError(
                f"Configuration file does not exist: {config_path}\n"
                f"Hint: Use create_if_not_exists=True to automatically create default configuration"
            )

    # Load configuration
    try:
        config = MemoryModuleConfig.from_yaml(str(config_path))
        logger.info(f"Successfully loaded configuration file: {config_path}")

        # Validate configuration
        if not config.validate():
            raise ValueError("Configuration validation failed, please check configuration items")

        return config

    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise ValueError(f"Configuration file format error: {e}") from e


def create_config_file(
    config_path: str, agent_name: str = "default_agent", enable_long_term: bool = True
) -> MemoryModuleConfig:
    """Create configuration file.

    Args:
        config_path: Configuration file save path
        agent_name: Agent name
        enable_long_term: Whether to enable long-term memory

    Returns:
        MemoryModuleConfig: Created configuration object
    """
    config_path = Path(config_path).expanduser().resolve()

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create configuration
    config = MemoryModuleConfig(
        short_term=ShortTermMemoryConfig(max_size=1000, auto_cleanup=True),
        long_term=LongTermMemoryConfig(
            agent_name=agent_name if enable_long_term else None,
            vector_store_path="./qdrant_storage",
        ),
        enable_short_term=True,
        enable_long_term=enable_long_term,
    )

    # Save configuration
    config.to_yaml(str(config_path))
    logger.info(f"Configuration file created: {config_path}")

    return config


def get_config_from_env() -> Optional[MemoryModuleConfig]:
    """Load configuration path from environment variable and load configuration.

    Environment variables:
        FLAGSCALE_MEMORY_CONFIG: Configuration file path

    Returns:
        MemoryModuleConfig: Configuration object, or None if environment variable is not set
    """
    config_path = os.getenv("FLAGSCALE_MEMORY_CONFIG")
    if config_path:
        return load_config_from_file(config_path)
    return None


def load_config(
    config_path: Optional[str] = None,
    agent_name: str = "default_agent",
    create_if_not_exists: bool = False,
) -> MemoryModuleConfig:
    """Load configuration.

    Configuration file path is determined by the following priority:
    1. If config_path is provided, use that path
    2. Read from environment variable FLAGSCALE_MEMORY_CONFIG
    3. If neither is provided, raise error

    Args:
        config_path: Configuration file path (optional)
        agent_name: Agent name (used when creating default configuration)
        create_if_not_exists: Whether to create default configuration if file does not exist

    Returns:
        MemoryModuleConfig: Configuration object

    Raises:
        ValueError: If configuration file path is not provided and environment variable is not set
    """
    # Use provided path if available
    if config_path:
        return load_config_from_file(
            config_path, create_if_not_exists=create_if_not_exists, default_agent_name=agent_name
        )

    # Otherwise, try reading from environment variable
    env_config = get_config_from_env()
    if env_config:
        return env_config

    # Neither provided, raise error
    raise ValueError(
        "Configuration file path not provided. Please use one of the following:\n"
        "1. Specify configuration file path: load_config(config_path='memory_config.yaml')\n"
        "2. Set environment variable: export FLAGSCALE_MEMORY_CONFIG=/path/to/config.yaml"
    )


# Model Factory


def _resolve_env_var(value: str) -> str:
    """Resolve environment variable reference.

    Args:
        value: String that may contain environment variable reference, format: ${ENV_VAR_NAME}

    Returns:
        Resolved value
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.getenv(env_var, "")
        if not resolved:
            logger.warning(f"Environment variable {env_var} is not set")
        return resolved
    return value


class ModelFactory:
    """Model factory for creating model instances from configuration."""

    @staticmethod
    def create_llm_model(config: Optional[LLMConfig]) -> Optional[Any]:
        """Create LLM model from configuration.

        Args:
            config: LLM configuration object

        Returns:
            LLM model instance, or None if config is None

        Raises:
            ValueError: If configuration is invalid or provider is not supported
        """
        if config is None:
            return None

        if config.provider is None:
            logger.warning("LLM provider not specified, skipping creation")
            return None

        provider = config.provider.lower()

        if provider == "openai":
            return ModelFactory._create_openai_llm(config)
        elif provider == "huggingface":
            return ModelFactory._create_huggingface_llm(config)
        elif provider == "custom":
            return ModelFactory._create_custom_llm(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def create_embedding_model(config: Optional[EmbeddingConfig]) -> Optional[Any]:
        """Create Embedding model from configuration.

        Args:
            config: Embedding configuration object

        Returns:
            Embedding model instance, or None if config is None

        Raises:
            ValueError: If configuration is invalid or provider is not supported
        """
        if config is None:
            return None

        if config.provider is None:
            logger.warning("Embedding provider not specified, skipping creation")
            return None

        provider = config.provider.lower()

        if provider == "openai":
            return ModelFactory._create_openai_embedding(config)
        elif provider == "huggingface":
            return ModelFactory._create_huggingface_embedding(config)
        elif provider == "custom":
            return ModelFactory._create_custom_embedding(config)
        else:
            raise ValueError(f"Unsupported Embedding provider: {provider}")

    @staticmethod
    def _create_openai_llm(config: LLMConfig) -> Any:
        """Create OpenAI-compatible LLM model."""
        try:
            from openai import AsyncOpenAI

            # If api_key is not specified in config, try to read from environment variables
            if config.api_key:
                api_key = _resolve_env_var(config.api_key)
            else:
                # Try to read from common environment variables
                api_key = (
                    os.getenv("OPENAI_API_KEY")
                    or os.getenv("OPENAI_KEY")
                    or os.getenv("ZHIPU_API_KEY")
                )

            api_base = config.api_base

            client = AsyncOpenAI(
                api_key=api_key if api_key else "EMPTY", base_url=api_base if api_base else None
            )

            model_name = config.model or "gpt-3.5-turbo"
            temperature = config.temperature
            max_tokens = config.max_tokens

            async def llm_model(messages: List[Dict[str, str]], tools: List[Dict] = None) -> str:
                """OpenAI LLM model call interface."""
                params = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if tools:
                    params["tools"] = tools

                # Merge extra parameters
                params.update(config.extra_params)

                response = await client.chat.completions.create(**params)
                return response.choices[0].message.content

            logger.info(f"Created OpenAI LLM: {model_name}")
            return llm_model

        except ImportError:
            logger.error("Failed to import OpenAI library, please install: pip install openai")
            raise

    @staticmethod
    def _create_openai_embedding(config: EmbeddingConfig) -> Any:
        """Create OpenAI-compatible Embedding model."""
        try:
            from openai import AsyncOpenAI

            # If api_key is not specified in config, try to read from environment variables
            if config.api_key:
                api_key = _resolve_env_var(config.api_key)
            else:
                # Try to read from common environment variables
                api_key = (
                    os.getenv("OPENAI_API_KEY")
                    or os.getenv("OPENAI_KEY")
                    or os.getenv("ZHIPU_API_KEY")
                )

            api_base = config.api_base

            client = AsyncOpenAI(
                api_key=api_key if api_key else "EMPTY", base_url=api_base if api_base else None
            )

            model_name = config.model or "text-embedding-3-small"
            dimensions = config.dimensions

            async def embedding_model(texts: List[str]) -> Union[List[float], List[List[float]]]:
                """OpenAI Embedding model call interface."""
                is_single = len(texts) == 1

                params = {"model": model_name, "input": texts}

                if dimensions:
                    params["dimensions"] = dimensions

                # Merge extra parameters
                params.update(config.extra_params)

                response = await client.embeddings.create(**params)
                embeddings = [item.embedding for item in response.data]

                return embeddings[0] if is_single else embeddings

            logger.info(f"Created OpenAI Embedding: {model_name}")
            return embedding_model

        except ImportError:
            logger.error("Failed to import OpenAI library, please install: pip install openai")
            raise

    @staticmethod
    def _create_huggingface_llm(config: LLMConfig) -> Any:
        """Create HuggingFace LLM model."""
        try:
            import torch

            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not config.model:
                raise ValueError("HuggingFace LLM requires model parameter to be specified")

            model_name = config.model
            device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading HuggingFace LLM model: {model_name} (device: {device})")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )

            if device == "cpu":
                model = model.to(device)

            max_tokens = config.max_tokens
            temperature = config.temperature

            def llm_model(messages: List[Dict[str, str]], tools: List[Dict] = None) -> str:
                """HuggingFace LLM model call interface (synchronous)."""
                # Convert messages to text
                text = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    text += f"{role}: {content}\n"
                text += "assistant:"

                inputs = tokenizer(text, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                return response.strip()

            logger.info(f"Created HuggingFace LLM: {model_name}")
            return llm_model

        except ImportError:
            logger.error(
                "Failed to import transformers library, please install: pip install transformers torch"
            )
            raise

    @staticmethod
    def _create_huggingface_embedding(config: EmbeddingConfig) -> Any:
        """Create HuggingFace Embedding model."""
        try:
            import torch

            from sentence_transformers import SentenceTransformer

            if not config.model:
                raise ValueError("HuggingFace Embedding requires model parameter to be specified")

            model_name = config.model
            device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading HuggingFace Embedding model: {model_name} (device: {device})")

            model = SentenceTransformer(model_name, device=device)

            def embedding_model(texts: List[str]) -> Union[List[float], List[List[float]]]:
                """HuggingFace Embedding model call interface (synchronous)."""
                is_single = len(texts) == 1
                embeddings = model.encode(texts, convert_to_numpy=True).tolist()
                return embeddings[0] if is_single else embeddings

            logger.info(f"Created HuggingFace Embedding: {model_name}")
            return embedding_model

        except ImportError:
            logger.error(
                "Failed to import sentence-transformers library, please install: pip install sentence-transformers"
            )
            raise

    @staticmethod
    def _create_custom_llm(config: LLMConfig) -> Any:
        """Create custom LLM model (requires user to provide model instance in code)."""
        logger.warning(
            "custom provider requires user to provide model instance in code, "
            "cannot be automatically created from config. Please use MemoryManager(..., llm_model=your_model)"
        )
        return None

    @staticmethod
    def _create_custom_embedding(config: EmbeddingConfig) -> Any:
        """Create custom Embedding model (requires user to provide model instance in code)."""
        logger.warning(
            "custom provider requires user to provide model instance in code, "
            "cannot be automatically created from config. Please use MemoryManager(..., embedding_model=your_model)"
        )
        return None
