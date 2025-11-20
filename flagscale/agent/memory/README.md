# FlagScale Agent Memory Module

Memory management module for FlagScale Agent, providing short-term and long-term memory capabilities.

## Overview

- **Short-term Memory (InMemoryMemory)**: Fast access to conversation history, stored in memory
- **Long-term Memory (Mem0LongTermMemory)**: Vector retrieval and semantic search based on mem0

### Difference from Collaborator

| Module | Layer | Responsibility | Storage | Scope |
|--------|-------|----------------|---------|-------|
| **Collaborator** | Coordination | Agent collaboration, state synchronization, message passing | Redis (distributed) | Cross-Agent |
| **Memory** | Cognitive | Conversation history, knowledge retrieval, context management | Local memory/Vector database | Single Agent internal |

## Quick Start

### Basic Usage (Short-term Memory Only)

```python
from flagscale.agent.memory import MemoryManager, Msg
import asyncio

async def main():
    memory = MemoryManager(enable_short_term=True, max_size=100)
    
    await memory.short_term_memory.add(Msg(role="user", content="Hello"))
    await memory.short_term_memory.add(Msg(role="assistant", content="Hi there!"))
    
    results = await memory.short_term_memory.retrieve("Hello", limit=5)
    print(f"Found {len(results)} related memories")

asyncio.run(main())
```

### Integration with ToolRegistry

```python
from flagscale.agent.tool_match import ToolRegistry
from flagscale.agent.memory import MemoryManager, register_memory_tools

memory = MemoryManager(enable_short_term=True, max_size=100)
registry = ToolRegistry()
registered = register_memory_tools(registry, memory)
```

### Using Configuration File

```python
from flagscale.agent.memory import load_config_from_file, MemoryManager

# Load configuration
config = load_config_from_file("memory_config.yaml")

# Create memory manager (LLM and Embedding models required)
memory = MemoryManager.from_config(
    config,
    llm_model=your_llm_model,
    embedding_model=your_embedding_model,
)
```

## Configuration

### Configuration File Format

```yaml
memory:
  # Short-term memory configuration
  short_term:
    max_size: 1000              # Maximum number of messages to store
    auto_cleanup: true           # Automatically clean up old messages
  
  # Long-term memory configuration
  long_term:
    provider: "mem0"            # Use mem0 as long-term memory provider
    agent_name: "my_agent"       # Agent name (required when enabling long-term memory)
    user_name: null              # User name (optional)
    run_name: null               # Run session name (optional)
    
    # Vector store configuration
    vector_store_type: "qdrant"  # Vector database type
    vector_store_path: "./qdrant_storage"  # Storage path
    collection_name: "memory_collection"  # Collection name
    on_disk: true                # Use disk persistence
    
    default_memory_type: null    # Default memory type
    
    # LLM model configuration (optional, provide in code if not configured)
    llm:
      provider: "openai"         # Provider: openai
      model: "gpt-3.5-turbo"     # Model name
      api_base: null             # API endpoint (null uses default endpoint)
      api_key: "${OPENAI_API_KEY}"  # API key (supports environment variable format)
      temperature: 0.7           # Temperature parameter
      max_tokens: 2000           # Maximum tokens
    
    # Embedding model configuration (optional, provide in code if not configured)
    embedding:
      provider: "openai"         # Provider: openai
      model: "text-embedding-3-small"  # Model name
      api_base: null             # API endpoint (null uses default endpoint)
      api_key: "${OPENAI_API_KEY}"  # API key (supports environment variable format)
      dimensions: 1536           # Vector dimensions
  
  # Global configuration
  enable_short_term: true        # Enable short-term memory
  enable_long_term: false       # Enable long-term memory
  auto_record: false            # Automatically record to long-term memory
  auto_record_threshold: 10      # Auto-record message count threshold
```

### Loading Configuration File

```python
from flagscale.agent.memory import load_config_from_file, load_config, MemoryManager

# Method 1: Direct path specification
config = load_config_from_file("memory_config.yaml")

# Method 2: Read path from environment variable
# export FLAGSCALE_MEMORY_CONFIG=/path/to/memory_config.yaml
config = load_config()

# Create from configuration
memory = MemoryManager.from_config(config, llm_model=llm, embedding_model=emb)
```

## LLM and Embedding Model Setup

Long-term memory requires LLM and Embedding models. Models must be callable and support synchronous or asynchronous interfaces.

### Embedding Dimension Adaptation

Different Embedding models have different output dimensions (e.g., OpenAI text-embedding-3-small is 1536-dimensional, text-embedding-3-large is 3072-dimensional). The system adapts through the following mechanisms:

1. **Configuration File Specification**: Specify dimensions in the `embedding.dimensions` field of the configuration file
   ```yaml
   embedding:
     provider: "openai"
     model: "text-embedding-3-small"
     dimensions: 1536  # Specify vector dimensions
   ```

2. **Automatic Transfer to Vector Store**: Dimension information is automatically passed to mem0's vector store configuration (`embedding_model_dims`), ensuring the vector database uses the correct dimensions

3. **OpenAI Models**: For OpenAI models that support dimension parameters, the `dimensions` parameter is passed to the API, allowing you to adjust output dimensions

4. **Custom Models**: When using custom Embedding models, you need to manually specify `dimensions` in the configuration, or pass it through the `embedding_dimensions` parameter:
   ```python
   memory = MemoryManager(
       config=config,
       embedding_model=your_custom_embedding_model,
       embedding_dimensions=1024  # Specify custom model dimensions
   )
   ```

**Note**: If dimensions are not specified, some vector databases may not initialize correctly. It is recommended to always explicitly specify dimensions in the configuration.

### Model Interfaces

**LLM Model Interface:**
```python
def llm_model(messages: List[Dict[str, str]], tools: List[Dict] = None) -> str:
    # messages: [{"role": "user", "content": "..."}, ...]
    # Returns: str or object containing content/text/message.content
    pass
```

**Embedding Model Interface:**
```python
from typing import List, Union

def embedding_model(texts: List[str]) -> Union[List[float], List[List[float]]]:
    # texts: List of texts
    # Returns: List[float] for single text, List[List[float]] for multiple texts
    pass
```

### Usage Example

```python
from openai import AsyncOpenAI
from flagscale.agent.memory import load_config_from_file, MemoryManager

config = load_config_from_file("memory_config.yaml")
client = AsyncOpenAI(api_key="your-api-key")

async def openai_llm(messages, tools=None):
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, tools=tools
    )
    return response.choices[0].message.content

async def openai_embedding(texts):
    response = await client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings[0] if len(texts) == 1 else embeddings

memory = MemoryManager.from_config(
    config,
    llm_model=openai_llm,
    embedding_model=openai_embedding,
)
```

## API Reference

### MemoryManager

```python
memory = MemoryManager(
    enable_short_term=True,         # Whether to enable short-term memory
    enable_long_term=False,          # Whether to enable long-term memory
    max_size=1000,                  # Maximum capacity of short-term memory
    llm_model=None,                 # LLM model (required for long-term memory)
    embedding_model=None,           # Embedding model (required for long-term memory)
    agent_name=None,                # Agent name
    vector_store_path="./qdrant_storage",  # Vector store path
)
```

### register_memory_tools

```python
registered_tools = register_memory_tools(
    tool_registry,      # ToolRegistry instance
    memory_manager,     # MemoryManager instance
    category="memory"   # Tool category
)
```

## FAQ

**Q: What if the configuration file is not found?**  
A: You must explicitly specify the configuration file path, or use `create_if_not_exists=True` to automatically create it.

**Q: Must models be callable?**  
A: Yes, models must implement the `__call__` method.

**Q: Are both synchronous and asynchronous interfaces supported?**  
A: Yes, the adapter automatically detects and handles both.

**Q: What dependencies are required for long-term memory?**  
A: You need to install `mem0ai >= 0.1.115` and `qdrant-client`.

**Q: What's the difference between short-term and long-term memory?**  
A: Short-term memory is memory-based for fast access to conversation history; long-term memory is vector-based retrieval requiring LLM and Embedding models.

## Dependencies

- `mem0ai >= 0.1.115`: Long-term memory backend (optional)
- `qdrant-client`: Vector database client (optional)
- `pyyaml`: Configuration file support
- `pydantic >= 2.0`: Data validation

## Notes

1. **Short-term Memory**: No additional dependencies required, can be used directly
2. **Long-term Memory**: Requires LLM and Embedding models, as well as the mem0 library
3. **Qdrant**: Uses embedded mode by default, no separate service deployment required
4. **Asynchronous Operations**: All memory operations are asynchronous and require `await`
