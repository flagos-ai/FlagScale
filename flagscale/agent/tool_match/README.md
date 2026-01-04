# Tool Match Module

Intelligent tool matching module for automatically selecting the most relevant tools based on task descriptions.

## Features

- **Multi-weight Scoring System**: Combines semantic similarity, keyword matching, and category relevance for comprehensive scoring
- **Intelligent Degradation Mechanism**: Automatically falls back to other scoring methods when certain components are unavailable (e.g., network issues, missing dependencies)
- **LRU Cache Optimization**: Uses Least Recently Used cache mechanism to optimize query performance
- **Category Management**: Supports tool categorization and category-based search
- **Flexible Configuration**: Configurable maximum tool count, minimum similarity threshold, and other parameters

## Core Components

### ToolMatcher
Intelligent tool matcher responsible for calculating the match score between tools and tasks.

**Main Functions:**
- Semantic similarity calculation (based on sentence-transformers)
- Keyword matching scoring
- Category relevance scoring
- Multi-weight comprehensive scoring
- Degradation mechanism management

### ToolRegistry
Tool registry that manages all available tools and provides search interfaces.

**Main Functions:**
- Tool registration and management
- Organize tools by category
- Tool search and matching
- Statistical information retrieval

## Quick Start

```python
from flagscale.agent.tool_match import ToolRegistry

# Create tool registry
registry = ToolRegistry(max_tools=5, min_similarity=0.1)

# Register tool
tool = {
    "function": {
        "name": "read_file",
        "description": "read file content and return text data",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "file path to read"}
            }
        }
    }
}
registry.register_tool(tool, category="file")

# Search for relevant tools
results = registry.search_tools("read file content")
for tool_name, score in results:
    print(f"{tool_name}: {score:.3f}")
```

## Scoring System

### Weight Configuration
- **Semantic Similarity**: 70% (requires sentence-transformers)
- **Keyword Matching**: 20%
- **Category Relevance**: 10%

### Degradation Mechanism
When certain components are unavailable, the system automatically degrades:
- Network unavailable → Disable semantic similarity
- Missing dependencies → Disable corresponding components
- Weights automatically re-normalized

## Tool Format

```python
tool = {
    "function": {
        "name": "tool_name",           # Tool name
        "description": "tool description",  # Tool description
        "parameters": {                # Parameter definition
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "param description"}
            }
        }
    }
}
```

## Category System

Supports the following predefined categories:
- `general`: General tools
- `file`: File operations
- `search`: Search related
- `data`: Data processing
- `network`: Network operations
- `system`: System commands

## API Reference

### ToolRegistry

#### Main Methods
- `register_tool(tool, category)`: Register a single tool
- `register_tools(tools, category)`: Register multiple tools in batch
- `search_tools(query, category=None)`: Search for tools
- `get_tool_by_name(name)`: Get tool by name
- `get_tools_by_category(category)`: Get tools by category
- `get_stats()`: Get statistical information

#### Degradation Control
- `set_degradation(component, degraded)`: Set component degradation status
- `get_degradation_status()`: Get degradation status
- `reset_degradation()`: Reset all degradation flags

### ToolMatcher

#### Configuration Parameters
- `max_tools`: Maximum number of tools to return (default: 3)
- `min_similarity`: Minimum similarity threshold (default: 0.1)

## Dependencies

- `sentence-transformers`: Semantic similarity calculation (optional, auto-degrades when missing)
- `numpy`: Numerical computation
- `torch`: Tensor operations (optional)

## Installation

```bash
# Full functionality (recommended)
pip install sentence-transformers torch numpy

# Basic functionality (no semantic matching)
pip install numpy
```

## Usage Examples

For detailed usage examples, please refer to the `fixed_test_tool_match.py` test script.

## Notes

1. The sentence-transformers model will be automatically downloaded on first use
2. The system automatically degrades to keyword matching when network is unavailable
3. It's recommended to use English keywords for tool descriptions for better matching results
4. Cache mechanism automatically manages memory usage, no manual cleanup required
