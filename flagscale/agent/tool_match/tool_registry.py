"""Tool registry for managing available tools"""

from typing import Any

from .tool_matcher import ToolMatcher


class ToolRegistry:
    """Registry for managing and searching tools"""

    def __init__(self, max_tools: int = 3, min_similarity: float = 0.1):
        self.tools = {}  # Changed to dict for O(1) lookups
        self.categories = {}
        self.matcher = ToolMatcher(max_tools, min_similarity)
        self._needs_refit = False

    def register_tool(self, tool: dict[str, Any], category: str = "general"):
        """Register a new tool"""
        self._register_tool_internal(tool, category)

        # Mark that refit is needed (lazy retraining)
        self._needs_refit = True

    def register_tools(self, tools: list[dict[str, Any]], category: str = "general"):
        """Register multiple tools efficiently"""
        for tool in tools:
            # Use internal method to avoid multiple refits
            self._register_tool_internal(tool, category)
        # Single refit after all tools are registered
        self._needs_refit = True

    def _register_tool_internal(self, tool: dict[str, Any], category: str = "general"):
        """Internal method for registering tool without refit"""
        if "function" not in tool:
            return

        func = tool["function"]
        if "name" not in func:
            return

        tool_name = func["name"]

        # Add category
        tool["category"] = category

        # Remove duplicate (O(1) operation now)
        if tool_name in self.tools:
            # Remove from old category if exists
            old_tool = self.tools[tool_name]
            old_category = old_tool.get("category", "general")
            if old_category in self.categories and tool_name in self.categories[old_category]:
                self.categories[old_category].remove(tool_name)
                if not self.categories[old_category]:  # Remove empty category
                    del self.categories[old_category]

        # Add tool to dictionary
        self.tools[tool_name] = tool

        # Update categories
        if category not in self.categories:
            self.categories[category] = []
        if tool_name not in self.categories[category]:
            self.categories[category].append(tool_name)

    def get_tool_by_name(self, name: str) -> dict[str, Any] | None:
        """Get tool by name - O(1) lookup"""
        return self.tools.get(name)

    def get_tools_by_category(self, category: str) -> list[dict[str, Any]]:
        """Get tools by category"""
        if category not in self.categories:
            return []
        return [
            self.tools[tool_name]
            for tool_name in self.categories[category]
            if tool_name in self.tools
        ]

    def search_tools(self, query: str, category: str | None = None) -> list[tuple[str, float]]:
        """Search for tools with lazy refit"""
        # Ensure matcher is up to date
        if self._needs_refit:
            self.matcher.fit(self.get_all_tools())
            self._needs_refit = False

        if category:
            # Perform search on all tools first, then filter by category
            all_matched_tools = self.matcher.match_tools(query)
            tool_names_in_category = set(self.categories.get(category, []))
            return [
                (name, score) for name, score in all_matched_tools if name in tool_names_in_category
            ]
        else:
            return self.matcher.match_tools(query)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools"""
        return list(self.tools.values())

    def get_categories(self) -> list[str]:
        """Get all categories"""
        return list(self.categories.keys())

    def clear_cache(self):
        """Clear matcher cache for fresh start"""
        if hasattr(self.matcher, "_query_cache"):
            self.matcher._query_cache.clear()
        self._needs_refit = True

    def set_degradation(self, component: str, degraded: bool = True):
        """Set degradation flag for a specific scoring component.

        Args:
            component: The scoring component ('semantic', 'keyword', 'category')
            degraded: Whether to degrade (set weight to 0) or restore the component
        """
        self.matcher.set_degradation(component, degraded)

    def get_degradation_status(self) -> dict[str, bool]:
        """Get current degradation status of all components."""
        return self.matcher.get_degradation_status()

    def reset_degradation(self):
        """Reset all degradation flags to False."""
        self.matcher.reset_degradation()

    def get_effective_weights(self) -> dict[str, float]:
        """Get effective weights considering degradation flags."""
        return self.matcher.get_effective_weights()

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "categories": len(self.categories),
            "category_breakdown": {cat: len(tools) for cat, tools in self.categories.items()},
            "needs_refit": self._needs_refit,
            "cache_size": len(getattr(self.matcher, "_query_cache", {})),
            "degradation_status": self.get_degradation_status(),
            "effective_weights": self.get_effective_weights(),
        }
