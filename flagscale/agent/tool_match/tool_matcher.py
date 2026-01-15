"""Intelligent tool matcher using semantic embeddings"""

import logging
from collections import OrderedDict
from typing import Any

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class ToolMatcher:
    """Semantic tool matcher with multi-weight scoring and degradation mechanism."""

    def __init__(self, max_tools: int = 3, min_similarity: float = 0.1):
        self.max_tools = max_tools
        self.min_similarity = min_similarity
        self.tools = []
        self.tool_embeddings = []
        self.model = None
        self._query_cache = OrderedDict()  # LRU Cache for query embeddings
        self._cache_max_size = 100  # Maximum cache size

        # Multi-weight scoring system
        self.scoring_weights = {
            "semantic": 0.7,  # Semantic similarity weight
            "keyword": 0.2,  # Keyword matching weight
            "category": 0.1,  # Category relevance weight
        }

        # Degradation flags - when True, corresponding weight is set to 0
        self.degradation_flags = {"semantic": False, "keyword": False, "category": False}

        self._init_model()

    def set_degradation(self, component: str, degraded: bool = True):
        """Set degradation flag for a specific scoring component.

        Args:
            component: The scoring component ('semantic', 'keyword', 'category')
            degraded: Whether to degrade (set weight to 0) or restore the component
        """
        if component in self.degradation_flags:
            self.degradation_flags[component] = degraded
            logger.info(f"Component '{component}' degradation set to {degraded}")
        else:
            raise ValueError(f"Unknown degradation component: '{component}'")

    def get_effective_weights(self) -> dict[str, float]:
        """Get effective weights considering degradation flags."""
        effective_weights = {}
        for component, weight in self.scoring_weights.items():
            if self.degradation_flags[component]:
                effective_weights[component] = 0.0
            else:
                effective_weights[component] = weight
        return effective_weights

    def normalize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights so they sum to 1.0."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            # If all weights are 0, distribute equally among non-degraded components
            non_degraded = {
                k: v for k, v in self.scoring_weights.items() if not self.degradation_flags[k]
            }
            if non_degraded:
                equal_weight = 1.0 / len(non_degraded)
                return {k: equal_weight for k in non_degraded.keys()}
            else:
                return weights

        return {k: v / total_weight for k, v in weights.items()}

    def _init_model(self):
        """Initialize sentence transformer model with improved error handling"""
        try:
            # Check if sentence-transformers is available first
            from sentence_transformers import SentenceTransformer

            # Check network connectivity with multiple fallbacks
            if not self._check_network_connectivity():
                logger.warning(
                    "Network unavailable for model download. Degrading semantic component."
                )
                self.model = None
                self.set_degradation("semantic", True)
                return

            # Try to load model with local cache first
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Semantic model initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to load model (may need download): {e}")
                logger.warning("Degrading semantic component.")
                self.model = None
                self.set_degradation("semantic", True)

        except ImportError:
            logger.warning(
                "sentence-transformers not available. Install with: pip install sentence-transformers"
            )
            logger.warning("Degrading semantic component.")
            self.model = None
            self.set_degradation("semantic", True)
        except Exception as e:
            logger.error(f"Failed to initialize semantic model: {e}. Degrading semantic component.")
            self.model = None
            self.set_degradation("semantic", True)

    def _check_network_connectivity(self) -> bool:
        """Check network connectivity with  endpoint"""
        import urllib.request

        endpoints = ["https://huggingface.co"]

        for endpoint in endpoints:
            try:
                urllib.request.urlopen(endpoint, timeout=3)
                return True
            except (TimeoutError, urllib.error.URLError):
                continue

        return False

    def _calculate_semantic_score(self, task: str, tool_index: int) -> float:
        """Calculate semantic similarity score for a tool."""
        if not self.model or len(self.tool_embeddings) == 0:
            return 0.0

        try:
            task_embedding = self._get_cached_embedding(task)
            similarity = self._cosine_similarity(
                task_embedding, self.tool_embeddings[tool_index : tool_index + 1]
            )
            # Handle both scalar and array results
            if hasattr(similarity, "__len__") and len(similarity) > 0:
                return float(similarity[0])
            elif hasattr(similarity, "item"):  # numpy scalar
                return float(similarity.item())
            else:
                return float(similarity)
        except Exception as e:
            logger.error(f"Semantic scoring failed: {e}")
            return 0.0

    def _calculate_keyword_score(self, task: str, tool: dict[str, Any]) -> float:
        """Calculate keyword matching score for a tool."""
        try:
            func = tool.get("function", {})
            name = func.get("name", "").lower()
            desc = func.get("description", "").lower()
            task_lower = task.lower()

            # Extract keywords from task (simple word splitting)
            task_keywords = set(task_lower.split())

            # Calculate keyword overlap
            tool_text = f"{name} {desc}"
            tool_keywords = set(tool_text.split())

            if not task_keywords:
                return 0.0

            overlap = len(task_keywords.intersection(tool_keywords))
            return overlap / len(task_keywords)
        except Exception as e:
            logger.error(f"Keyword scoring failed: {e}")
            return 0.0

    def _calculate_category_score(self, task: str, tool: dict[str, Any]) -> float:
        """Calculate category relevance score for a tool."""
        try:
            category = tool.get("category", "general").lower()
            task_lower = task.lower()

            # Simple category matching based on keywords
            category_keywords = {
                "general": 0.5,
                "file": (
                    1.0
                    if any(word in task_lower for word in ["file", "read", "write", "save", "load"])
                    else 0.0
                ),
                "search": (
                    1.0
                    if any(word in task_lower for word in ["search", "find", "look", "query"])
                    else 0.0
                ),
                "data": (
                    1.0
                    if any(
                        word in task_lower for word in ["data", "process", "analyze", "transform"]
                    )
                    else 0.0
                ),
                "network": (
                    1.0
                    if any(word in task_lower for word in ["network", "url", "http", "api"])
                    else 0.0
                ),
                "system": (
                    1.0
                    if any(word in task_lower for word in ["system", "command", "run", "execute"])
                    else 0.0
                ),
            }

            return category_keywords.get(category, 0.5)
        except Exception as e:
            logger.error(f"Category scoring failed: {e}")
            return 0.0

    def fit(self, tools: list[dict[str, Any]]):
        """Train matcher with tools"""
        self.tools = tools
        if self.model:
            self._fit_embeddings()

    def _fit_embeddings(self):
        """Generate embeddings for tools"""
        try:
            tool_texts = []
            for tool in self.tools:
                func = tool.get("function", {})
                name = func.get("name", "")
                desc = func.get("description", "")
                tool_texts.append(f"{name} {desc}".strip())

            if tool_texts:
                self.tool_embeddings = self.model.encode(tool_texts, convert_to_tensor=True)
                logger.info(f"Generated embeddings for {len(self.tools)} tools")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self.tool_embeddings = []

    def match_tools(self, task: str) -> list[tuple[str, float]]:
        """Match task with relevant tools using multi-weight scoring.
        When components are degraded, their weights are set to 0.
        """
        if not self.tools:
            return []

        # Get effective weights considering degradation
        effective_weights = self.get_effective_weights()
        normalized_weights = self.normalize_weights(effective_weights)

        # Calculate scores for all tools
        tool_scores = []
        for i, tool in enumerate(self.tools):
            name = tool.get("function", {}).get("name", f"tool_{i}")

            # Calculate individual component scores
            scores = {}
            scores["semantic"] = self._calculate_semantic_score(task, i)
            scores["keyword"] = self._calculate_keyword_score(task, tool)
            scores["category"] = self._calculate_category_score(task, tool)

            # Calculate weighted final score
            final_score = sum(
                scores[component] * normalized_weights[component] for component in scores
            )

            # Apply minimum similarity threshold
            if final_score >= self.min_similarity:
                tool_scores.append((name, final_score))

        # Sort by score and return top tools
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores[: self.max_tools]

    def get_degradation_status(self) -> dict[str, bool]:
        """Get current degradation status of all components."""
        return self.degradation_flags.copy()

    def reset_degradation(self):
        """Reset all degradation flags to False."""
        for component in self.degradation_flags:
            self.degradation_flags[component] = False
        logger.info("All degradation flags reset")

    def _get_cached_embedding(self, task: str):
        """Get embedding from cache or compute and cache it - LRU cache"""
        if task in self._query_cache:
            # Move to end (most recently used)
            self._query_cache.move_to_end(task)
            return self._query_cache[task]

        # Compute new embedding
        embedding = self.model.encode([task], convert_to_tensor=True)

        # Manage cache size with LRU eviction
        if len(self._query_cache) >= self._cache_max_size:
            # Remove least recently used entry (first item)
            self._query_cache.popitem(last=False)

        self._query_cache[task] = embedding
        return embedding

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity"""
        try:
            import torch

            if torch.is_tensor(a) and torch.is_tensor(b):
                a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
                b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
                similarity = torch.mm(a_norm, b_norm.t())
                result = similarity.squeeze().cpu().numpy()
                # Ensure result is always an array
                return np.array([result]) if result.ndim == 0 else result
        except ImportError:
            pass

        # Fallback to numpy
        try:
            a_np = a.cpu().numpy() if hasattr(a, "cpu") else np.array(a)
            b_np = b.cpu().numpy() if hasattr(b, "cpu") else np.array(b)

            a_norm = a_np / np.linalg.norm(a_np, axis=1, keepdims=True)
            b_norm = b_np / np.linalg.norm(b_np, axis=1, keepdims=True)

            similarity = np.dot(a_norm, b_norm.T)
            result = similarity.squeeze()
            # Ensure result is always an array
            return np.array([result]) if result.ndim == 0 else result
        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return np.zeros(len(self.tools))
