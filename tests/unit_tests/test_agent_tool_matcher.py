import importlib
import sys
import types
import urllib.request

import numpy as np
import pytest


def _matcher(monkeypatch, max_tools=3, min_similarity=0.0):
    module = importlib.import_module("flagscale.agent.tool_match.tool_matcher")
    monkeypatch.setattr(module.ToolMatcher, "_init_model", lambda self: None)
    return module.ToolMatcher(max_tools=max_tools, min_similarity=min_similarity)


def test_degradation_weights_and_errors(monkeypatch):
    matcher = _matcher(monkeypatch)

    matcher.set_degradation("semantic", True)
    assert matcher.get_degradation_status()["semantic"] is True
    assert matcher.get_effective_weights()["semantic"] == 0.0
    assert matcher.normalize_weights({"semantic": 0.0, "keyword": 0.2, "category": 0.1}) == {
        "semantic": 0.0,
        "keyword": pytest.approx(2 / 3),
        "category": pytest.approx(1 / 3),
    }

    with pytest.raises(ValueError, match="Unknown degradation"):
        matcher.set_degradation("missing", True)

    matcher.set_degradation("keyword", True)
    matcher.set_degradation("category", True)
    assert matcher.normalize_weights({"semantic": 0.0, "keyword": 0.0, "category": 0.0}) == {
        "semantic": 0.0,
        "keyword": 0.0,
        "category": 0.0,
    }
    matcher.reset_degradation()
    assert matcher.get_degradation_status() == {
        "semantic": False,
        "keyword": False,
        "category": False,
    }


def test_keyword_category_fit_and_match_without_semantic_model(monkeypatch):
    matcher = _matcher(monkeypatch, max_tools=2, min_similarity=0.01)
    tools = [
        {
            "function": {"name": "read_file", "description": "read file from disk"},
            "category": "file",
        },
        {
            "function": {"name": "search_docs", "description": "find query text"},
            "category": "search",
        },
        {"function": {"name": "run_command", "description": "execute system command"}},
    ]

    matcher.fit(tools)
    assert matcher.tools == tools
    assert matcher._calculate_semantic_score("read file", 0) == 0.0
    assert matcher._calculate_keyword_score("read file", tools[0]) == 1.0
    assert matcher._calculate_category_score("please read the file", tools[0]) == 1.0
    assert matcher._calculate_category_score("unknown", {"category": "general"}) == 0.5
    assert matcher._calculate_category_score("unknown", {"category": "custom"}) == 0.5

    results = matcher.match_tools("read file")
    assert results[0][0] == "read_file"
    assert len(results) <= 2
    assert matcher.match_tools("anything")

    empty_matcher = _matcher(monkeypatch)
    assert empty_matcher.match_tools("read file") == []


def test_embeddings_cache_lru_semantic_score_and_numpy_cosine(monkeypatch):
    matcher = _matcher(monkeypatch)
    fake_torch = types.ModuleType("torch")
    fake_torch.is_tensor = lambda value: False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, convert_to_tensor=True):
            self.calls.append(list(texts))
            return np.array([[1.0, 0.0]])

    model = FakeModel()
    matcher.model = model
    matcher.tools = [{"function": {"name": "tool"}}]
    matcher.tool_embeddings = np.array([[1.0, 0.0]])

    assert matcher._get_cached_embedding("task") is matcher._query_cache["task"]
    assert matcher._get_cached_embedding("task") is matcher._query_cache["task"]
    assert model.calls == [["task"]]

    matcher._cache_max_size = 1
    matcher._get_cached_embedding("other")
    assert list(matcher._query_cache.keys()) == ["other"]

    assert matcher._calculate_semantic_score("other", 0) == pytest.approx(1.0)
    assert matcher._cosine_similarity(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[
        0
    ] == pytest.approx(0.0)


def test_fit_embeddings_failures_and_score_exception_paths(monkeypatch):
    matcher = _matcher(monkeypatch)
    matcher.tools = [{"function": {"name": "tool", "description": "desc"}}]

    class FailingModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError("boom")

    matcher.model = FailingModel()
    matcher._fit_embeddings()
    assert matcher.tool_embeddings == []
    assert matcher._calculate_semantic_score("task", 0) == 0.0
    assert matcher._calculate_keyword_score("task", None) == 0.0
    assert matcher._calculate_category_score("task", None) == 0.0


def test_network_and_model_initialization_paths(monkeypatch):
    module = importlib.import_module("flagscale.agent.tool_match.tool_matcher")

    monkeypatch.setattr(urllib.request, "urlopen", lambda endpoint, timeout=3: object())
    assert module.ToolMatcher._check_network_connectivity(object()) is True

    class FakeSentenceTransformer:
        def __init__(self, name):
            self.name = name

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setattr(module.ToolMatcher, "_check_network_connectivity", lambda self: True)

    matcher = module.ToolMatcher()
    assert isinstance(matcher.model, FakeSentenceTransformer)
    assert matcher.get_degradation_status()["semantic"] is False


def test_model_initialization_degrades_when_network_or_import_fails(monkeypatch):
    module = importlib.reload(importlib.import_module("flagscale.agent.tool_match.tool_matcher"))

    class FakeSentenceTransformer:
        def __init__(self, name):
            raise RuntimeError("download failed")

    fake_sentence_transformers = types.ModuleType("sentence_transformers")
    fake_sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence_transformers)
    monkeypatch.setattr(module.ToolMatcher, "_check_network_connectivity", lambda self: False)

    matcher = module.ToolMatcher()
    assert matcher.model is None
    assert matcher.get_degradation_status()["semantic"] is True

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    module = importlib.reload(module)
    matcher = module.ToolMatcher()
    assert matcher.model is None
    assert matcher.get_degradation_status()["semantic"] is True
