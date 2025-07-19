import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from shared.search_engine import (  # noqa: E402
    CachedEnhancedSearchEngine,
    EnhancedHybridSearchEngine,
    HybridSearchEngine,
)


@pytest.fixture
def empty_kb(tmp_path):
    kb_path = tmp_path / "kb"
    (kb_path / "chunks").mkdir(parents=True)
    (kb_path / "metadata").mkdir()
    (kb_path / "embeddings").mkdir()
    (kb_path / "kb_metadata.json").write_text("{}", encoding="utf-8")
    return kb_path


def test_classify_query_intent_cached(monkeypatch, empty_kb):
    engine = CachedEnhancedSearchEngine(str(empty_kb))

    calls = {"count": 0}

    def fake_parent(self, query, client=None):
        calls["count"] += 1
        return {"primary_intent": "general"}

    monkeypatch.setattr(
        EnhancedHybridSearchEngine,
        "classify_query_intent",
        fake_parent,
        raising=False,
    )

    assert engine.classify_query_intent("hello") == {"primary_intent": "general"}
    assert engine.classify_query_intent("hello") == {"primary_intent": "general"}
    assert calls["count"] == 1


def test_search_uses_intent_cache(monkeypatch, empty_kb):
    engine = CachedEnhancedSearchEngine(str(empty_kb))

    calls = {"count": 0}

    def fake_parent(self, query, client=None):
        calls["count"] += 1
        return {}

    monkeypatch.setattr(
        EnhancedHybridSearchEngine,
        "classify_query_intent",
        fake_parent,
        raising=False,
    )
    monkeypatch.setattr(HybridSearchEngine, "search", lambda *a, **k: ([], True))

    engine.search("foo")
    engine.search("foo")
    assert calls["count"] == 1
