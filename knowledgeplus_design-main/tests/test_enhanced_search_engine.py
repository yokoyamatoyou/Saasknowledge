import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from shared.search_engine import (  # noqa: E402
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


def test_calculate_recency_weight(empty_kb):
    engine = EnhancedHybridSearchEngine(str(empty_kb))
    recent = {"version_info": {"effective_date": datetime.now().strftime("%Y-%m-%d")}}
    old = {"version_info": {"effective_date": "2000-01-01"}}
    assert engine.calculate_recency_weight(recent) > engine.calculate_recency_weight(
        old
    )


def test_filter_latest_versions(empty_kb):
    engine = EnhancedHybridSearchEngine(str(empty_kb))
    chunks = [
        {
            "id": "c1",
            "metadata": {"version_info": {"superseded_by": "c2", "status": "active"}},
        },
        {
            "id": "c2",
            "metadata": {"version_info": {"superseded_by": None, "status": "active"}},
        },
    ]
    result = engine.filter_latest_versions(chunks)
    assert [c["id"] for c in result] == ["c2"]


def test_search_filters_and_conflicts(monkeypatch, empty_kb):
    engine = EnhancedHybridSearchEngine(str(empty_kb))

    monkeypatch.setattr(
        engine,
        "classify_query_intent",
        lambda q, client=None: {
            "needs_latest": True,
            "temporal_requirement": "latest",
            "scope": "company_wide",
            "rule_type": "承認権限",
        },
    )

    base_results = [
        {
            "id": "old",
            "text": "x",
            "metadata": {
                "version_info": {
                    "effective_date": "2000-01-01",
                    "superseded_by": "new",
                    "status": "active",
                },
                "hierarchy_info": {"approval_level": "company", "authority_score": 1.0},
            },
            "similarity": 0.9,
        },
        {
            "id": "new",
            "text": "x",
            "metadata": {
                "version_info": {
                    "effective_date": "2024-01-01",
                    "superseded_by": None,
                    "status": "active",
                },
                "hierarchy_info": {"approval_level": "company", "authority_score": 1.0},
            },
            "similarity": 0.8,
        },
        {
            "id": "new2",
            "text": "x",
            "metadata": {
                "version_info": {
                    "effective_date": "2024-06-01",
                    "superseded_by": None,
                    "status": "active",
                },
                "hierarchy_info": {"approval_level": "company", "authority_score": 1.0},
            },
            "similarity": 0.7,
        },
    ]

    def fake_super(
        self,
        query,
        top_k=15,
        threshold=0.075,
        vector_weight=None,
        bm25_weight=None,
        client=None,
    ):
        return base_results, False

    monkeypatch.setattr(HybridSearchEngine, "search", fake_super)

    monkeypatch.setattr(
        engine,
        "detect_rule_conflicts",
        lambda chunks, client=None: [
            {
                "rule_type": "承認権限",
                "conflicting_chunks": ["new"],
                "explanation": "x",
                "recommendation": "y",
            }
        ],
    )

    results, not_found = engine.search("q")
    assert not not_found
    assert len(results) == 2
    ids = {r["id"] for r in results}
    assert ids == {"new", "new2"}
    for r in results:
        if r["id"] == "new":
            assert r.get("conflicts")
