import sys
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.metrics import SearchMetricsCollector, get_collector, get_report
from shared.search_engine import EnhancedHybridSearchEngine, HybridSearchEngine


def test_metrics_collector_updates_and_report():
    collector = SearchMetricsCollector()
    collector.log_search(
        "q1",
        [{"metadata": {"version_info": {"status": "active"}}}],
        1.0,
        cache_hit=True,
    )
    collector.log_search(
        "q2",
        [
            {"metadata": {}},
            {
                "metadata": {"version_info": {"status": "active"}},
                "conflicts": ["x"],
            },
        ],
        3.0,
        cache_hit=False,
    )
    report = collector.get_report()
    assert report["total_searches"] == 2
    assert abs(report["version_filtered"] - 1.5) < 1e-6
    assert report["conflict_detected"] == 1
    assert abs(report["avg_response_time"] - 2.0) < 1e-6
    assert abs(report["cache_hit_rate"] - 0.5) < 1e-6
    assert abs(report["conflict_rate"] - 0.5) < 1e-6
    assert any("ルール矛盾" in rec for rec in report["recommendations"])


@pytest.fixture
def empty_kb(tmp_path):
    kb_path = tmp_path / "kb"
    (kb_path / "chunks").mkdir(parents=True)
    (kb_path / "metadata").mkdir()
    (kb_path / "embeddings").mkdir()
    (kb_path / "kb_metadata.json").write_text("{}", encoding="utf-8")
    return kb_path


def test_enhanced_search_engine_logs_metrics(monkeypatch, empty_kb):
    collector = SearchMetricsCollector()
    import shared.metrics as metrics_module

    monkeypatch.setattr(metrics_module, "collector", collector, raising=False)
    monkeypatch.setattr(metrics_module, "get_collector", lambda: collector)

    engine = EnhancedHybridSearchEngine(str(empty_kb))

    monkeypatch.setattr(engine, "classify_query_intent", lambda q, client=None: {})
    monkeypatch.setattr(
        HybridSearchEngine,
        "search",
        lambda self, query, **kwargs: (
            [
                {
                    "id": "c1",
                    "text": "x",
                    "metadata": {"version_info": {"status": "active"}},
                    "similarity": 0.5,
                }
            ],
            False,
        ),
    )
    monkeypatch.setattr(engine, "detect_rule_conflicts", lambda chunks, client=None: [])

    engine.search("query")

    report = collector.get_report()
    assert report["total_searches"] == 1
    assert report["version_filtered"] > 0
