from __future__ import annotations

from typing import List


class SearchMetricsCollector:
    """Collect and report search metrics."""

    def __init__(self) -> None:
        self.metrics = {
            "total_searches": 0,
            "version_filtered": 0,
            "conflict_detected": 0,
            "avg_response_time": 0,
            "cache_hit_rate": 0,
        }

    def log_search(
        self, query: str, results: List[dict], execution_time: float, cache_hit: bool = False
    ) -> None:
        """Record a search execution."""
        self.metrics["total_searches"] += 1

        version_filtered = sum(
            1
            for r in results
            if r.get("metadata", {}).get("version_info", {}).get("status") == "active"
        )
        self.metrics["version_filtered"] += version_filtered / max(len(results), 1)

        conflicts_found = sum(1 for r in results if r.get("conflicts"))
        if conflicts_found > 0:
            self.metrics["conflict_detected"] += 1

        n = self.metrics["total_searches"]
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (n - 1) + execution_time) / n
        )

        if cache_hit:
            self.metrics["cache_hit_rate"] = (
                (self.metrics["cache_hit_rate"] * (n - 1) + 1) / n
            )
        else:
            self.metrics["cache_hit_rate"] = (
                (self.metrics["cache_hit_rate"] * (n - 1)) / n
            )

    def get_report(self) -> dict:
        """Return a metrics summary with improvement tips."""
        conflict_rate = self.metrics["conflict_detected"] / max(
            self.metrics["total_searches"], 1
        )
        self.metrics["conflict_rate"] = conflict_rate
        return {
            **self.metrics,
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate improvement suggestions from metrics."""
        recommendations: List[str] = []

        if self.metrics["avg_response_time"] > 2.0:
            recommendations.append(
                "検索速度が遅いです。インデックスの再構築を検討してください。"
            )

        if self.metrics.get("conflict_rate", 0) > 0.3:
            recommendations.append(
                "ルール矛盾が多く検出されています。文書の標準化を推進してください。"
            )

        if self.metrics["cache_hit_rate"] < 0.2:
            recommendations.append(
                "キャッシュヒット率が低いです。キャッシュサイズの拡大を検討してください。"
            )

        return recommendations


# default collector used by the search engine
collector = SearchMetricsCollector()


def get_collector() -> SearchMetricsCollector:
    """Return the global metrics collector."""
    return collector


def get_report() -> dict:
    """Return metrics report from the global collector."""
    return collector.get_report()
