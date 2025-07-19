import json
import logging
from pathlib import Path
from typing import Dict, Optional

_DEFAULT_METRICS = (
    Path(__file__).resolve().parents[1] / "data" / "experiment_metrics.json"
)
_DEFAULT_DEPLOY = Path(__file__).resolve().parents[1] / "data" / "active_algorithm.txt"

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Track algorithm performance and deploy the best one."""

    def __init__(
        self,
        metrics_path: Path = _DEFAULT_METRICS,
        deploy_path: Path = _DEFAULT_DEPLOY,
    ) -> None:
        self.metrics_path = metrics_path
        self.deploy_path = deploy_path
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict[str, Dict[str, int]]:
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {
                        k: {
                            "runs": int(v.get("runs", 0)),
                            "success": int(v.get("success", 0)),
                        }
                        for k, v in data.items()
                    }
            except Exception as e:  # pragma: no cover - log only
                logger.warning("Failed to load metrics: %s", e)
        return {}

    def _save_metrics(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

    def record_result(self, algorithm: str, success: bool) -> None:
        stats = self.metrics.setdefault(algorithm, {"runs": 0, "success": 0})
        stats["runs"] += 1
        if success:
            stats["success"] += 1
        self._save_metrics()

    def select_best(self, min_runs: int = 5) -> Optional[str]:
        best = None
        best_rate = -1.0
        for alg, stats in self.metrics.items():
            if stats["runs"] < min_runs:
                continue
            rate = stats["success"] / stats["runs"] if stats["runs"] else 0.0
            if rate > best_rate:
                best_rate = rate
                best = alg
        return best

    def deploy_best(self, min_runs: int = 5) -> Optional[str]:
        alg = self.select_best(min_runs)
        if alg:
            self.deploy_path.parent.mkdir(parents=True, exist_ok=True)
            self.deploy_path.write_text(alg, encoding="utf-8")
        return alg

    def get_active_algorithm(self, fallback: str) -> str:
        if self.deploy_path.exists():
            alg = self.deploy_path.read_text(encoding="utf-8").strip()
            if alg:
                return alg
        return fallback
