from pathlib import Path

from shared.experiment_manager import ExperimentManager


def test_record_and_deploy(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    deploy = tmp_path / "deploy.txt"
    mgr = ExperimentManager(metrics, deploy)

    for _ in range(10):
        mgr.record_result("algo_a", True)
    for _ in range(10):
        mgr.record_result("algo_b", False)

    best = mgr.deploy_best(min_runs=5)
    assert best == "algo_a"
    assert deploy.read_text(encoding="utf-8") == "algo_a"
    assert mgr.get_active_algorithm("default") == "algo_a"
