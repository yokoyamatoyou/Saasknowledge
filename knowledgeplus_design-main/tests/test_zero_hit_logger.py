# ruff: noqa: E402
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.zero_hit_logger import load_zero_hit_queries, log_zero_hit_query


def test_log_and_load(tmp_path):
    path = tmp_path / "zero.log"
    log_zero_hit_query("foo", path)
    log_zero_hit_query("bar", path)
    assert load_zero_hit_queries(path) == ["foo", "bar"]


def test_log_ignores_empty(tmp_path):
    path = tmp_path / "zero.log"
    log_zero_hit_query("", path)
    assert load_zero_hit_queries(path) == []
