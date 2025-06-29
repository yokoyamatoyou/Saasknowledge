import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared import upload_utils
import pytest

pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")
pytest.importorskip("sentence_transformers")
pytest.importorskip("nltk")
from shared.search_engine import HybridSearchEngine


def test_reindex_loads_new_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    kb_name = "kb"
    # save two chunks
    upload_utils.save_processed_data(kb_name, "1", chunk_text="apple", embedding=[1], metadata={})
    upload_utils.save_processed_data(kb_name, "2", chunk_text="banana", embedding=[2], metadata={})
    engine = HybridSearchEngine(str(tmp_path / kb_name))
    assert len(engine.chunks) == 2

    # save another chunk after engine initialization
    upload_utils.save_processed_data(kb_name, "3", chunk_text="orange", embedding=[3], metadata={})
    assert len(engine.chunks) == 2  # engine not aware yet

    engine.reindex()
    assert len(engine.chunks) == 3


def test_reindex_removes_deleted_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    kb_name = "kb"
    upload_utils.save_processed_data(kb_name, "1", chunk_text="apple", embedding=[1], metadata={})
    upload_utils.save_processed_data(kb_name, "2", chunk_text="banana", embedding=[2], metadata={})
    engine = HybridSearchEngine(str(tmp_path / kb_name))
    assert len(engine.chunks) == 2
