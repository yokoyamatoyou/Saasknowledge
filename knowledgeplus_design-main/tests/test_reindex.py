import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from shared import upload_utils  # noqa: E402

pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")
pytest.importorskip("sentence_transformers")
pytest.importorskip("nltk")
from shared.search_engine import HybridSearchEngine  # noqa: E402


def test_reindex_loads_new_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    kb_name = "kb"
    # save two chunks
    upload_utils.save_processed_data(
        kb_name, "1", chunk_text="apple", embedding=[1], metadata={}
    )
    upload_utils.save_processed_data(
        kb_name, "2", chunk_text="banana", embedding=[2], metadata={}
    )
    engine = HybridSearchEngine(str(tmp_path / kb_name))
    assert len(engine.chunks) == 2

    # save another chunk after engine initialization
    upload_utils.save_processed_data(
        kb_name, "3", chunk_text="orange", embedding=[3], metadata={}
    )
    assert len(engine.chunks) == 2  # engine not aware yet

    engine.reindex()
    assert len(engine.chunks) == 3


def test_reindex_removes_deleted_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    kb_name = "kb"
    upload_utils.save_processed_data(
        kb_name, "1", chunk_text="apple", embedding=[1], metadata={}
    )
    upload_utils.save_processed_data(
        kb_name, "2", chunk_text="banana", embedding=[2], metadata={}
    )
    engine = HybridSearchEngine(str(tmp_path / kb_name))
    assert len(engine.chunks) == 2

    # remove one chunk on disk after the engine has loaded
    (tmp_path / kb_name / "chunks" / "1.txt").unlink()
    (tmp_path / kb_name / "metadata" / "1.json").unlink()
    (tmp_path / kb_name / "embeddings" / "1.pkl").unlink()

    engine.reindex()

    # verify the deleted chunk is no longer present
    assert len(engine.chunks) == 1
    assert all(c["id"] != "1" for c in engine.chunks)
