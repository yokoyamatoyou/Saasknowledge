import json
import logging

# Adjust the import path to allow importing from shared
import sys
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from config import EMBEDDING_DIM  # noqa: E402
from core import mm_builder_utils  # noqa: E402
from shared.search_engine import HybridSearchEngine  # noqa: E402
from shared.search_engine import tokenize_text_for_bm25_internal  # noqa: E402

# Mock for OpenAI client


@pytest.fixture
def temp_kb_with_data(tmp_path):
    kb_path = tmp_path / "test_kb"
    chunks_path = kb_path / "chunks"
    metadata_path = kb_path / "metadata"
    embeddings_path = kb_path / "embeddings"

    chunks_path.mkdir(parents=True)
    metadata_path.mkdir(parents=True)
    embeddings_path.mkdir(parents=True)

    # Create dummy chunks, metadata, and embeddings
    dummy_chunks = [
        {
            "id": "doc1",
            "text": "apple banana orange",
            "metadata": {"filename": "doc1.txt"},
        },
        {
            "id": "doc2",
            "text": "grapefruit lemon lime",
            "metadata": {"filename": "doc2.txt"},
        },
        {
            "id": "doc3",
            "text": "strawberry blueberry raspberry",
            "metadata": {"filename": "doc3.txt"},
        },
    ]
    dummy_embeddings = {
        "doc1": [0.1] * EMBEDDING_DIM,
        "doc2": [0.9] * EMBEDDING_DIM,
        "doc3": [0.3] * EMBEDDING_DIM,
    }

    for chunk in dummy_chunks:
        with open(chunks_path / f"{chunk['id']}.txt", "w", encoding="utf-8") as f:
            f.write(chunk["text"])
        with open(metadata_path / f"{chunk['id']}.json", "w", encoding="utf-8") as f:
            json.dump(chunk["metadata"], f)
        with open(embeddings_path / f"{chunk['id']}.pkl", "wb") as f:
            import pickle

            pickle.dump({"embedding": dummy_embeddings[chunk["id"]]}, f)

    # Create a dummy kb_metadata.json
    with open(kb_path / "kb_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"embedding_model": "test-model"}, f)

    return kb_path, dummy_chunks, dummy_embeddings


def test_hybrid_search_returns_results(temp_kb_with_data, monkeypatch):
    kb_path, dummy_chunks, dummy_embeddings = temp_kb_with_data

    monkeypatch.setattr(
        mm_builder_utils,
        "load_model_and_processor",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        HybridSearchEngine,
        "get_clip_text_embedding",
        lambda self, text: [0.8] * EMBEDDING_DIM,
    )  # Query similar to doc2

    engine = HybridSearchEngine(str(kb_path))

    # Ensure BM25 index is built (it should be during init)
    assert engine.bm25_index is not None
    assert len(engine.chunks) == 3  # All chunks should be loaded

    query = "search for something"
    results, not_found = engine.search(query, top_k=2, threshold=0.1)

    assert not not_found
    assert len(results) == 2

    # Check if results are sorted by similarity (highest first)
    assert results[0]["similarity"] >= results[1]["similarity"]

    # Check if the most similar document (doc2) is among the top results
    # Based on the mocked embeddings, doc2 should be most similar to the query
    doc_ids = [r["id"] for r in results]
    assert "doc2" in doc_ids

    # Check structure of results
    for res in results:
        assert "id" in res
        assert "text" in res
        assert "metadata" in res
        assert "similarity" in res
        assert "vector_score" in res
        assert "bm25_score" in res
        assert "content_type" in res
        assert isinstance(res["similarity"], float)
        assert isinstance(res["vector_score"], float)
        assert isinstance(res["bm25_score"], float)


def test_reindex_functionality(temp_kb_with_data, monkeypatch):
    kb_path, dummy_chunks, dummy_embeddings = temp_kb_with_data

    monkeypatch.setattr(
        mm_builder_utils,
        "load_model_and_processor",
        lambda: (object(), object()),
    )
    monkeypatch.setattr(
        HybridSearchEngine,
        "get_clip_text_embedding",
        lambda self, text: [0.1] * EMBEDDING_DIM,
    )

    # Initialize engine
    engine = HybridSearchEngine(str(kb_path))
    assert len(engine.chunks) == 3

    # Add a new dummy chunk after initialization
    new_chunk_id = "doc4"
    new_chunk_text = "newly added document content"
    new_chunk_embedding = [0.2] * EMBEDDING_DIM

    with open(kb_path / "chunks" / f"{new_chunk_id}.txt", "w", encoding="utf-8") as f:
        f.write(new_chunk_text)
    with open(
        kb_path / "metadata" / f"{new_chunk_id}.json", "w", encoding="utf-8"
    ) as f:
        json.dump({"filename": f"{new_chunk_id}.txt"}, f)
    with open(kb_path / "embeddings" / f"{new_chunk_id}.pkl", "wb") as f:
        import pickle

        pickle.dump({"embedding": new_chunk_embedding}, f)

    # Reindex
    engine.reindex()

    # Verify the new chunk is loaded
    assert len(engine.chunks) == 4
    assert any(c["id"] == new_chunk_id for c in engine.chunks)
    assert new_chunk_id in engine.embeddings

    # Verify BM25 index is rebuilt and includes the new chunk
    # This is an indirect check, as direct inspection of BM25 internal state is hard
    # We can check if the tokenized corpus size matches the new chunk count
    assert len(engine.tokenized_corpus_for_bm25) == 4


def test_bm25_tokenization_internal():
    # Test with normal text
    tokens = tokenize_text_for_bm25_internal("This is a test sentence.")
    assert "test" in tokens
    assert "sentence" in tokens
    assert "this" not in tokens  # Stopword

    # Test with Japanese text (assuming SudachiPy is available and working)
    # If SudachiPy is not available, it falls back to regex, which might not remove Japanese stopwords
    japanese_text = "これはテストの文章です。"
    tokens_jp = tokenize_text_for_bm25_internal(japanese_text)
    # Depending on SudachiPy/regex fallback, results might vary.
    # Just ensure it's not empty and contains some expected tokens.
    assert len(tokens_jp) > 0
    assert "テスト" in tokens_jp or "test" in tokens_jp.lower()

    # Test with empty string
    tokens_empty = tokenize_text_for_bm25_internal("")
    assert tokens_empty == ["<bm25_empty_input_token>"]

    # Test with text containing only stopwords
    tokens_stopwords = tokenize_text_for_bm25_internal("the a is")
    assert tokens_stopwords == ["<bm25_all_stopwords_token>"]

    # Test with text that becomes empty after tokenization/stopwords removal
    tokens_empty_after_processing = tokenize_text_for_bm25_internal(" ")
    assert tokens_empty_after_processing == ["<bm25_empty_input_token>"]

    tokens_empty_after_processing_jp = tokenize_text_for_bm25_internal("の に は")
    assert tokens_empty_after_processing_jp == ["<bm25_all_stopwords_token>"]


def test_missing_kb_metadata_logs_warning(tmp_path, caplog):
    kb_dir = tmp_path / "kb_no_meta"
    (kb_dir / "chunks").mkdir(parents=True)
    (kb_dir / "metadata").mkdir()
    (kb_dir / "embeddings").mkdir()

    with caplog.at_level(logging.WARNING):
        HybridSearchEngine(str(kb_dir))

    assert any("メタデータファイルが見つかりません" in r.message for r in caplog.records)


def test_malformed_chunk_skipped(tmp_path):
    """Engine should ignore unreadable chunk files without crashing."""
    kb_dir = tmp_path / "mal_kb"
    chunks = kb_dir / "chunks"
    meta = kb_dir / "metadata"
    embeds = kb_dir / "embeddings"
    chunks.mkdir(parents=True)
    meta.mkdir()
    embeds.mkdir()

    # valid chunk
    (chunks / "good.txt").write_text("hello", encoding="utf-8")
    (meta / "good.json").write_text("{}", encoding="utf-8")
    with open(embeds / "good.pkl", "wb") as f:
        import pickle

        pickle.dump({"embedding": [0.1]}, f)

    # malformed chunk file that cannot be read as UTF-8
    with open(chunks / "bad.txt", "wb") as f:
        f.write(b"\xff\xfe\xfa")
    (meta / "bad.json").write_text("{}", encoding="utf-8")
    with open(embeds / "bad.pkl", "wb") as f:
        import pickle

        pickle.dump({"embedding": [0.2]}, f)

    engine = HybridSearchEngine(str(kb_dir))

    ids = [c["id"] for c in engine.chunks]
    assert "good" in ids
    assert "bad" not in ids
