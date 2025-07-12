import pytest
from pathlib import Path
import json
from unittest.mock import MagicMock, patch
import numpy as np
import logging

# Adjust the import path to allow importing from shared
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.search_engine import HybridSearchEngine, tokenize_text_for_bm25_internal

# Mock for OpenAI client
class MockOpenAIClient:
    def __init__(self):
        self.embeddings = MagicMock()

# Mock for OpenAI embeddings create method
class MockEmbeddingsResponse:
    def __init__(self, embedding_vectors):
        self.data = [MagicMock(embedding=v) for v in embedding_vectors]

@pytest.fixture
def mock_openai_client():
    client = MockOpenAIClient()
    # Mock a consistent embedding for testing
    def create(**kwargs):
        inputs = kwargs.get("input")
        if not isinstance(inputs, list):
            inputs = [inputs]
        return MockEmbeddingsResponse([[0.5] * 1536 for _ in inputs])

    client.embeddings.create.side_effect = create
    return client

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
        {"id": "doc1", "text": "apple banana orange", "metadata": {"filename": "doc1.txt"}},
        {"id": "doc2", "text": "grapefruit lemon lime", "metadata": {"filename": "doc2.txt"}},
        {"id": "doc3", "text": "strawberry blueberry raspberry", "metadata": {"filename": "doc3.txt"}},
    ]
    dummy_embeddings = {
        "doc1": [0.1] * 1536,
        "doc2": [0.9] * 1536,
        "doc3": [0.3] * 1536,
    }

    for chunk in dummy_chunks:
        with open(chunks_path / f"{chunk['id']}.txt", "w", encoding="utf-8") as f:
            f.write(chunk["text"])
        with open(metadata_path / f"{chunk['id']}.json", "w", encoding="utf-8") as f:
            json.dump(chunk["metadata"], f)
        with open(embeddings_path / f"{chunk['id']}.pkl", "wb") as f:
            import pickle
            pickle.dump({'embedding': dummy_embeddings[chunk['id']]}, f)
            
    # Create a dummy kb_metadata.json
    with open(kb_path / "kb_metadata.json", "w", encoding="utf-8") as f:
        json.dump({"embedding_model": "test-model"}, f)

    return kb_path, dummy_chunks, dummy_embeddings

def test_hybrid_search_returns_results(temp_kb_with_data, mock_openai_client, monkeypatch):
    kb_path, dummy_chunks, dummy_embeddings = temp_kb_with_data
    
    # Mock the internal get_embedding_from_openai to return a predictable query embedding
    monkeypatch.setattr(HybridSearchEngine, 'get_embedding_from_openai', 
                        lambda self, text, model_name=None, client=None: [0.8] * 1536) # Query similar to doc2

    engine = HybridSearchEngine(str(kb_path))
    
    # Ensure BM25 index is built (it should be during init)
    assert engine.bm25_index is not None
    assert len(engine.chunks) == 3 # All chunks should be loaded

    query = "search for something"
    results, not_found = engine.search(query, top_k=2, threshold=0.1)

    assert not not_found
    assert len(results) == 2
    
    # Check if results are sorted by similarity (highest first)
    assert results[0]['similarity'] >= results[1]['similarity']

    # Check if the most similar document (doc2) is among the top results
    # Based on the mocked embeddings, doc2 ([0.9]*1536) should be most similar to query ([0.8]*1536)
    doc_ids = [r['id'] for r in results]
    assert "doc2" in doc_ids
    
    # Check structure of results
    for res in results:
        assert "id" in res
        assert "text" in res
        assert "metadata" in res
        assert "similarity" in res
        assert "vector_score" in res
        assert "bm25_score" in res
        assert isinstance(res['similarity'], float)
        assert isinstance(res['vector_score'], float)
        assert isinstance(res['bm25_score'], float)

def test_reindex_functionality(temp_kb_with_data, monkeypatch):
    kb_path, dummy_chunks, dummy_embeddings = temp_kb_with_data
    
    # Initialize engine
    engine = HybridSearchEngine(str(kb_path))
    assert len(engine.chunks) == 3
    
    # Add a new dummy chunk after initialization
    new_chunk_id = "doc4"
    new_chunk_text = "newly added document content"
    new_chunk_embedding = [0.2] * 1536
    
    with open(kb_path / "chunks" / f"{new_chunk_id}.txt", "w", encoding="utf-8") as f:
        f.write(new_chunk_text)
    with open(kb_path / "metadata" / f"{new_chunk_id}.json", "w", encoding="utf-8") as f:
        json.dump({"filename": f"{new_chunk_id}.txt"}, f)
    with open(kb_path / "embeddings" / f"{new_chunk_id}.pkl", "wb") as f:
        import pickle
        pickle.dump({'embedding': new_chunk_embedding}, f)

    # Reindex
    engine.reindex()
    
    # Verify the new chunk is loaded
    assert len(engine.chunks) == 4
    assert any(c['id'] == new_chunk_id for c in engine.chunks)
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
    assert "this" not in tokens # Stopword

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

    assert any(
        "メタデータファイルが見つかりません" in r.message for r in caplog.records
    )
