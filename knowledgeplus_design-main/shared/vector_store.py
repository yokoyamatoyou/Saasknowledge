import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """Simple in-memory vector store backed by files on disk."""

    def __init__(self, kb_path: str | Path) -> None:
        """Initialize the store.

        Parameters
        ----------
        kb_path : str or Path
            Path to the knowledge base directory containing ``chunks``,
            ``metadata`` and ``embeddings`` subfolders.

        Raises
        ------
        OSError
            If the paths cannot be accessed or created.
        """
        self.kb_path = Path(kb_path)
        self.chunks_path = self.kb_path / "chunks"
        self.metadata_path = self.kb_path / "metadata"
        self.embeddings_path = self.kb_path / "embeddings"

        # データを読み込み
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()

    def _load_chunks(self) -> Dict[str, Dict[str, Any]]:
        """Load chunk text and metadata from disk.

        Returns
        -------
        dict
            Mapping of chunk identifier to a dictionary containing ``text`` and
            ``metadata`` keys.

        Raises
        ------
        OSError
            If reading the files fails.
        json.JSONDecodeError
            If a metadata file contains invalid JSON.
        """
        chunks: Dict[str, Dict[str, Any]] = {}

        if self.chunks_path.exists():
            for chunk_file in self.chunks_path.glob("*.txt"):
                chunk_id = chunk_file.stem

                # チャンクテキストの読み込み
                with open(chunk_file, "r", encoding="utf-8") as f:
                    chunk_text = f.read()

                # メタデータの読み込み
                metadata = {}
                metadata_file = self.metadata_path / f"{chunk_id}.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)

                chunks[chunk_id] = {"text": chunk_text, "metadata": metadata}

        return chunks

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load chunk embeddings from disk.

        Returns
        -------
        dict
            Mapping of chunk identifier to embedding vectors stored as
            ``numpy.ndarray``.

        Raises
        ------
        OSError
            If the pickle files cannot be read.
        pickle.UnpicklingError
            If an embedding file is corrupted.
        """
        embeddings: Dict[str, np.ndarray] = {}

        if self.embeddings_path.exists():
            for emb_file in self.embeddings_path.glob("*.pkl"):
                chunk_id = emb_file.stem

                with open(emb_file, "rb") as f:
                    embeddings[chunk_id] = pickle.load(f)

        return embeddings

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Search for the most similar chunks to ``query_vector``.

        Parameters
        ----------
        query_vector : numpy.ndarray
            Embedding vector representing the query.
        top_k : int, optional
            Maximum number of results to return. ``5`` by default.
        threshold : float, optional
            Minimum cosine similarity required to include a result. ``0.6`` by
            default.

        Returns
        -------
        list of dict
            Search results sorted in descending order of similarity. Each result
            dictionary contains ``id``, ``text``, ``metadata`` and ``similarity``
            keys.

        Raises
        ------
        ValueError
            If ``query_vector`` has an incompatible shape.
        """
        results: List[Dict[str, Any]] = []

        for chunk_id, embedding in self.embeddings.items():
            if chunk_id in self.chunks:
                # コサイン類似度を計算
                similarity = cosine_similarity([query_vector], [embedding])[0][0]

                if similarity >= threshold:
                    results.append(
                        {
                            "id": chunk_id,
                            "text": self.chunks[chunk_id]["text"],
                            "metadata": self.chunks[chunk_id]["metadata"],
                            "similarity": float(similarity),
                        }
                    )

        # 類似度でソート（降順）
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # top_k件を返す
        return results[:top_k]


def initialize_vector_store(kb_path: str | Path) -> VectorStore:
    """Return a :class:`VectorStore` instance for ``kb_path``.

    Parameters
    ----------
    kb_path : str or Path
        Directory containing the vector store files.

    Returns
    -------
    VectorStore
        Initialized store for the given path.
    """
    return VectorStore(kb_path)
