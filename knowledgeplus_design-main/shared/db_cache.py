import json
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List


def _db_path(kb_name_or_path: str | Path) -> Path:
    path = Path(kb_name_or_path)
    if path.is_absolute() or os.sep in str(kb_name_or_path):
        kb_dir = path
    else:
        from . import upload_utils

        kb_dir = upload_utils.BASE_KNOWLEDGE_DIR / path
    kb_dir.mkdir(parents=True, exist_ok=True)
    return kb_dir / "kb_cache.db"


def init_db(kb_name: str | Path) -> Path:
    """Initialize the cache database for a knowledge base."""
    path = _db_path(kb_name)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings (id TEXT PRIMARY KEY, vector BLOB)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS token_tokens (id TEXT PRIMARY KEY, tokens TEXT)"
    )
    conn.commit()
    conn.close()
    return path


def save_embedding(kb_name: str | Path, chunk_id: str, vector: List[float]) -> None:
    """Persist an embedding vector for ``chunk_id``."""
    init_db(kb_name)
    path = _db_path(kb_name)
    conn = sqlite3.connect(path)
    conn.execute(
        "REPLACE INTO embeddings (id, vector) VALUES (?, ?)",
        (chunk_id, sqlite3.Binary(pickle.dumps(vector))),
    )
    conn.commit()
    conn.close()


def load_embeddings(kb_name: str | Path) -> Dict[str, List[float]]:
    """Load all embeddings for ``kb_name`` from the cache."""
    path = _db_path(kb_name)
    if not path.exists():
        return {}
    conn = sqlite3.connect(path)
    cur = conn.execute("SELECT id, vector FROM embeddings")
    rows = cur.fetchall()
    conn.close()
    result: Dict[str, List[float]] = {}
    for cid, blob in rows:
        try:
            result[cid] = pickle.loads(blob)
        except Exception:
            continue
    return result


def save_token_list(kb_name: str | Path, chunk_id: str, tokens: List[str]) -> None:
    """Persist tokenized text for ``chunk_id``."""
    init_db(kb_name)
    path = _db_path(kb_name)
    conn = sqlite3.connect(path)
    conn.execute(
        "REPLACE INTO token_tokens (id, tokens) VALUES (?, ?)",
        (chunk_id, json.dumps(tokens, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def load_token_lists(kb_name: str | Path) -> Dict[str, List[str]]:
    """Return mapping of chunk IDs to token lists."""
    path = _db_path(kb_name)
    if not path.exists():
        return {}
    conn = sqlite3.connect(path)
    cur = conn.execute("SELECT id, tokens FROM token_tokens")
    rows = cur.fetchall()
    conn.close()
    result: Dict[str, List[str]] = {}
    for cid, tokens_json in rows:
        try:
            result[cid] = json.loads(tokens_json)
        except Exception:
            continue
    return result


def clear_cache(kb_name: str | Path) -> None:
    """Remove all cached embeddings and token lists for ``kb_name``."""
    path = _db_path(kb_name)
    if not path.exists():
        return
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM embeddings")
    conn.execute("DELETE FROM token_tokens")
    conn.commit()
    conn.close()
