import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any

# Base directory for all knowledge bases
BASE_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
BASE_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_openai_key():
    """Return the OpenAI API key or raise an informative error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set")
    return api_key


def _ensure_dirs(kb_dir: Path):
    """Create standard subdirectories for a knowledge base."""
    subdirs = {}
    for name in ["chunks", "embeddings", "metadata", "images", "files"]:
        path = kb_dir / name
        path.mkdir(parents=True, exist_ok=True)
        subdirs[name] = path
    return subdirs


def save_processed_data(
    kb_name: str,
    chunk_id: str,
    chunk_text: str = None,
    embedding=None,
    metadata: dict = None,
    original_filename: str = None,
    original_bytes: bytes = None,
    image_bytes: bytes = None,
):
    """Save processed chunk, embedding and metadata in a unified layout.

    Returned paths are also embedded inside the saved metadata so that other
    components can easily reference stored files.
    """
    kb_dir = BASE_KNOWLEDGE_DIR / kb_name
    dirs = _ensure_dirs(kb_dir)

    paths: Dict[str, Any] = {}

    if chunk_text is not None:
        chunk_path = dirs["chunks"] / f"{chunk_id}.txt"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk_text)
        paths["chunk_path"] = str(chunk_path)

    if embedding is not None:
        emb_path = dirs["embeddings"] / f"{chunk_id}.pkl"
        with open(emb_path, "wb") as f:
            pickle.dump({"embedding": embedding}, f)
        paths["embedding_path"] = str(emb_path)

    if image_bytes is not None:
        img_path = dirs["images"] / f"{chunk_id}.jpg"
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        paths["image_path"] = str(img_path)

    if original_bytes is not None and original_filename:
        file_path = dirs["files"] / original_filename
        if file_path.exists():
            existing_bytes = file_path.read_bytes()
            if existing_bytes != original_bytes:
                base = file_path.stem
                ext = file_path.suffix
                version = 1
                while True:
                    new_name = f"{base}_v{version}{ext}"
                    new_path = file_path.with_name(new_name)
                    if not new_path.exists():
                        file_path = new_path
                        break
                    version += 1
        with open(file_path, "wb") as f:
            f.write(original_bytes)
        paths["original_file_path"] = str(file_path)

    if metadata is not None:
        meta = metadata.copy()
        if original_filename:
            meta.setdefault("original_file", original_filename)
        meta["paths"] = paths
        meta_path = dirs["metadata"] / f"{chunk_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        paths["metadata_path"] = str(meta_path)

    return paths


def save_user_metadata(kb_name: str, item_id: str, title: str, tags: list[str]) -> str:
    """Store user-provided metadata for an item and return the file path."""
    kb_dir = BASE_KNOWLEDGE_DIR / kb_name / "metadata"
    kb_dir.mkdir(parents=True, exist_ok=True)
    meta_path = kb_dir / f"{item_id}_user.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "tags": tags}, f, ensure_ascii=False, indent=2)
    return str(meta_path)
