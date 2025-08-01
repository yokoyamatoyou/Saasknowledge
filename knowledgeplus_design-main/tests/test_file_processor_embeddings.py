import io
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from config import EMBEDDING_DIM  # noqa: E402
from shared import upload_utils  # noqa: E402
from shared.file_processor import FileProcessor  # noqa: E402
from shared.kb_builder import KnowledgeBuilder  # noqa: E402


def _make_png():
    try:
        from PIL import Image
    except Exception:
        pytest.skip("Pillow not installed", allow_module_level=True)
    buf = io.BytesIO()
    Image.new("RGB", (5, 5), "green").save(buf, format="PNG")
    buf.seek(0)
    buf.name = "img.png"
    return buf


def test_process_image_saves_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    builder = KnowledgeBuilder(FileProcessor(), lambda: None, lambda *_: None)
    monkeypatch.setattr(
        builder,
        "generate_image_embedding",
        lambda b: [0.1] * EMBEDDING_DIM,
    )
    buf = _make_png()
    FileProcessor.process_file(buf, kb_name="kb1", builder=builder)
    emb_dir = tmp_path / "kb1" / "embeddings"
    files = list(emb_dir.glob("*.pkl"))
    assert len(files) == 1
    data = pickle.loads(files[0].read_bytes())
    assert len(data["embedding"]) == EMBEDDING_DIM
    assert data["embedding"][0] == 0.1
    chunk_dir = tmp_path / "kb1" / "chunks"
    chunks = list(chunk_dir.glob("*.txt"))
    assert len(chunks) == 1
    assert chunks[0].read_text(encoding="utf-8") == "img.png"


def test_process_document_saves_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    builder = KnowledgeBuilder(FileProcessor(), lambda: None, lambda *_: None)
    monkeypatch.setattr(
        builder, "generate_text_embedding", lambda t: [0.5] * EMBEDDING_DIM
    )
    buf = io.BytesIO(b"hello world")
    buf.name = "note.txt"
    FileProcessor.process_file(buf, kb_name="kb2", builder=builder)
    emb_dir = tmp_path / "kb2" / "embeddings"
    files = list(emb_dir.glob("*.pkl"))
    assert len(files) == 1
    data = pickle.loads(files[0].read_bytes())
    assert len(data["embedding"]) == EMBEDDING_DIM
    assert data["embedding"][0] == 0.5
