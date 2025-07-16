import io
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

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
    monkeypatch.setattr(
        KnowledgeBuilder,
        "generate_image_embedding",
        staticmethod(lambda b: [0.1, 0.2]),
    )
    buf = _make_png()
    FileProcessor.process_file(buf, kb_name="kb1")
    emb_dir = tmp_path / "kb1" / "embeddings"
    files = list(emb_dir.glob("*.pkl"))
    assert len(files) == 1
    data = pickle.loads(files[0].read_bytes())
    assert data["embedding"] == [0.1, 0.2]


def test_process_document_saves_embedding(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(
        KnowledgeBuilder,
        "generate_text_embedding",
        staticmethod(lambda t: [0.5]),
    )
    buf = io.BytesIO(b"hello world")
    buf.name = "note.txt"
    FileProcessor.process_file(buf, kb_name="kb2")
    emb_dir = tmp_path / "kb2" / "embeddings"
    files = list(emb_dir.glob("*.pkl"))
    assert len(files) == 1
    data = pickle.loads(files[0].read_bytes())
    assert data["embedding"] == [0.5]
