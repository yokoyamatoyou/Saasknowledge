import io
import sys
import types
from pathlib import Path

import pytest

# Ensure heavy deps are available; otherwise skip
pytest.importorskip("numpy")
pytest.importorskip("rank_bm25")
pytest.importorskip("sentence_transformers")
pytest.importorskip("nltk")

sys.modules.setdefault(
    "knowledge_gpt_app.app",
    types.SimpleNamespace(refresh_search_engine=lambda *a, **k: None),
)

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from config import EMBEDDING_DIM  # noqa: E402
from core import mm_builder_utils  # noqa: E402
from shared import upload_utils  # noqa: E402
from shared.search_engine import HybridSearchEngine  # noqa: E402
from ui_modules import management_ui  # noqa: E402


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


def _setup_streamlit(monkeypatch, files):
    class DummyCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            pass

    dummy = DummyCtx()
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st, "divider", lambda *a, **k: None)
    monkeypatch.setattr(st, "tabs", lambda labels: [dummy for _ in labels])
    monkeypatch.setattr(st, "expander", lambda *a, **k: dummy)
    monkeypatch.setattr(st, "popover", lambda *a, **k: dummy)
    monkeypatch.setattr(st, "image", lambda *a, **k: None)
    monkeypatch.setattr(st, "radio", lambda *a, **k: "個別処理")
    monkeypatch.setattr(st, "text_area", lambda *a, **k: "d")
    monkeypatch.setattr(st, "text_input", lambda *a, **k: "t")
    monkeypatch.setattr(st, "selectbox", lambda *a, **k: "技術文書")
    monkeypatch.setattr(st, "select_slider", lambda *a, **k: "中")
    monkeypatch.setattr(st, "spinner", lambda *a, **k: dummy)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "toast", lambda *a, **k: None)

    class DummyProgress:
        def progress(self, *a, **k):
            pass

    monkeypatch.setattr(st, "progress", lambda *a, **k: DummyProgress())
    monkeypatch.setattr(st, "button", lambda label, *a, **k: True)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: files)


def test_render_management_mode_embeddings_and_index(tmp_path, monkeypatch):
    st.session_state.clear()
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(management_ui, "display_thumbnail_grid", lambda *a, **k: None)
    monkeypatch.setattr(management_ui, "analyze_image_with_gpt4o", lambda *a, **k: {})
    monkeypatch.setattr(management_ui, "generate_faq", lambda *a, **k: 0)
    monkeypatch.setattr(
        mm_builder_utils,
        "get_text_embedding",
        lambda text, model=None, processor=None: [0.1] * EMBEDDING_DIM,
    )
    monkeypatch.setattr(
        mm_builder_utils,
        "get_image_embedding",
        lambda img, model=None, processor=None: [0.2] * EMBEDDING_DIM,
    )
    monkeypatch.setattr(
        mm_builder_utils, "load_model_and_processor", lambda: (object(), object())
    )

    text_file = io.BytesIO(b"hello world")
    text_file.name = "doc.txt"
    image_file = _make_png()

    _setup_streamlit(monkeypatch, [text_file, image_file])

    management_ui.render_management_mode()

    kb_dir = tmp_path / "default_kb"
    emb_files = list((kb_dir / "embeddings").glob("*.pkl"))
    assert len(emb_files) >= 2

    monkeypatch.setattr(
        HybridSearchEngine, "_load_or_build_bm25_index", lambda self: None
    )
    monkeypatch.setattr(HybridSearchEngine, "_integrate_faq_chunks", lambda self: None)
    engine = HybridSearchEngine(str(kb_dir))
    assert len(engine.embeddings) >= 2
