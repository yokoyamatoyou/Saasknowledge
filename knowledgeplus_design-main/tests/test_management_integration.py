import io
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault(
    "knowledge_gpt_app.app",
    types.SimpleNamespace(refresh_search_engine=lambda *a, **k: None),
)
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from ui_modules import management_ui  # noqa: E402


def test_render_management_mode_mixed_files(monkeypatch):
    st.session_state.clear()

    # Patch basic Streamlit widgets used in the management UI
    class DummyCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            pass

    dummy_ctx = DummyCtx()
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st, "divider", lambda *a, **k: None)
    monkeypatch.setattr(st, "tabs", lambda labels: [dummy_ctx for _ in labels])
    monkeypatch.setattr(st, "expander", lambda *a, **k: dummy_ctx)
    monkeypatch.setattr(st, "radio", lambda *a, **k: "個別処理")
    monkeypatch.setattr(st, "text_input", lambda *a, **k: "kb")
    monkeypatch.setattr(st, "number_input", lambda *a, **k: 1)
    monkeypatch.setattr(st, "spinner", lambda *a, **k: dummy_ctx)
    monkeypatch.setattr(st, "error", lambda *a, **k: None)
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "toast", lambda *a, **k: None)

    class DummyProgress:
        def progress(self, *a, **k):
            pass

    monkeypatch.setattr(st, "progress", lambda *a, **k: DummyProgress())

    def fake_button(label, *a, **k):
        return label == "選択したファイルの処理を開始"

    monkeypatch.setattr(st, "button", fake_button)

    text_file = io.BytesIO(b"hello")
    text_file.name = "doc.txt"
    image_file = io.BytesIO(b"img")
    image_file.name = "pic.png"
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [text_file, image_file])

    monkeypatch.setattr(management_ui, "display_thumbnail_grid", lambda *a, **k: None)

    def fake_process_file(cls, uploaded_file):
        if uploaded_file.name.endswith(".txt"):
            return {
                "type": "document",
                "text": "hello",
                "images": [],
                "metadata": {},
            }
        return {
            "image_base64": f"b64_{uploaded_file.name}",
            "metadata": {},
            "type": "image",
        }

    monkeypatch.setattr(
        management_ui.FileProcessor, "process_file", classmethod(fake_process_file)
    )

    analysis_calls = []

    def fake_analyze(image_b64, filename, cad_meta):
        analysis_calls.append(filename)
        return {"file": filename}

    monkeypatch.setattr(management_ui, "analyze_image_with_gpt4o", fake_analyze)

    builders = []

    class DummyBuilder:
        def __init__(self, *a, **k):
            self.calls = []
            self.results = []
            builders.append(self)

        def build_from_file(
            self, uploaded_file, analysis, image_base64, user_additions, cad_metadata
        ):
            item = {"filename": uploaded_file.name, "stats": {"vector_dimensions": 1}}
            self.calls.append(uploaded_file.name)
            self.results.append(item)
            return item

        def refresh_search_engine(self, *a, **k):
            pass

    monkeypatch.setattr(management_ui, "KnowledgeBuilder", DummyBuilder)

    management_ui.render_management_mode()

    assert builders, "KnowledgeBuilder was not instantiated"
    builder = builders[0]
    assert builder.calls == ["pic.png"]
    assert analysis_calls == ["pic.png"]
