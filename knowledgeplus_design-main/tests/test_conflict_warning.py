import io
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault(
    "knowledge_gpt_app.app",
    types.SimpleNamespace(
        refresh_search_engine=lambda *a, **k: None,
        get_search_engine=lambda name: DummyEngine(),
    ),
)

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402
from ui_modules import management_ui  # noqa: E402


class DummyEngine:
    def __init__(self):
        self.chunks = []
        self.detect_called = False

    def detect_rule_conflicts(self, chunks, client=None):
        self.detect_called = True
        return [
            {
                "rule_type": "承認権限",
                "conflicting_chunks": ["old"],
                "explanation": "x",
            }
        ]


def _setup(monkeypatch, buttons):
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
    monkeypatch.setattr(st, "button", lambda label, *a, **k: label in buttons)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: [_make_png()])


try:
    from PIL import Image
except Exception:
    Image = None


def _make_png():
    if Image is None:
        pytest.skip("Pillow not installed", allow_module_level=True)
    buf = io.BytesIO()
    Image.new("RGB", (5, 5), "blue").save(buf, format="PNG")
    buf.seek(0)
    buf.name = "img.png"
    return buf


def test_conflict_warning(monkeypatch):
    st.session_state.clear()
    engine = DummyEngine()

    sys.modules["knowledge_gpt_app.app"].get_search_engine = lambda name: engine

    _setup(monkeypatch, {"選択したファイルの処理を開始", "ナレッジベースに登録"})

    monkeypatch.setattr(management_ui, "analyze_image_with_gpt4o", lambda *a, **k: {})
    monkeypatch.setattr(management_ui, "display_thumbnail_grid", lambda *a, **k: None)
    monkeypatch.setattr(
        management_ui.FileProcessor,
        "process_file",
        classmethod(
            lambda cls, uploaded_file, *, builder=None: {
                "type": "image",
                "image_base64": "b64",
            }
        ),
    )

    builders = []

    class DummyBuilder:
        def __init__(self, *a, **k):
            builders.append(self)

        def _create_comprehensive_search_chunk(self, analysis, adds):
            return "chunk"

        def _create_structured_metadata(self, analysis, adds, filename):
            return {"rule_info": {"rule_types": ["A"], "contains_rules": True}}

        def build_from_file(self, *a, **k):
            self.saved = True

        def refresh_search_engine(self, *a, **k):
            pass

    monkeypatch.setattr(management_ui, "KnowledgeBuilder", DummyBuilder)

    management_ui.render_management_mode()

    assert "conflict_state_0" in st.session_state
    assert not getattr(builders[0], "saved", False)

    # Confirm button would clear the warning in a real session
