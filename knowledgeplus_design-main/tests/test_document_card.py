import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402


def test_render_document_card_outputs_html(monkeypatch):
    outputs = []
    monkeypatch.setattr(
        st, "markdown", lambda text, unsafe_allow_html=False: outputs.append(text)
    )
    mod = importlib.import_module("ui_modules.document_card")
    doc = {"metadata": {"title": "foo"}, "text": "hello world", "similarity": 0.5}
    mod.render_document_card(doc)
    assert outputs
    assert "foo" in outputs[0]
    assert "Score" in outputs[0]
