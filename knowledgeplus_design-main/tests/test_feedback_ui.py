import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402


class DummyCol:
    def __init__(self, click_label=None):
        self.click_label = click_label

    def button(self, label, *a, **k):
        return label == self.click_label


def _setup_cols(monkeypatch, click_label):
    cols = [
        DummyCol("役に立った" if click_label == "役に立った" else None),
        DummyCol("役に立たなかった" if click_label == "役に立たなかった" else None),
        DummyCol(),
    ]
    monkeypatch.setattr(st, "columns", lambda *a, **k: cols)


def _load_module(monkeypatch):
    dummy_app = types.ModuleType("knowledge_gpt_app.app")
    dummy_app.list_knowledge_bases = lambda: []
    dummy_app.search_multiple_knowledge_bases = lambda *a, **k: ([], False)
    monkeypatch.setitem(sys.modules, "knowledge_gpt_app.app", dummy_app)
    sys.modules.pop("ui_modules.search_ui", None)
    return importlib.import_module("ui_modules.search_ui")


def test_positive_feedback(monkeypatch):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "render_document_card", lambda *a, **k: None)
    records = []
    monkeypatch.setattr(mod.feedback_store, "record_feedback", lambda cid, score=1: records.append((cid, score)))
    monkeypatch.setattr(st, "toast", lambda *a, **k: None)
    _setup_cols(monkeypatch, "役に立った")

    doc = {"id": "c1", "metadata": {}, "text": ""}
    mod.render_result_with_feedback(doc)
    assert records == [("c1", 1)]


def test_negative_feedback(monkeypatch):
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "render_document_card", lambda *a, **k: None)
    records = []
    monkeypatch.setattr(mod.feedback_store, "record_feedback", lambda cid, score=1: records.append((cid, score)))
    monkeypatch.setattr(st, "toast", lambda *a, **k: None)
    _setup_cols(monkeypatch, "役に立たなかった")

    doc = {"id": "c2", "metadata": {}, "text": ""}
    mod.render_result_with_feedback(doc)
    assert records == [("c2", -1)]
