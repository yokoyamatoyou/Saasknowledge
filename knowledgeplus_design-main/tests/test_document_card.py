import base64
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


def test_render_document_card_image_and_download(tmp_path, monkeypatch):
    outputs = []
    images = []
    downloads = []
    monkeypatch.setattr(
        st, "markdown", lambda text, unsafe_allow_html=False: outputs.append(text)
    )
    monkeypatch.setattr(st, "image", lambda *a, **k: images.append(True))
    monkeypatch.setattr(st, "download_button", lambda *a, **k: downloads.append(True))

    file_path = tmp_path / "test.txt"
    file_path.write_text("x")

    mod = importlib.import_module("ui_modules.document_card")
    doc = {
        "metadata": {
            "title": "foo",
            "preview_image": base64.b64encode(b"img").decode("utf-8"),
            "paths": {"original_file_path": str(file_path)},
        },
        "text": "hello world",
        "similarity": 0.5,
    }
    mod.render_document_card(doc)
    assert images
    assert downloads


def test_render_document_card_missing_file(monkeypatch):
    errors = []
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "image", lambda *a, **k: None)
    monkeypatch.setattr(st, "download_button", lambda *a, **k: None)
    monkeypatch.setattr(st, "error", lambda msg: errors.append(msg))

    mod = importlib.import_module("ui_modules.document_card")
    doc = {
        "metadata": {"title": "foo", "paths": {"original_file_path": "none"}},
        "text": "hello",
        "similarity": 0.1,
    }
    mod.render_document_card(doc)
    assert errors
    assert "リンク先に存在しません" in errors[0]
