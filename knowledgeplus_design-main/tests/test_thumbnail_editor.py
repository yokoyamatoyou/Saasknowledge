from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")

import streamlit as st

from shared import upload_utils
from ui_modules import thumbnail_editor as te


def _compute_snippet(text: str) -> str:
    mid = len(text) // 2
    snippet = text[max(0, mid - 6) : mid + 6]
    return snippet.strip()


def _create_items(tmp_dir: Path, count: int):
    kb = "kb"
    texts = []
    ids = []
    for i in range(count):
        cid = str(i)
        text = f"text {i} for testing"
        texts.append(text)
        paths = upload_utils.save_processed_data(
            kb,
            cid,
            chunk_text=text,
            metadata={},
        )
        meta_path = Path(paths["metadata_path"])
        new_meta = meta_path.with_name(f"metadata_{meta_path.name}")
        meta_path.rename(new_meta)
        ids.append(new_meta.stem) # IDをファイル名から取得するように変更
    return kb, ids, texts


def test_load_items(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(te, "BASE_KNOWLEDGE_DIR", tmp_path)

    kb, ids, texts = _create_items(tmp_path, 3)

    items = te._load_items(kb)
    assert len(items) == 3
    items.sort(key=lambda it: it["id"])
    for cid, text, item in zip(ids, texts, items):
        assert item["id"] == cid
        assert item["snippet"] == _compute_snippet(text)


def test_display_thumbnail_grid_pagination(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    monkeypatch.setattr(te, "BASE_KNOWLEDGE_DIR", tmp_path)

    kb, _, _ = _create_items(tmp_path, 10)

    class DummyCol:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(st, "columns", lambda n: [DummyCol() for _ in range(n)])
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "info", lambda *a, **k: None)
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st, "text_input", lambda *a, **k: "")
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "rerun", lambda: None)

    def fake_button(label, *a, **k):
        return label == "Next"

    monkeypatch.setattr(st, "button", fake_button)

    st.session_state.clear()
    te.display_thumbnail_grid(kb)
    assert st.session_state.get(f"thumb_page_{kb}") == 1

    monkeypatch.setattr(st, "button", lambda *a, **k: False)
    te.display_thumbnail_grid(kb)
    assert st.session_state.get(f"thumb_page_{kb}") == 1
