import json
import logging
import sys
import types
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

pytest.importorskip("streamlit")

import generate_faq  # noqa: E402


def test_generate_faq_cli(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb"
    (kb_dir / "chunks").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir()
    (kb_dir / "metadata").mkdir()
    (kb_dir / "files").mkdir()
    (kb_dir / "chunks" / "1.txt").write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(generate_faq, "BASE_KNOWLEDGE_DIR", tmp_path)

    def fake_generate(name, max_tokens=1000, num_pairs=3, client=None):
        out = tmp_path / name / "faqs.json"
        out.write_text(
            json.dumps([{"id": "faq1", "question": "q", "answer": "a"}]),
            encoding="utf-8",
        )
        (tmp_path / name / "chunks" / "faq1.txt").write_text("q a")
        (tmp_path / name / "embeddings" / "faq1.pkl").write_bytes(b"0")
        (tmp_path / name / "metadata" / "faq1.json").write_text("{}")
        return 1

    monkeypatch.setattr(generate_faq, "generate_faqs_from_chunks", fake_generate)

    generate_faq.main(["kb"])

    faq_file = kb_dir / "faqs.json"
    assert faq_file.exists()
    data = json.loads(faq_file.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data
    first = data[0]
    assert "question" in first and "answer" in first


def test_generate_faq_imports_helper(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb2"
    (kb_dir / "chunks").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir()
    (kb_dir / "metadata").mkdir()
    (kb_dir / "files").mkdir()

    (kb_dir / "chunks" / "1.txt").write_text("dummy text", encoding="utf-8")

    monkeypatch.setattr(generate_faq, "BASE_KNOWLEDGE_DIR", tmp_path)

    import sys
    import types

    kgapp = types.ModuleType("knowledge_gpt_app.app")
    monkeypatch.setitem(sys.modules, "knowledge_gpt_app.app", kgapp)

    called = {}

    def fake_get_embedding(text, client=None):
        called["text"] = text
        return [0.0]

    monkeypatch.setattr(kgapp, "get_embedding", fake_get_embedding, raising=False)

    def fake_create(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='[{"question":"q","answer":"a"}]'
                    )
                )
            ]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    monkeypatch.setattr(generate_faq, "save_processed_data", lambda *a, **k: None)

    count = generate_faq.generate_faqs_from_chunks("kb2", client=client)

    assert count == 1
    assert called["text"].startswith("Q:")


def test_malformed_json_logged(tmp_path, monkeypatch, caplog):
    kb_dir = tmp_path / "kb3"
    (kb_dir / "chunks").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir()
    (kb_dir / "metadata").mkdir()
    (kb_dir / "files").mkdir()
    (kb_dir / "chunks" / "1.txt").write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(generate_faq, "BASE_KNOWLEDGE_DIR", tmp_path)

    def fake_create(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content="not-json"))
            ]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    monkeypatch.setattr(generate_faq, "save_processed_data", lambda *a, **k: None)

    with caplog.at_level(logging.ERROR):
        count = generate_faq.generate_faqs_from_chunks("kb3", client=client)

    assert count == 0
    assert any("JSON decode failed" in r.message for r in caplog.records)
