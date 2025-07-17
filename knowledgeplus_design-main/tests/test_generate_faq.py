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

    def fake_generate(name, max_tokens=1000, num_pairs=3, client=None, source=None):
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

    called = {}

    monkeypatch.setattr(
        generate_faq.mm_builder_utils,
        "get_text_embedding",
        lambda text: called.setdefault("text", text) or [0.0],
    )

    captured = {}

    def fake_create(**kwargs):
        captured.setdefault("temps", []).append(kwargs.get("temperature"))
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='[{"category":"c","question":"q","answer":"a"}]'
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


def test_generate_from_source_url(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb_url"
    (kb_dir / "chunks").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir()
    (kb_dir / "metadata").mkdir()
    (kb_dir / "files").mkdir()

    monkeypatch.setattr(generate_faq, "BASE_KNOWLEDGE_DIR", tmp_path)

    called = {}

    captured = {}

    def fake_get(url, timeout=10):
        called["url"] = url

        class Resp:
            text = "<html><body>Hello world</body></html>"

            def raise_for_status(self):
                pass

        return Resp()

    monkeypatch.setattr(generate_faq, "requests", types.SimpleNamespace(get=fake_get))

    monkeypatch.setattr(
        generate_faq.mm_builder_utils, "get_text_embedding", lambda t: [0.0]
    )

    def fake_create(**kwargs):
        captured.setdefault("temps", []).append(kwargs.get("temperature"))
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='[{"category":"c","question":"q","answer":"a"}]'
                    )
                )
            ]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    count = generate_faq.generate_faqs_from_chunks(
        "kb_url", client=client, source="http://example.com"
    )

    assert count == 1
    assert called["url"] == "http://example.com"
    assert captured["temps"] == [0.0]

    data = json.loads((kb_dir / "faqs.json").read_text(encoding="utf-8"))
    assert data and "question" in data[0] and "answer" in data[0]


def test_generate_from_source_text(tmp_path, monkeypatch):
    kb_dir = tmp_path / "kb_text"
    for sub in ["chunks", "embeddings", "metadata", "files"]:
        (kb_dir / sub).mkdir(parents=True)

    monkeypatch.setattr(generate_faq, "BASE_KNOWLEDGE_DIR", tmp_path)

    monkeypatch.setattr(
        generate_faq.mm_builder_utils, "get_text_embedding", lambda t: [0.0]
    )
    monkeypatch.setattr(generate_faq, "save_processed_data", lambda *a, **k: None)

    captured = {}

    def fake_create(**kwargs):
        captured["temp"] = kwargs.get("temperature")
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='[{"category":"c","question":"q","answer":"a"}]'
                    )
                )
            ]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    count = generate_faq.generate_faqs_from_chunks(
        "kb_text", client=client, source="Just some text"
    )

    assert count == 1
    assert captured["temp"] == 0.0

    data = json.loads((kb_dir / "faqs.json").read_text(encoding="utf-8"))
    assert isinstance(data, list) and data[0]["question"] == "q"


def test_temperature_increment(monkeypatch):
    captured = {}

    def fake_create(**kwargs):
        captured["temp"] = kwargs.get("temperature")
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content='[{"category":"c","question":"q","answer":"a"}]'
                    )
                )
            ]
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    pairs, count = generate_faq.generate_faq_from_source(
        "hello", 1, client, max_tokens=100, q_count=5
    )

    assert captured["temp"] == 0.01
    assert count == 6
    assert pairs and pairs[0]["answer"] == "a"
