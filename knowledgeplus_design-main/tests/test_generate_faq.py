import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
pytest.importorskip("streamlit")

import generate_faq


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
        out.write_text(json.dumps([{"id": "faq1", "question": "q", "answer": "a"}]), encoding="utf-8")
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
