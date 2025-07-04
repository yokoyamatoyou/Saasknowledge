import json
from pathlib import Path
import io
from unittest.mock import MagicMock
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.kb_builder import KnowledgeBuilder
from shared.file_processor import FileProcessor
from shared import upload_utils
from core import faq_utils


def test_build_and_generate_faq(tmp_path, monkeypatch):
    kb_dir = tmp_path / "knowledge_base" / "test_kb"
    for sub in ["chunks", "embeddings", "metadata", "images", "files"]:
        (kb_dir / sub).mkdir(parents=True)

    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path / "knowledge_base")
    monkeypatch.setattr(faq_utils, "BASE_KNOWLEDGE_DIR", tmp_path / "knowledge_base", raising=False)

    # Stub refresh_search_engine to track calls
    refresh_calls = []
    def fake_refresh(name):
        refresh_calls.append(name)
    import types
    stub_app = types.ModuleType("knowledge_gpt_app.app")
    stub_app.refresh_search_engine = fake_refresh
    monkeypatch.setitem(sys.modules, "knowledge_gpt_app.app", stub_app)

    # Prepare mock OpenAI client
    class MockClient:
        def __init__(self):
            self.embeddings = MagicMock()
            self.embeddings.create.return_value = MagicMock(data=[MagicMock(embedding=[0.1]*5)])
            self.chat = MagicMock()
            self.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content='{"image_type": "図面", "main_content": "dummy", "keywords": ["x"], "description_for_search": "d"}'))])
    client = MockClient()

    # Mock file_processor
    processor = MagicMock(spec=FileProcessor)
    processor.process_file.return_value = ("b64", {"file_type": "mock"})

    builder = KnowledgeBuilder(processor, lambda: client, fake_refresh)

    uploaded_file = io.BytesIO(b"data")
    uploaded_file.name = "img.png"

    analysis = {"image_type": "写真", "main_content": "sample", "keywords": ["x"], "description_for_search": "d"}
    user_additions = {"title": "Title", "additional_keywords": ["y"]}

    item = builder.build_from_file(uploaded_file, analysis, "b64", user_additions, None)
    assert item is not None
    assert refresh_calls == ["test_kb"]

    def fake_generate(name, max_tokens=1000, num_pairs=3, client=None):
        out = tmp_path / "knowledge_base" / name / "faqs.json"
        out.write_text(json.dumps([{"id": "faq1", "question": "q", "answer": "a"}]), encoding="utf-8")
        (tmp_path / "knowledge_base" / name / "chunks" / "faq1.txt").write_text("q a")
        (tmp_path / "knowledge_base" / name / "embeddings" / "faq1.pkl").write_bytes(b"0")
        (tmp_path / "knowledge_base" / name / "metadata" / "faq1.json").write_text("{}")
        return 1

    monkeypatch.setattr(faq_utils, "generate_faqs_from_chunks", fake_generate)

    count = faq_utils.generate_faq("test_kb", 1000, 1, client=client)
    assert count == 1
    assert refresh_calls == ["test_kb", "test_kb"]
    faq_file = kb_dir / "faqs.json"
    assert faq_file.exists()
    data = json.loads(faq_file.read_text(encoding="utf-8"))
    assert data and data[0]["answer"] == "a"
