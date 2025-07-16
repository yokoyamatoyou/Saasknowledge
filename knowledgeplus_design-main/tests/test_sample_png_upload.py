import io
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.file_processor import FileProcessor  # noqa: E402
from shared.kb_builder import KnowledgeBuilder  # noqa: E402


class MockOpenAIClient:
    def __init__(self):
        self.embeddings = MagicMock()
        self.chat = MagicMock()


class MockEmbeddingsResponse:
    def __init__(self, embedding_vectors):
        self.data = [MagicMock(embedding=v) for v in embedding_vectors]


@pytest.fixture
def mock_openai_client():
    client = MockOpenAIClient()

    def create(**kwargs):
        inputs = kwargs.get("input")
        if not isinstance(inputs, list):
            inputs = [inputs]
        return MockEmbeddingsResponse([[0.1] * 1536 for _ in inputs])

    client.embeddings.create.side_effect = create
    client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"image_type": "図面", "main_content": "テスト", "keywords": ["テスト"], "description_for_search": "テスト"}'
                )
            )
        ]
    )
    return client


@pytest.fixture
def mock_refresh_search_engine():
    return MagicMock()


@pytest.fixture
def mock_file_processor():
    processor = MagicMock(spec=FileProcessor)
    processor.process_file.return_value = ("dummy_base64_image", {"file_type": "mock"})
    return processor


@pytest.fixture
def kb_builder_instance(
    mock_file_processor, mock_openai_client, mock_refresh_search_engine
):
    return KnowledgeBuilder(
        mock_file_processor, lambda: mock_openai_client, mock_refresh_search_engine
    )


@pytest.fixture
def temp_kb_dir(tmp_path):
    kb_dir = tmp_path / "knowledge_base" / "test_kb"
    for sub in ["chunks", "embeddings", "metadata", "images", "files"]:
        (kb_dir / sub).mkdir(parents=True)
    return kb_dir


def test_sample_png_upload(kb_builder_instance, temp_kb_dir, monkeypatch):
    monkeypatch.setattr("shared.upload_utils.BASE_KNOWLEDGE_DIR", temp_kb_dir.parent)

    try:
        from PIL import Image
    except Exception:
        pytest.skip("Pillow not installed", allow_module_level=True)

    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "blue").save(buf, format="PNG")
    buf.seek(0)
    buf.name = "sample_png.png"
    uploaded_file = buf

    analysis_result = {
        "image_type": "図面",
        "main_content": "サンプルPNG画像",
        "keywords": ["sample"],
        "description_for_search": "sample png",
    }
    user_additions = {"title": "サンプルPNG", "additional_keywords": ["test"]}

    saved_item = kb_builder_instance.build_from_file(
        uploaded_file,
        analysis=analysis_result,
        image_base64="dummy_base64_image",
        user_additions=user_additions,
        cad_metadata=None,
    )

    assert saved_item is not None
    item_id = saved_item["id"]
    assert (temp_kb_dir / "chunks" / f"{item_id}.txt").exists()
    assert (temp_kb_dir / "embeddings" / f"{item_id}.pkl").exists()
    assert (temp_kb_dir / "metadata" / f"{item_id}.json").exists()

    with open(temp_kb_dir / "chunks" / f"{item_id}.txt", "r", encoding="utf-8") as f:
        chunk_text = f.read()
        assert "画像タイプ: 図面" in chunk_text
        assert "主要内容: サンプルPNG画像" in chunk_text

    with open(temp_kb_dir / "metadata" / f"{item_id}.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        assert meta["display_metadata"]["title"] == "サンプルPNG"
        assert "sample" in meta["display_metadata"]["keywords"]
        assert "test" in meta["display_metadata"]["keywords"]

    with open(temp_kb_dir / "embeddings" / f"{item_id}.pkl", "rb") as f:
        embedding_data = pickle.load(f)
        assert len(embedding_data["embedding"]) == 1536
