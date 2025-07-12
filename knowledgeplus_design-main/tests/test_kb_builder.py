import io
import json

# Adjust the import path to allow importing from shared
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from shared.file_processor import FileProcessor  # noqa: E402
from shared.kb_builder import KnowledgeBuilder  # noqa: E402


# Mock for OpenAI client
class MockOpenAIClient:
    def __init__(self):
        self.embeddings = MagicMock()
        self.chat = MagicMock()


# Mock for OpenAI embeddings create method
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
    # Mock process_file to return dummy base64 and metadata for image/CAD
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
    # Create a temporary knowledge base directory structure
    kb_dir = tmp_path / "knowledge_base" / "test_kb"
    (kb_dir / "chunks").mkdir(parents=True)
    (kb_dir / "embeddings").mkdir(parents=True)
    (kb_dir / "metadata").mkdir(parents=True)
    (kb_dir / "images").mkdir(parents=True)
    (kb_dir / "files").mkdir(parents=True)
    return kb_dir


def test_build_from_file_image(kb_builder_instance, temp_kb_dir, monkeypatch):
    # Mock save_processed_data to use the temporary directory
    monkeypatch.setattr("shared.upload_utils.BASE_KNOWLEDGE_DIR", temp_kb_dir.parent)

    # Create a dummy image file
    dummy_image_content = b"dummy_image_bytes"
    uploaded_file = io.BytesIO(dummy_image_content)
    uploaded_file.name = "test_image.png"

    analysis_result = {
        "image_type": "写真",
        "main_content": "これはテスト画像です",
        "keywords": ["テスト", "画像"],
        "description_for_search": "テスト用の画像",
    }
    user_additions = {"title": "テスト画像", "additional_keywords": ["追加"]}
    cad_metadata = None

    saved_item = kb_builder_instance.build_from_file(
        uploaded_file,
        analysis=analysis_result,
        image_base64="dummy_base64_image",
        user_additions=user_additions,
        cad_metadata=cad_metadata,
    )

    assert saved_item is not None
    assert saved_item["type"] == "image"
    assert saved_item["filename"] == "test_image.png"

    # Verify files are created in the temporary knowledge base directory
    item_id = saved_item["id"]
    assert (temp_kb_dir / "chunks" / f"{item_id}.txt").exists()
    assert (temp_kb_dir / "embeddings" / f"{item_id}.pkl").exists()  # Should be .pkl
    assert (temp_kb_dir / "metadata" / f"{item_id}.json").exists()
    assert (temp_kb_dir / "images" / f"{item_id}.jpg").exists()
    assert (temp_kb_dir / "files" / f"{item_id}_info.json").exists()

    # Verify content of chunk file
    with open(temp_kb_dir / "chunks" / f"{item_id}.txt", "r", encoding="utf-8") as f:
        chunk_text = f.read()
        assert "画像タイプ: 写真" in chunk_text
        assert "主要内容: これはテスト画像です" in chunk_text

    # Verify content of metadata file
    with open(temp_kb_dir / "metadata" / f"{item_id}.json", "r", encoding="utf-8") as f:
        metadata_data = json.load(f)
        assert metadata_data["display_metadata"]["title"] == "テスト画像"
        assert "テスト" in metadata_data["display_metadata"]["keywords"]
        assert "追加" in metadata_data["display_metadata"]["keywords"]

    # Verify refresh_search_engine was called
    kb_builder_instance.refresh_search_engine.assert_called_once_with(temp_kb_dir.name)
