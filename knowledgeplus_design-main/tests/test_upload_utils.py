import json
import sys
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from shared import db_cache, upload_utils  # noqa: E402


def test_save_processed_data_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    paths = upload_utils.save_processed_data(
        "kb",
        "1",
        chunk_text="hello",
        embedding=[0.1, 0.2],
        metadata={"foo": "bar"},
        original_filename="orig.txt",
        original_bytes=b"data",
        image_bytes=b"img",
    )
    meta = json.loads(Path(paths["metadata_path"]).read_text(encoding="utf-8"))
    assert meta["paths"]["chunk_path"] == paths["chunk_path"]
    assert "original_file_path" in meta["paths"]


def test_save_processed_data_stores_embedding_in_db(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    embed = [0.3, 0.4]
    upload_utils.save_processed_data(
        "kb",
        "123",
        chunk_text="x",
        embedding=embed,
        metadata={},
    )
    loaded = db_cache.load_embeddings("kb")
    assert loaded == {"123": embed}


def test_save_processed_data_version(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    first = upload_utils.save_processed_data(
        "kb",
        "1",
        metadata={},
        original_filename="file.txt",
        original_bytes=b"a",
    )
    second = upload_utils.save_processed_data(
        "kb",
        "2",
        metadata={},
        original_filename="file.txt",
        original_bytes=b"a",
    )
    assert Path(first["original_file_path"]) == Path(second["original_file_path"])
    third = upload_utils.save_processed_data(
        "kb",
        "3",
        metadata={},
        original_filename="file.txt",
        original_bytes=b"b",
    )
    assert Path(third["original_file_path"]) != Path(first["original_file_path"])


def test_save_user_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    path = upload_utils.save_user_metadata("kb", "abc123", "Title", ["x", "y"])
    assert Path(path).exists()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data == {"title": "Title", "tags": ["x", "y"]}


def test_load_user_metadata_and_list(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_utils, "BASE_KNOWLEDGE_DIR", tmp_path)
    kb_dir = tmp_path / "kb" / "metadata"
    kb_dir.mkdir(parents=True)
    meta_file = kb_dir / "1.json"
    meta_file.write_text(json.dumps({"paths": {}}), encoding="utf-8")
    user_file = kb_dir / "1_user.json"
    user_file.write_text(json.dumps({"title": "T"}), encoding="utf-8")

    items = upload_utils.list_metadata_items("kb")
    assert items == [("1", {"paths": {}})]

    data = upload_utils.load_user_metadata("kb", "1")
    assert data == {"title": "T"}
    missing = upload_utils.load_user_metadata("kb", "nope")
    assert missing == {}


def test_ensure_openai_key_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError) as exc:
        upload_utils.ensure_openai_key()
    assert "OPENAI_API_KEY" in str(exc.value)
