import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared import upload_utils


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
