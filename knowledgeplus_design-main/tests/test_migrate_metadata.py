import json
import logging
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[2]))

from migrate_metadata import migrate_knowledge_base, migrate_metadata_file  # noqa: E402


def test_migrate_metadata_file_adds_keys(tmp_path):
    meta = {
        "filename": "company_doc.txt",
        "created_at": "2024-01-01",
        "meta_info": {"summary": "承認手続き", "keywords": []},
    }
    meta_path = tmp_path / "1.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    result = migrate_metadata_file(meta_path)
    assert result is True

    migrated = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "version_info" in migrated
    assert "hierarchy_info" in migrated
    assert "rule_info" in migrated


def test_migrate_knowledge_base_logs_counts(tmp_path, caplog):
    kb_dir = tmp_path / "kb"
    metadata_dir = kb_dir / "metadata"
    chunks_dir = kb_dir / "chunks"
    metadata_dir.mkdir(parents=True)
    chunks_dir.mkdir()

    valid_meta = {"filename": "file.txt", "created_at": "2024-01-01"}
    (metadata_dir / "1.json").write_text(
        json.dumps(valid_meta, ensure_ascii=False), encoding="utf-8"
    )
    (chunks_dir / "1.txt").write_text("text", encoding="utf-8")

    (metadata_dir / "bad.json").write_text("{", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        migrate_knowledge_base(str(kb_dir))

    assert any("移行完了: 成功 1件, エラー 1件" in r.message for r in caplog.records)
