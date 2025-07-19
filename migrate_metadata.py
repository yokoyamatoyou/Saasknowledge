import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_metadata_file(metadata_path: Path, chunk_text: str | None = None) -> bool:
    """Migrate a single metadata JSON file to the expanded schema."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if "version_info" not in metadata:
            metadata["version_info"] = {
                "version": "1.0.0",
                "effective_date": metadata.get(
                    "created_at", datetime.now().strftime("%Y-%m-%d")
                ).split()[0],
                "expiry_date": None,
                "supersedes": [],
                "superseded_by": None,
                "status": "active",
            }

        if "hierarchy_info" not in metadata:
            filename = metadata.get("filename", "").lower()
            keywords = metadata.get("meta_info", {}).get("keywords", [])
            approval_level = "local"
            department = "未分類"
            authority_score = 0.5
            if any(k in filename for k in ["全社", "会社", "company"]):
                approval_level = "company"
                authority_score = 0.9
            elif any(k in filename for k in ["部門", "営業部", "総務部", "department"]):
                approval_level = "department"
                authority_score = 0.7
                if "営業" in filename:
                    department = "営業部"
                elif "総務" in filename:
                    department = "総務部"
                elif "経理" in filename:
                    department = "経理部"
            metadata["hierarchy_info"] = {
                "approval_level": approval_level,
                "department": department,
                "location": "未設定",
                "authority_score": authority_score,
            }

        if "rule_info" not in metadata:
            rule_keywords = ["承認", "権限", "規定", "手続き", "ルール", "条件"]
            meta_info = metadata.get("meta_info", {})
            contains_rules = False
            rule_types: list[str] = []
            summary = meta_info.get("summary", "")
            keywords = meta_info.get("keywords", [])
            if any(kw in summary or kw in keywords for kw in rule_keywords):
                contains_rules = True
                if "見積" in summary or "見積" in keywords:
                    rule_types.append("見積り条件")
                if "承認" in summary or "承認" in keywords:
                    rule_types.append("承認権限")
                if "手続" in summary or "手続" in keywords:
                    rule_types.append("手続き")
            metadata["rule_info"] = {
                "contains_rules": contains_rules,
                "rule_types": rule_types,
                "extracted_rules": [],
            }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("メタデータ移行完了: %s", metadata_path.name)
        return True
    except Exception as e:  # pragma: no cover - simple script
        logger.error("メタデータ移行エラー (%s): %s", metadata_path, e)
        return False


def migrate_knowledge_base(kb_path: str) -> None:
    """Migrate all metadata files under a knowledge base."""
    kb = Path(kb_path)
    metadata_dir = kb / "metadata"
    chunks_dir = kb / "chunks"
    if not metadata_dir.exists():
        logger.error("メタデータディレクトリが見つかりません: %s", metadata_dir)
        return

    success = 0
    error = 0
    for meta_file in metadata_dir.glob("*.json"):
        chunk_text = None
        chunk_file = chunks_dir / f"{meta_file.stem}.txt"
        if chunk_file.exists():
            try:
                chunk_text = chunk_file.read_text(encoding="utf-8")
            except Exception as e:  # pragma: no cover - optional
                logger.warning("チャンクテキスト読み込みエラー (%s): %s", meta_file.stem, e)
        if migrate_metadata_file(meta_file, chunk_text):
            success += 1
        else:
            error += 1
    logger.info("移行完了: 成功 %d件, エラー %d件", success, error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="メタデータ移行スクリプト")
    parser.add_argument("--kb-path", required=True, help="ナレッジベースのパス")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    migrate_knowledge_base(args.kb_path)
