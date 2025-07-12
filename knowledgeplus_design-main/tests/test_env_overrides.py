import importlib
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))


def test_directory_overrides(tmp_path, monkeypatch):
    kb_dir = tmp_path / "custom_kb"
    hist_dir = tmp_path / "custom_hist"
    monkeypatch.setenv("KNOWLEDGE_BASE_DIR", str(kb_dir))
    monkeypatch.setenv("CHAT_HISTORY_DIR", str(hist_dir))

    from shared import chat_history_utils, upload_utils

    importlib.reload(upload_utils)
    importlib.reload(chat_history_utils)

    assert upload_utils.BASE_KNOWLEDGE_DIR == kb_dir
    assert chat_history_utils.CHAT_HISTORY_DIR == hist_dir

    upload_utils.save_processed_data("kb", "1", chunk_text="hi", metadata={})
    hid = chat_history_utils.create_history()

    assert (kb_dir / "kb" / "chunks" / "1.txt").exists()
    assert (hist_dir / f"{hid}.json").exists()

    monkeypatch.delenv("KNOWLEDGE_BASE_DIR", raising=False)
    monkeypatch.delenv("CHAT_HISTORY_DIR", raising=False)
    importlib.reload(upload_utils)
    importlib.reload(chat_history_utils)
