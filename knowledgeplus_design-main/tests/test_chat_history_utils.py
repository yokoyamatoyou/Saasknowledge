import json
from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from shared import chat_history_utils as chu


def test_chat_history_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(chu, "CHAT_HISTORY_DIR", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    hid = chu.create_history({"mode": "test"})
    chu.append_message(hid, "user", "hello")
    chu.append_message(hid, "assistant", "hi")
    chu.update_title(hid, "Session")
    data = json.loads((tmp_path / f"{hid}.json").read_text(encoding="utf-8"))
    assert data["title"] == "Session"
    assert data["settings"] == {"mode": "test"}
    assert len(data["messages"]) == 2

    histories = chu.load_chat_histories()
    assert len(histories) == 1
    h = histories[0]
    assert h["id"] == hid
    assert h["title"] == "Session"
    assert h["settings"] == {"mode": "test"}
    assert len(h["messages"]) == 2

    assert chu.delete_history(hid) is True
    assert not (tmp_path / f"{hid}.json").exists()


def test_load_history(tmp_path, monkeypatch):
    monkeypatch.setattr(chu, "CHAT_HISTORY_DIR", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    hid = chu.create_history()
    chu.append_message(hid, "user", "hi")

    data = chu.load_history(hid)
    assert data is not None
    assert data["messages"][0]["content"] == "hi"

    assert chu.load_history("missing") is None
