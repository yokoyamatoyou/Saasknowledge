import json
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from shared import ui_state  # noqa: E402


def test_load_and_save_ui_state(tmp_path, monkeypatch):
    path = tmp_path / "ui.json"
    monkeypatch.setattr(ui_state, "UI_STATE_FILE", path)
    state = {"sidebar_visible": True}
    ui_state.save_ui_state(state)
    loaded = ui_state.load_ui_state()
    assert loaded == state

