import json
import os
from pathlib import Path
from typing import Any, Dict

# Store UI state JSON alongside chat histories
UI_STATE_FILE = Path(
    os.getenv(
        "UI_STATE_FILE",
        str(Path(__file__).resolve().parents[2] / "chat_history" / "ui_state.json"),
    )
)


def load_ui_state() -> Dict[str, Any]:
    """Return the persisted UI state dictionary or an empty dict."""
    if not UI_STATE_FILE.exists():
        return {}
    try:
        with open(UI_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_ui_state(state: Dict[str, Any]) -> None:
    """Persist the UI state to disk."""
    UI_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(UI_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
