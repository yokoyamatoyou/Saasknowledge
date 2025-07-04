from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Store histories at the repository root so they are shared across all apps.
# Allow override via the ``CHAT_HISTORY_DIR`` environment variable so the
# location can be customized for different deployments.
CHAT_HISTORY_DIR = Path(
    os.getenv(
        "CHAT_HISTORY_DIR",
        str(Path(__file__).resolve().parents[2] / "chat_history"),
    )
)
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def get_history_path(history_id: str) -> Path:
    """Return the file path for the given history ID."""
    return CHAT_HISTORY_DIR / f"{history_id}.json"


def load_chat_histories() -> List[Dict]:
    """Return a list of available chat histories sorted by creation time."""
    histories = []
    for path in CHAT_HISTORY_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            histories.append({
                "id": path.stem,
                "title": data.get("title", "会話"),
                "created_at": data.get("created_at"),
                "settings": data.get("settings", {}),
                "messages": data.get("messages", []),
            })
        except Exception:
            continue
    histories.sort(key=lambda h: h.get("created_at", ""), reverse=True)
    return histories


def list_history_ids() -> List[str]:
    """Return history file IDs sorted by creation time."""
    return [h["id"] for h in load_chat_histories()]


def load_history(history_id: str) -> Optional[Dict]:
    """Return a single chat history dict or ``None`` if unavailable."""
    path = get_history_path(history_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def create_history(settings: Optional[Dict] = None) -> str:
    """Create a new chat history file and return its ID."""
    history_id = str(uuid.uuid4())
    data = {
        "title": "新しい会話",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings or {},
        "messages": [],
    }
    path = get_history_path(history_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return history_id


def append_message(history_id: str, role: str, content: str) -> None:
    """Append ``role`` and ``content`` to ``history_id``'s message list.

    The JSON file ``<history_id>.json`` is updated in place if it exists.
    """
    path = get_history_path(history_id)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("messages", []).append({"role": role, "content": content})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_title(history_id: str, title: str) -> None:
    """Update the title for the specified history file if it exists."""
    path = get_history_path(history_id)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["title"] = title
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def delete_history(history_id: str) -> bool:
    """Delete the specified history file.

    Returns True if the file existed and was removed, False otherwise.
    """
    path = get_history_path(history_id)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False
