from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Store histories at the repository root so they are shared across all apps
CHAT_HISTORY_DIR = Path(__file__).resolve().parents[2] / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


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


def create_history(settings: Optional[Dict] = None) -> str:
    """Create a new chat history file and return its ID."""
    history_id = str(uuid.uuid4())
    data = {
        "title": "新しい会話",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings or {},
        "messages": [],
    }
    with open(CHAT_HISTORY_DIR / f"{history_id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return history_id


def append_message(history_id: str, role: str, content: str) -> None:
    path = CHAT_HISTORY_DIR / f"{history_id}.json"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("messages", []).append({"role": role, "content": content})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_title(history_id: str, title: str) -> None:
    path = CHAT_HISTORY_DIR / f"{history_id}.json"
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["title"] = title
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
