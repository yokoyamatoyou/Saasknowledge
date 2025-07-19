import json
from pathlib import Path
from typing import Dict

_FEEDBACK_PATH = Path(__file__).resolve().parents[1] / "data" / "feedback.json"


def load_feedback(path: Path | None = None) -> Dict[str, int]:
    p = Path(path) if path else _FEEDBACK_PATH
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): int(v) for k, v in data.items()}
        except Exception:
            pass
    return {}


def save_feedback(feedback: Dict[str, int], path: Path | None = None) -> None:
    p = Path(path) if path else _FEEDBACK_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)


def record_feedback(chunk_id: str, score: int = 1, path: Path | None = None) -> None:
    data = load_feedback(path)
    data[chunk_id] = data.get(chunk_id, 0) + int(score)
    save_feedback(data, path)
