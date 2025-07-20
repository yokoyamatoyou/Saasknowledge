from pathlib import Path
from typing import List

_ZERO_HIT_PATH = Path(__file__).resolve().parents[1] / "data" / "zero_hit_queries.log"


def log_zero_hit_query(query: str, path: Path | None = None) -> None:
    """Append a search query to the zero-hit log."""
    if not query:
        return
    p = Path(path) if path else _ZERO_HIT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(query.strip() + "\n")


def load_zero_hit_queries(path: Path | None = None) -> List[str]:
    """Return the list of recorded zero-hit queries."""
    p = Path(path) if path else _ZERO_HIT_PATH
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
