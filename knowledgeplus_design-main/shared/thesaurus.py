import json
from pathlib import Path
from typing import Dict, List

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "synonyms.json"


def load_synonyms(path: Path | None = None) -> Dict[str, List[str]]:
    p = Path(path) if path else _DEFAULT_PATH
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): list(v) for k, v in data.items()}
        except Exception:
            pass
    return {}


def expand_query(query: str, synonyms: Dict[str, List[str]]) -> str:
    tokens = query.split()
    expanded = []
    for t in tokens:
        expanded.append(t)
        if t in synonyms:
            expanded.extend(synonyms[t])
    return " ".join(expanded)


def save_synonyms(data: Dict[str, List[str]], path: Path | None = None) -> None:
    """Write a synonyms dictionary to disk."""
    p = Path(path) if path else _DEFAULT_PATH
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_synonyms(
    term: str, words: List[str], path: Path | None = None
) -> Dict[str, List[str]]:
    """Add or update a term's synonyms and persist them."""
    syns = load_synonyms(path)
    syns[str(term)] = list(words)
    save_synonyms(syns, path)
    return syns
