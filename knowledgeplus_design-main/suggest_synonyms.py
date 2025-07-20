import argparse
import json
from pathlib import Path
from typing import Dict, List

from shared.openai_utils import get_openai_client
from shared.thesaurus import load_synonyms, save_synonyms
from shared.zero_hit_logger import load_zero_hit_queries


def suggest_synonyms(log_path: Path, thesaurus_path: Path) -> Dict[str, List[str]]:
    queries = load_zero_hit_queries(log_path)
    existing = load_synonyms(thesaurus_path)
    client = get_openai_client()
    suggestions: Dict[str, List[str]] = {}
    for q in queries:
        if not q or q in existing:
            continue
        words: List[str] = []
        if client is not None:
            prompt = f"以下の単語の類義語や表記揺れを3件教えてください。" f"JSONリストで回答してください。\n単語: {q}"
            try:
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "synonym suggester"},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                if isinstance(data, list):
                    words = [str(w) for w in data]
                else:
                    words = [str(w) for w in data.get("synonyms", [])]
            except Exception:
                words = []
        suggestions[q] = words
    return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-hit synonym suggester")
    parser.add_argument("--log", type=Path, default=None, help="zero hit log path")
    parser.add_argument(
        "--thesaurus", type=Path, default=None, help="synonyms json path"
    )
    parser.add_argument(
        "--update", action="store_true", help="update thesaurus with suggestions"
    )
    args = parser.parse_args()

    sugg = suggest_synonyms(args.log, args.thesaurus)
    if not sugg:
        print("No new suggestions")
        return
    for term, words in sugg.items():
        print(term, "->", words)
    if args.update:
        syns = load_synonyms(args.thesaurus)
        for term, words in sugg.items():
            if words:
                syns[term] = words
        save_synonyms(syns, args.thesaurus)
        print("Thesaurus updated")


if __name__ == "__main__":
    main()
