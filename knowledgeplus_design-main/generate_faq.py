import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from shared.upload_utils import BASE_KNOWLEDGE_DIR, save_processed_data

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai may not be installed in tests
    OpenAI = None


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def generate_faqs_from_chunks(kb_name: str, max_tokens: int = 1000, num_pairs: int = 3, client=None) -> int:
    kb_dir = BASE_KNOWLEDGE_DIR / kb_name
    chunks_dir = kb_dir / "chunks"
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

    if client is None:
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client unavailable")

    faq_entries = []
    for chunk_file in sorted(chunks_dir.glob("*.txt")):
        text = chunk_file.read_text(encoding="utf-8")[:max_tokens]
        prompt = (
            f"You are a helpful assistant. Based on the following text, "
            f"generate {num_pairs} question and answer pairs as JSON in the form "
            f"[{{'question': '...', 'answer': '...'}}, ...].\nText:\n{text}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            pairs = json.loads(content)
        except Exception:
            continue
        for pair in pairs:
            q = pair.get("question")
            a = pair.get("answer")
            if not q or not a:
                continue
            faq_id = f"faq_{uuid4().hex}"
            combined = f"Q: {q}\nA: {a}"
            try:
                from mm_kb_builder.app import get_embedding as _get_embedding
            except Exception:
                raise RuntimeError("Embedding function unavailable")
            embedding = _get_embedding(combined, client)
            save_processed_data(
                kb_name,
                faq_id,
                chunk_text=combined,
                embedding=embedding,
                metadata={"faq": True, "question": q, "answer": a},
            )
            faq_entries.append({"id": faq_id, "question": q, "answer": a})
    if faq_entries:
        with open(kb_dir / "faqs.json", "w", encoding="utf-8") as f:
            json.dump(faq_entries, f, ensure_ascii=False, indent=2)
    return len(faq_entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate FAQs for a knowledge base")
    parser.add_argument("kb_name", help="Knowledge base name")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args(argv)
    count = generate_faqs_from_chunks(args.kb_name, args.max_tokens, args.pairs)
    print(f"Generated {count} FAQ entries")


if __name__ == "__main__":
    main()
