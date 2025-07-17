import argparse
import json
import logging
from uuid import uuid4

import requests
from core import mm_builder_utils
from shared.logging_utils import configure_logging
from shared.openai_utils import get_openai_client
from shared.upload_utils import BASE_KNOWLEDGE_DIR, save_processed_data

configure_logging()
logger = logging.getLogger(__name__)


def generate_faqs_from_chunks(
    kb_name: str,
    max_tokens: int = 1000,
    num_pairs: int = 3,
    client=None,
    source: str | None = None,
) -> int:
    kb_dir = BASE_KNOWLEDGE_DIR / kb_name
    chunks_dir = kb_dir / "chunks"

    texts: list[str] = []
    if source:
        text = source.strip()
        if text.startswith("http://") or text.startswith("https://"):
            try:
                resp = requests.get(text, timeout=10)
                resp.raise_for_status()
                text = resp.text
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to fetch URL: %s", e)
        texts.append(text)
    else:
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
        texts = [
            p.read_text(encoding="utf-8") for p in sorted(chunks_dir.glob("*.txt"))
        ]

    if client is None:
        client = get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client unavailable")

    faq_entries = []
    for raw_text in texts:
        text = raw_text[:max_tokens]
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
        except json.JSONDecodeError as e:
            logger.error("JSON decode failed: %s", getattr(e, "msg", e))
            logger.error("Raw response: %s", content)
            continue
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
                embedding = mm_builder_utils.get_text_embedding(combined)
            except Exception:
                raise RuntimeError("Embedding function unavailable")
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
    logger.info("Generated %s FAQ entries", count)


if __name__ == "__main__":
    main()
