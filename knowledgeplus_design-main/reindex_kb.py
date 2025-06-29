import argparse
from pathlib import Path
from shared.search_engine import HybridSearchEngine


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild indexes for an existing knowledge base"
    )
    parser.add_argument(
        "kb_name",
        help="Name of the knowledge base directory under knowledge_base/",
    )
    args = parser.parse_args()

    kb_path = Path("knowledge_base") / args.kb_name
    if not kb_path.exists():
        parser.error(f"Knowledge base '{args.kb_name}' not found at {kb_path}")

    engine = HybridSearchEngine(str(kb_path))
    engine.reindex()
    print(f"Reindex complete for '{args.kb_name}'")


if __name__ == "__main__":
    main()
