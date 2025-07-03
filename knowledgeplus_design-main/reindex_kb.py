import argparse
from pathlib import Path
import logging
from shared.search_engine import HybridSearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info("Reindex complete for '%s'", args.kb_name)


if __name__ == "__main__":
    main()
