import json
import sys
from pathlib import Path

# Ensure shared package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "knowledgeplus_design-main"))

from shared.logging_utils import configure_logging  # noqa: E402
from shared.metrics import get_report  # noqa: E402


def main() -> None:
    """Print search metrics as JSON."""
    configure_logging()
    report = get_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
