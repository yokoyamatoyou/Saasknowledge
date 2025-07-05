import sys
from pathlib import Path

STUBS_DIR = Path(__file__).resolve().parent / "stubs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Prefer real libraries when available. The lightweight stubs are kept for
# environments without optional dependencies installed.
try:  # pragma: no cover - environment check
    import numpy  # noqa: F401
    import nltk  # noqa: F401
    USE_STUBS = False
except Exception:  # pragma: no cover - fallback to stubs
    USE_STUBS = True

if USE_STUBS:
    sys.path.insert(0, str(STUBS_DIR))
sys.path.insert(1, str(PROJECT_ROOT))
