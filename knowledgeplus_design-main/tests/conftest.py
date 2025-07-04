import sys
from pathlib import Path

STUBS_DIR = Path(__file__).resolve().parent / "stubs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(STUBS_DIR))
sys.path.insert(1, str(PROJECT_ROOT))
