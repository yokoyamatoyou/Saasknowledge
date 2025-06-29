import runpy
import sys
from pathlib import Path

# Allow imports from repository root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Execute the unified app located at the repository root
runpy.run_path(str(ROOT_DIR / "unified_app.py"), run_name="__main__")
