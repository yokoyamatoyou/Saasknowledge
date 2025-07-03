import runpy
import sys
from pathlib import Path

# Allow running the app from the repository root
ROOT_DIR = Path(__file__).resolve().parent / "knowledgeplus_design-main"
sys.path.insert(0, str(ROOT_DIR))

runpy.run_path(str(ROOT_DIR / "app.py"), run_name="__main__")
