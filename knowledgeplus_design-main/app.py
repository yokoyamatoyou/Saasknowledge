import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Delegate execution to the main unified application
runpy.run_path(str(ROOT_DIR / "unified_app.py"), run_name="__main__")
