import importlib
import sys
import types
import warnings
from pathlib import Path

import pytest

STUBS_DIR = Path(__file__).resolve().parent / "stubs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.modules.pop("numpy", None)
if str(STUBS_DIR) in sys.path:
    sys.path.remove(str(STUBS_DIR))
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
np = importlib.import_module("numpy")

sys.path.insert(0, str(STUBS_DIR))
sys.path.insert(1, str(PROJECT_ROOT))

if not hasattr(np, "__version__"):
    np.__version__ = "1.24.0"

sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=lambda *a, **k: object()),
)

pytest.importorskip("streamlit")
pytest.importorskip("sudachipy")


def test_app_import_warns_once(monkeypatch):
    sys.modules.setdefault("pandas", types.SimpleNamespace())
    tokenize_mod = types.ModuleType("nltk.tokenize")
    tokenize_mod.word_tokenize = lambda t: t.split()
    nltk_mod = types.ModuleType("nltk")
    nltk_mod.tokenize = tokenize_mod
    sys.modules.setdefault("nltk", nltk_mod)
    sys.modules.setdefault("nltk.tokenize", tokenize_mod)

    warnings_called = []

    def fake_warn(*args, **kwargs):
        msg = kwargs.get("message", args[0] if args else None)
        warnings_called.append(msg)

    monkeypatch.setattr(warnings, "warn", fake_warn)
    if str(STUBS_DIR) in sys.path:
        sys.path.remove(str(STUBS_DIR))
    sys.modules.pop("knowledge_gpt_app.app", None)
    importlib.import_module("knowledge_gpt_app.app")
    sys.path.insert(0, str(STUBS_DIR))

    msgs = [m for m in warnings_called if m and "PyPDF2" in str(m)]
    assert len(msgs) <= 1
