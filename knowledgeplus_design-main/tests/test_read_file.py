import types
import base64
from io import BytesIO
from pathlib import Path
import sys
import pytest
import importlib
STUBS_DIR = Path(__file__).resolve().parent / "stubs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.modules.pop('numpy', None)
# Use real numpy and nltk by temporarily removing stub paths
if str(STUBS_DIR) in sys.path:
    sys.path.remove(str(STUBS_DIR))
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
np = importlib.import_module('numpy')
import nltk
sys.path.insert(0, str(STUBS_DIR))
sys.path.insert(1, str(PROJECT_ROOT))
import numpy.random
import numpy.core

if not hasattr(np, "__version__"):
    np.__version__ = "1.24.0"

# Stub heavy dependencies before importing the app
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=lambda *a, **k: object())
)

pytest.importorskip("streamlit")
pytest.importorskip("sudachipy")



def _load_app_with_real_numpy():
    sys.modules.setdefault("pandas", types.SimpleNamespace())
    tokenize_mod = types.ModuleType("nltk.tokenize")
    tokenize_mod.word_tokenize = lambda t: t.split()
    nltk_mod = types.ModuleType("nltk")
    nltk_mod.tokenize = tokenize_mod
    sys.modules.setdefault("nltk", nltk_mod)
    sys.modules.setdefault("nltk.tokenize", tokenize_mod)
    if str(STUBS_DIR) in sys.path:
        sys.path.remove(str(STUBS_DIR))
    app = importlib.import_module("knowledge_gpt_app.app")
    sys.path.insert(0, str(STUBS_DIR))
    return app


def test_read_file_excel(monkeypatch):
    kgapp = _load_app_with_real_numpy()
    fake_sheet = types.SimpleNamespace(
        title="Sheet1",
        _images=[],
        iter_rows=lambda values_only=True: [[1, 2], [3, 4]],
    )
    fake_wb = types.SimpleNamespace(worksheets=[fake_sheet])

    def fake_load(fileobj, data_only=True):
        return fake_wb

    monkeypatch.setattr(kgapp, "EXCEL_SUPPORT", True)
    monkeypatch.setattr(kgapp, "openpyxl", types.SimpleNamespace(load_workbook=fake_load))

    buf = BytesIO(b"dummy")
    buf.name = "test.xlsx"
    buf.seek(0)
    text = kgapp.read_file(buf)
    assert "Sheet1" in text
    assert "1\t2" in text


def test_read_file_markdown_image(monkeypatch):
    kgapp = _load_app_with_real_numpy()  # Moved import inside function
    b64 = base64.b64encode(b"imgdata").decode("ascii")
    content = f"hello ![](data:image/png;base64,{b64})"
    buf = BytesIO(content.encode("utf-8"))
    buf.name = "a.md"

    monkeypatch.setattr(kgapp, "OCR_SUPPORT", True)
    monkeypatch.setattr(kgapp, "Image", types.SimpleNamespace(open=lambda b: b))
    monkeypatch.setattr(kgapp, "pytesseract", types.SimpleNamespace(image_to_string=lambda i, lang=None: "ocr"))

    text = kgapp.read_file(buf)
    assert "hello" in text
    assert "ocr" in text


def test_read_file_pdf_simple(monkeypatch):
    kgapp = _load_app_with_real_numpy()
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(40, 10, "simple text")
    data = pdf.output(dest="S").encode("latin1")

    buf = BytesIO(data)
    buf.name = "file.pdf"

    text = kgapp.read_file(buf)
    assert "simple text" in text

