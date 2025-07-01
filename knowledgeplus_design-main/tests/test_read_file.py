import types
import base64
from io import BytesIO
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("streamlit")
pytest.importorskip("sudachipy")



def test_read_file_excel(monkeypatch):
    import knowledge_gpt_app.app as kgapp
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
    import knowledge_gpt_app.app as kgapp # Moved import inside function
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

