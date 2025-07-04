import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from PIL import Image
import importlib
import types

# Remove stubs to load real python-docx
STUBS_DIR = Path(__file__).resolve().parent / "stubs"
if str(STUBS_DIR) in sys.path:
    sys.path.remove(str(STUBS_DIR))
docx = importlib.import_module("docx")
sys.path.insert(0, str(STUBS_DIR))

from shared.file_processor import FileProcessor


def _create_docx_with_image(tmp_path):
    doc = docx.Document()
    doc.add_paragraph("hello world")
    img_path = tmp_path / "img.png"
    Image.new("RGB", (5, 5), "red").save(img_path)
    doc.add_picture(str(img_path))
    out_path = tmp_path / "sample.docx"
    doc.save(out_path)
    return out_path


def test_extract_text_images_metadata(tmp_path, monkeypatch):
    doc_path = _create_docx_with_image(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "pytesseract",
        types.SimpleNamespace(image_to_string=lambda i, lang=None: "ocr text"),
    )

    with open(doc_path, "rb") as f:
        text, images, meta = FileProcessor.extract_text_images_metadata(f)

    assert "hello world" in text
    assert "ocr text" in text
    assert len(images) == 1
    assert meta["image_count"] == 1
    assert meta["summary"].startswith("hello world")
    assert meta.get("preview_image") == images[0]
    assert meta.get("ocr_snippets") == ["ocr text"]
