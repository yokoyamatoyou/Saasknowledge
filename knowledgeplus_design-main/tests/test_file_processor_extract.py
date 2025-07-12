import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

try:
    from PIL import Image
except Exception:
    pytest.skip("Pillow not installed", allow_module_level=True)

# Temporarily remove stub path to import real python-docx
STUBS_DIR = Path(__file__).resolve().parent / "stubs"
if str(STUBS_DIR) in sys.path:
    sys.path.remove(str(STUBS_DIR))
docx = importlib.import_module("docx")
sys.path.insert(0, str(STUBS_DIR))

from shared.file_processor import FileProcessor  # noqa: E402


def _create_docx_with_image(tmp_path):
    doc = docx.Document()
    doc.add_paragraph("hello world")
    img_path = tmp_path / "img.png"
    Image.new("RGB", (5, 5), "red").save(img_path)
    doc.add_picture(str(img_path))
    out_path = tmp_path / "sample.docx"
    doc.save(out_path)
    return out_path


def test_extract_text_and_images_docx(tmp_path):
    doc_path = _create_docx_with_image(tmp_path)
    with open(doc_path, "rb") as f:
        text, images = FileProcessor.extract_text_and_images(f)
    assert "hello world" in text
    assert len(images) == 1
