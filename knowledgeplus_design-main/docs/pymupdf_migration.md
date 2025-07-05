# PyMuPDF Migration Instructions

This document outlines the steps to replace the `pdf2image` dependency with **PyMuPDF** so the project no longer requires the external poppler utilities.

## Background
`pdf2image` currently converts PDF pages by calling out to the `pdftoppm` tool from poppler. This causes errors when poppler is missing. PyMuPDF (imported as `fitz`) links directly to MuPDF for rendering and text extraction, removing the external dependency.

## Tasks for Codex
1. **Add PyMuPDF**
   - Update `knowledgeplus_design-main/requirements.txt`, `requirements-extra.txt` and `mm_kb_builder/requirements.txt` to include `PyMuPDF>=1.23.0`.
   - Remove all `pdf2image` entries from these files.
2. **Refactor PDF handling code**
   - Replace `from pdf2image import convert_from_bytes` in `knowledge_gpt_app/app.py` with PyMuPDF usage. When a PDF page lacks text, open the PDF with `fitz.open(stream=data, filetype="pdf")`, render the page with `page.get_pixmap()`, convert to a `PIL.Image` object and perform OCR on that image.
   - In `shared/file_processor.py`, adjust `_encode_image_to_base64` and `extract_text_and_images` so PDF pages are rendered via PyMuPDF instead of `pdf2image.convert_from_bytes`.
3. **Clean up references**
   - Remove the `PDF_SUPPORT` flag and related checks for `pdf2image`.
   - Update any comments and error messages that mention `pdf2image` or poppler.
4. **Verify tests**
   - After refactoring, run `pytest -q` to ensure all existing tests still pass.

These changes will eliminate the poppler requirement and make PDF processing self‑contained.
