# Development Progress Log

## 2024-06-30
- Installed missing dependency `sudachipy` required for tests.
- Ensured all tests run successfully via `pytest -q`.
- Reviewed repository structure and confirmed unified app entry point at `unified_app.py`.

## 2025-07-01
- Implemented sidebar toggle in `unified_app.py` so the sidebar slides in and out using `<<` and `>>`.
- Verified automated tests still pass.


## 2025-07-03
- Added unit test for `save_user_metadata` to verify metadata files are stored correctly.
- All tests continue to pass with `pytest -q`.

## 2025-07-04
- Integrated thumbnail metadata editor into the management tab using `display_thumbnail_grid`.
- Added regression test to confirm the call is present in `unified_app.py`.
- All tests pass with `pytest -q`.

## 2025-07-05
- Removed temporary print statements from `shared/search_engine.py` and cleaned up minor style issues.

## 2025-07-06
- Installed dependencies using `scripts/install_light.sh` followed by `scripts/install_extra.sh` to resolve missing modules without network errors.
- Verified that `pytest -q` completes successfully with 33 tests.

## 2025-07-07
- Added `extract_text_and_images()` helper in `shared/file_processor.py` to parse DOCX/PDF files and return embedded images.
- Created `test_file_processor_extract.py` covering DOCX extraction.
- All tests pass with `pytest -q` (34 tests).

## 2025-07-08
- Added `test_full_workflow.py` to simulate building a knowledge item and generating FAQs using stubs.
- Verified all tests pass with `pytest -q` (40 tests).

## 2025-07-09
- Added PyMuPDF migration instructions in docs/pymupdf_migration.md for removing the pdf2image/poppler dependency.

## 2025-07-10
- Replaced pdf2image with PyMuPDF for PDF OCR fallback.
- Updated tests to prefer real dependencies when installed.
- Verified all 48 tests pass with pytest.

## 2025-07-11
- Made PyMuPDF optional in `FileProcessor` so missing dependencies no longer
  break imports.
- Image extraction tests now skip when Pillow isn't installed.
- Confirmed the suite runs cleanly with and without optional packages.

## 2025-07-12
- Switched to mandatory PyMuPDF in `shared/file_processor.py` and removed old pdf2image checks.
- PDF thumbnails now render via `fitz` directly.
- Verified all 54 tests pass with `pytest -q`.

## 2025-07-13
- Added PyMuPDF to `requirements-light.txt` so CI installs it by default.
- Updated README to note PyMuPDF is mandatory and pytesseract remains optional.


## 2025-07-17
- Fixed a ValueError when uploading mixed file types in `mm_kb_builder`. `FileProcessor.process_file` now returns dictionaries so the app uses `get()` to access `image_base64` and `metadata`. Unsupported files show a warning and are skipped. Recorded this change because merge conflicts had occurred at the same code section previously.


## 2025-07-17 (2)
- Reformatted `mm_kb_builder/app.py` with Black to resolve pre-commit failure.

## 2025-07-17 (3)
- Ran pre-commit hooks locally and confirmed Black, ruff and isort all pass.
- Verified full test suite executes successfully after installing light dependencies.
