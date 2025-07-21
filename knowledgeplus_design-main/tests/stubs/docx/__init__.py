"""Very small stub of the ``python-docx`` ``Document`` class.

This is intentionally tiny and only implements the pieces needed by the test
suite.  It allows creating ``Document`` objects, adding paragraphs and images
and saving/loading that minimal structure.  The goal is to avoid pulling in the
real ``python-docx`` dependency when running the lightweight test environment.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Iterable, Optional


class Document:
    def __init__(self, file: Optional[IO | str] = None):
        """Create a new stub document or load from ``file`` if provided."""

        self.paragraphs: list[SimpleNamespace] = []
        self._images: list[bytes] = []
        self.part = SimpleNamespace(related_parts={})

        if file is not None:
            data: bytes
            if hasattr(file, "read"):
                data = file.read()
                file.seek(0)
            else:
                with open(file, "rb") as f:
                    data = f.read()

            try:
                obj = json.loads(data.decode("utf-8"))
            except Exception:
                # Not a stub DOCX file; leave document empty
                return

            for text in obj.get("paragraphs", []):
                self.add_paragraph(text)
            for b64 in obj.get("images", []):
                img = base64.b64decode(b64)
                self._add_image_bytes(img)

    def add_paragraph(self, text: str = ""):
        para = SimpleNamespace(text=text)
        self.paragraphs.append(para)
        return para

    def _add_image_bytes(self, data: bytes):
        key = f"image{len(self.part.related_parts)}"
        self.part.related_parts[key] = SimpleNamespace(
            content_type="image/png", blob=data
        )
        self._images.append(data)

    def add_picture(self, image_path: str):
        with open(image_path, "rb") as f:
            data = f.read()
        self._add_image_bytes(data)

    def save(self, file_path: str | Path):
        obj = {
            "paragraphs": [p.text for p in self.paragraphs],
            "images": [base64.b64encode(i).decode("utf-8") for i in self._images],
        }
        Path(file_path).write_bytes(json.dumps(obj).encode("utf-8"))
