from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from shared.kb_builder import KnowledgeBuilder

try:
    import fitz  # PyMuPDF

    FITZ_SUPPORT = True
except ImportError:  # pragma: no cover - optional
    fitz = None
    FITZ_SUPPORT = False

try:
    import docx  # python-docx

    DOCX_SUPPORT = True
except ImportError:  # pragma: no cover - optional
    DOCX_SUPPORT = False

try:
    from PIL import Image

    PIL_SUPPORT = True
except ImportError:  # pragma: no cover - optional
    PIL_SUPPORT = False

try:
    import PyPDF2

    PDF_TEXT_SUPPORT = True
except ImportError:  # pragma: no cover - optional
    PDF_TEXT_SUPPORT = False

import streamlit as st
from config import DEFAULT_KB_NAME

# CAD processing libraries
try:
    import ezdxf
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle

    DXF_SUPPORT = True
except ImportError:
    DXF_SUPPORT = False

try:
    import trimesh

    STL_SUPPORT = True
except ImportError:
    STL_SUPPORT = False

try:
    import cadquery as cq

    STEP_SUPPORT = True
except ImportError:
    STEP_SUPPORT = False

# PDF processing library is now a required dependency
logger = logging.getLogger(__name__)


class FileProcessor:
    SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
    SUPPORTED_DOCUMENT_TYPES = ["pdf", "docx", "doc", "xlsx", "xls", "txt"]
    SUPPORTED_CAD_TYPES = [
        "dxf",
        "stl",
        "ply",
        "obj",
        "step",
        "stp",
        "iges",
        "igs",
        "3ds",
    ]

    @staticmethod
    def _encode_image_to_base64(image_file):
        """画像ファイルをbase64エンコード"""
        try:
            if (
                hasattr(image_file, "type")
                and image_file.type == "application/pdf"
                and FITZ_SUPPORT
                and PIL_SUPPORT
            ):
                data = image_file.read()
                pdf_doc = fitz.open(stream=data, filetype="pdf")
                page = pdf_doc.load_page(0)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                pdf_doc.close()
                image_file.seek(0)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            elif (
                hasattr(image_file, "type")
                and image_file.type == "application/pdf"
                and not FITZ_SUPPORT
            ):
                st.error("PyMuPDF がインストールされていないため PDF を画像化できません")
                return None
            else:
                image_file.seek(0)
                image_bytes = image_file.read()
                image_file.seek(0)
                return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"画像base64エンコードエラー: {e}")
            st.error(f"画像の処理中にエラーが発生しました: {e}")
            return None

    @staticmethod
    def load_pdf(file_obj):
        """Return ``PyPDF2.PdfReader`` and raw bytes for ``file_obj``."""
        if not PDF_TEXT_SUPPORT:
            raise ImportError("PyPDF2 is required for PDF support")

        data = file_obj.read()
        file_obj.seek(0)
        reader = PyPDF2.PdfReader(BytesIO(data))
        return reader, data

    @staticmethod
    def _process_dxf_file(dxf_file):
        """DXFファイルを処理して画像とメタデータを生成"""
        if not DXF_SUPPORT:
            return None, {"error": "DXF処理ライブラリが利用できません"}

        try:
            dxf_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as temp_file:
                temp_file.write(dxf_file.read())
                temp_file_path = temp_file.name
            dxf_file.seek(0)

            doc = ezdxf.readfile(temp_file_path)
            msp = doc.modelspace()

            entities_info = {
                "lines": [],
                "circles": [],
                "arcs": [],
                "texts": [],
                "dimensions": [],
                "blocks": [],
            }

            for entity in msp:
                if entity.dxftype() == "LINE":
                    entities_info["lines"].append(
                        {
                            "start": tuple(entity.dxf.start[:2]),
                            "end": tuple(entity.dxf.end[:2]),
                        }
                    )
                elif entity.dxftype() == "CIRCLE":
                    entities_info["circles"].append(
                        {
                            "center": tuple(entity.dxf.center[:2]),
                            "radius": entity.dxf.radius,
                        }
                    )
                elif entity.dxftype() == "TEXT":
                    entities_info["texts"].append(
                        {
                            "text": entity.dxf.text,
                            "position": tuple(entity.dxf.insert[:2]),
                        }
                    )

            fig, ax = plt.subplots(figsize=(12, 8))

            for line in entities_info["lines"]:
                ax.plot(
                    [line["start"][0], line["end"][0]],
                    [line["start"][1], line["end"][1]],
                    "b-",
                    linewidth=1,
                )

            for circle in entities_info["circles"]:
                circle_patch = Circle(
                    circle["center"],
                    circle["radius"],
                    fill=False,
                    edgecolor="red",
                    linewidth=1,
                )
                ax.add_patch(circle_patch)

            for text in entities_info["texts"]:
                ax.text(
                    text["position"][0],
                    text["position"][1],
                    text["text"],
                    fontsize=8,
                    ha="left",
                )

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"DXF Drawing: {dxf_file.name}")

            buffer = io.BytesIO()
            plt.savefig(buffer, format="PNG", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            plt.close()

            os.unlink(temp_file_path)

            metadata = {
                "file_type": "DXF",
                "total_entities": len(list(msp)),
                "lines_count": len(entities_info["lines"]),
                "circles_count": len(entities_info["circles"]),
                "texts_count": len(entities_info["texts"]),
                "layers": [layer.dxf.name for layer in doc.layers],
                "drawing_units": doc.header.get("$INSUNITS", "Unknown"),
                "entities_detail": entities_info,
            }

            return image_base64, metadata

        except Exception as e:
            logger.error(f"DXF処理エラー: {e}")
            return None, {"error": f"DXF処理中にエラーが発生しました: {e}"}

    @staticmethod
    def _process_stl_file(stl_file):
        """STLファイルを処理して複数角度の画像を生成"""
        if not STL_SUPPORT:
            return None, {"error": "STL処理ライブラリが利用できません"}

        try:
            stl_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
                temp_file.write(stl_file.read())
                temp_file_path = temp_file.name
            stl_file.seek(0)

            mesh = trimesh.load_mesh(temp_file_path)

            metadata = {
                "file_type": "STL",
                "vertices_count": len(mesh.vertices),
                "faces_count": len(mesh.faces),
                "volume": float(mesh.volume),
                "surface_area": float(mesh.area),
                "bounds": mesh.bounds.tolist(),
                "center_mass": mesh.center_mass.tolist(),
                "is_watertight": mesh.is_watertight,
                "is_valid": mesh.is_valid,
            }

            angles = [(0, 0), (90, 0), (0, 90), (45, 45)]
            images = []

            for i, (azimuth, elevation) in enumerate(angles):
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")

                ax.plot_trisurf(
                    mesh.vertices[:, 0],
                    mesh.vertices[:, 1],
                    mesh.vertices[:, 2],
                    triangles=mesh.faces,
                    alpha=0.8,
                    cmap="viridis",
                )

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"STL Model - View {i+1} ({azimuth}°, {elevation}°)")
                ax.view_init(elev=elevation, azim=azimuth)

                buffer = io.BytesIO()
                plt.savefig(buffer, format="PNG", dpi=150, bbox_inches="tight")
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                images.append(image_base64)
                plt.close()

            os.unlink(temp_file_path)

            return images[0], {**metadata, "additional_views": images[1:]}

        except Exception as e:
            logger.error(f"STL処理エラー: {e}")
            return None, {"error": f"STL処理中にエラーが発生しました: {e}"}

    @staticmethod
    def _process_step_file(step_file):
        """STEPファイルを処理（CadQuery使用）"""
        if not STEP_SUPPORT:
            return None, {"error": "STEP処理ライブラリ（CadQuery）が利用できません"}

        try:
            step_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as temp_file:
                temp_file.write(step_file.read())
                temp_file_path = temp_file.name
            step_file.seek(0)

            result = cq.importers.importStep(temp_file_path)

            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as stl_temp:
                stl_temp_path = stl_temp.name

            cq.exporters.export(result, stl_temp_path)

            with open(stl_temp_path, "rb") as stl_temp_file:
                image_base64, metadata = FileProcessor._process_stl_file(stl_temp_file)

            if metadata and "error" not in metadata:
                metadata["file_type"] = "STEP"
                metadata["original_format"] = "STEP/STP"

            os.unlink(temp_file_path)
            os.unlink(stl_temp_path)

            return image_base64, metadata

        except Exception as e:
            logger.error(f"STEP処理エラー: {e}")
            return None, {"error": f"STEP処理中にエラーが発生しました: {e}"}

    @staticmethod
    def extract_text_and_images(file_obj):
        """Return text and embedded images from DOCX or PDF.

        Images are returned as base64 encoded PNG strings. Unsupported files
        fall back to plain text extraction.
        """
        ext = Path(file_obj.name).suffix.lower().lstrip(".")
        text = ""
        images = []
        try:
            if ext == "docx" and DOCX_SUPPORT:
                doc = docx.Document(file_obj)
                for para in doc.paragraphs:
                    text += para.text + "\n"
                if PIL_SUPPORT:
                    for rel in doc.part.related_parts.values():
                        if "image" in rel.content_type:
                            img = Image.open(BytesIO(rel.blob))
                            buf = BytesIO()
                            img.save(buf, format="PNG")
                            images.append(
                                base64.b64encode(buf.getvalue()).decode("utf-8")
                            )
            elif ext == "pdf" and PDF_TEXT_SUPPORT:
                data = file_obj.read()
                file_obj.seek(0)
                reader = PyPDF2.PdfReader(BytesIO(data))
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                if PIL_SUPPORT and FITZ_SUPPORT:
                    pdf_doc = fitz.open(stream=data, filetype="pdf")
                    for page in pdf_doc:
                        pix = page.get_pixmap()
                        img = Image.frombytes(
                            "RGB", [pix.width, pix.height], pix.samples
                        )
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
                    pdf_doc.close()
                elif PIL_SUPPORT and not FITZ_SUPPORT:
                    logger.warning(
                        "PyMuPDF not available, skipping PDF image extraction"
                    )
            else:
                text = file_obj.read().decode("utf-8", errors="replace")
                file_obj.seek(0)
        except Exception as e:  # pragma: no cover - parsing failures
            logger.error(f"Document extraction error: {e}")
        return text, images

    @staticmethod
    def create_metadata_from_text(text: str, images: list[str]) -> dict:
        """Return minimal metadata using the given text and images.

        This helper generates a short summary from the beginning of the text and
        records the number of extracted images.  A preview of the first image is
        also included so the UI can display a thumbnail without re-reading the
        document.  Heavy model calls are avoided so that PDF/DOCX processing
        remains lightweight.
        """
        summary = text.replace("\n", " ")[:100]
        meta = {"summary": summary, "image_count": len(images)}
        if images:
            meta["preview_image"] = images[0]
        return meta

    @staticmethod
    def extract_text_images_metadata(file_obj):
        """Return text, images and simple metadata for DOCX or PDF files.

        If ``pytesseract`` is available, basic OCR is performed on embedded
        images so that the returned text includes any visible captions.  The
        recognized strings are stored under ``ocr_snippets`` in the metadata.
        This keeps the implementation lightweight while capturing additional
        context when images contain text.
        """
        text, images = FileProcessor.extract_text_and_images(file_obj)
        meta = FileProcessor.create_metadata_from_text(text, images)

        ocr_snippets: list[str] = []
        if images and PIL_SUPPORT:
            try:  # optional dependency
                import pytesseract  # type: ignore
            except Exception:  # pragma: no cover - pytesseract missing
                pytesseract = None

            if pytesseract is not None:
                for img_b64 in images[:3]:  # limit to avoid heavy loops
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        img = Image.open(BytesIO(img_bytes))
                        text_snip = pytesseract.image_to_string(img, lang="jpn+eng")
                        if text_snip.strip():
                            ocr_snippets.append(text_snip.strip())
                    except Exception:  # pragma: no cover - OCR failure
                        continue

        if ocr_snippets:
            meta["ocr_snippets"] = ocr_snippets
            text = text + "\n" + "\n".join(ocr_snippets)

        return text, images, meta

    @classmethod
    def process_file(
        cls,
        file,
        kb_name: str = DEFAULT_KB_NAME,
        builder: Optional["KnowledgeBuilder"] = None,
    ):
        """Return a normalized representation of ``file``.

        The return value is a dictionary compatible with
        :meth:`KnowledgeBuilder.build_and_save_knowledge`.  It contains a
        ``type`` key describing the detected file type and either a ``text``
        field for document content or an ``image_base64`` field for image data.
        CAD metadata is returned under ``metadata`` when applicable.
        """

        file_extension = Path(file.name).suffix.lower().replace(".", "")

        if file_extension in cls.SUPPORTED_IMAGE_TYPES:
            image_b64 = cls._encode_image_to_base64(file)
            try:
                from shared import upload_utils
                from shared.kb_builder import KnowledgeBuilder  # local import

                file.seek(0)
                img_bytes = file.read()
                file.seek(0)
                if builder is None:
                    builder = KnowledgeBuilder(
                        cls(), lambda: None, refresh_search_engine_func=None
                    )
                embedding = builder.generate_image_embedding(img_bytes)
                if embedding is not None:
                    chunk_id = str(uuid.uuid4())
                    chunk_text = Path(file.name).name
                    upload_utils.save_processed_data(
                        kb_name,
                        chunk_id,
                        chunk_text=chunk_text,
                        embedding=embedding,
                    )
            except Exception as e:  # pragma: no cover - optional deps missing
                logger.error("画像ベクトル保存エラー: %s", e)

            return {"type": "image", "image_base64": image_b64, "metadata": None}

        if file_extension in {"pdf", "docx"}:
            text, images, meta = cls.extract_text_images_metadata(file)
            if text.strip():
                try:
                    from shared import upload_utils
                    from shared.kb_builder import KnowledgeBuilder  # local import

                    if builder is None:
                        builder = KnowledgeBuilder(
                            cls(), lambda: None, refresh_search_engine_func=None
                        )
                    embedding = builder.generate_text_embedding(text)
                    if embedding is not None:
                        chunk_id = str(uuid.uuid4())
                        upload_utils.save_processed_data(
                            kb_name,
                            chunk_id,
                            chunk_text=text,
                            embedding=embedding,
                            metadata=meta,
                        )
                except Exception as e:  # pragma: no cover - optional deps missing
                    logger.error("文書ベクトル保存エラー: %s", e)

                return {
                    "type": "document",
                    "text": text,
                    "images": images,
                    "metadata": meta,
                }

            image_b64 = cls._encode_image_to_base64(file)
            meta["file_type"] = "image_file"
            return {
                "type": "image",
                "image_base64": image_b64,
                "metadata": meta,
            }

        if file_extension in cls.SUPPORTED_DOCUMENT_TYPES:
            try:
                text = file.read().decode("utf-8", errors="replace")
            finally:
                file.seek(0)
            meta = cls.create_metadata_from_text(text, [])
            try:
                from shared import upload_utils
                from shared.kb_builder import KnowledgeBuilder

                if builder is None:
                    builder = KnowledgeBuilder(
                        cls(), lambda: None, refresh_search_engine_func=None
                    )
                embedding = builder.generate_text_embedding(text)
                if embedding is not None:
                    chunk_id = str(uuid.uuid4())
                    upload_utils.save_processed_data(
                        kb_name,
                        chunk_id,
                        chunk_text=text,
                        embedding=embedding,
                        metadata=meta,
                    )
            except Exception as e:  # pragma: no cover - optional deps missing
                logger.error("文書ベクトル保存エラー: %s", e)

            return {"type": "document", "text": text, "images": [], "metadata": meta}

        if file_extension == "dxf":
            image_b64, cad_meta = cls._process_dxf_file(file)
            return {"type": "cad", "image_base64": image_b64, "metadata": cad_meta}
        if file_extension == "stl":
            image_b64, cad_meta = cls._process_stl_file(file)
            return {"type": "cad", "image_base64": image_b64, "metadata": cad_meta}
        if file_extension in ["step", "stp"]:
            image_b64, cad_meta = cls._process_step_file(file)
            return {"type": "cad", "image_base64": image_b64, "metadata": cad_meta}
        if file_extension in ["ply", "obj"] and STL_SUPPORT:
            image_b64, cad_meta = cls._process_stl_file(file)
            return {"type": "cad", "image_base64": image_b64, "metadata": cad_meta}

        return {
            "type": "unsupported",
            "image_base64": None,
            "metadata": {"error": f"未対応のファイル形式です: {file_extension}"},
        }
