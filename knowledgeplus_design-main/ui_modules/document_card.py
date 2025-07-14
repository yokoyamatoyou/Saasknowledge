import base64
from html import escape
from pathlib import Path
from typing import Any, Dict

import streamlit as st


def render_document_card(doc: Dict[str, Any]) -> None:
    """Display a single search result using the `doc-card` style."""
    meta = doc.get("metadata", {}) or {}
    display_meta = meta.get("display_metadata", {}) or {}
    title = (
        meta.get("title")
        or display_meta.get("title")
        or meta.get("filename", "No title")
    )
    snippet = doc.get("text", "")[:120].replace("\n", " ")
    similarity = doc.get("similarity")

    preview_b64 = meta.get("preview_image")
    image_path = meta.get("paths", {}).get("image_path")
    if preview_b64:
        try:
            img_bytes = base64.b64decode(preview_b64)
            st.image(img_bytes, use_container_width=True)
        except Exception:
            pass
    elif image_path and Path(image_path).exists():
        st.image(image_path, use_container_width=True)

    body = f"<div class='doc-card'><strong>{escape(title)}</strong>"
    if similarity is not None:
        body += f"<div>Score: {similarity:.3f}</div>"
    body += f"<div>{escape(snippet)}...</div></div>"
    st.markdown(body, unsafe_allow_html=True)

    file_path = meta.get("paths", {}).get("original_file_path")
    if file_path:
        file = Path(file_path)
        if file.exists():
            st.download_button(
                label="画像ダウンロード",
                data=file.read_bytes(),
                file_name=file.name,
            )
        else:
            st.error("データ取得時から移動されており、リンク先に存在しません")
