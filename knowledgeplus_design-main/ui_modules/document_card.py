import streamlit as st
from html import escape
from typing import Dict, Any


def render_document_card(doc: Dict[str, Any]) -> None:
    """Display a single search result using the `doc-card` style."""
    meta = doc.get("metadata", {}) or {}
    display_meta = meta.get("display_metadata", {}) or {}
    title = meta.get("title") or display_meta.get("title") or meta.get("filename", "No title")
    snippet = doc.get("text", "")[:120].replace("\n", " ")
    similarity = doc.get("similarity")
    body = f"<div class='doc-card'><strong>{escape(title)}</strong>"
    if similarity is not None:
        body += f"<div>Score: {similarity:.3f}</div>"
    body += f"<div>{escape(snippet)}...</div></div>"
    st.markdown(body, unsafe_allow_html=True)
