"""Shared UI theming utilities."""

from pathlib import Path


def apply_intel_theme(st):
    """Inject Intel themed CSS into a Streamlit app."""
    css_path = Path(__file__).parent / "static" / "theme.css"
    css = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
