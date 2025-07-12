import os

import streamlit as st

DEFAULT_SIDEBAR_WIDTH = os.getenv("SIDEBAR_WIDTH", "18rem")
# Allow the initial visibility to be configured so different deployments
# can start with the sidebar expanded or collapsed. The value is treated
# as a boolean where "1", "true" or "yes" enable the sidebar.
DEFAULT_SIDEBAR_VISIBLE = os.getenv("SIDEBAR_DEFAULT_VISIBLE", "false").lower() in {
    "1",
    "true",
    "yes",
}


def render_sidebar_toggle(
    key: str = "toggle_sidebar",
    collapsed_label: str = "＞＞",
    expanded_label: str = "＜＜",
    sidebar_width: str = DEFAULT_SIDEBAR_WIDTH,
) -> None:
    """Display a toggle button to collapse or expand the sidebar.

    Parameters
    ----------
    key: str
        Session state key for the toggle button.
    collapsed_label: str
        Label shown when the sidebar is hidden.
    expanded_label: str
        Label shown when the sidebar is visible.
    sidebar_width: str
        CSS width of the sidebar.  The default value can be overridden by
        the ``SIDEBAR_WIDTH`` environment variable to customize layouts.
    """
    if "sidebar_visible" not in st.session_state:
        st.session_state["sidebar_visible"] = DEFAULT_SIDEBAR_VISIBLE

    toggle_label = (
        collapsed_label if not st.session_state.sidebar_visible else expanded_label
    )
    button_clicked = False
    sidebar_button = getattr(getattr(st, "sidebar", None), "button", None)
    if st.session_state.sidebar_visible:
        if sidebar_button and sidebar_button(toggle_label, key=key, help="サイドバーを折りたたむ"):
            button_clicked = True
    else:
        if st.button(toggle_label, key=key, help="サイドバーを表示"):
            st.markdown(
                f"<style>button[id='{key}'] {{position: fixed; top: 0.5rem; left: 0.5rem; z-index: 1000;}}</style>",
                unsafe_allow_html=True,
            )
            button_clicked = True

    if button_clicked:
        st.session_state.sidebar_visible = not st.session_state.sidebar_visible
        st.rerun()

    margin = "0" if st.session_state.sidebar_visible else f"-{sidebar_width}"
    st.markdown(
        f"""
        <style>
        [data-testid='stSidebar'] {{
            transition: margin-left 0.3s ease;
            margin-left: {margin};
            width: {sidebar_width};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
