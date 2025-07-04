import streamlit as st


def render_sidebar_toggle(
    key: str = "toggle_sidebar",
    collapsed_label: str = "＞＞",
    expanded_label: str = "＜＜",
    sidebar_width: str = "18rem",
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
        CSS width of the sidebar; allow overrides for custom layouts.
    """
    if "sidebar_visible" not in st.session_state:
        st.session_state["sidebar_visible"] = False

    toggle_label = collapsed_label if not st.session_state.sidebar_visible else expanded_label
    if st.button(toggle_label, key=key, help="サイドバーの表示切替"):
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
