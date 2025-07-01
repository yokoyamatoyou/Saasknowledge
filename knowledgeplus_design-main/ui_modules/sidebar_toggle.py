import streamlit as st


def render_sidebar_toggle(
    key: str = "toggle_sidebar",
    collapsed_label: str = ">>",
    expanded_label: str = "<<",
) -> None:
    """Display a toggle button to collapse or expand the sidebar."""
    if "sidebar_visible" not in st.session_state:
        st.session_state["sidebar_visible"] = False

    toggle_label = collapsed_label if not st.session_state.sidebar_visible else expanded_label
    if st.button(toggle_label, key=key, help="サイドバーの表示切替"):
        st.session_state.sidebar_visible = not st.session_state.sidebar_visible
        st.rerun()

    st.markdown(
        f"""
        <style>
        [data-testid='stSidebar'] {{
            transition: margin-left 0.3s ease;
            margin-left: {'0' if st.session_state.sidebar_visible else '-18rem'};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
