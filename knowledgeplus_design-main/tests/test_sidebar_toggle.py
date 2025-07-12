import importlib
import sys
from pathlib import Path

pytest = __import__("pytest")
pytest.importorskip("streamlit")
import streamlit as st  # noqa: E402

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))


def test_sidebar_toggle_updates_state(monkeypatch):
    calls = {"rerun": False}
    monkeypatch.setattr(st, "button", lambda label, key=None, help=None: True)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "rerun", lambda: calls.__setitem__("rerun", True))
    sidebar_toggle = importlib.import_module("ui_modules.sidebar_toggle")

    st.session_state.clear()
    sidebar_toggle.render_sidebar_toggle(key="test_toggle")
    assert st.session_state.get("sidebar_visible") is True
    assert calls["rerun"] is True


def test_sidebar_toggle_custom_width(monkeypatch):
    captured = {}
    monkeypatch.setattr(st, "button", lambda *a, **k: False)
    monkeypatch.setattr(st, "rerun", lambda: None)
    monkeypatch.setattr(
        st, "markdown", lambda text, **k: captured.setdefault("css", text)
    )
    sidebar_toggle = importlib.import_module("ui_modules.sidebar_toggle")

    st.session_state.clear()
    sidebar_toggle.render_sidebar_toggle(key="toggle_w", sidebar_width="25rem")
    assert "25rem" in captured.get("css", "")


def test_sidebar_toggle_initial_state_from_env(monkeypatch):
    monkeypatch.setattr(st, "button", lambda *a, **k: False)
    monkeypatch.setattr(st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(st, "rerun", lambda: None)
    monkeypatch.setenv("SIDEBAR_DEFAULT_VISIBLE", "true")

    # Reload module so the environment variable is read
    if "ui_modules.sidebar_toggle" in sys.modules:
        del sys.modules["ui_modules.sidebar_toggle"]
    sidebar_toggle = importlib.import_module("ui_modules.sidebar_toggle")

    st.session_state.clear()
    sidebar_toggle.render_sidebar_toggle(key="toggle_env")
    assert st.session_state.get("sidebar_visible") is True
