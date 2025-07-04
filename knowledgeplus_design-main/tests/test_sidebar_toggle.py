import importlib
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

pytest = __import__('pytest')

pytest.importorskip("streamlit")
import streamlit as st


def test_sidebar_toggle_updates_state(monkeypatch):
    calls = {'rerun': False}
    monkeypatch.setattr(st, 'button', lambda label, key=None, help=None: True)
    monkeypatch.setattr(st, 'markdown', lambda *a, **k: None)
    monkeypatch.setattr(st, 'rerun', lambda: calls.__setitem__('rerun', True))
    sidebar_toggle = importlib.import_module('ui_modules.sidebar_toggle')

    st.session_state.clear()
    sidebar_toggle.render_sidebar_toggle(key="test_toggle")
    assert st.session_state.get('sidebar_visible') is True
    assert calls['rerun'] is True
