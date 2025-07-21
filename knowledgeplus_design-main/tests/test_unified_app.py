import importlib
import sys
import types
from pathlib import Path

import pytest
from config import DEFAULT_KB_NAME

sys.modules.pop("numpy", None)
np = importlib.import_module("numpy")

# Get the project root directory (one level up from the 'tests' directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(1, str(PROJECT_ROOT))
sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=lambda *a, **k: object()),
)


def _load_app_with_real_numpy():
    STUBS_DIR = Path(__file__).resolve().parent / "stubs"
    sys.modules.setdefault("pandas", types.SimpleNamespace())
    tokenize_mod = types.ModuleType("nltk.tokenize")
    tokenize_mod.word_tokenize = lambda t: t.split()
    nltk_mod = types.ModuleType("nltk")
    nltk_mod.tokenize = tokenize_mod
    sys.modules.setdefault("nltk", nltk_mod)
    sys.modules.setdefault("nltk.tokenize", tokenize_mod)
    if str(STUBS_DIR) in sys.path:
        sys.path.remove(str(STUBS_DIR))
    app = importlib.import_module("knowledge_gpt_app.app")
    sys.path.insert(0, str(STUBS_DIR))
    sys.path.insert(1, str(PROJECT_ROOT))
    return app


# Provide minimal UI module stubs so unified_app can import
sys.modules["ui_modules.search_ui"] = types.ModuleType("ui_modules.search_ui")
sys.modules["ui_modules.search_ui"].render_search_mode = lambda *a, **k: None
sys.modules["ui_modules.management_ui"] = types.ModuleType("ui_modules.management_ui")
sys.modules["ui_modules.management_ui"].render_management_mode = lambda *a, **k: None
sys.modules["ui_modules.chat_ui"] = types.ModuleType("ui_modules.chat_ui")
sys.modules["ui_modules.chat_ui"].render_chat_mode = lambda *a, **k: None


def test_sidebar_has_faq_button():
    app_path = PROJECT_ROOT / "ui_modules" / "management_ui.py"
    text = app_path.read_text(encoding="utf-8")
    assert "FAQ生成" in text
    assert "処理モード" in text
    assert "個別処理" in text
    assert "まとめて処理" in text
    assert "インデックス更新" in text
    assert "自動(処理後)" in text
    assert "手動" in text
    assert "検索インデックス更新" in text


def test_manual_refresh_call_present():
    app_path = PROJECT_ROOT / "ui_modules" / "management_ui.py"
    text = app_path.read_text(encoding="utf-8")
    import re

    pattern = r'if st\.button\("検索インデックス更新"\).*refresh_search_engine\(DEFAULT_KB_NAME\)'
    assert re.search(pattern, text, re.DOTALL)


def test_sidebar_toggle_button_present():
    app_path = PROJECT_ROOT / "unified_app.py"
    text = app_path.read_text(encoding="utf-8")
    assert "toggle_sidebar" in text
    assert "＞＞" in text and "＜＜" in text


def test_prompt_advice_option_present():
    app_path = PROJECT_ROOT / "unified_app.py"
    text = app_path.read_text(encoding="utf-8")
    assert "アドバイスを有効化" in text


def test_delete_history_button_present():
    app_path = PROJECT_ROOT / "unified_app.py"
    text = app_path.read_text(encoding="utf-8")
    assert "delete_history" in text
    assert "削除" in text


def test_prompt_advice_saved_to_history():
    app_path = PROJECT_ROOT / "ui_modules" / "chat_ui.py"
    text = app_path.read_text(encoding="utf-8")
    assert (
        'append_message(st.session_state.current_chat_id, "info", advice_text)' in text
    )


def test_safe_generate_handles_error(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("sudachipy")

    import streamlit as st

    # Mock st.error to capture its calls
    mock_st_error_messages = []
    monkeypatch.setattr(st, "error", lambda msg: mock_st_error_messages.append(msg))

    import importlib

    monkeypatch.setattr("ui_modules.theme.apply_intel_theme", lambda *a, **k: None)
    monkeypatch.setattr(st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(st, "title", lambda *a, **k: None)
    monkeypatch.setattr(
        st,
        "columns",
        lambda *a, **k: (
            types.SimpleNamespace(button=lambda *a, **k: False),
            types.SimpleNamespace(button=lambda *a, **k: False),
        ),
    )
    sidebar = types.SimpleNamespace(radio=lambda *a, **k: "FAQ")
    monkeypatch.setattr(st, "sidebar", sidebar)
    monkeypatch.setattr(st, "info", lambda *a, **k: None)

    # Mock streamlit.session_state before importing unified_app
    # This mock now correctly simulates the structure used by the app
    mock_chat_controller = types.SimpleNamespace(
        generate_gpt_response=lambda *a, **k: "mocked response"
    )

    class MockSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

        def __setattr__(self, name, value):
            self[name] = value

    # Initialize with the structure the app expects
    mock_session_state = MockSessionState(
        chat_controller=mock_chat_controller, search_engines={}
    )
    monkeypatch.setattr(st, "session_state", mock_session_state)

    # We need to reload unified_app for the monkeypatching to take effect
    mod = importlib.reload(__import__("unified_app"))

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(mock_chat_controller, "generate_gpt_response", boom)
    result = mod.safe_generate_gpt_response(
        "prompt",
        conversation_history=[],
        persona="default",
        temperature=0.1,
        response_length="簡潔",
        client=None,
    )
    assert result is None
    assert len(mock_st_error_messages) > 0  # Check if st.error was called


def test_safe_generate_missing_client(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("sudachipy")

    import streamlit as st

    mock_messages = []
    monkeypatch.setattr(st, "error", lambda msg: mock_messages.append(msg))

    import importlib

    monkeypatch.setattr("ui_modules.theme.apply_intel_theme", lambda *a, **k: None)
    monkeypatch.setattr(st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(st, "title", lambda *a, **k: None)
    monkeypatch.setattr(
        st,
        "columns",
        lambda *a, **k: (
            types.SimpleNamespace(button=lambda *a, **k: False),
            types.SimpleNamespace(button=lambda *a, **k: False),
        ),
    )
    sidebar = types.SimpleNamespace(radio=lambda *a, **k: "FAQ")
    monkeypatch.setattr(st, "sidebar", sidebar)
    monkeypatch.setattr(st, "info", lambda *a, **k: None)

    from shared.chat_controller import ChatController

    monkeypatch.setattr(
        ChatController, "_get_openai_client_internal", staticmethod(lambda: None)
    )

    controller = ChatController(None)  # type: ignore[arg-type]

    class MockSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(f"{type(self).__name__} has no attribute {name}")

        def __setattr__(self, name, value):
            self[name] = value

    mock_session_state = MockSessionState(chat_controller=controller, search_engines={})
    monkeypatch.setattr(st, "session_state", mock_session_state)

    mod = importlib.reload(__import__("unified_app"))

    result = mod.safe_generate_gpt_response("prompt")
    from shared.errors import OpenAIClientError

    with pytest.raises(OpenAIClientError):
        next(result)


def test_refresh_search_engine_reloads_engine(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("sudachipy")
    import streamlit as st

    # Mock st.session_state
    class MockSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

        def __setattr__(self, name, value):
            self[name] = value

    # Initialize with the structure the app expects
    mock_session_state = MockSessionState(search_engines={})
    monkeypatch.setattr(st, "session_state", mock_session_state)

    # Mock HybridSearchEngine
    class MockHybridSearchEngine:
        def __init__(self, path):
            self.reindex_called = False

        def reindex(self):
            self.reindex_called = True

    monkeypatch.setattr(
        "shared.search_engine.HybridSearchEngine", MockHybridSearchEngine
    )

    # Import the function to test using real numpy
    app_mod = _load_app_with_real_numpy()
    refresh_search_engine = app_mod.refresh_search_engine

    # We also need to mock get_search_engine to control the engine instance
    def mock_get_search_engine(kb_name):
        if kb_name not in st.session_state.search_engines:
            st.session_state.search_engines[kb_name] = MockHybridSearchEngine(None)
        return st.session_state.search_engines[kb_name]

    monkeypatch.setattr(
        "knowledge_gpt_app.app.get_search_engine", mock_get_search_engine
    )

    # Call the function
    refresh_search_engine(DEFAULT_KB_NAME)

    # Assert that reindex was called on the correct engine instance
    assert st.session_state.search_engines[DEFAULT_KB_NAME].reindex_called


def test_thumbnail_grid_call_present():
    app_path = PROJECT_ROOT / "ui_modules" / "management_ui.py"
    text = app_path.read_text(encoding="utf-8")
    assert "display_thumbnail_grid(DEFAULT_KB_NAME)" in text
