import logging
import os
import uuid
from datetime import datetime

import streamlit as st
from shared.chat_controller import get_persona_list
from shared.chat_history_utils import (
    create_history,
    delete_history,
    load_chat_histories,
)
from shared.env import load_env
from shared.upload_utils import ensure_openai_key
from ui_modules.chat_ui import render_chat_mode
from ui_modules.management_ui import render_management_mode
from ui_modules.search_ui import render_search_mode
from ui_modules.sidebar_toggle import render_sidebar_toggle
from ui_modules.theme import apply_intel_theme

logger = logging.getLogger(__name__)

# Load environment variables from .env if present
load_env()

# Early check for OpenAI API key
try:
    ensure_openai_key()
except Exception as e:
    st.error(f"OpenAI API key error: {e}")
    st.stop()

# Global page config and styling
# Default to showing the sidebar unless explicitly disabled
sidebar_visible = os.getenv("SIDEBAR_DEFAULT_VISIBLE", "true").lower() in {
    "1",
    "true",
    "yes",
}
initial_state = "expanded" if sidebar_visible else "collapsed"
st.set_page_config(
    layout="wide",
    page_title="ナレッジ＋",
    initial_sidebar_state=initial_state,
)

apply_intel_theme(st)

TOGGLE_SIDEBAR_KEY = "toggle_sidebar"
TOGGLE_SIDEBAR_COLLAPSED = "＞＞"
TOGGLE_SIDEBAR_EXPANDED = "＜＜"

render_sidebar_toggle(
    key=TOGGLE_SIDEBAR_KEY,
    collapsed_label=TOGGLE_SIDEBAR_COLLAPSED,
    expanded_label=TOGGLE_SIDEBAR_EXPANDED,
)


def safe_generate_gpt_response(
    user_input,
    conversation_history=None,
    persona="default",
    temperature=None,
    response_length=None,
    client=None,
):
    """Return a response generator or ``None`` on failure."""
    try:
        gen = st.session_state.chat_controller.generate_gpt_response(
            user_input,
            conversation_history=conversation_history,
            persona=persona,
            temperature=temperature,
            response_length=response_length,
            client=client,
        )
        return gen
    except Exception as e:
        st.error(f"GPT応答生成エラー: {e}")
        logger.error("GPT response generation error", exc_info=True)
        return None


# --- Session State Initialization ---
if "current_mode" not in st.session_state:
    st.session_state["current_mode"] = "検索"  # Default to Search mode
if "search_executed" not in st.session_state:
    st.session_state["search_executed"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "gpt_conversation_id" not in st.session_state:
    st.session_state["gpt_conversation_id"] = str(uuid.uuid4())
if "gpt_conversation_title" not in st.session_state:
    st.session_state["gpt_conversation_title"] = "新しい会話"
if "persona" not in st.session_state:
    st.session_state["persona"] = "default"
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7
if "response_length" not in st.session_state:
    st.session_state["response_length"] = "普通"
if "use_knowledge_search" not in st.session_state:
    # Flag controlling knowledge base lookup
    st.session_state["use_knowledge_search"] = True
if "prompt_advice" not in st.session_state:
    st.session_state["prompt_advice"] = False
if "title_generated" not in st.session_state:
    st.session_state["title_generated"] = False
if "chat_histories" not in st.session_state:
    st.session_state["chat_histories"] = load_chat_histories()
if "current_chat_id" not in st.session_state:
    if st.session_state["chat_histories"]:
        hist = st.session_state["chat_histories"][0]
        st.session_state["current_chat_id"] = hist["id"]
        st.session_state["chat_history"] = hist["messages"]
        st.session_state["gpt_conversation_title"] = hist["title"]
    else:
        st.session_state["current_chat_id"] = create_history({})

# --- Sidebar Navigation ---
# Use a key to persist selection and keep labels consistent across screens
mode_options = {
    "チャット": "chatGPT",
    "検索": "ナレッジ検索",
    "管理": "管理",
}

selected_mode_display = st.sidebar.radio(
    "メニュー",
    list(mode_options.values()),
    index=list(mode_options.keys()).index(st.session_state["current_mode"]),
    key="sidebar_mode_radio",
    help="アプリケーションのモードを選択します。",
)
# Convert the selected display label back to the internal key. If the label is
# not recognized (e.g. in tests that monkeypatch the sidebar), keep the
# existing mode to avoid errors.
if selected_mode_display in mode_options.values():
    st.session_state["current_mode"] = list(mode_options.keys())[
        list(mode_options.values()).index(selected_mode_display)
    ]
else:
    logger.warning("Invalid mode selection: %s", selected_mode_display)

# Chat related sidebar controls. Some unit tests monkeypatch `st.sidebar` with a
# simple object that only provides `radio()`. Guard these calls so the app can
# be imported even when sidebar methods are missing.
sidebar = st.sidebar
if hasattr(sidebar, "markdown"):
    sidebar.markdown("---")
if hasattr(sidebar, "button") and sidebar.button("＋ 新しいチャット", key="new_chat_btn"):
    new_id = create_history(
        {
            "persona": st.session_state.get("persona"),
            "temperature": st.session_state.get("temperature"),
            "prompt_advice": st.session_state.get("prompt_advice"),
        }
    )
    st.session_state.current_chat_id = new_id
    st.session_state.chat_history = []
    st.session_state.gpt_conversation_title = "新しい会話"
    st.session_state.title_generated = False
    st.session_state.chat_histories.insert(
        0,
        {
            "id": new_id,
            "title": "新しい会話",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {},
            "messages": [],
        },
    )

if st.session_state.chat_histories and hasattr(sidebar, "expander"):
    with sidebar.expander("過去の会話", expanded=False):
        for hist in list(st.session_state.chat_histories):
            col_load, col_del = st.columns([4, 1])
            if col_load.button(hist["title"], key=f"load_{hist['id']}"):
                st.session_state.current_chat_id = hist["id"]
                st.session_state.chat_history = hist["messages"]
                st.session_state.gpt_conversation_title = hist["title"]
                st.session_state.title_generated = True
                st.rerun()
            if col_del.button("削除", key=f"del_{hist['id']}"):
                delete_history(hist["id"])
                st.session_state.chat_histories = [
                    h for h in st.session_state.chat_histories if h["id"] != hist["id"]
                ]
                if st.session_state.current_chat_id == hist["id"]:
                    if st.session_state.chat_histories:
                        new_hist = st.session_state.chat_histories[0]
                        st.session_state.current_chat_id = new_hist["id"]
                        st.session_state.chat_history = new_hist["messages"]
                        st.session_state.gpt_conversation_title = new_hist["title"]
                        st.session_state.title_generated = True
                    else:
                        new_id = create_history({})
                        st.session_state.current_chat_id = new_id
                        st.session_state.chat_history = []
                        st.session_state.gpt_conversation_title = "新しい会話"
                        st.session_state.title_generated = False
                st.rerun()

if hasattr(sidebar, "expander"):
    with sidebar.expander("チャット設定", expanded=False):
        personas = get_persona_list()
        persona_ids = [p["id"] for p in personas]
        persona_names = {p["id"]: p.get("name", p["id"]) for p in personas}
        current_id = st.session_state.get("persona", persona_ids[0])
        selected_id = st.selectbox(
            "AIペルソナ",
            persona_ids,
            index=persona_ids.index(current_id),
            format_func=lambda x: persona_names.get(x, x),
        )
        st.session_state.persona = selected_id
        st.session_state.temperature = st.slider(
            "温度", 0.0, 1.0, float(st.session_state.get("temperature", 0.7)), 0.05
        )
        st.session_state.use_knowledge_search = st.checkbox(
            "全てのナレッジから検索する",
            value=st.session_state.get("use_knowledge_search", True),
        )

if hasattr(sidebar, "expander"):
    with sidebar.expander("プロンプトアドバイス", expanded=False):
        st.session_state.prompt_advice = st.checkbox(
            "アドバイスを有効化",
            value=st.session_state.get("prompt_advice", False),
        )

# --- Main Content Area based on Mode ---
st.markdown("<h1 class='app-title'>ナレッジ＋</h1>", unsafe_allow_html=True)

if st.session_state["current_mode"] == "検索":
    render_search_mode(safe_generate_gpt_response)
elif st.session_state["current_mode"] == "管理":
    render_management_mode()
elif st.session_state["current_mode"] == "チャット":
    render_chat_mode(safe_generate_gpt_response)
