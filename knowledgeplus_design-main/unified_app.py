import streamlit as st
import logging
import uuid
from datetime import datetime

from shared.env import load_env

# Import shared modules
from shared.upload_utils import ensure_openai_key, BASE_KNOWLEDGE_DIR
from ui_modules.theme import apply_intel_theme
from shared.chat_history_utils import (
    load_chat_histories,
    create_history,
    append_message,
    update_title,
    delete_history,
)
from ui_modules.sidebar_toggle import render_sidebar_toggle
from ui_modules.search_ui import render_search_mode
from ui_modules.management_ui import render_management_mode
from ui_modules.chat_ui import render_chat_mode

# Import functions from knowledge_gpt_app.app (some might be moved later)
from shared.openai_utils import get_openai_client

from config import DEFAULT_KB_NAME

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
st.set_page_config(
    layout="wide", page_title="KNOWLEDGE+", initial_sidebar_state="collapsed"
)

apply_intel_theme(st)

st.markdown("""
<style>
/* General styling for Google-like simplicity */
html, body, [class*="st-"] {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
    color: #3C4043; /* Google Grey 800 */
    background-color: #FFFFFF; /* White background */
}

/* Main container to center content and limit width */
.main .block-container {
    max-width: 850px; /* Similar to Google search results width */
    padding-top: 2rem; 
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Title styling - smaller and more subtle */
h1 {
    font-size: 2.2rem; /* Adjusted for smaller, more Google-like title */
    color: #202124; /* Google Grey 900 */
    text-align: center;
    margin-bottom: 2rem;
}

/* Search input styling - rounded, subtle border */
[data-testid="stTextInput"] input {
    border-color: #dfe1e5; /* Google Grey 300 */
    border-radius: 24px; /* Pill shape */
    padding: 10px 20px;
    box-shadow: none; /* Remove default Streamlit shadow */
    transition: box-shadow 0.3s ease-in-out, border-color 0.3s ease-in-out;
}
[data-testid="stTextInput"] input:hover {
    box-shadow: 0 1px 1px rgba(0,0,0,0.1); /* Subtle shadow on hover */
}
[data-testid="stTextInput"] input:focus {
    border-color: #1a73e8; /* Google Blue 500 */
    box-shadow: 0 1px 1px rgba(0,0,0,0.1), 0 0 0 1px #1a73e8; /* Blue glow on focus */
    outline: none; /* Remove default outline */
}

/* Primary Button styling - Google Blue */
[data-testid="stButton"] > button {
    background-color: #1a73e8; /* Google Blue 500 */
    color: #FFFFFF; /* White text */
    border-radius: 4px;
    border: none;
    padding: 10px 24px;
    font-weight: 500;
    transition: background-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
}
[data-testid="stButton"] > button:hover {
    background-color: #1765cc; /* Darker blue on hover */
    box-shadow: 0 1px 2px rgba(0,0,0,0.15);
}
[data-testid="stButton"] > button:active {
    background-color: #145cb8; /* Even darker on active */
}

/* Secondary Button styling */
[data-testid="stButton"] > button.secondary {
    background-color: #f8f9fa; /* Google Grey 100 */
    color: #3C4043; /* Google Grey 800 */
    border: 1px solid #dadce0; /* Google Grey 300 */
}
[data-testid="stButton"] > button.secondary:hover {
    background-color: #f0f0f0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

/* Card styling for search results */
.doc-card {
    border: 1px solid #dfe1e5;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 1px 2px 0 rgba(60,64,67,.3); /* Subtle shadow */
    background-color: #FFFFFF;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: #f8f9fa; /* Light grey header */
    border-radius: 8px;
    border: 1px solid #dadce0;
    padding: 10px 16px;
}

/* Chat input styling - improved visibility */
.st-chat-input {
    border: 1px solid #dadce0; /* Add a clear border */
    border-radius: 8px;
    padding: 8px;
    background-color: #f8f9fa; /* Slightly different background */
}
.st-chat-input input {
    border: none; /* Remove inner input border */
    box-shadow: none; /* Remove inner input shadow */
}

/* Chat message styling */
.stChatMessage {
    padding: 10px 15px;
    border-radius: 18px;
    margin-bottom: 10px;
}
.stChatMessage.user {
    background-color: #e6f4ea; /* Light green for user */
    border-bottom-right-radius: 2px;
}
.stChatMessage.assistant {
    background-color: #e8f0fe; /* Light blue for assistant */
    border-bottom-left-radius: 2px;
}

/* Maximize chat space */
[data-testid="stVerticalBlock"] > div:nth-child(2) {
    flex: 1; /* Allow chat history to take available vertical space */
    overflow-y: auto; /* Enable scrolling for chat history */
}
</style>
""", unsafe_allow_html=True)

TOGGLE_SIDEBAR_KEY = "toggle_sidebar"
TOGGLE_SIDEBAR_COLLAPSED = "＞＞"
TOGGLE_SIDEBAR_EXPANDED = "＜＜"

render_sidebar_toggle(
    key=TOGGLE_SIDEBAR_KEY,
    collapsed_label=TOGGLE_SIDEBAR_COLLAPSED,
    expanded_label=TOGGLE_SIDEBAR_EXPANDED,
)

def safe_generate_gpt_response(user_input, conversation_history=None, persona="default", temperature=None, response_length=None, client=None):
    """Wrapper around ChatController.generate_gpt_response with error handling."""
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
    st.session_state["current_mode"] = "検索" # Default to Search mode
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
# Use a key to persist selection and add emoji icons
mode_options = {
    "検索": "検索",
    "管理": "管理",
    "チャット": "チャット",
}

selected_mode_display = st.sidebar.radio(
    "メニュー", 
    list(mode_options.values()), 
    index=list(mode_options.keys()).index(st.session_state["current_mode"]),
    key="sidebar_mode_radio",
    help="アプリケーションのモードを選択します。"
)
# Convert the selected display label back to the internal key. If the label is
# not recognized (e.g. in tests that monkeypatch the sidebar), keep the
# existing mode to avoid errors.
if selected_mode_display in mode_options.values():
    st.session_state["current_mode"] = list(mode_options.keys())[list(mode_options.values()).index(selected_mode_display)]
else:
    logger.warning("Invalid mode selection: %s", selected_mode_display)

# Chat related sidebar controls. Some unit tests monkeypatch `st.sidebar` with a
# simple object that only provides `radio()`. Guard these calls so the app can
# be imported even when sidebar methods are missing.
sidebar = st.sidebar
if hasattr(sidebar, "markdown"):
    sidebar.markdown("---")
if hasattr(sidebar, "button") and sidebar.button("＋ 新しいチャット", key="new_chat_btn"):
    new_id = create_history({
        "persona": st.session_state.get("persona"),
        "temperature": st.session_state.get("temperature"),
        "prompt_advice": st.session_state.get("prompt_advice"),
    })
    st.session_state.current_chat_id = new_id
    st.session_state.chat_history = []
    st.session_state.gpt_conversation_title = "新しい会話"
    st.session_state.title_generated = False
    st.session_state.chat_histories.insert(0, {
        "id": new_id,
        "title": "新しい会話",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {},
        "messages": [],
    })

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
            "AIペルソナ", persona_ids,
            index=persona_ids.index(current_id),
            format_func=lambda x: persona_names.get(x, x),
        )
        st.session_state.persona = selected_id
        st.session_state.temperature = st.slider(
            "温度", 0.0, 1.0, float(st.session_state.get("temperature", 0.7)), 0.05
        )

if hasattr(sidebar, "expander"):
    with sidebar.expander("プロンプトアドバイス", expanded=False):
        st.session_state.prompt_advice = st.checkbox(
            "アドバイスを有効化",
            value=st.session_state.get("prompt_advice", False),
        )

# --- Main Content Area based on Mode ---
st.title("KNOWLEDGE+")  # Always show the main title

if st.session_state["current_mode"] == "検索":
    render_search_mode(safe_generate_gpt_response)
elif st.session_state["current_mode"] == "管理":
    render_management_mode()
elif st.session_state["current_mode"] == "チャット":
    render_chat_mode(safe_generate_gpt_response)
