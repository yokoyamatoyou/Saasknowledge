import streamlit as st
import logging
import uuid
from datetime import datetime

# Import shared modules
from shared.chat_controller import ChatController, get_persona_list
from shared.search_engine import HybridSearchEngine
from shared.file_processor import FileProcessor
from shared.upload_utils import ensure_openai_key, BASE_KNOWLEDGE_DIR
from ui_modules.theme import apply_intel_theme
from shared.chat_history_utils import (
    load_chat_histories,
    create_history,
    append_message,
    update_title,
)

# Import functions from knowledge_gpt_app.app (some might be moved later)
from knowledge_gpt_app.app import (
    list_knowledge_bases,
    semantic_chunking,
    get_openai_client,
    refresh_search_engine,
    read_file as app_read_file, # Rename to avoid conflict with FileProcessor
    search_multiple_knowledge_bases
)

# Import FAQ generation (assuming it's a standalone script)
from generate_faq import generate_faqs_from_chunks

from config import DEFAULT_KB_NAME

logger = logging.getLogger(__name__)

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

if "sidebar_visible" not in st.session_state:
    st.session_state["sidebar_visible"] = False

toggle_label = ">>" if not st.session_state.sidebar_visible else "<<"
if st.button(toggle_label, key="toggle_sidebar", help="サイドバーの表示切替"):
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible
    st.rerun()

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        transition: margin-left 0.3s ease;
        margin-left: {'0' if st.session_state.sidebar_visible else '-18rem'};
    }}
    </style>
    """,
    unsafe_allow_html=True,
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


def render_document_card(doc: dict) -> None:
    """Display a single search result using the `doc-card` style."""
    meta = doc.get("metadata", {}) or {}
    display_meta = meta.get("display_metadata", {}) or {}
    title = meta.get("title") or display_meta.get("title") or meta.get("filename", "No title")
    snippet = doc.get("text", "")[:120].replace("\n", " ")
    similarity = doc.get("similarity")
    from html import escape
    body = f"<div class='doc-card'><strong>{escape(title)}</strong>"
    if similarity is not None:
        body += f"<div>Score: {similarity:.3f}</div>"
    body += f"<div>{escape(snippet)}...</div></div>"
    st.markdown(body, unsafe_allow_html=True)

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
if "rag_enabled" not in st.session_state:
    st.session_state["rag_enabled"] = True # Default to RAG enabled
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
st.session_state["current_mode"] = list(mode_options.keys())[list(mode_options.values()).index(selected_mode_display)]

# Chat related sidebar controls
st.sidebar.markdown("---")
if st.sidebar.button("＋ 新しいチャット", key="new_chat_btn"):
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

if st.session_state.chat_histories:
    with st.sidebar.expander("過去の会話", expanded=False):
        for hist in st.session_state.chat_histories:
            if st.button(hist["title"], key=f"load_{hist['id']}"):
                st.session_state.current_chat_id = hist["id"]
                st.session_state.chat_history = hist["messages"]
                st.session_state.gpt_conversation_title = hist["title"]
                st.session_state.title_generated = True
                st.rerun()

with st.sidebar.expander("チャット設定", expanded=False):
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
    st.session_state.prompt_advice = st.checkbox(
        "アドバイスを有効化", value=st.session_state.get("prompt_advice", False)
    )

# --- Main Content Area based on Mode ---
st.title("KNOWLEDGE+") # Always show the main title

if st.session_state["current_mode"] == "検索":
    # Search mode specific UI
    query = st.text_input(
        "main_search_box",
        placeholder="🔍 キーワードで検索、またはAIへの質問を入力...",
        label_visibility="collapsed",
        help="ナレッジベースから情報を検索します。"
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("検索", type="primary", help="入力されたキーワードでナレッジベースを検索します。"):
            st.session_state["search_executed"] = True
            kb_names = [kb["name"] for kb in list_knowledge_bases()]
            st.session_state["results"], _ = search_multiple_knowledge_bases(
                query, kb_names
            )
            st.session_state["last_query"] = query
    with col2:
        if st.button("クリア", help="検索ボックスと結果をクリアします。"):
            st.session_state["search_executed"] = False
            st.session_state["results"] = []
            st.session_state["last_query"] = ""
            st.rerun()

    if st.session_state.get("search_executed"):
        st.markdown("\n---") # Separator
        tabs = st.tabs(["AIによる要約", "関連ナレッジ一覧"])
        with tabs[0]:
            results = st.session_state.get("results", [])
            if results:
                client = get_openai_client()
                if client:
                    context = "\n".join(r.get("text", "") for r in results[:3])
                    prompt = (
                        f"次の情報から質問『{st.session_state.get('last_query','')}』への"
                        f"要約回答を生成してください:\n{context}"
                    )
                    # Streaming for AI Summary
                    st.write("AIが要約を生成中...")
                    summary_placeholder = st.empty()
                    full_summary = ""
                    gen = safe_generate_gpt_response(
                        prompt,
                        conversation_history=[],
                        persona="default",
                        temperature=0.3,
                        response_length="簡潔",
                        client=client,
                    )
                    if gen:
                        for chunk in gen:
                            full_summary += chunk
                            summary_placeholder.markdown(full_summary + "▌")
                    summary_placeholder.markdown(full_summary) # Final content without cursor
                else:
                    st.info("要約生成に失敗しました。")
            else:
                st.info("検索結果がありません。")
        with tabs[1]:
            for doc in st.session_state.get("results", []):
                render_document_card(doc)




if st.session_state["current_mode"] == "管理":
    st.subheader("管理")
    tabs = st.tabs(["ナレッジベース構築", "FAQ自動生成"])

    with tabs[0]:
        st.divider()
        with st.expander("ナレッジを追加する", expanded=True):
            process_mode = st.radio("処理モード", ["個別処理", "まとめて処理"], help="ファイルを個別に処理するか、まとめて処理するかを選択します。")
            index_mode = st.radio("インデックス更新", ["自動(処理後)", "手動"], help="ファイル処理後に検索インデックスを自動で更新するか、手動で更新するかを選択します。")

            files = st.file_uploader(
                "ファイルを選択",
                type=FileProcessor.SUPPORTED_IMAGE_TYPES + FileProcessor.SUPPORTED_DOCUMENT_TYPES + FileProcessor.SUPPORTED_CAD_TYPES,
                accept_multiple_files=process_mode == "まとめて処理",
                help="サポートされている画像、ドキュメント、CADファイルをアップロードします。",
            )

            if files:
                if not isinstance(files, list):
                    files = [files]

                for file in files:
                    with st.spinner(f"ファイルを解析中: {file.name}..."):
                        text = app_read_file(file)
                    with st.spinner(f"ベクトル化しています: {file.name}..."):
                        if text:
                            client = get_openai_client()
                            if client:
                                semantic_chunking(
                                    text,
                                    15,
                                    "C",
                                    "auto",
                                    DEFAULT_KB_NAME,
                                    client,
                                    original_filename=file.name,
                                    original_bytes=file.getvalue(),
                                    refresh=index_mode == "自動(処理後)" and process_mode == "個別処理",
                                )

                if process_mode == "まとめて処理" and index_mode == "自動(処理後)":
                    refresh_search_engine(DEFAULT_KB_NAME)

                st.toast("アップロード完了")

            if index_mode == "手動":
                if st.button("検索インデックス更新"):
                    with st.spinner("検索エンジン更新中..."):
                        refresh_search_engine(DEFAULT_KB_NAME)
                    st.toast("検索インデックスを更新しました")

    with tabs[1]:
        kb_name = st.text_input("Knowledge base name", value=DEFAULT_KB_NAME, help="FAQを生成するナレッジベースの名前を入力します。")
        max_tokens = st.number_input("Max tokens per chunk", 100, 2000, 1000, 100, help="チャンクあたりの最大トークン数を設定します。")
        pairs = st.number_input("Pairs per chunk", 1, 10, 3, 1, help="各チャンクから生成するQ&Aペアの数を設定します。")
        if st.button("◎ FAQ生成", key="generate_faqs_btn", type="primary", help="設定に基づいてFAQを生成し、ナレッジベースに保存します。"):
            client = get_openai_client()
            if not client:
                st.error("OpenAIクライアントの取得に失敗しました。")
            else:
                with st.spinner("FAQを生成中..."):
                    count = generate_faqs_from_chunks(kb_name, int(max_tokens), int(pairs), client=client)
                    refresh_search_engine(kb_name)
                st.success(f"{count}件のFAQを生成しました。")

if st.session_state["current_mode"] == "チャット":
    st.subheader("チャット")  # Subheader for current mode

    use_kb = st.checkbox(
        "全てのナレッジから検索する",
        value=st.session_state.get("rag_enabled", True),
    )
    st.session_state["rag_enabled"] = use_kb

    # チャット履歴表示エリア
    chat_container = st.container(height=None) # Maximize vertical space

    with chat_container:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_msg = st.chat_input("メッセージを送信")
    if user_msg:
        st.session_state["chat_history"].append({"role": "user", "content": user_msg})
        append_message(st.session_state.current_chat_id, "user", user_msg)

        if st.session_state.get("prompt_advice"):
            client = get_openai_client()
            if client:
                advice_gen = safe_generate_gpt_response(
                    f"以下のユーザープロンプトを、より明確で効果的なプロンプトにするための改善案を、簡潔な箇条書きのMarkdown形式で提案してください:\n\n---\n{user_msg}\n---",
                    conversation_history=[],
                    persona="default",
                    temperature=0.0,
                    response_length="簡潔",
                    client=client,
                )
                advice_text = ""
                if advice_gen:
                    for chunk in advice_gen:
                        advice_text += chunk
                st.info(f"💡 プロンプトアドバイス:\n{advice_text}")
        
        context = ""
        if use_kb:
            # ナレッジ検索が有効な場合のみナレッジベースを読み込み、検索を実行
            # ChatControllerのインスタンス化をここで行うことで、RAG無効時は不要な初期化を避ける
            if "chat_controller" not in st.session_state or not isinstance(st.session_state.chat_controller, ChatController):
                try:
                    engine = HybridSearchEngine(str(BASE_KNOWLEDGE_DIR / DEFAULT_KB_NAME))
                    st.session_state.chat_controller = ChatController(engine)
                except Exception as e:
                    st.error(f"ナレッジベースの初期化に失敗しました: {e}")
                    st.session_state.chat_controller = None # 初期化失敗時はNoneを設定

            if st.session_state.chat_controller:
                results, _ = search_multiple_knowledge_bases(user_msg, [DEFAULT_KB_NAME])
                context = "\n".join(r.get("text", "") for r in results[:3])
                if not context:
                    st.info("ナレッジ検索で関連情報が見つかりませんでした。AIの一般的な知識で回答します。")
            else:
                st.warning("ナレッジ検索が無効化されているか、ナレッジベースの初期化に失敗したため、検索は行われません。")

        client = get_openai_client()
        if client:
            prompt = (
                f"次の情報を参考にユーザーの質問に答えてください:\n{context}\n\n質問:{user_msg}"
                if use_kb and context
                else user_msg
            )
            chat_temp = 0.2 if use_kb else float(st.session_state.get("temperature", 0.7))
            chat_persona = st.session_state.get("persona", "default")
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # ChatControllerが初期化されていない場合（RAG無効時など）は、直接GPT応答を生成
                if "chat_controller" not in st.session_state or st.session_state.chat_controller is None:
                    # Fallback to direct GPT response without RAG
                    gen = safe_generate_gpt_response(
                        user_msg,
                        conversation_history=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["chat_history"][:-1]
                            if m["role"] in ("user", "assistant")
                        ],
                        persona=chat_persona,
                        temperature=chat_temp,
                        response_length="普通",
                        client=client,
                    )
                    if gen:
                        for chunk in gen:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                else:
                    gen = safe_generate_gpt_response(
                        prompt,
                        conversation_history=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state["chat_history"][:-1]
                            if m["role"] in ("user", "assistant")
                        ],
                        persona=chat_persona,
                        temperature=chat_temp,
                        response_length="普通",
                        client=client,
                    )
                    if gen:
                        for chunk in gen:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response) # Final content without cursor
            answer = full_response
        else:
            answer = "OpenAIクライアントを初期化できませんでした。"
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        append_message(st.session_state.current_chat_id, "assistant", answer)
        
        # 4. 会話タイトル生成/更新
        if (
            not st.session_state.get("title_generated")
            and len(st.session_state.chat_history) >= 2
            and client
        ):
            current_history_for_title_gen = [
                m for m in st.session_state.chat_history if m["role"] in ["user", "assistant"]
            ]
            if current_history_for_title_gen:
                try:
                    if "chat_controller" in st.session_state and st.session_state.chat_controller:
                        new_title_val = st.session_state.chat_controller.generate_conversation_title(
                            current_history_for_title_gen, client
                        )
                        st.session_state.gpt_conversation_title = new_title_val
                        update_title(st.session_state.current_chat_id, new_title_val)
                        for h in st.session_state.chat_histories:
                            if h["id"] == st.session_state.current_chat_id:
                                h["title"] = new_title_val
                                break
                        st.session_state.title_generated = True
                        logger.info(f"会話タイトルを更新: {new_title_val}")
                except Exception as e:
                    logger.error(f"会話タイトル生成エラー: {e}", exc_info=True)

        st.rerun()


