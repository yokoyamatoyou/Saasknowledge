import streamlit as st
from ui_modules.document_card import render_document_card
from shared.openai_utils import get_openai_client
from knowledge_gpt_app.app import list_knowledge_bases, search_multiple_knowledge_bases
from config import DEFAULT_KB_NAME


def render_search_mode(safe_generate_gpt_response):
    """Render the search interface."""
    query = st.text_input(
        "main_search_box",
        placeholder="🔍 キーワードで検索、またはAIへの質問を入力...",
        label_visibility="collapsed",
        help="ナレッジベースから情報を検索します。",
    )

    col1, col2, _ = st.columns([1, 1, 4])
    if col1.button("検索", type="primary", help="入力されたキーワードでナレッジベースを検索します。"):
        st.session_state["search_executed"] = True
        kb_names = [kb["name"] for kb in list_knowledge_bases()]
        st.session_state["results"], _ = search_multiple_knowledge_bases(query, kb_names)
        st.session_state["last_query"] = query
    if col2.button("クリア", help="検索ボックスと結果をクリアします。"):
        st.session_state["search_executed"] = False
        st.session_state["results"] = []
        st.session_state["last_query"] = ""
        st.rerun()

    if st.session_state.get("search_executed"):
        st.markdown("\n---")
        tabs = st.tabs(["AIによる要約", "関連ナレッジ一覧"])
        with tabs[0]:
            results = st.session_state.get("results", [])
            if results:
                client = get_openai_client()
                if client:
                    context = "\n".join(r.get("text", "") for r in results[:3])
                    prompt = (
                        f"次の情報から質問『{st.session_state.get('last_query','')}』への要約回答を生成してください:\n{context}"
                    )
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
                    summary_placeholder.markdown(full_summary)
                else:
                    st.info("要約生成に失敗しました。")
            else:
                st.info("検索結果がありません。")
        with tabs[1]:
            for doc in st.session_state.get("results", []):
                render_document_card(doc)
