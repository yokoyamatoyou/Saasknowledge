import streamlit as st
from config import HYBRID_BM25_WEIGHT, HYBRID_VECTOR_WEIGHT
from knowledge_gpt_app.app import list_knowledge_bases, search_multiple_knowledge_bases
from shared import feedback_store
from shared.openai_utils import get_openai_client
from shared.zero_hit_logger import log_zero_hit_query
from ui_modules.document_card import render_document_card


def render_result_with_feedback(doc):
    """Render a search result and capture feedback from the user."""
    render_document_card(doc)
    chunk_id = str(
        doc.get("id")
        or doc.get("metadata", {}).get("id")
        or doc.get("metadata", {}).get("chunk_id", "")
    )
    if not chunk_id:
        return
    col_good, col_bad, _ = st.columns([0.15, 0.15, 0.7])
    if col_good.button("役に立った", key=f"useful_{chunk_id}"):
        feedback_store.record_feedback(chunk_id, 1)
        st.toast("フィードバックを保存しました")
    if col_bad.button("役に立たなかった", key=f"not_useful_{chunk_id}"):
        feedback_store.record_feedback(chunk_id, -1)
        st.toast("フィードバックを保存しました")


def render_search_mode(safe_generate_gpt_response):
    """Refactored search interface with a clean layout and simplified controls."""
    st.subheader("ナレッジ検索")

    sidebar = st.sidebar
    if hasattr(sidebar, "checkbox"):
        if "filter_latest" not in st.session_state:
            st.session_state["filter_latest"] = False
        if "filter_company_rules" not in st.session_state:
            st.session_state["filter_company_rules"] = False
        with sidebar.expander("検索フィルター", expanded=False):
            st.session_state["filter_latest"] = st.checkbox(
                "最新バージョンのみ", value=st.session_state["filter_latest"]
            )
            st.session_state["filter_company_rules"] = st.checkbox(
                "会社規定のみ", value=st.session_state["filter_company_rules"]
            )

    # Apply custom CSS for buttons
    st.markdown(
        """
    <style>
        div.stButton > button[kind="primary"] {
            background-color: #007bff;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            width: 100%;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #0056b3;
            color: white;
        }
        div.stButton > button[kind="secondary"] {
            background-color: #6c757d;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            width: 100%;
        }
        div.stButton > button[kind="secondary"]:hover {
            background-color: #5a6268;
            color: white;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Main search input
    query = st.text_input(
        "main_search_box",
        placeholder="🔍 キーワードで検索、またはAIへの質問を入力...",
        label_visibility="collapsed",
    )

    # Create two columns for the buttons
    col1, col2 = st.columns(2)

    # "Execute" button
    if col1.button(
        "実行",
        type="primary",
        help="入力されたキーワードでナレッジベースを検索します。",
    ):
        st.session_state["search_executed"] = True
        try:
            kb_names = [kb["name"] for kb in list_knowledge_bases()]
            if kb_names:
                results, _ = search_multiple_knowledge_bases(
                    query,
                    kb_names,
                    vector_weight=HYBRID_VECTOR_WEIGHT,
                    bm25_weight=HYBRID_BM25_WEIGHT,
                )
                if st.session_state.get("filter_latest"):
                    results = [
                        r
                        for r in results
                        if not r.get("metadata", {})
                        .get("version_info", {})
                        .get("superseded_by")
                    ]
                if st.session_state.get("filter_company_rules"):
                    results = [
                        r
                        for r in results
                        if r.get("metadata", {})
                        .get("hierarchy_info", {})
                        .get("approval_level")
                        == "company"
                    ]
                st.session_state["results"] = results
                if not results:
                    log_zero_hit_query(query)
            else:
                st.session_state["results"] = []
                st.warning("検索可能なナレッジベースがありません。")
        except Exception as e:
            st.error(f"検索中にエラーが発生しました: {e}")
            st.session_state["results"] = []

        st.session_state["last_query"] = query

    # "Reset" button
    if col2.button(
        "リセット",
        type="secondary",
        help="検索ボックスと結果をクリアします。",
    ):
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
                    prompt = f"次の情報から質問『{st.session_state.get('last_query','')}』への要約回答を生成してください:\n{context}"
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
                render_result_with_feedback(doc)
