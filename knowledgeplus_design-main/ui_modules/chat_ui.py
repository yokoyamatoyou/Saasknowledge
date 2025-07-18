import streamlit as st
from config import DEFAULT_KB_NAME, HYBRID_BM25_WEIGHT, HYBRID_VECTOR_WEIGHT
from knowledge_gpt_app.app import search_multiple_knowledge_bases
from shared.chat_controller import ChatController
from shared.chat_history_utils import append_message, update_title
from shared.openai_utils import get_openai_client
from shared.prompt_advisor import generate_prompt_advice
from shared.search_engine import HybridSearchEngine
from shared.upload_utils import BASE_KNOWLEDGE_DIR


def render_chat_mode(safe_generate_gpt_response):
    """Render the chat interface."""
    st.subheader("chatGPT")
    # Display current conversation title underneath the header
    st.markdown(f"### {st.session_state.get('gpt_conversation_title', '新しい会話')}")
    use_kb = st.session_state.get("use_knowledge_search", True)

    # Use configured search weights without showing the slider
    vec_weight = HYBRID_VECTOR_WEIGHT
    bm25_weight = HYBRID_BM25_WEIGHT

    chat_container = st.container(height=700)
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
                advice_text = generate_prompt_advice(user_msg, client=client)
                if advice_text:
                    st.info(f"💡 プロンプトアドバイス:\n{advice_text}")
                    st.session_state["chat_history"].append(
                        {"role": "info", "content": advice_text}
                    )
                    # append_message(st.session_state.current_chat_id, "info", advice_text)
                    append_message(
                        st.session_state.current_chat_id, "info", advice_text
                    )

        context = ""
        if use_kb:
            if "chat_controller" not in st.session_state or not isinstance(
                st.session_state.chat_controller, ChatController
            ):
                try:
                    engine = HybridSearchEngine(
                        str(BASE_KNOWLEDGE_DIR / DEFAULT_KB_NAME)
                    )
                    st.session_state.chat_controller = ChatController(engine)
                except Exception as e:
                    st.error(f"ナレッジベースの初期化に失敗しました: {e}")
                    st.session_state.chat_controller = None

            if st.session_state.chat_controller:
                results, _ = search_multiple_knowledge_bases(
                    user_msg,
                    [DEFAULT_KB_NAME],
                    vector_weight=vec_weight,
                    bm25_weight=bm25_weight,
                )
                context = "\n".join(r.get("text", "") for r in results[:3])
                if not context:
                    st.info(
                        "ナレッジ検索で関連情報が見つかりませんでした。AIの一般的な知識で回答します。"
                    )
            else:
                st.warning(
                    "ナレッジ検索が無効化されているか、ナレッジベースの初期化に失敗したため、検索は行われません。"
                )

        client = get_openai_client()
        if client:
            prompt = (
                f"次の情報を参考にユーザーの質問に答えてください:\n{context}\n\n質問:{user_msg}"
                if use_kb and context
                else user_msg
            )
            chat_temp = (
                0.2 if use_kb else float(st.session_state.get("temperature", 0.7))
            )
            chat_persona = st.session_state.get("persona", "default")

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
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
                message_placeholder.markdown(full_response)
            answer = full_response
        else:
            answer = "OpenAIクライアントを初期化できませんでした。"
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer}
        )
        append_message(st.session_state.current_chat_id, "assistant", answer)

        if (
            not st.session_state.get("title_generated")
            and len(st.session_state.chat_history) >= 2
            and client
        ):
            current_history_for_title_gen = [
                m
                for m in st.session_state.chat_history
                if m["role"] in ["user", "assistant"]
            ]
            if current_history_for_title_gen:
                try:
                    if (
                        "chat_controller" in st.session_state
                        and st.session_state.chat_controller
                    ):
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
                except Exception as e:
                    st.error(f"会話タイトル生成エラー: {e}")
        st.rerun()
