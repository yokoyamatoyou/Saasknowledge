import streamlit as st
from config import DEFAULT_KB_NAME
from core.faq_utils import generate_faq
from knowledge_gpt_app.app import read_file as app_read_file
from knowledge_gpt_app.app import refresh_search_engine, semantic_chunking
from shared.file_processor import FileProcessor
from shared.openai_utils import get_openai_client
from ui_modules.thumbnail_editor import display_thumbnail_grid


def render_management_mode():
    """Render the management interface including FAQ generation."""
    st.subheader("管理")
    tabs = st.tabs(["ナレッジベース構築", "FAQ自動生成"])

    with tabs[0]:
        st.divider()
        with st.expander("ナレッジを追加する", expanded=True):
            process_mode = st.radio(
                "処理モード", ["個別処理", "まとめて処理"], help="ファイルを個別に処理するか、まとめて処理するかを選択します。"
            )
            index_mode = st.radio(
                "インデックス更新",
                ["自動(処理後)", "手動"],
                help="ファイル処理後に検索インデックスを自動で更新するか、手動で更新するかを選択します。",
            )

            files = st.file_uploader(
                "ファイルを選択",
                type=FileProcessor.SUPPORTED_IMAGE_TYPES
                + FileProcessor.SUPPORTED_DOCUMENT_TYPES
                + FileProcessor.SUPPORTED_CAD_TYPES,
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
                                    refresh=index_mode == "自動(処理後)"
                                    and process_mode == "個別処理",
                                )

                if process_mode == "まとめて処理" and index_mode == "自動(処理後)":
                    refresh_search_engine(DEFAULT_KB_NAME)

                st.toast("アップロード完了")

            if index_mode == "手動":
                if st.button("検索インデックス更新"):
                    with st.spinner("検索エンジン更新中..."):
                        refresh_search_engine(DEFAULT_KB_NAME)
                    st.toast("検索インデックスを更新しました")

        display_thumbnail_grid(DEFAULT_KB_NAME)

    with tabs[1]:
        kb_name = st.text_input(
            "Knowledge base name",
            value=DEFAULT_KB_NAME,
            help="FAQを生成するナレッジベースの名前を入力します。",
        )
        max_tokens = st.number_input(
            "Max tokens per chunk", 100, 2000, 1000, 100, help="チャンクあたりの最大トークン数を設定します。"
        )
        pairs = st.number_input(
            "Pairs per chunk", 1, 10, 3, 1, help="各チャンクから生成するQ&Aペアの数を設定します。"
        )
        if st.button(
            "◎ FAQ生成",
            key="generate_faqs_btn",
            type="primary",
            help="設定に基づいてFAQを生成し、ナレッジベースに保存します。",
        ):
            client = get_openai_client()
            if not client:
                st.error("OpenAIクライアントの取得に失敗しました。")
            else:
                with st.spinner("FAQを生成中..."):
                    count = generate_faq(
                        kb_name, int(max_tokens), int(pairs), client=client
                    )
                st.success(f"{count}件のFAQを生成しました。")
