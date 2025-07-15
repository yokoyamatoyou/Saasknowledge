import logging
import time

import streamlit as st
from config import DEFAULT_KB_NAME
from core.faq_utils import generate_faq
from core.mm_builder_utils import analyze_image_with_gpt4o
from knowledge_gpt_app.app import refresh_search_engine
from shared.file_processor import FileProcessor
from shared.kb_builder import KnowledgeBuilder
from shared.openai_utils import get_openai_client
from ui_modules.thumbnail_editor import display_thumbnail_grid

logger = logging.getLogger(__name__)


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

            file_processor = FileProcessor()
            kb_builder = KnowledgeBuilder(
                file_processor,
                get_openai_client_func=get_openai_client,
                refresh_search_engine_func=refresh_search_engine,
            )

            files = st.file_uploader(
                "ファイルを選択",
                type=file_processor.SUPPORTED_IMAGE_TYPES
                + file_processor.SUPPORTED_DOCUMENT_TYPES
                + file_processor.SUPPORTED_CAD_TYPES,
                accept_multiple_files=process_mode == "まとめて処理",
                help="サポートされている画像、ドキュメント、CADファイルをアップロードします。",
            )

            if files:
                if not isinstance(files, list):
                    files = [files]

                if st.button("選択したファイルの処理を開始", type="primary"):
                    progress_bar = st.progress(0, "処理を開始します...")
                    start_time = time.time()

                    original_refresh = kb_builder.refresh_search_engine
                    if process_mode == "まとめて処理" or index_mode == "手動":
                        kb_builder.refresh_search_engine = lambda *_: None

                    for i, uploaded_file in enumerate(files):
                        file_name = uploaded_file.name
                        progress_text = f"({i+1}/{len(files)}) {file_name} を処理中..."
                        progress_bar.progress((i + 1) / len(files), text=progress_text)

                        try:
                            with st.spinner(progress_text):
                                image_b64, cad_meta = file_processor.process_file(
                                    uploaded_file
                                )

                                if not image_b64:
                                    st.error(f"ファイルの処理に失敗しました: {file_name}")
                                    logger.error(
                                        f"File processing failed for {file_name}"
                                    )
                                    continue

                                analysis = analyze_image_with_gpt4o(
                                    image_b64, uploaded_file.name, cad_meta
                                )

                                kb_builder.build_from_file(
                                    uploaded_file,
                                    analysis=analysis,
                                    image_base64=image_b64,
                                    user_additions={},
                                    cad_metadata=cad_meta,
                                )

                                st.success(f"✓ ナレッジを追加しました: {file_name}")
                                logger.info(
                                    f"Successfully added knowledge for {file_name}"
                                )
                        except Exception as e:
                            st.error(f"処理中に予期せぬエラーが発生しました ({file_name}): {e}")
                            logger.error(
                                f"Unhandled error processing {file_name}: {e}",
                                exc_info=True,
                            )

                    kb_builder.refresh_search_engine = original_refresh

                    if process_mode == "まとめて処理" and index_mode == "自動(処理後)":
                        with st.spinner("検索エンジン更新中..."):
                            refresh_search_engine(DEFAULT_KB_NAME)

                    end_time = time.time()
                    total_time = end_time - start_time
                    progress_bar.progress(
                        1.0, f"全ての処理が完了しました！ (合計時間: {total_time:.2f}秒)"
                    )

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
