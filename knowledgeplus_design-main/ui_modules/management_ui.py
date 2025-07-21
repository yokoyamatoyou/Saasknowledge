import base64
import io
import logging
import time

import streamlit as st
from config import DEFAULT_KB_NAME
from core.faq_utils import generate_faq
from core.mm_builder_utils import analyze_image_with_gpt4o
from knowledge_gpt_app.app import get_search_engine, refresh_search_engine
from shared.file_processor import FileProcessor
from shared.kb_builder import KnowledgeBuilder
from shared.openai_utils import get_openai_client
from shared.thesaurus import load_synonyms, update_synonyms
from shared.zero_hit_logger import load_zero_hit_queries
from ui_modules.thumbnail_editor import display_thumbnail_grid

logger = logging.getLogger(__name__)


def render_management_mode():
    """Render the management interface including FAQ generation."""
    st.subheader("管理")
    tabs = st.tabs(["ナレッジベース構築", "FAQ自動生成", "同義語管理", "ゼロヒットクエリ"])

    with tabs[0]:
        st.divider()
        with st.expander("ナレッジを追加する", expanded=True):
            process_mode = st.radio(
                "処理モード",
                ["個別処理", "まとめて処理"],
                help="ファイルを個別に処理するか、まとめて処理するかを選択します。",
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

                if "pending_uploads" not in st.session_state:
                    st.session_state.pending_uploads = []

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
                                processed_data = file_processor.process_file(
                                    uploaded_file,
                                    builder=kb_builder,
                                )
                                proc_type = processed_data.get("type")
                                image_b64 = processed_data.get("image_base64")
                                cad_meta = processed_data.get("metadata")

                                if proc_type in ("image", "cad"):
                                    if not image_b64:
                                        st.error(f"ファイルの処理に失敗しました: {file_name}")
                                        logger.error(
                                            f"File processing failed for {file_name}"
                                        )
                                        continue

                                    analysis = analyze_image_with_gpt4o(
                                        image_b64, uploaded_file.name, cad_meta
                                    )

                                    if process_mode == "まとめて処理":
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
                                    else:
                                        st.session_state.pending_uploads.append(
                                            {
                                                "filename": file_name,
                                                "bytes": uploaded_file.getvalue(),
                                                "image_base64": image_b64,
                                                "analysis": analysis,
                                                "cad_metadata": cad_meta,
                                            }
                                        )
                                        st.success(
                                            f"✓ {file_name} の解析が完了しました。下部で詳細を確認してください"
                                        )

                                elif proc_type == "document":
                                    st.success(f"✓ ドキュメントを追加しました: {file_name}")
                                    logger.info(
                                        f"Successfully added document for {file_name}"
                                    )
                                else:
                                    st.error(f"ファイルの処理に失敗しました: {file_name}")
                                    logger.error(
                                        f"Unsupported file type for {file_name}"
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
                        1.0,
                        f"全ての処理が完了しました！ (合計時間: {total_time:.2f}秒)",
                    )

                    # 個別処理の場合は解析結果を編集・登録するUIを表示
                    if process_mode == "個別処理" and st.session_state.pending_uploads:
                        st.markdown("---")
                        st.subheader("解析結果の確認とメタデータ入力")
                        for idx, item in enumerate(
                            list(st.session_state.pending_uploads)
                        ):
                            try:
                                image_bytes = base64.b64decode(item["image_base64"])
                                st.image(
                                    image_bytes,
                                    caption=item["filename"],
                                    use_container_width=True,
                                )
                            except Exception:
                                st.image(
                                    item["image_base64"],
                                    caption=item["filename"],
                                    use_container_width=True,
                                )
                            with st.popover("AI解析結果"):
                                st.json(item["analysis"])

                            title = st.text_input("タイトル", key=f"title_{idx}")
                            add_desc = st.text_area("補足説明", key=f"desc_{idx}")
                            purpose = st.text_input("用途・目的", key=f"purpose_{idx}")
                            context = st.text_area("文脈・背景", key=f"context_{idx}")
                            related_documents = st.text_input("関連文書", key=f"docs_{idx}")
                            keywords_str = st.text_input(
                                "追加キーワード (カンマ区切り)", key=f"kw_{idx}"
                            )
                            category = st.selectbox(
                                "カテゴリ",
                                [
                                    "技術文書",
                                    "組織図",
                                    "フローチャート",
                                    "データ図表",
                                    "写真",
                                    "地図",
                                    "その他",
                                ],
                                key=f"cat_{idx}",
                            )
                            importance = st.select_slider(
                                "重要度",
                                options=["低", "中", "高", "最重要"],
                                key=f"imp_{idx}",
                            )

                            user_additions = {
                                "title": title,
                                "additional_description": add_desc,
                                "purpose": purpose,
                                "context": context,
                                "related_documents": related_documents,
                                "additional_keywords": [
                                    k.strip()
                                    for k in keywords_str.split(",")
                                    if k.strip()
                                ],
                                "category": category,
                                "importance": importance,
                            }

                            if st.button("プレビュー生成", key=f"preview_{idx}"):
                                preview_chunk = (
                                    kb_builder._create_comprehensive_search_chunk(
                                        item["analysis"], user_additions
                                    )
                                )
                                preview_meta = kb_builder._create_structured_metadata(
                                    item["analysis"], user_additions, item["filename"]
                                )
                                st.text_area(
                                    "検索チャンク",
                                    preview_chunk,
                                    height=120,
                                    disabled=True,
                                )
                                st.json(preview_meta)

                            conflict_key = f"conflict_state_{idx}"
                            if conflict_key in st.session_state:
                                conf = st.session_state[conflict_key]["conflicts"]
                                st.warning("ルールの矛盾が検出されました。内容を確認してください。")
                                st.json(conf)
                                col1, col2 = st.columns(2)
                                if col1.button("登録を続ける", key=f"confirm_{idx}"):
                                    buf = io.BytesIO(item["bytes"])
                                    buf.name = item["filename"]
                                    kb_builder.build_from_file(
                                        buf,
                                        analysis=item["analysis"],
                                        image_base64=item["image_base64"],
                                        user_additions=st.session_state[conflict_key][
                                            "user_additions"
                                        ],
                                        cad_metadata=item.get("cad_metadata"),
                                    )
                                    st.success(f"✓ ナレッジを追加しました: {item['filename']}")
                                    st.session_state.pending_uploads.remove(item)
                                    del st.session_state[conflict_key]
                                    st.rerun()
                                elif col2.button("キャンセル", key=f"cancel_{idx}"):
                                    del st.session_state[conflict_key]
                                    st.info("登録をキャンセルしました")
                            elif st.button("ナレッジベースに登録", key=f"register_{idx}"):
                                preview_chunk = (
                                    kb_builder._create_comprehensive_search_chunk(
                                        item["analysis"], user_additions
                                    )
                                )
                                preview_meta = kb_builder._create_structured_metadata(
                                    item["analysis"], user_additions, item["filename"]
                                )
                                engine = get_search_engine(DEFAULT_KB_NAME)
                                conflicts: list[dict] = []
                                if engine is not None:
                                    rtypes = preview_meta.get("rule_info", {}).get(
                                        "rule_types", []
                                    )
                                    if rtypes:
                                        existing = [
                                            ch
                                            for ch in engine.chunks
                                            if any(
                                                rt
                                                in ch.get("metadata", {})
                                                .get("rule_info", {})
                                                .get("rule_types", [])
                                                for rt in rtypes
                                            )
                                        ]
                                        conflicts = engine.detect_rule_conflicts(
                                            existing
                                            + [
                                                {
                                                    "id": "new",
                                                    "text": preview_chunk,
                                                    "metadata": preview_meta,
                                                }
                                            ]
                                        )
                                if conflicts:
                                    st.session_state[conflict_key] = {
                                        "conflicts": conflicts,
                                        "user_additions": user_additions,
                                    }
                                    st.warning("ルールの矛盾が検出されました。再度ボタンを押して確定してください。")
                                else:
                                    buf = io.BytesIO(item["bytes"])
                                    buf.name = item["filename"]
                                    kb_builder.build_from_file(
                                        buf,
                                        analysis=item["analysis"],
                                        image_base64=item["image_base64"],
                                        user_additions=user_additions,
                                        cad_metadata=item.get("cad_metadata"),
                                    )
                                    st.success(f"✓ ナレッジを追加しました: {item['filename']}")
                                    st.session_state.pending_uploads.remove(item)
                                    st.rerun()

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
        faq_source = st.text_area(
            "FAQ生成用テキストまたはURL",
            key="faq_source",
            height=100,
            help="直接入力したテキストまたは取得したいWebページのURLを指定します。",
        )
        max_tokens = st.number_input(
            "Max tokens per chunk",
            100,
            2000,
            1000,
            100,
            help="チャンクあたりの最大トークン数を設定します。",
        )
        pairs = st.number_input(
            "Pairs per chunk",
            1,
            10,
            3,
            1,
            help="各チャンクから生成するQ&Aペアの数を設定します。",
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
                        kb_name,
                        int(max_tokens),
                        int(pairs),
                        client=client,
                        source=faq_source,
                    )

                st.success(f"{count}件のFAQを生成しました。")

    with tabs[2]:
        st.divider()
        st.subheader("同義語辞書の編集")
        current = load_synonyms()
        st.json(current)
        term = st.text_input("キーワード", key="syn_term")
        words = st.text_input("同義語 (カンマ区切り)", key="syn_words")
        if st.button("保存", key="syn_save"):
            new_words = [w.strip() for w in words.split(",") if w.strip()]
            updated = update_synonyms(term.strip(), new_words)
            engine = get_search_engine(DEFAULT_KB_NAME)
            if engine is not None:
                engine.synonyms = updated
            st.toast("同義語を更新しました")

    with tabs[3]:
        st.divider()
        st.subheader("未ヒット検索クエリ")
        queries = load_zero_hit_queries()
        if queries:
            unique = sorted(set(queries))
            st.write("記録されたクエリ:")
            for q in unique:
                st.markdown(f"- {q}")
            selected = st.selectbox("クエリを選択", unique)
            words = st.text_input("同義語 (カンマ区切り)", key="zero_syn_words")
            if st.button("同義語追加", key="zero_syn_save"):
                new_words = [w.strip() for w in words.split(",") if w.strip()]
                update_synonyms(selected.strip(), new_words)
                engine = get_search_engine(DEFAULT_KB_NAME)
                if engine is not None:
                    engine.synonyms = load_synonyms()
                st.toast("同義語を更新しました")
        else:
            st.info("記録されたゼロヒットクエリはありません。")
