import base64
import io
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from config import DEFAULT_KB_NAME, EMBEDDING_DIMENSIONS
from core.mm_builder_utils import analyze_image_with_gpt4o
from PIL import Image
from shared.file_processor import FileProcessor
from shared.kb_builder import KnowledgeBuilder
from shared.logging_utils import configure_logging
from shared.openai_utils import get_openai_client
from shared.upload_utils import BASE_KNOWLEDGE_DIR as SHARED_KB_DIR
from ui_modules.theme import apply_intel_theme


def _refresh_search_engine(kb_name: str) -> None:
    """Reload the chatbot's search index for the given knowledge base.

    The heavy index refresh logic lives in ``knowledge_gpt_app``. This
    helper imports ``refresh_search_engine`` only when needed to avoid
    circular dependencies and then calls it with ``kb_name``.
    """
    try:
        from knowledge_gpt_app.app import refresh_search_engine as _refresh
    except Exception as e:  # pragma: no cover - import may fail in isolation
        logging.error(f"refresh_search_engine import failed: {e}")
        return
    _refresh(kb_name)


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ページ設定
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
if not st.session_state.get("_page_configured", False):
    st.set_page_config(
        page_title="マルチモーダルナレッジ構築ツール",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["_page_configured"] = True

# インテル風テーマ適用
apply_intel_theme(st)

# 設定
current_dir = Path(__file__).resolve().parent

# 親ディレクトリ(リポジトリルート)をパスに追加
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# ロギング設定
configure_logging()
logger = logging.getLogger(__name__)

# 定数
GPT4O_MODEL = "gpt-4.1"

# Use the lists defined in FileProcessor so supported types remain
# consistent across the repository.
SUPPORTED_IMAGE_TYPES = FileProcessor.SUPPORTED_IMAGE_TYPES
SUPPORTED_DOCUMENT_TYPES = FileProcessor.SUPPORTED_DOCUMENT_TYPES
SUPPORTED_CAD_TYPES = FileProcessor.SUPPORTED_CAD_TYPES

# 共通ナレッジベースディレクトリ
# centralize configuration using the shared utility
BASE_KNOWLEDGE_DIR = SHARED_KB_DIR
BASE_KNOWLEDGE_DIR.mkdir(exist_ok=True)

# データディレクトリ
DATA_DIR = BASE_KNOWLEDGE_DIR

# セッション状態初期化
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processed_images" not in st.session_state:
    st.session_state.processed_images = {}
if "current_editing_id" not in st.session_state:
    st.session_state.current_editing_id = None

# Shared processing utilities
_file_processor = FileProcessor()
_kb_builder = KnowledgeBuilder(
    _file_processor,
    get_openai_client_func=get_openai_client,
    refresh_search_engine_func=_refresh_search_engine,
)


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# メインUI
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

st.title("◊ マルチモーダルナレッジ構築ツール")
st.markdown("画像・図面からナレッジベースを構築するための改良版ツールです。")

# サイドバー設定
st.sidebar.header("⚡ 設定")
show_debug = st.sidebar.checkbox("デバッグ情報を表示", value=False)
embedding_dims = st.sidebar.selectbox(
    "埋め込み次元数",
    [1536, 3072],
    index=1 if EMBEDDING_DIMENSIONS == 3072 else 0,
    help="1536: コスト効率重視、3072: 精度重視",
)

# メインタブ
tab1, tab2, tab3 = st.tabs(["↑ 画像アップロード", "∠ 内容編集・ナレッジ化", "≡ ナレッジベース管理"])

with tab1:
    st.header("画像・図面のアップロード")

    # ファイル形式の説明
    with st.expander("⟐ 対応ファイル形式"):
        st.markdown(
            f"""
        **画像ファイル**: {', '.join(SUPPORTED_IMAGE_TYPES)}
        **文書ファイル**: {', '.join(SUPPORTED_DOCUMENT_TYPES)}
        **CADファイル**: {', '.join(SUPPORTED_CAD_TYPES)}
        """
        )

    # ファイルアップロード
    uploaded_files = st.file_uploader(
        "画像・CADファイルを選択してください",
        type=SUPPORTED_IMAGE_TYPES + SUPPORTED_DOCUMENT_TYPES + SUPPORTED_CAD_TYPES,
        accept_multiple_files=True,
        help="複数ファイルの同時アップロードが可能です。CADファイルは自動的に画像に変換されます。",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)}個のファイルがアップロードされました")

        if st.button("⌕ AI解析を開始", type="primary"):
            client = get_openai_client()
            if not client:
                st.error("OpenAIクライアントに接続できません")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    file_bytes = uploaded_file.getvalue()
                    uploaded_file.seek(0)
                    status_text.text(
                        f"処理中: {uploaded_file.name} ({i+1}/{len(uploaded_files)})"
                    )

                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    is_cad_file = file_extension in SUPPORTED_CAD_TYPES

                    processed = _file_processor.process_file(
                        uploaded_file,
                        builder=_kb_builder,
                    )
                    image_base64 = processed.get("image_base64")
                    cad_metadata = processed.get("metadata")
                    proc_type = processed.get("type")

                    if proc_type not in ("image", "cad"):
                        st.warning(f"{uploaded_file.name} はこの画面では処理できないファイル形式です")
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        continue

                    if image_base64 is None:
                        st.error(
                            f"ファイル処理エラー: {uploaded_file.name} - {cad_metadata.get('error', '不明なエラー')}"
                        )
                        continue

                    # GPT-4o解析
                    with st.spinner(f"GPT-4.1で解析中: {uploaded_file.name}"):
                        analysis = analyze_image_with_gpt4o(
                            image_base64, uploaded_file.name, cad_metadata, client
                        )

                    if "error" not in analysis:
                        image_id = str(uuid.uuid4())
                        st.session_state.processed_images[image_id] = {
                            "filename": uploaded_file.name,
                            "file_extension": file_extension,
                            "is_cad_file": is_cad_file,
                            "image_base64": image_base64,
                            "analysis": analysis,
                            "cad_metadata": cad_metadata,
                            "user_additions": {},
                            "is_finalized": False,
                            "original_bytes": file_bytes,
                        }

                        file_type_display = "CADファイル" if is_cad_file else "画像"
                        st.success(
                            f"◎ {uploaded_file.name} ({file_type_display}) の解析完了"
                        )
                    else:
                        st.error(
                            f"× {uploaded_file.name} の解析失敗: {analysis.get('error')}"
                        )

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.text("全ての処理が完了しました")

with tab2:
    st.header("内容編集・ナレッジ化")

    if not st.session_state.processed_images:
        st.info("「画像アップロード」タブで画像を処理してください")
    else:
        # 画像選択
        image_options = {
            f"{data['filename']} (ID: {img_id[:8]}...)": img_id
            for img_id, data in st.session_state.processed_images.items()
        }

        selected_display = st.selectbox(
            "編集する画像を選択", list(image_options.keys()), index=0
        )

        if selected_display:
            selected_id = image_options[selected_display]
            image_data = st.session_state.processed_images[selected_id]

            # ★★★ 3列レイアウト：画像、AIデータ、ユーザー編集 ★★★
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.subheader("◊ 画像プレビュー")

                try:
                    image_bytes = base64.b64decode(image_data["image_base64"])
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(
                        image, caption=image_data["filename"], use_container_width=True
                    )
                except Exception as e:
                    st.error(f"画像表示エラー: {e}")

                # ファイル情報
                if image_data.get("is_cad_file", False):
                    st.info(f"⚙ CADファイル: {image_data['file_extension'].upper()}")

                    if image_data.get("cad_metadata"):
                        cad_meta = image_data["cad_metadata"]
                        with st.expander("⚙ CAD技術情報"):
                            if cad_meta.get("file_type"):
                                st.write(f"**形式**: {cad_meta['file_type']}")
                            if cad_meta.get("total_entities"):
                                st.write(f"**エンティティ数**: {cad_meta['total_entities']}")
                            if cad_meta.get("vertices_count"):
                                st.write(f"**頂点数**: {cad_meta['vertices_count']}")
                            if cad_meta.get("volume"):
                                st.write(f"**体積**: {cad_meta['volume']:.3f}")

            with col2:
                st.subheader("∷ AI解析結果")
                analysis = image_data["analysis"]

                st.write(f"**画像タイプ**: {analysis.get('image_type', 'N/A')}")

                with st.expander("∠ AI抽出内容（参考）", expanded=False):
                    if analysis.get("main_content"):
                        st.write(f"**主要内容**: {analysis['main_content']}")

                    if analysis.get("detected_elements"):
                        st.write("**検出要素**:")
                        for element in analysis["detected_elements"][:5]:  # 上位5つのみ表示
                            st.write(f"- {element}")

                    if analysis.get("text_content"):
                        st.write(f"**画像内テキスト**: {analysis['text_content'][:200]}...")

                    if analysis.get("keywords"):
                        st.write(
                            f"**AI生成キーワード**: {', '.join(analysis['keywords'][:8])}..."
                        )

                    if analysis.get("technical_details"):
                        st.write(f"**技術詳細**: {analysis['technical_details']}")

                # ★★★ リアルタイムプレビュー ★★★
                st.subheader("⌕ ナレッジプレビュー")

                if st.button("⟐ 最新プレビューを生成"):
                    user_additions = image_data.get("user_additions", {})

                    # プレビューチャンク作成
                    preview_chunk = _kb_builder._create_comprehensive_search_chunk(
                        analysis, user_additions
                    )
                    preview_metadata = _kb_builder._create_structured_metadata(
                        analysis, user_additions, image_data["filename"]
                    )

                    st.markdown("**□ ベクトル化される検索チャンク:**")
                    with st.container():
                        st.text_area(
                            "",
                            preview_chunk,
                            height=150,
                            disabled=True,
                            key="preview_chunk",
                        )
                        st.caption(f"文字数: {len(preview_chunk)}")

                    st.markdown("**≡ 構造化メタデータ（検索結果表示用）:**")
                    with st.popover("メタデータ詳細"):
                        st.json(preview_metadata)

            with col3:
                st.subheader("∠ ユーザー追加情報")

                user_additions = image_data.get("user_additions", {})

                # タイトル
                title = st.text_input(
                    "∠ タイトル",
                    value=user_additions.get("title", ""),
                    help="検索結果に表示されるタイトル",
                )

                # 補足説明
                additional_description = st.text_area(
                    "⟐ 補足説明",
                    value=user_additions.get("additional_description", ""),
                    help="AIの解析に追加したい詳細な説明（重要：検索対象になります）",
                    height=100,
                )

                # 用途・目的
                purpose = st.text_input(
                    "◉ 用途・目的",
                    value=user_additions.get("purpose", ""),
                    help="この画像の用途や目的",
                )

                # 文脈・背景
                context = st.text_area(
                    "≈ 文脈・背景",
                    value=user_additions.get("context", ""),
                    help="この画像の背景情報や文脈",
                    height=80,
                )

                # 関連文書
                related_documents = st.text_input(
                    "⟐ 関連文書",
                    value=user_additions.get("related_documents", ""),
                    help="関連する文書やファイル名",
                )

                # 追加キーワード
                additional_keywords_str = st.text_input(
                    "⊞ 追加キーワード（カンマ区切り）",
                    value=", ".join(user_additions.get("additional_keywords", [])),
                    help="検索用の追加キーワード（重要：検索性能向上）",
                )
                additional_keywords = [
                    kw.strip()
                    for kw in additional_keywords_str.split(",")
                    if kw.strip()
                ]

                # カテゴリと重要度
                col3_1, col3_2 = st.columns(2)
                with col3_1:
                    category_options = [
                        "技術文書",
                        "組織図",
                        "フローチャート",
                        "データ図表",
                        "写真",
                        "地図",
                        "その他",
                    ]
                    selected_category = st.selectbox(
                        "≣ カテゴリ",
                        category_options,
                        index=(
                            category_options.index(
                                user_additions.get("category", "技術文書")
                            )
                            if user_additions.get("category") in category_options
                            else 0
                        ),
                    )

                with col3_2:
                    importance = st.select_slider(
                        "◇ 重要度",
                        options=["低", "中", "高", "最重要"],
                        value=user_additions.get("importance", "中"),
                    )

                # 情報更新
                if st.button("□ 情報を更新", type="secondary"):
                    st.session_state.processed_images[selected_id]["user_additions"] = {
                        "title": title,
                        "additional_description": additional_description,
                        "purpose": purpose,
                        "context": context,
                        "related_documents": related_documents,
                        "additional_keywords": additional_keywords,
                        "category": selected_category,
                        "importance": importance,
                    }
                    st.success("◎ 情報が更新されました")
                    st.rerun()

                st.markdown("---")

                # ★★★ ナレッジベース登録 ★★★
                st.subheader("⤴ ナレッジベース登録")

                if not image_data.get("is_finalized", False):
                    if st.button("◎ ナレッジベースに登録", type="primary"):
                        with st.spinner("ナレッジベース登録中..."):
                            current_user_additions = st.session_state.processed_images[
                                selected_id
                            ]["user_additions"]
                            buf = io.BytesIO(image_data.get("original_bytes", b""))
                            buf.name = image_data["filename"]
                            saved_item = _kb_builder.build_from_file(
                                buf,
                                analysis=analysis,
                                image_base64=image_data["image_base64"],
                                user_additions=current_user_additions,
                                cad_metadata=image_data.get("cad_metadata"),
                            )

                            if saved_item:
                                st.session_state.processed_images[selected_id][
                                    "is_finalized"
                                ] = True
                                st.success("◎ ナレッジベースに登録完了！")

                                with st.expander(
                                    "≡ 登録されたデータ（既存RAGシステム互換）",
                                    expanded=True,
                                ):
                                    st.write(f"**ID**: {saved_item['id']}")
                                    st.write(f"**ファイルリンク**: {saved_item['file_link']}")
                                    st.write(
                                        f"**ベクトル次元数**: {saved_item['stats']['vector_dimensions']}"
                                    )
                                    st.write(
                                        f"**キーワード数**: {saved_item['stats']['keywords_count']}"
                                    )
                                    st.write(
                                        f"**チャンク文字数**: {saved_item['stats']['chunk_length']}"
                                    )

                                    st.markdown("**⟐ 保存先ファイル:**")
                                    st.code(
                                        f"""
chunks/{saved_item['id']}.json      # 検索用テキストチャンク
embeddings/{saved_item['id']}.json  # ベクトルデータ
metadata/{saved_item['id']}.json    # メタ情報
images/{saved_item['id']}.jpg       # 画像ファイル
files/{saved_item['id']}_info.json  # ファイル情報
                                    """
                                    )
                            else:
                                st.error("× ナレッジベース登録中にエラーが発生しました")
                else:
                    st.success("◎ この画像は既にナレッジベース登録済みです")

                    if st.button("⟲ 再登録（情報更新）", help="情報を更新して再度登録します"):
                        st.session_state.processed_images[selected_id][
                            "is_finalized"
                        ] = False
                        st.rerun()

with tab3:
    st.header("ナレッジベース管理")

    # ナレッジベース選択
    kb_name = st.selectbox("ナレッジベース選択", [DEFAULT_KB_NAME], index=0)  # 将来複数KB対応可能

    # 既存RAGシステム互換のディレクトリ構造
    kb_dir = DATA_DIR / kb_name
    chunks_dir = kb_dir / "chunks"
    embeddings_dir = kb_dir / "embeddings"
    metadata_dir = kb_dir / "metadata"
    images_dir = kb_dir / "images"
    files_dir = kb_dir / "files"

    if metadata_dir.exists():
        metadata_files = list(metadata_dir.glob("*.json"))
        if metadata_files:
            # ナレッジベース統計表示
            kb_metadata_path = kb_dir / "kb_metadata.json"
            if kb_metadata_path.exists():
                try:
                    with open(kb_metadata_path, "r", encoding="utf-8") as f:
                        kb_info = json.load(f)

                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        st.metric(
                            "≡ 総アイテム数",
                            kb_info.get("total_items", len(metadata_files)),
                        )
                    with col_stat2:
                        st.metric(
                            "◊ 画像",
                            kb_info.get("item_types", {}).get(
                                "image", len(metadata_files)
                            ),
                        )
                    with col_stat3:
                        st.metric(
                            "⟐ テキスト",
                            kb_info.get("item_types", {}).get("text_chunk", 0),
                        )
                    with col_stat4:
                        st.metric(
                            "⟲ 最終更新",
                            (
                                kb_info.get("last_updated", "")[:10]
                                if kb_info.get("last_updated")
                                else "N/A"
                            ),
                        )
                except Exception:
                    st.info(f"≡ ナレッジベース登録データ: {len(metadata_files)}件")
            else:
                st.info(f"≡ ナレッジベース登録データ: {len(metadata_files)}件")

            # ディレクトリ構造表示
            with st.expander("⟐ ディレクトリ構造（既存RAGシステム互換）", expanded=False):
                st.code(
                    f"""
⟐ {kb_name}/
├── ⟐ chunks/      ({len(list(chunks_dir.glob('*.json')) if chunks_dir.exists() else [])}件)
├── ≈ embeddings/  ({len(list(embeddings_dir.glob('*.json')) if embeddings_dir.exists() else [])}件)
├── ≡ metadata/    ({len(list(metadata_dir.glob('*.json')) if metadata_dir.exists() else [])}件)
├── ◊ images/      ({len(list(images_dir.glob('*.*')) if images_dir.exists() else [])}件)
├── ⟐ files/       ({len(list(files_dir.glob('*.json')) if files_dir.exists() else [])}件)
└── ⟐ kb_metadata.json
                """
                )

            # データ一覧表示
            data_list = []
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    display_meta = data.get("display_metadata", {})
                    stats = data.get("stats", {})

                    data_list.append(
                        {
                            "ID": data["id"][:8] + "...",
                            "ファイル名": data.get("filename", "N/A"),
                            "タイトル": (
                                display_meta.get("title", "N/A")[:30] + "..."
                                if len(display_meta.get("title", "")) > 30
                                else display_meta.get("title", "N/A")
                            ),
                            "画像タイプ": display_meta.get("image_type", "N/A"),
                            "カテゴリ": display_meta.get("category", "N/A"),
                            "重要度": display_meta.get("importance", "N/A"),
                            "キーワード数": stats.get("keywords_count", 0),
                            "チャンク文字数": stats.get("chunk_length", 0),
                            "ベクトル次元": stats.get("vector_dimensions", 0),
                            "ファイルリンク": data.get("file_link", "N/A"),
                            "作成日時": data.get("created_at", "")[:19],
                        }
                    )
                except Exception as e:
                    logger.error(f"メタデータ読み込みエラー {metadata_file}: {e}")

            if data_list:
                df = pd.DataFrame(data_list)
                st.dataframe(df, use_container_width=True)

                # エクスポート機能
                if st.button("↓ ナレッジベースをCSVエクスポート"):
                    csv = df.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="□ CSVダウンロード",
                        data=csv,
                        file_name=f"{kb_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                # ★★★ 改良版ナレッジ検索（分離構造対応） ★★★
                st.subheader("⌕ ナレッジ検索")

                col_search1, col_search2 = st.columns([3, 1])
                with col_search1:
                    search_query = st.text_input(
                        "検索クエリを入力",
                        placeholder="例: WEB版 比較項目、技術仕様、組織図",
                    )

                with col_search2:
                    search_top_k = st.selectbox("表示件数", [5, 10, 15, 20], index=1)

                if search_query and st.button("⌕ 検索実行", type="primary"):
                    client = get_openai_client()
                    if client:
                        with st.spinner("検索中..."):
                            # クエリのベクトル化
                            query_embedding = _kb_builder._get_embedding(
                                search_query, client, dimensions=embedding_dims
                            )

                            if query_embedding is not None:
                                # 類似度計算（分離構造対応）
                                similarities = []

                                for metadata_file in metadata_files:
                                    try:
                                        # メタデータ読み込み
                                        with open(
                                            metadata_file, "r", encoding="utf-8"
                                        ) as f:
                                            metadata = json.load(f)

                                        item_id = metadata["id"]

                                        # 対応するベクトルファイル読み込み
                                        embedding_file = (
                                            embeddings_dir / f"{item_id}.json"
                                        )
                                        if embedding_file.exists():
                                            with open(
                                                embedding_file, "r", encoding="utf-8"
                                            ) as f:
                                                embedding_data = json.load(f)

                                            doc_embedding = embedding_data.get("vector")
                                            if doc_embedding is not None and len(
                                                doc_embedding
                                            ) == len(query_embedding):
                                                # コサイン類似度計算
                                                similarity = np.dot(
                                                    query_embedding, doc_embedding
                                                ) / (
                                                    np.linalg.norm(query_embedding)
                                                    * np.linalg.norm(doc_embedding)
                                                )

                                                # チャンクデータも読み込み（検索結果表示用）
                                                chunk_file = (
                                                    chunks_dir / f"{item_id}.json"
                                                )
                                                chunk_data = {}
                                                if chunk_file.exists():
                                                    with open(
                                                        chunk_file,
                                                        "r",
                                                        encoding="utf-8",
                                                    ) as f:
                                                        chunk_data = json.load(f)

                                                similarities.append(
                                                    {
                                                        "metadata": metadata,
                                                        "chunk_data": chunk_data,
                                                        "embedding_data": embedding_data,
                                                        "similarity": similarity,
                                                    }
                                                )
                                    except Exception as e:
                                        logger.error(f"検索エラー {metadata_file}: {e}")

                                # 類似度順でソート
                                similarities.sort(
                                    key=lambda x: x["similarity"], reverse=True
                                )

                                # 検索結果表示
                                st.write(
                                    f"**◉ 検索結果（上位{min(search_top_k, len(similarities))}件）**"
                                )

                                if similarities:
                                    for i, result in enumerate(
                                        similarities[:search_top_k]
                                    ):
                                        metadata = result["metadata"]
                                        display_meta = metadata.get(
                                            "display_metadata", {}
                                        )
                                        stats = metadata.get("stats", {})
                                        chunk_data = result.get("chunk_data", {})

                                        with st.expander(
                                            f"{i+1}. {display_meta.get('title', metadata.get('filename', 'N/A'))} (類似度: {result['similarity']:.3f})"
                                        ):
                                            # 4列レイアウト
                                            (
                                                result_col1,
                                                result_col2,
                                                result_col3,
                                                result_col4,
                                            ) = st.columns([1, 1, 1, 1])

                                            with result_col1:
                                                st.markdown("**⟐ 基本情報**")
                                                st.write(
                                                    f"**ファイル名**: {metadata.get('filename', 'N/A')}"
                                                )
                                                st.write(
                                                    f"**タイプ**: {display_meta.get('image_type', 'N/A')}"
                                                )
                                                st.write(
                                                    f"**カテゴリ**: {display_meta.get('category', 'N/A')}"
                                                )
                                                st.write(
                                                    f"**重要度**: {display_meta.get('importance', 'N/A')}"
                                                )

                                                # ★ ファイルリンク表示
                                                file_link = metadata.get(
                                                    "file_link", ""
                                                )
                                                if file_link:
                                                    st.write(
                                                        f"**⟐ ファイルリンク**: `{file_link}`"
                                                    )

                                            with result_col2:
                                                st.markdown("**≡ 統計・スコア**")
                                                st.metric(
                                                    "類似度",
                                                    f"{result['similarity']:.3f}",
                                                )
                                                st.write(
                                                    f"**キーワード数**: {stats.get('keywords_count', 0)}"
                                                )
                                                st.write(
                                                    f"**チャンク文字数**: {stats.get('chunk_length', 0)}"
                                                )
                                                st.write(
                                                    f"**ベクトル次元**: {stats.get('vector_dimensions', 0)}"
                                                )
                                                st.write(
                                                    f"**作成日**: {metadata.get('created_at', '')[:10]}"
                                                )

                                            with result_col3:
                                                st.markdown("**◊ 画像プレビュー**")
                                                try:
                                                    item_id = metadata["id"]
                                                    image_file = (
                                                        images_dir / f"{item_id}.jpg"
                                                    )
                                                    if image_file.exists():
                                                        image = Image.open(image_file)
                                                        st.image(image, width=150)
                                                    else:
                                                        st.write("画像ファイルなし")
                                                except Exception:
                                                    st.write("画像プレビュー不可")

                                            with result_col4:
                                                st.markdown("**⟐ ファイル構造**")
                                                item_id = metadata["id"]
                                                st.write(
                                                    f"**チャンク**: `chunks/{item_id}.json`"
                                                )
                                                st.write(
                                                    f"**ベクトル**: `embeddings/{item_id}.json`"
                                                )
                                                st.write(
                                                    f"**メタデータ**: `metadata/{item_id}.json`"
                                                )
                                                st.write(
                                                    f"**画像**: `images/{item_id}.jpg`"
                                                )

                                            # 詳細情報
                                            if display_meta.get("main_content"):
                                                st.markdown("**∠ 内容**")
                                                content = display_meta["main_content"]
                                                st.write(
                                                    content[:200] + "..."
                                                    if len(content) > 200
                                                    else content
                                                )

                                            if display_meta.get("keywords"):
                                                st.markdown("**⊞ キーワード**")
                                                keywords = display_meta["keywords"]
                                                keywords_display = ", ".join(
                                                    keywords[:10]
                                                )
                                                if len(keywords) > 10:
                                                    keywords_display += (
                                                        f" （他{len(keywords)-10}個）"
                                                    )
                                                st.write(keywords_display)

                                            if display_meta.get("purpose"):
                                                st.markdown("**◉ 用途**")
                                                st.write(display_meta["purpose"])

                                            # デバッグ情報
                                            if show_debug:
                                                with st.popover("⚙ デバッグ: チャンクとファイルパス"):
                                                    st.markdown("**検索チャンク:**")
                                                    st.text_area(
                                                        "",
                                                        chunk_data.get("content", ""),
                                                        height=100,
                                                        disabled=True,
                                                        key=f"debug_chunk_{i}",
                                                    )
                                                    st.markdown("**ファイルパス:**")
                                                    st.write(
                                                        f"chunks: {chunks_dir / f'{item_id}.json'}"
                                                    )
                                                    st.write(
                                                        f"embeddings: {embeddings_dir / f'{item_id}.json'}"
                                                    )
                                                    st.write(
                                                        f"metadata: {metadata_dir / f'{item_id}.json'}"
                                                    )
                                else:
                                    st.info("検索結果が見つかりませんでした。検索語を変更してみてください。")
                            else:
                                st.error("× 検索クエリのベクトル化に失敗しました")
                    else:
                        st.error("× OpenAIクライアントに接続できません")
        else:
            st.info("ナレッジベースにデータがありません")
    else:
        st.info(f"ナレッジベース '{kb_name}' が見つかりません")

    # 現在のセッション統計
    if st.session_state.processed_images:
        st.subheader("≈ 現在のセッション統計")

        total_images = len(st.session_state.processed_images)
        finalized_images = sum(
            1
            for data in st.session_state.processed_images.values()
            if data.get("is_finalized", False)
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("処理済み画像", total_images)
        with col2:
            st.metric("登録済み", finalized_images)
        with col3:
            st.metric("未登録", total_images - finalized_images)
        with col4:
            st.metric("ベクトル次元", embedding_dims)

# フッター
st.markdown("---")
st.markdown(
    '<div class="footer-text">マルチモーダルナレッジ構築ツール v3.0.0 - 既存RAGシステム統合対応版</div>',
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    os.chdir(current_dir)
