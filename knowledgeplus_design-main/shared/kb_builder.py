import base64
import io
import logging
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
from openai import OpenAI
import json

from config import (
    DEFAULT_KB_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_BATCH_SIZE,
)
from shared.openai_utils import get_embeddings_batch
from shared.file_processor import FileProcessor
from shared import upload_utils
from shared.upload_utils import save_processed_data

logger = logging.getLogger(__name__)

class KnowledgeBuilder:
    def __init__(self, file_processor: FileProcessor, get_openai_client_func, refresh_search_engine_func):
        self.file_processor = file_processor
        self.get_openai_client = get_openai_client_func
        self.refresh_search_engine = refresh_search_engine_func

    def build_from_file(self, uploaded_file, analysis, image_base64, user_additions, cad_metadata):
        client = self.get_openai_client()
        if not client:
            st.error("OpenAIクライアントに接続できません")
            return None

        search_chunk = self._create_comprehensive_search_chunk(analysis, user_additions)
        embedding = self._get_embedding(search_chunk, client)

        if embedding is None:
            st.error("埋め込みベクトルの生成に失敗しました。")
            return None

        image_id = str(uuid.uuid4())
        filename = uploaded_file.name
        original_bytes = uploaded_file.getvalue()

        success, saved_item = self._save_unified_knowledge_item(
            image_id,
            analysis,
            user_additions,
            embedding,
            filename,
            image_base64=image_base64,
            original_bytes=original_bytes,
            refresh=True,
        )

        if success:
            return saved_item
        else:
            return None

    def _create_comprehensive_search_chunk(self, analysis_result, user_additions):
        """★ ベクトル化用の包括的検索チャンクを作成"""
        chunk_parts = []
        
        # 基本情報
        if analysis_result.get('image_type'):
            chunk_parts.append(f"画像タイプ: {analysis_result['image_type']}")
        
        if analysis_result.get('main_content'):
            chunk_parts.append(f"主要内容: {analysis_result['main_content']}")
        
        # 検出要素
        elements = analysis_result.get('detected_elements', [])
        if elements:
            chunk_parts.append(f"主要要素: {', '.join(elements)}")
        
        # 技術詳細
        tech_details = analysis_result.get('technical_details', '')
        if tech_details:
            chunk_parts.append(f"技術詳細: {tech_details}")
        
        # 技術仕様（CAD用）
        tech_specs = analysis_result.get('technical_specifications', '')
        if tech_specs:
            chunk_parts.append(f"技術仕様: {tech_specs}")
        
        # 寸法情報
        dimensions_info = analysis_result.get('dimensions_info', '')
        if dimensions_info:
            chunk_parts.append(f"寸法情報: {dimensions_info}")
        
        # GPTが読み取ったテキスト内容（重要：検索対象）
        text_content = analysis_result.get('text_content', '')
        if text_content and text_content.strip():
            chunk_parts.append(f"画像内テキスト: {text_content}")
        
        # 注記・記号
        annotations = analysis_result.get('annotations', '')
        if annotations:
            chunk_parts.append(f"注記・記号: {annotations}")
        
        # ユーザー追加情報
        if user_additions.get('additional_description'):
            chunk_parts.append(f"補足説明: {user_additions['additional_description']}")
        
        if user_additions.get('purpose'):
            chunk_parts.append(f"用途・目的: {user_additions['purpose']}")
        
        if user_additions.get('context'):
            chunk_parts.append(f"文脈・背景: {user_additions['context']}")
        
        if user_additions.get('related_documents'):
            chunk_parts.append(f"関連文書: {user_additions['related_documents']}")
        
        # キーワード統合
        keywords = analysis_result.get('keywords', [])
        user_keywords = user_additions.get('additional_keywords', [])
        search_terms = analysis_result.get('search_terms', [])
        all_keywords = keywords + user_keywords + search_terms
        if all_keywords:
            chunk_parts.append(f"キーワード: {', '.join(set(all_keywords))}")  # 重複除去
        
        # 関連トピック
        related_topics = analysis_result.get('related_topics', [])
        if related_topics:
            chunk_parts.append(f"関連トピック: {', '.join(related_topics)}")

        return "\n".join(chunk_parts)

    def _get_embeddings_batch(
        self,
        texts: list[str],
        client,
        dimensions: int = EMBEDDING_DIMENSIONS,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> list[list[float] | None]:
        """Return embeddings for the provided texts in batches."""
        if client is None:
            return []
        results: list[list[float] | None] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = get_embeddings_batch(batch, client, EMBEDDING_MODEL, dimensions)
            if vectors:
                results.extend(vectors)
            else:
                results.extend([None] * len(batch))
        return results

    def _get_embedding(self, text, client, dimensions=EMBEDDING_DIMENSIONS):
        """テキストの埋め込みベクトルを生成"""
        if client is None:
            return None

        try:
            if not text or not text.strip():
                return None

            if len(text) > 30000:
                text = text[:30000]

            result = self._get_embeddings_batch([text], client, dimensions)[0]
            return result
        except Exception as e:
            logger.error(f"埋め込みベクトル生成エラー: {e}")
            return None

    def _save_unified_knowledge_item(
        self,
        image_id,
        analysis_result,
        user_additions,
        embedding,
        filename,
        image_base64=None,
        original_bytes=None,
        refresh: bool = True,
    ):
        """★ 統一ナレッジアイテムとして保存（RAGシステム互換構造）"""
        try:
            search_chunk = self._create_comprehensive_search_chunk(analysis_result, user_additions)
            structured_metadata = self._create_structured_metadata(analysis_result, user_additions, filename)
            kb_name = DEFAULT_KB_NAME
            kb_path = upload_utils.BASE_KNOWLEDGE_DIR / kb_name
            if not kb_path.exists():
                subdirs = [p.name for p in upload_utils.BASE_KNOWLEDGE_DIR.iterdir() if p.is_dir()]
                if len(subdirs) == 1:
                    kb_name = subdirs[0]
                    kb_path = upload_utils.BASE_KNOWLEDGE_DIR / kb_name
            if image_base64:
                try:
                    image_bytes = base64.b64decode(image_base64)
                except Exception:
                    image_bytes = None
            else:
                image_bytes = None

            full_metadata = {
                "filename": filename,
                "display_metadata": structured_metadata,
                "analysis_data": {
                    "gpt_analysis": analysis_result,
                    "cad_metadata": analysis_result.get('cad_metadata', {}),
                    "user_additions": user_additions,
                },
            }

            paths = save_processed_data(
                kb_name,
                image_id,
                chunk_text=search_chunk,
                embedding=embedding,
                metadata=full_metadata,
                original_filename=filename,
                original_bytes=original_bytes,
                image_bytes=image_bytes,
            )
            # Write a companion info file with basic details
            info_path = (upload_utils.BASE_KNOWLEDGE_DIR / kb_name / "files" / f"{image_id}_info.json")
            info = {
                "file_path": paths.get("original_file_path"),
                "created_at": datetime.now().isoformat(),
            }
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            if refresh:
                self.refresh_search_engine(kb_name)
            file_link = paths.get("original_file_path", "")

            return True, {
                "id": image_id,
                "type": "image",
                "filename": filename,
                "chunk_path": paths.get("chunk_path"),
                "embedding_path": paths.get("embedding_path"),
                "metadata_path": paths.get("metadata_path"),
                "image_path": paths.get("image_path"),
                "file_link": file_link,
                "stats": {
                    "vector_dimensions": len(embedding) if embedding is not None else 0,
                    "keywords_count": len(structured_metadata.get("keywords", [])),
                    "chunk_length": len(search_chunk),
                }
            }
            
        except Exception as e:
            logger.error(f"ナレッジデータ保存エラー: {e}")
            return False, None

    def _create_structured_metadata(self, analysis_result, user_additions, filename):
        """★ 構造化メタデータを作成（検索結果表示用）"""
        return {
            # 基本情報
            "filename": filename,
            "image_type": analysis_result.get('image_type', ''),
            "category": user_additions.get('category', ''),
            "importance": user_additions.get('importance', '中'),
            
            # 表示用説明
            "title": user_additions.get('title', analysis_result.get('main_content', '')[:50] + '...'),
            "description_for_search": analysis_result.get('description_for_search', ''),
            "main_content": analysis_result.get('main_content', ''),
            
            # ユーザー情報
            "purpose": user_additions.get('purpose', ''),
            "context": user_additions.get('context', ''),
            "related_documents": user_additions.get('related_documents', ''),
            
            # 分類・タグ
            "keywords": analysis_result.get('keywords', []) + user_additions.get('additional_keywords', []),
            "search_terms": analysis_result.get('search_terms', []),
            "category_tags": analysis_result.get('category_tags', []),
            "related_topics": analysis_result.get('related_topics', []),
            
            # 技術情報
            "technical_details": analysis_result.get('technical_details', ''),
            "technical_specifications": analysis_result.get('technical_specifications', ''),
            "dimensions_info": analysis_result.get('dimensions_info', ''),
            "drawing_standards": analysis_result.get('drawing_standards', ''),
            "manufacturing_info": analysis_result.get('manufacturing_info', ''),
            
            # テキスト内容
            "text_content": analysis_result.get('text_content', ''),
            "annotations": analysis_result.get('annotations', ''),
            
            # 要素
            "detected_elements": analysis_result.get('detected_elements', []),
            
            # CADメタデータ
            "cad_metadata": analysis_result.get('cad_metadata', {})
        }