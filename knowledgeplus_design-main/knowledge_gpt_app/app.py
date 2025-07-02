import os
import sys
import streamlit as st
from openai import OpenAI
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, DEFAULT_KB_NAME
import PyPDF2
import docx
import pandas as pd
import json
import re
import numpy as np
from io import BytesIO
from sudachipy import tokenizer, dictionary
import base64
import tempfile
import shutil
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import pickle
from pathlib import Path
import logging
import traceback
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Optional libraries for PDF/Docx OCR handling
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

import pytesseract
OCR_SUPPORT = True

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import openpyxl
    EXCEL_SUPPORT = True
except Exception:
    openpyxl = None
    EXCEL_SUPPORT = False
import time
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from shared.upload_utils import (
    save_processed_data,
    BASE_KNOWLEDGE_DIR as SHARED_KB_DIR,
    ensure_openai_key,
)
from generate_faq import generate_faqs_from_chunks
from ui_modules.theme import apply_intel_theme
from shared.logging_utils import configure_logging

# ロギング設定を最初に行う
configure_logging()
logger = logging.getLogger(__name__)

# スクリプトディレクトリの解決
current_dir = Path(__file__).resolve().parent

# 親ディレクトリ(リポジトリルート)をパスに追加
repo_root = current_dir.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# 強化版NLTKリソースダウンロード関数
def ensure_nltk_resources():
    """必要なNLTKリソースが確実にダウンロードされるようにする"""
    try:
        logger.info("NLTKリソースの確認とダウンロードを開始...")
        resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                if resource in ['punkt', 'stopwords']:
                     nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                elif resource == 'averaged_perceptron_tagger':
                    nltk.data.find(f'taggers/{resource}')
                elif resource == 'wordnet' or resource == 'omw-1.4':
                     nltk.data.find(f'corpora/{resource}')
                else:
                     nltk.data.find(resource)
                logger.info(f"リソース '{resource}' は既にダウンロード済みです")
            except LookupError:
                logger.info(f"リソース '{resource}' をダウンロード中...")
                nltk.download(resource, quiet=True)
                logger.info(f"リソース '{resource}' のダウンロードが完了しました")
        return True
    except Exception as e:
        logger.error(f"NLTKリソースのダウンロード中にエラーが発生しました: {e}")
        return False

ensure_nltk_resources()

# 自作モジュールのインポート
try:
    from shared.search_engine import (
        search_knowledge_base,
        HybridSearchEngine,
        ensure_nltk_resources as ensure_nltk_resources_kb,
    )
    from shared.chat_controller import (
        ChatController,
        get_persona_list,
        load_persona,
        generate_conversation_title,
    )
    from shared.vector_store import initialize_vector_store
    
    ensure_nltk_resources_kb()
    logger.info("自作モジュールのインポートに成功しました")
except Exception as e:
    logger.error(f"自作モジュールのインポートに失敗しました: {e}")
    traceback.print_exc()

# OpenAIクライアントを取得する関数 (app.py内で共通して使用)
@st.cache_resource
def get_openai_client():
    """OpenAIクライアントを取得する関数"""
    try:
        api_key = ensure_openai_key()
        client = OpenAI(api_key=api_key)
        logger.info("OpenAIクライアントの取得に成功しました。")
        return client
    except Exception as e:
        logger.error(f"OpenAI クライアント初期化エラー: {e}")
        st.error(f"OpenAI クライアントの初期化中にエラーが発生しました: {e}")
        return None

# 固定モデル名
GPT4_MODEL = "gpt-4.1-2025-04-14"
GPT4_MINI_MODEL = "gpt-4.1-mini-2025-04-14"

# 共通ナレッジベースディレクトリ
BASE_KNOWLEDGE_DIR = SHARED_KB_DIR
BASE_KNOWLEDGE_DIR.mkdir(exist_ok=True)
logger.info(f"Knowledge base directory: {BASE_KNOWLEDGE_DIR}")

# 互換のための変数名
RAG_BASE_DIR = BASE_KNOWLEDGE_DIR

# ワークスペースディレクトリと会話ディレクトリ
DATA_DIR = current_dir / "data"
DATA_DIR.mkdir(exist_ok=True)
CONVERSATION_DIR = DATA_DIR / "conversations"
CONVERSATION_DIR.mkdir(exist_ok=True)
logger.info(f"会話データディレクトリ: {CONVERSATION_DIR}")


# === 会話管理関数 ===
def ensure_conversation_directories():
    DATA_DIR.mkdir(exist_ok=True)
    CONVERSATION_DIR.mkdir(exist_ok=True)

def save_conversation(conversation_id, messages, title="新しい会話"):
    ensure_conversation_directories()
    file_path = CONVERSATION_DIR / f"{conversation_id}.json"
    conversation_data = {
        "id": conversation_id,
        "title": title,
        "messages": messages,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=4)
        logger.info(f"会話を保存しました: {conversation_id} (タイトル: {title})")
    except Exception as e:
        logger.error(f"会話の保存に失敗しました ({conversation_id}): {e}")

def load_conversation(conversation_id):
    ensure_conversation_directories()
    file_path = CONVERSATION_DIR / f"{conversation_id}.json"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            messages = data.get("messages", [])
            title = data.get("title", "会話") 
            logger.info(f"会話を読み込みました: {conversation_id} (タイトル: {title})")
            return messages, title
    except FileNotFoundError:
        logger.warning(f"会話ファイルが見つかりません: {conversation_id}")
        return None, None
    except json.JSONDecodeError:
        logger.error(f"会話ファイルのJSONデコードエラー: {conversation_id}")
        return None, None
    except Exception as e:
        logger.error(f"会話ファイルの読み込みエラー ({conversation_id}): {e}")
        return None, None


def list_conversations():
    ensure_conversation_directories()
    conversations = []
    if not CONVERSATION_DIR.exists():
        return []
        
    for filename_path in CONVERSATION_DIR.glob("*.json"):
        try:
            with open(filename_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                conversations.append({
                    "id": data.get("id", filename_path.stem),
                    "title": data.get("title", "会話"),
                    "date": data.get("saved_at", "不明な日付") 
                })
        except Exception as e:
            logger.error(f"会話メタデータの読み込みエラー ({filename_path.name}): {e}")
            conversations.append({
                "id": filename_path.stem,
                "title": "会話（読込エラー）",
                "date": "不明"
            })
    conversations.sort(key=lambda x: x.get("date"), reverse=True)
    return conversations

# -------------------- Streaming Helpers --------------------
def stream_markdown(text: str, delay: float = 0.02) -> None:
    """Display markdown text with a simple streaming effect."""
    placeholder = st.empty()
    rendered = ""
    for ch in text:
        rendered += ch
        placeholder.markdown(rendered)
        time.sleep(delay)

# セッション状態の初期化
if 'app_mode' not in st.session_state:
    st.session_state['app_mode'] = "ナレッジ検索"
if 'overlap_ratio' not in st.session_state:
    st.session_state['overlap_ratio'] = 15
if 'sudachi_mode' not in st.session_state:
    st.session_state['sudachi_mode'] = 'C'
if 'recommended_params' not in st.session_state:
    st.session_state['recommended_params'] = None
if 'processed_chunks' not in st.session_state:
    st.session_state['processed_chunks'] = None
if 'detected_doc_type' not in st.session_state:
    st.session_state['detected_doc_type'] = None
if 'knowledge_base_name' not in st.session_state:
    st.session_state['knowledge_base_name'] = DEFAULT_KB_NAME
if 'max_chunk_size' not in st.session_state:
    st.session_state['max_chunk_size'] = 1000
if 'forced_overlap_ratio' not in st.session_state:
    st.session_state['forced_overlap_ratio'] = 25
if 'embedding_model' not in st.session_state:
    st.session_state['embedding_model'] = EMBEDDING_MODEL
if 'selected_kb_option' not in st.session_state:
    st.session_state['selected_kb_option'] = None
if 'selected_kbs' not in st.session_state:
    st.session_state['selected_kbs'] = []
if 'search_engines' not in st.session_state:
    st.session_state.search_engines = {}

# chatGPTモード専用の状態
if 'gpt_messages' not in st.session_state:
    st.session_state['gpt_messages'] = []
if 'gpt_conversation_id' not in st.session_state:
    st.session_state['gpt_conversation_id'] = str(uuid.uuid4())
if 'gpt_conversation_title' not in st.session_state:
    st.session_state['gpt_conversation_title'] = "新しい会話"

# 共通のペルソナ・設定
if 'persona' not in st.session_state:
    st.session_state['persona'] = "default"
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 0.7
if 'response_length' not in st.session_state:
    st.session_state['response_length'] = "普通"

# Streamlit UIの設定
if "_page_configured" not in st.session_state:
    st.set_page_config(
        page_title="RAGシステム統合ツール",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["_page_configured"] = True

# カスタムCSSを適用
apply_intel_theme(st)

st.title("RAGシステム統合ツール")

# アプリケーションモード選択
mode_options = ["ナレッジ検索", "ナレッジ構築", "FAQ作成", "chatGPT"]
current_mode = st.session_state.get('app_mode', 'ナレッジ検索')
app_mode_index = mode_options.index(current_mode) if current_mode in mode_options else 0
app_mode = st.sidebar.radio(
    "モード選択",
    mode_options,
    index=app_mode_index
)
st.session_state['app_mode'] = app_mode


# ===================== 既存関数の定義 (OpenAIクライアントの渡し方に注意) =====================

def read_file(file):
    content = ""
    file_type = file.name.split('.')[-1].lower()
    try:
        if file_type == 'pdf':
            data = file.read()
            file.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(data)
                temp_path = temp_file.name
            pdf_reader = PyPDF2.PdfReader(BytesIO(data))
            for idx, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    content += page_text + "\n"
                elif PDF_SUPPORT and OCR_SUPPORT:
                    images = convert_from_bytes(data, first_page=idx + 1, last_page=idx + 1)
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0], lang='jpn+eng')
                        if ocr_text.strip():
                            content += ocr_text + "\n"
            os.unlink(temp_path)
        elif file_type == 'docx':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                shutil.copyfileobj(file, temp_file)
                temp_path = temp_file.name
            file.seek(0)
            doc = docx.Document(temp_path)
            for para in doc.paragraphs:
                content += para.text + "\n"
            if OCR_SUPPORT and Image:
                for rel in doc.part.related_parts.values():
                    if "image" in rel.content_type:
                        img = Image.open(BytesIO(rel.blob))
                        ocr_text = pytesseract.image_to_string(img, lang='jpn+eng')
                        if ocr_text.strip():
                            content += ocr_text + "\n"
            os.unlink(temp_path)
        elif file_type in ['xlsx', 'xls']:
            file_bytes = BytesIO(file.read())
            file.seek(0)
            if EXCEL_SUPPORT:
                wb = openpyxl.load_workbook(file_bytes, data_only=True)
                for sheet in wb.worksheets:
                    content += f"# シート: {sheet.title}\n"
                    for row in sheet.iter_rows(values_only=True):
                        cells = ["" if c is None else str(c) for c in row]
                        content += "\t".join(cells) + "\n"
                    if OCR_SUPPORT and Image and getattr(sheet, "_images", []):
                        for img in sheet._images:
                            try:
                                img_bytes = img._data()
                                img_obj = Image.open(BytesIO(img_bytes))
                                ocr = pytesseract.image_to_string(img_obj, lang='jpn+eng')
                                if ocr.strip():
                                    content += ocr + "\n"
                            except Exception:
                                pass
            else:
                excel_file = pd.ExcelFile(file_bytes)
                for sheet_name in excel_file.sheet_names:
                    sheet_df = excel_file.parse(sheet_name)
                    content += f"# シート: {sheet_name}\n"
                    content += sheet_df.to_string() + "\n\n"
        elif file_type in ['md', 'markdown', 'html', 'htm']:
            content = file.read().decode('utf-8', errors='replace')
            file.seek(0)
            if OCR_SUPPORT and Image:
                for m in re.finditer(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", content):
                    try:
                        img_bytes = base64.b64decode(m.group(1))
                        img = Image.open(BytesIO(img_bytes))
                        ocr = pytesseract.image_to_string(img, lang='jpn+eng')
                        if ocr.strip():
                            content += "\n" + ocr
                    except Exception:
                        pass
        elif file_type == 'doc':
            st.error("doc形式はdocxに変換してから再度アップロードしてください。")
            return None
        elif file_type == 'txt':
            content = file.read().decode('utf-8', errors='replace')
            file.seek(0)
        else:
            st.error(f"サポートされていないファイル形式: {file_type}")
            return None
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました ({file.name}): {e}")
        logger.error(f"ファイル読み込みエラー ({file.name}): {e}", exc_info=True)
        return None
    return content

def estimate_tokens(text):
    if not text: return 0
    japanese_ratio = len(re.findall(r'[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text)) / max(len(text), 1)
    chars_per_token = 2.0 if japanese_ratio > 0.5 else 3.5
    return int(len(text) / chars_per_token * 1.1)

def is_mostly_japanese(text):
    if not text: return False
    japanese_ratio = len(re.findall(r'[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text)) / max(len(text), 1)
    return japanese_ratio > 0.5

_sudachi_tokenizer_obj = None
def analyze_with_sudachi(text, mode_str='C'):
    global _sudachi_tokenizer_obj
    if not text: return [], []
    mode_map = {
        'A': tokenizer.Tokenizer.SplitMode.A,
        'B': tokenizer.Tokenizer.SplitMode.B,
        'C': tokenizer.Tokenizer.SplitMode.C
    }
    mode = mode_map.get(mode_str, tokenizer.Tokenizer.SplitMode.C)
    if _sudachi_tokenizer_obj is None:
        try:
            _sudachi_tokenizer_obj = dictionary.Dictionary().create()
        except Exception as e:
            logger.error(f"Sudachi辞書の初期化に失敗: {e}.")
            st.error(f"Sudachi形態素解析器の初期化に失敗しました: {e}")
            return [], []
    tokens_surface = [m.surface() for m in _sudachi_tokenizer_obj.tokenize(text, mode)]
    important_tokens = []
    for m in _sudachi_tokenizer_obj.tokenize(text, mode):
        pos = m.part_of_speech()[0]
        if pos in ['名詞', '動詞', '形容詞', '副詞']:
            if pos in ['動詞', '形容詞']:
                important_tokens.append(m.dictionary_form())
            else:
                important_tokens.append(m.surface())
    return tokens_surface, important_tokens

def safe_tokenize(text):
    if not text: return ["dummy_token"]
    try:
        tokens = word_tokenize(text)
        if not tokens:
            tokens = re.findall(r'\b\w+\b', text)
            if not tokens: tokens = ["dummy_token"]
        return tokens
    except Exception as e:
        logger.error(f"NLTKトークン化エラー: {e}")
        tokens = re.findall(r'\b\w+\b', text)
        if not tokens: tokens = ["dummy_token"]
        return tokens

def get_embedding(text, model=None, client=None):
    if client is None:
        client = get_openai_client() 
        if client is None:
            logger.error("OpenAIクライアントが利用できません (get_embedding)")
            st.error("OpenAI APIに接続できません (get_embedding)。APIキーを確認してください。")
            return None
    effective_model = model if model else st.session_state.get('embedding_model', EMBEDDING_MODEL)
    if not text or not text.strip():
        logger.warning("空のテキストに対する埋め込み要求をスキップしました。")
        return None
    try:
        max_input_chars = 25000 
        if len(text) > max_input_chars: 
            logger.warning(f"入力テキスト長が{max_input_chars}文字を超過。切り詰めます。")
            text_to_embed = text[:max_input_chars]
        else:
            text_to_embed = text
        response = client.embeddings.create(
            model=effective_model,
            input=text_to_embed,
            dimensions=EMBEDDING_DIMENSIONS
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"埋め込みベクトル作成エラー ({effective_model}): {e}", exc_info=True)
        st.error(f"埋め込みベクトルの作成中にエラーが発生しました ({effective_model}): {e}")
        return None

def detect_document_type(text_sample, client=None):
    if client is None:
        client = get_openai_client()
        if client is None:
            logger.error("OpenAIクライアントが利用できません (detect_document_type)")
            return {"doc_type": "一般文書", "confidence": 0.5, "reasoning": "OpenAIクライアント利用不可"}
    if not text_sample or not text_sample.strip():
        return {"doc_type": "一般文書", "confidence": 0.1, "reasoning": "テキストサンプルが空です。"}
    try:
        response = client.chat.completions.create(
            model=GPT4_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "あなたは文書分類の専門家です。与えられたテキストからドキュメントの種類を判別してください。"},
                {"role": "user", "content": f"""
                このテキストがどのような種類の文書か判別してください。
                以下のカテゴリーの中から最も適切なものを1つ選び、JSONで返してください：
                - 一般文書, 営業マニュアル, 就業規則, ISO文書, 技術文書, 契約書, 研究論文, 製品マニュアル, QA文書, 法律文書, その他
                テキストの一部: {text_sample[:min(len(text_sample), 3000)]}...
                JSON形式: {{"doc_type": "文書タイプ", "confidence": 0.0-1.0, "reasoning": "判断理由"}}
                """}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"文書タイプの自動判別中にエラーが発生しました: {e}")
        logger.error(f"文書タイプ判別エラー: {e}", exc_info=True)
        return {"doc_type": "一般文書", "confidence": 0.5, "reasoning": "自動判別に失敗したためデフォルト値を使用します。"}

def get_recommended_parameters(text_sample, document_type, client=None):
    if client is None:
        client = get_openai_client()
        if client is None:
            logger.error("OpenAIクライアントが利用できません (get_recommended_parameters)")
            return {"overlap": 15, "sudachi_mode": "C", "reasoning": "OpenAIクライアント利用不可"}
    if not text_sample or not text_sample.strip():
        return {"overlap": 15, "sudachi_mode": "C", "reasoning": "テキストサンプルが空です。"}
    try:
        response = client.chat.completions.create(
            model=GPT4_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "あなたは意味ベースのテキストチャンク分割の専門家です。GPT-4.1-miniでナレッジ検索を行うことを想定したパラメータを推奨してください。"},
                {"role": "user", "content": f"""
                このドキュメントの特性に最適なチャンク分けパラメータを推奨してください。
                文書タイプ: {document_type}
                1チャンクは最大{st.session_state.get('max_chunk_size',1000)}トークン目安。
                JSON形式: {{"overlap": 5-30%, "sudachi_mode": "A,B,C", "reasoning": "推奨理由"}}
                ドキュメントサンプル: {text_sample[:min(len(text_sample), 3000)]}...
                """}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        st.error(f"パラメータ推奨値の取得中にエラーが発生しました: {e}")
        logger.error(f"パラメータ推奨エラー: {e}", exc_info=True)
        return {"overlap": 15, "sudachi_mode": "C", "reasoning": f"エラーのためデフォルト値を使用: {e}"}

def get_search_engine(kb_name: str) -> HybridSearchEngine:
    if kb_name not in st.session_state.search_engines or st.session_state.search_engines[kb_name] is None:
        kb_path_str = str(RAG_BASE_DIR / kb_name)
        if not os.path.exists(kb_path_str):
            logger.error(f"ナレッジベースのパスが見つかりません: {kb_path_str} (KB名: {kb_name})")
            st.session_state.search_engines[kb_name] = None
            return None
        try:
            logger.info(f"Initializing HybridSearchEngine for '{kb_name}' at path '{kb_path_str}'...")
            engine = HybridSearchEngine(kb_path_str)
            st.session_state.search_engines[kb_name] = engine
            st.session_state.chat_controller = ChatController(engine) # ChatControllerのインスタンスを生成
            logger.info(f"HybridSearchEngine for '{kb_name}' initialized and cached.")
            return engine
        except Exception as e:
            logger.error(f"Failed to initialize HybridSearchEngine for '{kb_name}': {e}", exc_info=True)
            st.session_state.search_engines[kb_name] = None
            return None
    return st.session_state.search_engines[kb_name]

def refresh_search_engine(kb_name: str) -> None:
    """Rebuild indexes for a knowledge base if a search engine is available."""
    engine = get_search_engine(kb_name)
    if engine is not None:
        try:
            engine.reindex()
        except Exception as e:
            logger.error(f"Failed to refresh index for '{kb_name}': {e}", exc_info=True)

def list_knowledge_bases():
    kb_list = []
    if RAG_BASE_DIR.exists():
        for kb_dir_path in RAG_BASE_DIR.iterdir():
            if kb_dir_path.is_dir():
                kb_name = kb_dir_path.name
                metadata_path = kb_dir_path / "kb_metadata.json"
                kb_info = {"name": kb_name, "created_at": "不明", "doc_type": "不明", "num_chunks": 0, "embedding_model": "不明"}
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        kb_info.update(metadata)
                    except Exception as e:
                        logger.warning(f"メタデータ読み込み失敗 ({kb_name}): {e}", exc_info=False)
                kb_list.append(kb_info)
    return kb_list

def create_run_script():
    try:
        script_dir = current_dir
        with open(script_dir / 'start_app.bat', 'w', encoding='utf-8') as f:
            f.write('@echo off\nchcp 65001 > nul\n')
            f.write('echo RAG System Tool Startup...\n')
            f.write('if not exist .env (\n')
            f.write('  echo APIキーが.envファイルに見つかりません。入力してください。\n')
            f.write('  set /p OPENAI_API_KEY=OpenAI API Key: \n')
            f.write('  echo OPENAI_API_KEY=%OPENAI_API_KEY% > .env\n')
            f.write('  echo APIキーを.envに保存しました。\n')
            f.write(') else (\n')
            f.write('  echo .envファイルからAPIキーを読み込みます。\n')
            f.write('  for /f "delims=" %%x in (.env) do set "%%x"\n')
            f.write(')\n')
            f.write('cd /d "%~dp0"\nstart "" "http://localhost:8501"\nstreamlit run app.py\npause\n')
        with open(script_dir / 'start_app.sh', 'w', encoding='utf-8', newline='\n') as f:
            f.write('#!/bin/bash\necho "RAG System Tool Startup..."\n')
            f.write('if [ ! -f .env ]; then\n')
            f.write('  echo "APIキーが.envファイルに見つかりません。入力してください。"\n')
            f.write('  read -p "OpenAI API Key: " OPENAI_API_KEY_INPUT\n')
            f.write('  echo "OPENAI_API_KEY=$OPENAI_API_KEY_INPUT" > .env\n')
            f.write('  echo "APIキーを.envに保存しました。"\n')
            f.write('fi\n')
            f.write('source .env\n')
            f.write('DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n')
            f.write('if [[ "$OSTYPE" == "darwin"* ]]; then\n  open http://localhost:8501\nelif [[ "$OSTYPE" == "linux-gnu"* ]]; then\n  xdg-open http://localhost:8501 || sensible-browser http://localhost:8501\nfi\n')
            f.write('streamlit run "$DIR/app.py"\n')
        if os.name != 'nt':
            os.chmod(script_dir / 'start_app.sh', 0o755)
        return "起動スクリプトを作成しました（start_app.bat / start_app.sh）。"
    except Exception as e:
        logger.error(f"起動スクリプト作成エラー: {e}", exc_info=True)
        return f"起動スクリプト作成中にエラーが発生しました: {e}"

def export_knowledge_base(kb_name):
    kb_dir = RAG_BASE_DIR / kb_name
    if not kb_dir.exists() or not kb_dir.is_dir():
        st.warning(f"ナレッジベース '{kb_name}' は存在しないか、ディレクトリではありません。")
        return None
    try:
        with tempfile.TemporaryDirectory() as export_dir_temp:
            export_dir_path = Path(export_dir_temp)
            archive_base_name = export_dir_path / f"{kb_name}_export"
            zip_file_path = shutil.make_archive(str(archive_base_name), 'zip', root_dir=RAG_BASE_DIR, base_dir=kb_name)
            with open(zip_file_path, 'rb') as f:
                data = f.read()
            return data
    except Exception as e:
        st.error(f"エクスポート中にエラーが発生しました ({kb_name}): {e}")
        logger.error(f"エクスポートエラー ({kb_name}): {e}", exc_info=True)
        return None

def generate_chunk_metadata(chunk_text, document_type, client=None):
    if client is None:
        client = get_openai_client()
        if client is None: return {"summary": "メタデータ生成失敗(クライアント無)", "keywords": [], "tags": [], "search_queries": [], "synonyms": {}, "semantic_connections": [], "mini_context": ""}
    if not chunk_text or not chunk_text.strip():
        return {"summary": "チャンク空のためメタデータ生成スキップ", "keywords": [], "tags": [], "search_queries": [], "synonyms": {}, "semantic_connections": [], "mini_context": ""}
    try:
        prompt = f"""
        このテキストチャンクについて以下の情報を抽出・生成して詳細なメタデータをJSON形式で返してください。
        文書タイプ: {document_type}
        1. summary: 3文以内の簡潔な説明
        2. keywords: 重要キーワードリスト（最大10個）
        3. tags: 分類タグリスト（最大5個）
        4. search_queries: ユーザーが使いそうな検索クエリ（5つ）
        5. synonyms: キーワードの同義語辞書
        6. semantic_connections: 意味的に関連する概念（3つ）
        7. mini_context: GPT-4.1-mini検索結果用コンテキスト（50-80文字）
        チャンク内容: {chunk_text[:min(len(chunk_text),3500)]}...
        JSON形式で返してください。
        """
        response = client.chat.completions.create(
            model=GPT4_MODEL, response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "あなたはテキスト分析の専門家です。詳細なメタデータを抽出・生成してください。"},
                {"role": "user", "content": prompt}
            ]
        )
        metadata = json.loads(response.choices[0].message.content)
        return metadata
    except Exception as e:
        logger.error(f"メタデータ生成エラー: {e}")
        st.error(f"メタデータの生成中にエラーが発生しました: {e}")
        return {"summary": "メタデータ生成失敗", "keywords": [], "tags": [], "search_queries": [], "synonyms": {}, "semantic_connections": [], "mini_context": "コンテキスト生成失敗"}

def optimize_chunk_for_mini(chunk_text, document_type, metadata, client=None):
    if client is None:
        client = get_openai_client()
        if client is None: return "チャンク内容最適化失敗(クライアント無)"
    if not chunk_text or not chunk_text.strip(): return "チャンク空のため最適化スキップ"
    try:
        keywords_str = ", ".join(metadata.get("keywords", []))
        summary = metadata.get("summary", "要約なし")
        prompt = f"""
        以下のテキストチャンクをGPT-4.1-miniでの検索結果として最適な形式に最適化してください。
        文書タイプ: {document_type}, 要約: {summary}, キーワード: {keywords_str}
        原文: {chunk_text[:min(len(chunk_text),2000)]}...
        考慮点: 重要情報焦点、専門用語展開、理解容易な構造、段落/箇条書き、最大800文字。
        最適化されたテキストのみを返してください。
        """
        response = client.chat.completions.create(
            model=GPT4_MINI_MODEL,
            messages=[
                {"role": "system", "content": "あなたはテキスト最適化の専門家です。簡潔かつ理解しやすく再構成してください。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"チャンク最適化エラー: {e}")
        return chunk_text[:min(len(chunk_text),800)] + "..."

def segment_text_by_meaning(text, sudachi_mode='C', document_type="一般文書", client=None):
    if client is None:
        client = get_openai_client()
        if client is None: return [p for p in text.split('\n\n') if p.strip()]
    if not text or not text.strip(): return []
    if len(text) < 1000: return [text.strip()]
    try:
        sample_text = text[:min(len(text),7000)]
        analysis_prompt = f"""
        テキストを意味のまとまりで分割するための最適なアプローチ分析。文書タイプ: {document_type}
        分析対象: {sample_text}
        JSON形式: {{"text_structure": "全体構造", "segment_markers": "区切り特徴", "segmentation_approach": "最適アプローチ", "optimal_segment_length": "推奨文字数範囲"}}
        """
        response_analysis = client.chat.completions.create(
            model=GPT4_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "テキスト分析と構造化の専門家。"}, {"role": "user", "content": analysis_prompt}]
        )
        structure_analysis = json.loads(response_analysis.choices[0].message.content)
        segmentation_prompt = f"""
        テキストを意味のまとまりに基づいて分割してください。以下の分析結果と指示に従ってください。
        文書タイプ: {document_type}
        テキスト構造: {structure_analysis.get('text_structure', '不明')}
        分割マーカー: {structure_analysis.get('segment_markers', '不明')}
        分割アプローチ: {structure_analysis.get('segmentation_approach', '不明')}
        指示:
        1. テキストを意味のまとまりで分割し、各セグメントをJSON配列として返してください
        2. 各セグメントは独立して理解できる内容を持つようにしてください
        3. 見出しとその内容は同じセグメントに含めてください
        4. 論理的な区切り（章、節、トピックの変更点など）で分割してください
        5. 各セグメントの文字数は約{structure_analysis.get('optimal_segment_length', '500-1500')}文字を目安にしてください
        テキスト全体（最初の{min(len(text), 15000)}文字）:
        {text[:min(len(text), 15000)]} 
        分割されたセグメントをJSON配列で返してください。例: {{"segments": ["セグメント1", "セグメント2", ...]}}
        """
        response_segment = client.chat.completions.create(
            model=GPT4_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "テキストを意味のまとまりで適切に分割。"}, {"role": "user", "content": segmentation_prompt}]
        )
        result = json.loads(response_segment.choices[0].message.content)
        segments = result.get("segments", [])
        if not segments and result and isinstance(result, dict) and len(result.keys()) == 1:
             first_key = list(result.keys())[0]
             if isinstance(result[first_key], list) and all(isinstance(s, str) for s in result[first_key]):
                 segments = result[first_key]
        if not segments:
            logger.warning("GPTによるセグメント分割失敗。フォールバック使用。")
            segments = [p.strip() for p in text.split('\n\n') if p.strip()]
        return [s for s in segments if s.strip()]
    except Exception as e:
        logger.error(f"テキスト分割エラー: {e}", exc_info=True)
        return [p.strip() for p in text.split('\n\n') if p.strip()]

def create_overlapping_chunks(segments, overlap_ratio_percent=15, max_chunk_tokens=1000):
    if not segments: return []
    chunks_text = []
    current_chunk_text = ""
    overlap_ratio = overlap_ratio_percent / 100.0
    forced_overlap_ratio = st.session_state.get('forced_overlap_ratio', 25) / 100.0
    estimated_chars_per_token = 2.5
    max_chunk_chars = int(max_chunk_tokens * estimated_chars_per_token * 0.9)

    for i, segment_text in enumerate(segments):
        if not segment_text.strip(): continue
        if not current_chunk_text:
            current_chunk_text = segment_text
        else:
            potential_chunk_text = current_chunk_text + "\n" + segment_text
            if estimate_tokens(potential_chunk_text) > max_chunk_tokens:
                chunks_text.append(current_chunk_text.strip())
                sentences_in_completed_chunk = re.split(r'(?<=[。！？.!?])\s*', current_chunk_text.strip())
                sentences_in_completed_chunk = [s for s in sentences_in_completed_chunk if s.strip()]
                overlap_text_content = ""
                if sentences_in_completed_chunk:
                    target_overlap_chars = int(len(current_chunk_text) * overlap_ratio)
                    temp_overlap = []
                    accumulated_len = 0
                    for sent in reversed(sentences_in_completed_chunk):
                        temp_overlap.append(sent)
                        accumulated_len += len(sent)
                        if accumulated_len >= target_overlap_chars and len(temp_overlap) > 0 :
                            break
                    overlap_text_content = " ".join(reversed(temp_overlap)).strip()
                current_chunk_text = (overlap_text_content + "\n" + segment_text).strip() if overlap_text_content else segment_text
            else:
                current_chunk_text = potential_chunk_text
        
        while estimate_tokens(current_chunk_text) > max_chunk_tokens:
            sentences_to_split = re.split(r'(?<=[。！？.!?])\s*', current_chunk_text.strip())
            sentences_to_split = [s for s in sentences_to_split if s.strip()]
            if not sentences_to_split:
                chunks_text.append(current_chunk_text[:max_chunk_chars].strip())
                current_chunk_text = current_chunk_text[max_chunk_chars:].strip()
                if not current_chunk_text: break
                continue
            newly_formed_chunk_part = ""
            split_after_sentence_index = 0
            for idx, sentence_part in enumerate(sentences_to_split):
                tentative_new_chunk = (newly_formed_chunk_part + "\n" + sentence_part).strip() if newly_formed_chunk_part else sentence_part.strip()
                if estimate_tokens(tentative_new_chunk) <= max_chunk_tokens:
                    newly_formed_chunk_part = tentative_new_chunk
                    split_after_sentence_index = idx + 1
                else:
                    if idx == 0:
                        newly_formed_chunk_part = sentence_part[:max_chunk_chars].strip()
                        split_after_sentence_index = 1
                    break
            if newly_formed_chunk_part:
                chunks_text.append(newly_formed_chunk_part.strip())
                remaining_sentences_text = "\n".join(sentences_to_split[split_after_sentence_index:]).strip()
                if remaining_sentences_text:
                    sentences_in_new_chunk = re.split(r'(?<=[。！？.!?])\s*', newly_formed_chunk_part.strip())
                    sentences_in_new_chunk = [s for s in sentences_in_new_chunk if s.strip()]
                    forced_overlap_content = ""
                    if sentences_in_new_chunk:
                        target_forced_overlap_chars = int(len(newly_formed_chunk_part) * forced_overlap_ratio)
                        temp_forced_overlap = []
                        accumulated_forced_len = 0
                        for sent_f in reversed(sentences_in_new_chunk):
                            temp_forced_overlap.append(sent_f)
                            accumulated_forced_len += len(sent_f)
                            if accumulated_forced_len >= target_forced_overlap_chars and len(temp_forced_overlap) > 0:
                                break
                        forced_overlap_content = " ".join(reversed(temp_forced_overlap)).strip()
                    current_chunk_text = (forced_overlap_content + "\n" + remaining_sentences_text).strip() if forced_overlap_content else remaining_sentences_text
                else:
                    current_chunk_text = ""
            else:
                current_chunk_text = ""
            if not current_chunk_text: break
    
    if current_chunk_text.strip():
        chunks_text.append(current_chunk_text.strip())
    return [c for c in chunks_text if c.strip() and estimate_tokens(c) > 5]

def save_chunk_to_files(
    chunk_content,
    chunk_id,
    folder_name,
    base_filename,
    metadata,
    embedding,
    kb_dir_path: Path,
    refresh: bool = True,
):
    try:
        paths = save_processed_data(
            kb_dir_path.name,
            chunk_id,
            chunk_text=chunk_content,
            embedding=embedding,
            metadata=metadata,
        )
        if refresh:
            refresh_search_engine(kb_dir_path.name)
        return [
            paths.get("chunk_path"),
            paths.get("metadata_path"),
            paths.get("embedding_path"),
        ]
    except Exception as e:
        logger.error(f"チャンク保存エラー ({chunk_id}): {e}", exc_info=True)
        st.error(f"チャンクの保存中にエラーが発生しました ({chunk_id}): {e}")
        return []

def update_kb_metadata(kb_dir_path: Path, document_type, num_chunks, embedding_model):
    try:
        metadata_kb = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_type": document_type,
            "num_chunks": num_chunks,
            "embedding_model": embedding_model,
            "original_filename": st.session_state.get('_last_uploaded_filename_for_doc_type', 'N/A')
        }
        with open(kb_dir_path / "kb_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata_kb, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"KBメタデータ更新エラー ({kb_dir_path.name}): {e}", exc_info=True)
        st.error(f"ナレッジベースメタデータの更新中にエラーが発生しました: {e}")
        return False

def generate_folder_structure(text_sample, document_type, client=None):
    if client is None:
        client = get_openai_client()
        if client is None: return {"folder_name": f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "base_filename": "chunk", "description": "クライアント無"}
    if not text_sample or not text_sample.strip():
        return {"folder_name": f"doc_empty_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "base_filename": "empty_chunk", "description": "テキストサンプル空"}
    try:
        prompt = f"""
        テキスト内容に基づき、RAG検索に適したフォルダ名とベースファイル名を提案。
        フォルダ名、ファイル名には日本語を含めても良いが、OSで使える文字のみ。スペースはアンダースコアに置換。
        文書タイプ: {document_type}
        JSON形式: {{"folder_name": "内容がわかる名前", "base_filename": "ベース名", "description": "提案理由"}}
        テキストサンプル: {text_sample[:min(len(text_sample),5000)]}...
        """
        response = client.chat.completions.create(
            model=GPT4_MODEL, response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "ドキュメント整理専門家。適切なフォルダ構造とファイル名を提案。"},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        safe_chars_pattern = r'[\\/*?:"<>|\t\n\r\f\v]'
        result['folder_name'] = re.sub(r'\s+', "_", result.get('folder_name', f"folder_{datetime.now().strftime('%Y%m%d')}"))
        result['folder_name'] = re.sub(safe_chars_pattern, "_", result['folder_name'])
        result['base_filename'] = re.sub(r'\s+', "_", result.get('base_filename', "chunk"))
        result['base_filename'] = re.sub(safe_chars_pattern, "_", result['base_filename'])
        return result
    except Exception as e:
        logger.error(f"フォルダ構造生成エラー: {e}", exc_info=True)
        st.error(f"フォルダ構造の生成中にエラーが発生しました: {e}")
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        return {"folder_name": f"doc_err_{timestamp_str}", "base_filename": f"chunk_err_{timestamp_str}", "description": f"エラーのためデフォルト: {e}"}

def expand_search_query(query, client=None):
    if client is None:
        client = get_openai_client()
        if client is None: return query
    if not query or not query.strip(): return query
    try:
        prompt = f"""
        検索クエリを拡張。同義語、関連語、別表現、表記ゆれ考慮。検索エンジンに最適化。
        元のクエリ: {query}
        拡張クエリのみ返却。元のクエリの意図から離れすぎないように。
        """
        response = client.chat.completions.create(
            model=GPT4_MINI_MODEL,
            messages=[{"role": "system", "content": "検索クエリ最適化専門家。"}, {"role": "user", "content": prompt}]
        )
        expanded_query = response.choices[0].message.content.strip()
        logger.info(f"クエリ拡張: '{query}' -> '{expanded_query}'")
        return expanded_query
    except Exception as e:
        logger.error(f"クエリ拡張エラー: {e}", exc_info=True)
        return query

def search_multiple_knowledge_bases(query, kb_names, top_k=5, threshold=0.15, client=None):
    all_results = []
    not_found_overall = True
    if client is None:
        client = get_openai_client()
        if client is None:
            st.error("OpenAI APIクライアントの初期化に失敗しました。検索できません。")
            logger.error("OpenAI client failed to initialize in search_multiple_knowledge_bases.")
            return [], True
    if not kb_names:
        st.warning("検索対象のナレッジベースが指定されていません。")
        return [], True
    for kb_name in kb_names:
        engine = get_search_engine(kb_name)
        if engine:
            try:
                results_for_kb, not_found_for_kb = engine.search(query, top_k=top_k, threshold=threshold, client=client)
                if not not_found_for_kb and results_for_kb:
                    for r in results_for_kb: r['kb_name'] = kb_name
                    all_results.extend(results_for_kb)
                    not_found_overall = False
            except Exception as e:
                logger.error(f"ナレッジベース '{kb_name}' 検索エラー: {e}", exc_info=True)
                st.warning(f"ナレッジベース '{kb_name}' での検索中にエラー。")
        else:
            st.warning(f"ナレッジベース '{kb_name}' の検索エンジン利用不可。")
    all_results.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
    final_results = all_results[:top_k] 
    return final_results, len(final_results) == 0

def semantic_chunking(
    text,
    overlap_ratio,
    sudachi_mode,
    document_type,
    knowledge_base_name,
    client=None,
    original_filename=None,
    original_bytes=None,
    refresh: bool = True,
):
    if client is None:
        client = get_openai_client()
        if client is None:
            st.error("OpenAI API接続不可。チャンク分け中止。")
            return []
    if not text or not text.strip():
        st.warning("入力テキストが空のため、チャンク分けをスキップします。")
        return []
    try:
        kb_dir_path = RAG_BASE_DIR / knowledge_base_name
        kb_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ナレッジベースディレクトリ: {kb_dir_path}")
        if original_filename and original_bytes:
            save_processed_data(
                kb_dir_path.name,
                f"source_{uuid.uuid4()}",
                original_filename=original_filename,
                original_bytes=original_bytes,
            )
        with st.status("フォルダ構造を生成中...", expanded=False) as status_fs:
            folder_structure = generate_folder_structure(text[:min(len(text),5000)], document_type, client)
            folder_name = folder_structure.get("folder_name")
            base_filename = folder_structure.get("base_filename")
            st.info(f"生成フォルダ名: {folder_name}, ベースファイル名: {base_filename}")
            status_fs.update(label="フォルダ構造生成完了", state="complete")
        with st.status("テキストを意味単位で分割中...", expanded=False) as status_seg:
            segments = segment_text_by_meaning(text, sudachi_mode, document_type, client)
            if not segments:
                st.warning("意味単位での分割結果が空です。段落分割にフォールバックします。")
                segments = [p.strip() for p in text.split('\n\n') if p.strip()]
            st.success(f"{len(segments)}個のセグメントに分割完了")
            status_seg.update(label="意味単位分割完了", state="complete")
        with st.status("オーバーラップチャンクを作成中...", expanded=False) as status_chk:
            chunks = create_overlapping_chunks(segments, overlap_ratio, st.session_state.get('max_chunk_size',1000))
            st.success(f"{len(chunks)}個のチャンクを作成完了")
            status_chk.update(label="チャンク作成完了", state="complete")
        if not chunks:
            st.warning("作成されたチャンクがありません。処理を終了します。")
            return []
        processed_chunks_info = []
        embedding_model_for_kb = st.session_state.get('embedding_model', EMBEDDING_MODEL)
        progress_bar = st.progress(0, text="チャンク処理中...")
        total_chunks = len(chunks)
        for i, chunk_content in enumerate(chunks):
            if not chunk_content or not chunk_content.strip(): 
                logger.warning(f"空のチャンク {i+1} をスキップします。")
                progress_bar.progress((i + 1) / total_chunks, text=f"チャンク {i+1}/{total_chunks} 処理中 (スキップ)")
                continue
            chunk_id_str = str(i + 1).zfill(max(4, len(str(total_chunks))))
            with st.status(f"チャンク {chunk_id_str}/{total_chunks} を処理中...", expanded=False) as status_item:
                st.write(f"メタデータ生成中 (チャンク {chunk_id_str})...")
                metadata = generate_chunk_metadata(chunk_content, document_type, client)
                st.write(f"GPT-mini用コンテキスト最適化中 (チャンク {chunk_id_str})...")
                mini_context = optimize_chunk_for_mini(chunk_content, document_type, metadata, client)
                metadata["mini_context"] = mini_context
                _, important_tokens = analyze_with_sudachi(chunk_content, sudachi_mode)
                metadata["sudachi_tokens"] = important_tokens[:50]
                st.write(f"埋め込みベクトル生成中 (チャンク {chunk_id_str})...")
                embedding = get_embedding(chunk_content, embedding_model_for_kb, client)
                if embedding is None:
                    st.warning(f"チャンク {chunk_id_str} の埋め込みベクトル生成に失敗しました。このチャンクのベクトル情報はスキップされます。")
                st.write(f"ファイル保存中 (チャンク {chunk_id_str})...")
                saved_files = save_chunk_to_files(
                    chunk_content,
                    chunk_id_str,
                    folder_name,
                    base_filename,
                    {
                        "id": chunk_id_str,
                        "filename": f"{base_filename}_{chunk_id_str}",
                        "token_count": estimate_tokens(chunk_content),
                        "char_count": len(chunk_content),
                        "meta_info": metadata,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    embedding,
                    kb_dir_path,
                    refresh=False,
                )
                processed_chunks_info.append({
                    "id": chunk_id_str, "content": chunk_content, "meta_info": metadata,
                    "token_count": estimate_tokens(chunk_content), "filename": f"{base_filename}_{chunk_id_str}",
                })
                status_item.update(label=f"チャンク {chunk_id_str}/{total_chunks} 処理完了", state="complete")
            progress_bar.progress((i + 1) / total_chunks, text=f"チャンク {i+1}/{total_chunks} 処理完了")
        progress_bar.empty()
        update_kb_metadata(
            kb_dir_path,
            document_type,
            len(processed_chunks_info),
            embedding_model_for_kb,
        )
        if refresh:
            refresh_search_engine(kb_dir_path.name)
        return processed_chunks_info
    except Exception as e:
        logger.error(f"意味ベースチャンク分けエラー: {e}", exc_info=True)
        st.error(f"意味ベースチャンク分け実行中にエラー: {e}")
        if 'progress_bar' in locals(): progress_bar.empty()
        return []

# ===================== UI部分の実装 =====================

# 現在のモードを表示
st.markdown(f"""<div class="mode-header">現在のモード: {app_mode}</div>""", unsafe_allow_html=True)

if app_mode == "ナレッジ構築":
    st.markdown("ドキュメントをアップロードして意味ベースのチャンク分けを実行し、RAG用のナレッジベースを構築します。")
    tab1, tab2 = st.tabs(["ナレッジベース作成", "ナレッジベース管理"])
    with tab1:
        st.header("ナレッジベース作成")
        kb_name_input = st.text_input("ナレッジベース名*", value=st.session_state.get('knowledge_base_name', f"kb_{datetime.now().strftime('%Y%m%d')}"))
        st.session_state['knowledge_base_name'] = kb_name_input
        embedding_model_options = [EMBEDDING_MODEL, "text-embedding-3-small", "text-embedding-ada-002"]
        current_embedding_model = st.session_state.get('embedding_model', EMBEDDING_MODEL)
        embedding_model_idx = embedding_model_options.index(current_embedding_model) if current_embedding_model in embedding_model_options else 0
        selected_embedding_model = st.selectbox("埋め込みモデル*", embedding_model_options, index=embedding_model_idx)
        st.session_state['embedding_model'] = selected_embedding_model
        uploaded_file = st.file_uploader("対象ファイルを選択 (PDF, DOCX, XLSX, TXT)*", type=['pdf', 'docx', 'xlsx', 'xls', 'txt'])
        document_type_options = ['一般文書', '営業マニュアル', '就業規則', 'ISO文書', '技術文書', '契約書', '研究論文', '製品マニュアル', 'QA文書', '法律文書', 'その他']
        auto_detect_doc_type = st.checkbox("文書タイプを自動判別する", value=True)
        detected_doc_type_info = st.session_state.get('detected_doc_type')
        document_type_selection_from_state = None
        if detected_doc_type_info is not None:
            document_type_selection_from_state = detected_doc_type_info.get('doc_type')
        if auto_detect_doc_type and uploaded_file is not None:
            last_uploaded_file = st.session_state.get('_last_uploaded_filename_for_doc_type')
            if last_uploaded_file != uploaded_file.name or detected_doc_type_info is None:
                with st.spinner("文書タイプを自動判別中..."):
                    file_content_for_detection = read_file(uploaded_file) 
                    if file_content_for_detection:
                        client_for_detection = get_openai_client()
                        if client_for_detection:
                            detection_result = detect_document_type(file_content_for_detection[:min(len(file_content_for_detection),10000)], client_for_detection)
                            st.session_state['detected_doc_type'] = detection_result
                            st.session_state['_last_uploaded_filename_for_doc_type'] = uploaded_file.name 
                            document_type_selection_from_state = detection_result.get('doc_type') 
                            st.success(f"文書タイプ自動判別: {detection_result.get('doc_type')} (確信度: {detection_result.get('confidence', 0):.2f})")
                            with st.expander("判断理由を見る"):
                                st.info(f"{detection_result.get('reasoning', 'N/A')}")
                        else:
                            st.warning("文書タイプ自動判別不可: OpenAIクライアントエラー。手動で選択してください。")
                            st.session_state['detected_doc_type'] = None
                            document_type_selection_from_state = None
                    else:
                        st.warning("ファイル内容の読み込みに失敗したため、文書タイプ判別をスキップします。")
                        st.session_state['detected_doc_type'] = None
                        document_type_selection_from_state = None
        doc_type_final_idx = 0
        if document_type_selection_from_state and document_type_selection_from_state in document_type_options:
            doc_type_final_idx = document_type_options.index(document_type_selection_from_state)
        final_document_type_selected = st.selectbox("文書タイプ*", document_type_options, index=doc_type_final_idx, help="文書の種類を選択または自動判別結果を修正してください。")
        st.sidebar.header("チャンク分けパラメータ")
        overlap_ratio_ui_val = st.sidebar.slider("オーバーラップ率 (%)", 0, 50, st.session_state.get('overlap_ratio', 15), 5)
        st.session_state['overlap_ratio'] = overlap_ratio_ui_val
        sudachi_options_map_ui = {'A (最小分割)': 'A', 'B (中間)': 'B', 'C (最大分割)': 'C'}
        current_sudachi_mode_val_ui = st.session_state.get('sudachi_mode', 'C')
        sudachi_display_options_ui = list(sudachi_options_map_ui.keys())
        current_sudachi_display_ui = next((k for k, v in sudachi_options_map_ui.items() if v == current_sudachi_mode_val_ui), sudachi_display_options_ui[2])
        sudachi_mode_display_selected_ui = st.sidebar.selectbox("形態素解析の粒度 (日本語)", sudachi_display_options_ui, index=sudachi_display_options_ui.index(current_sudachi_display_ui))
        st.session_state['sudachi_mode'] = sudachi_options_map_ui[sudachi_mode_display_selected_ui]
        st.sidebar.subheader("詳細設定")
        max_chunk_size_ui_val = st.sidebar.slider("最大チャンクサイズ (トークン)", 200, 8000, st.session_state.get('max_chunk_size', 1000), 100, help="1チャンクあたりの最大トークン数。モデルのコンテキスト長も考慮してください。")
        st.session_state['max_chunk_size'] = max_chunk_size_ui_val
        forced_overlap_ratio_ui_val = st.sidebar.slider("強制分割時のオーバーラップ率 (%)", 0, 50, st.session_state.get('forced_overlap_ratio', 25), 5, help="チャンクが最大サイズを超えた場合に強制分割する際のオーバーラップ。")
        st.session_state['forced_overlap_ratio'] = forced_overlap_ratio_ui_val
        if st.session_state.get('recommended_params'):
            rec_p = st.session_state['recommended_params']
            with st.sidebar.expander("GPTによる推奨設定案を見る", expanded=False):
                st.markdown(f"推奨オーバーラップ率: **{rec_p.get('overlap', 'N/A')}%**")
                st.markdown(f"推奨形態素解析粒度: **{rec_p.get('sudachi_mode', 'N/A')}**")
                st.caption(f"理由: {rec_p.get('reasoning', 'N/A')}")
                if st.button("この推奨設定を適用", key="apply_rec_params_sidebar"):
                    st.session_state['overlap_ratio'] = rec_p.get('overlap', st.session_state['overlap_ratio'])
                    st.session_state['sudachi_mode'] = rec_p.get('sudachi_mode', st.session_state['sudachi_mode'])
                    st.rerun()
        st.markdown("---")
        if uploaded_file is not None and kb_name_input:
            if st.button("◎ ナレッジベース構築実行", key="execute_chunking_button_main", type="primary", help="上記の設定でナレッジベースの構築を開始します。"):
                with st.spinner("ナレッジベース構築処理を開始します..."):
                    file_bytes = uploaded_file.getvalue()
                    uploaded_file.seek(0)
                    main_content_for_chunking = read_file(uploaded_file)
                if main_content_for_chunking:
                    client_for_pipeline = get_openai_client()
                    if not client_for_pipeline:
                        st.error("OpenAIクライアントの取得に失敗。処理を中止します。")
                    else:
                        if auto_detect_doc_type and not st.session_state.get('recommended_params'):
                            if st.session_state.get('_last_uploaded_filename_for_param_rec') != uploaded_file.name:
                                with st.spinner("最適パラメータを分析中..."):
                                    rec_params_val = get_recommended_parameters(main_content_for_chunking[:min(len(main_content_for_chunking),10000)], final_document_type_selected, client_for_pipeline)
                                    st.session_state['recommended_params'] = rec_params_val
                                    st.session_state['_last_uploaded_filename_for_param_rec'] = uploaded_file.name
                                    st.info(f"推奨オーバーラップ率: {rec_params_val.get('overlap')}% / 推奨Sudachiモード: {rec_params_val.get('sudachi_mode')}")
                        st.session_state['processed_chunks'] = semantic_chunking(
                            main_content_for_chunking,
                            st.session_state['overlap_ratio'],
                            st.session_state['sudachi_mode'],
                            final_document_type_selected,
                            st.session_state['knowledge_base_name'],
                            client_for_pipeline,
                            original_filename=uploaded_file.name,
                            original_bytes=file_bytes,
                        )
                else:
                    st.error("ファイル内容の読み込みに失敗したため、処理を中止しました。")
        elif not kb_name_input:
            st.warning("ナレッジベース名を入力してください。")
        elif not uploaded_file:
            st.warning("ファイルをアップロードしてください。")
        if st.session_state.get('processed_chunks'):
            chunks_to_display = st.session_state['processed_chunks']
            st.success(f"ナレッジベース「{st.session_state['knowledge_base_name']}」に {len(chunks_to_display)}個のチャンクを保存しました。")
            st.subheader("作成されたチャンクのプレビュー")
            for i, chunk_item_disp in enumerate(chunks_to_display[:5]):
                with st.expander(f"チャンク {chunk_item_disp.get('id', i+1)}: {chunk_item_disp.get('filename', '')} (トークン: {chunk_item_disp.get('token_count')})"):
                    st.markdown(f"**要約**: {chunk_item_disp.get('meta_info', {}).get('summary', 'なし')}")
                    tab_c_cont, tab_c_meta = st.tabs(["内容プレビュー", "メタ情報"])
                    with tab_c_cont: st.text_area("内容 (先頭500文字)", chunk_item_disp.get("content", "")[:500] + "...", height=150, key=f"chunk_disp_content_{chunk_item_disp.get('id')}")
                    with tab_c_meta: st.json(chunk_item_disp.get('meta_info', {}), expanded=False)
            if len(chunks_to_display) > 5:
                st.info(f"全 {len(chunks_to_display)} 件中、最初の5件を表示しています。")
            if st.button("プレビューをクリアして次の準備をする", key="clear_processed_chunks"):
                st.session_state['processed_chunks'] = None
                st.session_state['detected_doc_type'] = None
                st.session_state['recommended_params'] = None
                st.session_state['_last_uploaded_filename_for_doc_type'] = None
                st.session_state['_last_uploaded_filename_for_param_rec'] = None
                st.rerun()
    with tab2:
        st.header("既存ナレッジベースの管理")
        kb_list_for_mng = list_knowledge_bases()
        if kb_list_for_mng:
            st.info(f"現在 {len(kb_list_for_mng)} 個のナレッジベースが利用可能です。")
            df_kb = pd.DataFrame(kb_list_for_mng)
            if not df_kb.empty:
                df_kb_display = df_kb[['name', 'num_chunks', 'doc_type', 'embedding_model', 'created_at', 'updated_at']].copy()
                df_kb_display.rename(columns={
                    'name': 'KB名', 'num_chunks': 'チャンク数', 'doc_type': '文書タイプ',
                    'embedding_model': '埋込モデル', 'created_at': '作成日時', 'updated_at': '最終更新日時'
                }, inplace=True)
                st.dataframe(df_kb_display, use_container_width=True)
            st.markdown("---")
            selected_kb_to_manage = st.selectbox("管理するナレッジベースを選択", [kb['name'] for kb in kb_list_for_mng], index=None, placeholder="ナレッジベースを選択...")
            if selected_kb_to_manage:
                st.subheader(f"ナレッジベース「{selected_kb_to_manage}」の操作")
                col_mng1, col_mng2 = st.columns(2)
                with col_mng1:
                    if st.button(f"↓ 「{selected_kb_to_manage}」をエクスポート", key=f"export_kb_btn_{selected_kb_to_manage}", use_container_width=True):
                        with st.spinner(f"'{selected_kb_to_manage}' をエクスポート準備中..."):
                            export_zip_data = export_knowledge_base(selected_kb_to_manage)
                            if export_zip_data:
                                st.download_button(label=f"◎ {selected_kb_to_manage}_export.zip をダウンロード", data=export_zip_data,
                                                   file_name=f"{selected_kb_to_manage}_export.zip", mime="application/zip", use_container_width=True)
                            else:
                                st.error("エクスポートに失敗しました。")
                with col_mng2:
                    if st.button(f"× 「{selected_kb_to_manage}」を削除", type="secondary", key=f"delete_kb_btn_{selected_kb_to_manage}", use_container_width=True, help="この操作は元に戻せません！"):
                        confirm_delete = st.checkbox(f"本当に「{selected_kb_to_manage}」を削除しますか？", key=f"confirm_del_{selected_kb_to_manage}")
                        if confirm_delete:
                            if selected_kb_to_manage in st.session_state.get('search_engines', {}):
                                del st.session_state.search_engines[selected_kb_to_manage]
                                logger.info(f"検索エンジンキャッシュから {selected_kb_to_manage} を削除しました。")
                            try:
                                shutil.rmtree(RAG_BASE_DIR / selected_kb_to_manage)
                                st.success(f"ナレッジベース '{selected_kb_to_manage}' を完全に削除しました。")
                                if st.session_state.get('selected_kb_option') == selected_kb_to_manage:
                                    st.session_state.selected_kb_option = None
                                if selected_kb_to_manage in st.session_state.get('selected_kbs', []):
                                    st.session_state.selected_kbs.remove(selected_kb_to_manage)
                                st.rerun()
                            except Exception as e:
                                st.error(f"ナレッジベース「{selected_kb_to_manage}」の削除中にエラーが発生しました: {e}")
                                logger.error(f"KB削除エラー ({selected_kb_to_manage}): {e}", exc_info=True)
        else:
            st.info("利用可能なナレッジベースがありません。「ナレッジベース作成」タブから新しいナレッジベースを作成してください。")
    with st.expander("ℹ RAGシステムのベストプラクティスとヒント"): 
        st.markdown("""
        - **チャンクサイズ**: GPTモデルのコンテキストウィンドウと検索対象ドキュメントの性質を考慮して調整します。通常500-1500トークン程度が目安です。
        - **オーバーラップ**: 文脈の連続性を保つために重要です。10-25%程度が一般的ですが、文書構造に依存します。
        - **メタデータ**: 各チャンクに豊富なメタデータ（キーワード、要約、出典など）を付与すると検索精度が向上します。
        - **埋め込みモデル**: `text-embedding-3-large` は高品質ですが高コスト。`text-embedding-3-small` はバランス型。`ada-002` は旧世代ですが軽量。
        - **ハイブリッド検索**: ベクトル検索とキーワード検索（BM25など）を組み合わせることで、より頑健な検索が実現できます。
        """)
    with st.expander("⚡ ワンクリック起動スクリプトを作成"):
        if st.button("Windows/Mac/Linux用 起動スクリプト生成", key="create_startup_script_btn"):
            script_msg = create_run_script()
            st.success(script_msg)
            st.info("生成されたスクリプト（start_app.bat / start_app.sh）をダブルクリック（または実行）するだけで、次回から簡単にアプリを起動できます。初回起動時にAPIキーの入力を求められることがあります（.envファイルに保存されます）。")

elif app_mode == "ナレッジ検索":
    st.header("∷ ナレッジ検索")
    st.sidebar.header("ナレッジベース設定")
    kb_list_for_search_ui = list_knowledge_bases()
    kb_names_for_search_ui = [kb["name"] for kb in kb_list_for_search_ui]
    if kb_names_for_search_ui:
        all_kb_option_display_ui = "すべてのナレッジベース"
        options_for_select_ui = [all_kb_option_display_ui] + kb_names_for_search_ui
        current_selected_kb_option_val = st.session_state.get('selected_kb_option', all_kb_option_display_ui)
        if current_selected_kb_option_val not in options_for_select_ui:
            logger.warning(f"セッションのKBオプション '{current_selected_kb_option_val}' が見つかりません。デフォルトを使用します。")
            current_selected_kb_option_val = all_kb_option_display_ui
            st.session_state['selected_kb_option'] = current_selected_kb_option_val
        selected_kb_option_idx = options_for_select_ui.index(current_selected_kb_option_val)
        selected_kb_display_option_ui = st.sidebar.selectbox(
            "検索対象ナレッジベース", options_for_select_ui,
            index=selected_kb_option_idx,
            help="検索したいナレッジベースを選択してください。「すべて」を選択すると、ロード済みの全KBから横断検索します。"
        )
        st.session_state['selected_kb_option'] = selected_kb_display_option_ui
        if selected_kb_display_option_ui == all_kb_option_display_ui:
            st.session_state['selected_kbs'] = kb_names_for_search_ui
        else:
            st.session_state['selected_kbs'] = [selected_kb_display_option_ui]
        loaded_engine_names = [name for name, engine in st.session_state.get('search_engines', {}).items() if engine is not None]
        if loaded_engine_names:
            st.sidebar.caption(f"準備済エンジン: {', '.join(loaded_engine_names)}")
        else:
            st.sidebar.caption("検索エンジンは検索実行時に自動準備されます。")
    else:
        st.sidebar.warning("利用可能なナレッジベースがありません。「ナレッジ構築」モードでナレッジベースを作成してください。")
    st.sidebar.header("検索パラメータ")
    search_threshold_val_ui = st.sidebar.slider("検索類似度閾値", 0.0, 1.0, 0.15, 0.01, help="この値以上の類似度を持つ結果を表示します。低いほど多くの結果が出ますが、関連性が低いものも混ざります。")
    top_k_val_ui = st.sidebar.slider("最大検索結果数", 1, 20, 5, help="表示する検索結果の最大数。")
    generate_gpt_answer_flag_ui = st.sidebar.checkbox("検索結果からGPTで回答を要約生成する", value=True)
    if generate_gpt_answer_flag_ui:
        st.sidebar.markdown("##### GPT回答生成設定")
        persona_details_list_kb_gpt = get_persona_list()
        persona_map_kb_gpt = {p['name']: p['id'] for p in persona_details_list_kb_gpt}
        persona_display_names_kb_gpt = [p['name'] for p in persona_details_list_kb_gpt]
        current_persona_id_kb_gpt = st.session_state.get('persona', 'default')
        current_persona_name_display_kb_gpt = next((p['name'] for p in persona_details_list_kb_gpt if p['id'] == current_persona_id_kb_gpt), \
                                            persona_display_names_kb_gpt[0] if persona_display_names_kb_gpt else "標準アシスタント")
        selected_persona_name_kb_gpt_ui = st.sidebar.selectbox(
            "AIペルソナ", persona_display_names_kb_gpt, 
            index=persona_display_names_kb_gpt.index(current_persona_name_display_kb_gpt) if current_persona_name_display_kb_gpt in persona_display_names_kb_gpt else 0,
            key="persona_kb_gpt_selectbox"
        )
        st.session_state['persona'] = persona_map_kb_gpt[selected_persona_name_kb_gpt_ui]
        temp_kb_gpt_ui = st.sidebar.slider("温度", 0.0, 1.0, st.session_state.get('temperature', 0.3), 0.05, key="temp_kb_gpt", help="値が高いほど創造的、低いほど事実に基づいた回答。")
        st.session_state['temperature'] = temp_kb_gpt_ui
        resp_len_options_kb_gpt = ["簡潔", "普通", "詳細"]
        current_resp_len_kb_gpt = st.session_state.get('response_length', "普通")
        resp_len_idx_kb_gpt = resp_len_options_kb_gpt.index(current_resp_len_kb_gpt) if current_resp_len_kb_gpt in resp_len_options_kb_gpt else 1
        resp_len_kb_gpt_ui = st.sidebar.radio("応答の長さ", resp_len_options_kb_gpt, index=resp_len_idx_kb_gpt, key="resp_len_kb_gpt", horizontal=True)
        st.session_state['response_length'] = resp_len_kb_gpt_ui
    search_query_input_val = st.text_input("検索クエリを入力してください", placeholder="例: 最新の休暇制度について教えて")
    if st.button("◎ 検索実行", type="primary", key="search_execute_main_button", use_container_width=True):
        if not search_query_input_val.strip():
            st.warning("検索クエリを入力してください。")
        elif not st.session_state.get('selected_kbs'):
            st.warning("検索対象のナレッジベースが選択されていません。サイドバーで選択してください。")
        else:
            ready_to_search_kbs_list = []
            for kb_name_check_search in st.session_state['selected_kbs']:
                if get_search_engine(kb_name_check_search) is not None:
                    ready_to_search_kbs_list.append(kb_name_check_search)
                else:
                    st.warning(f"ナレッジベース '{kb_name_check_search}' の検索エンジン準備に失敗。このKBは検索対象外となります。")
            if not ready_to_search_kbs_list:
                 st.error("検索可能なナレッジベースがありません。")
            else:
                with st.spinner(f"「{search_query_input_val}」で検索中 (対象KB: {', '.join(ready_to_search_kbs_list)})..."):
                    client_for_search_and_gpt = get_openai_client()
                    if not client_for_search_and_gpt:
                        st.error("OpenAIクライアントの取得に失敗しました。検索およびGPT回答生成は実行できません。")
                    else:
                        search_results_data, not_found_flag_val = search_multiple_knowledge_bases(
                            search_query_input_val,
                            ready_to_search_kbs_list,
                            top_k=top_k_val_ui,
                            threshold=search_threshold_val_ui,
                            client=client_for_search_and_gpt
                        )
                if not_found_flag_val or not search_results_data:
                    st.warning(f"検索結果が見つかりませんでした。クエリを変えるか、類似度閾値を調整してみてください。")
                else:
                    st.success(f"{len(search_results_data)}件の関連情報が見つかりました（類似度閾値: {search_threshold_val_ui}）。")
                    if generate_gpt_answer_flag_ui and search_results_data:
                        with st.spinner("検索結果に基づいてGPTが回答を生成中..."):
                            search_context_str_gpt = f"ユーザーは「{search_query_input_val}」と質問しています。\n以下の検索結果情報を参照し、質問に対する回答を生成してください。\n\n"
                            for idx_res, res_item_gpt in enumerate(search_results_data):
                                meta_gpt = res_item_gpt.get('metadata', {}).get('meta_info', {})
                                search_context_str_gpt += f"【情報 {idx_res+1}】(出典KB: {res_item_gpt.get('kb_name', '不明')}, ファイル: {res_item_gpt.get('metadata',{}).get('filename', 'N/A')}, 類似度: {res_item_gpt.get('similarity',0):.2f})\n"
                                search_context_str_gpt += f"- 要約: {meta_gpt.get('summary', 'なし')}\n"
                                search_context_str_gpt += f"- GPT-mini用コンテキスト: {meta_gpt.get('mini_context', 'なし')}\n"
                            gpt_prompt_for_answer = f"""{search_context_str_gpt}
指示:
- 上記の【情報】セクションのみを参照し、ユーザーの質問に対する包括的で分かりやすい回答を作成してください。
- 自身の知識や外部情報で補完しないでください。
- 各情報を適切に組み合わせ、一つのスムーズな回答としてください。
- 関連性の低い情報は無視してください。
- 回答には、どの情報源（出典KB、ファイル名など）に基づいているかを明確に示してください。(例: 「KB 'X' のファイル 'Y' によると...」)
- 検索結果に適切な情報がない場合は、「提供された情報の中には、ご質問に直接回答できる内容は見つかりませんでした。」と回答してください。
ユーザーの質問: 「{search_query_input_val}」
回答:
"""
                            gpt_answer_text = st.session_state.chat_controller.generate_gpt_response(
                                gpt_prompt_for_answer, 
                                conversation_history=[],
                                persona=st.session_state['persona'],
                                temperature=st.session_state['temperature'],
                                response_length=st.session_state['response_length'],
                                client=client_for_search_and_gpt
                            )
                            st.markdown("---")
                            st.subheader("⟲ AIによる検索結果の要約回答")
                            stream_markdown(gpt_answer_text)
                            st.markdown("---")
                    st.subheader(f"詳細な検索結果 ({len(search_results_data)}件)")
                    for i_disp, res_detail_disp in enumerate(search_results_data):
                        similarity_disp = res_detail_disp.get('similarity', 0)
                        kb_name_disp = res_detail_disp.get('kb_name', '不明')
                        file_name_disp = res_detail_disp.get('metadata',{}).get('filename', 'N/A')
                        chunk_id_disp = res_detail_disp.get('metadata',{}).get('id', 'N/A')
                        expander_title = f"結果 {i_disp+1}: {file_name_disp} (ID: {chunk_id_disp}) / KB: {kb_name_disp} / 類似度: {similarity_disp:.3f}"
                        with st.expander(expander_title):
                            meta_info_disp = res_detail_disp.get('metadata', {}).get('meta_info', {})
                            st.markdown(f"**要約:** {meta_info_disp.get('summary', 'なし')}")
                            if meta_info_disp.get('mini_context'):
                                st.markdown(f"**GPT-mini用コンテキスト:** {meta_info_disp.get('mini_context')}")
                            st.text_area(f"チャンク内容 (結果 {i_disp+1})", res_detail_disp.get('text', ''), height=200, key=f"search_res_text_{i_disp}")
                            with st.popover("詳細メタデータを見る...", use_container_width=True):
                                st.json(meta_info_disp, expanded=True)
                        time.sleep(0.05)
    with st.expander("💡 ナレッジ検索の使い方とヒント"):
        st.markdown("""
        - **検索対象の選択**: サイドバーで検索したいナレッジベースを選択します。「すべて」を選ぶと横断検索が可能です。
        - **クエリ入力**: 具体的なキーワードや自然な質問文で検索できます。
        - **AI要約回答**: 「GPTで回答を要約生成する」をオンにすると、検索結果を元にAIが回答をまとめてくれます。
        - **類似度閾値**: この値を調整することで、検索結果の厳密さを変更できます。低くするとより多くの結果が、高くするとより関連性の高い（と思われる）結果が得られます。
        - **結果の確認**: 各検索結果はクリックで詳細（チャンク内容やメタデータ）を確認できます。
        - **検索エンジン**: 検索実行時に、対象KBの検索エンジンが自動的に準備・利用されます。
        """)

elif app_mode == "FAQ作成":
    st.header("FAQ作成モード")

    kb_list_for_faq = list_knowledge_bases()
    kb_names_for_faq = [kb["name"] for kb in kb_list_for_faq]
    if not kb_names_for_faq:
        st.info("利用可能なナレッジベースがありません。まず『ナレッジ構築』モードで作成してください。")
    else:
        current_faq_kb = st.session_state.get("faq_kb_name", kb_names_for_faq[0])
        if current_faq_kb not in kb_names_for_faq:
            current_faq_kb = kb_names_for_faq[0]
        selected_kb = st.selectbox("対象ナレッジベース", kb_names_for_faq, index=kb_names_for_faq.index(current_faq_kb))
        st.session_state["faq_kb_name"] = selected_kb

        st.sidebar.header("FAQ生成設定")
        max_tokens = st.sidebar.number_input(
            "最大トークン数",
            min_value=100,
            max_value=4000,
            value=int(st.session_state.get("faq_max_tokens", 1000)),
            step=100,
        )
        st.session_state["faq_max_tokens"] = int(max_tokens)
        num_pairs = st.sidebar.number_input(
            "Q&A数",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("faq_num_pairs", 3)),
            step=1,
        )
        st.session_state["faq_num_pairs"] = int(num_pairs)

        if st.button("◎ FAQを生成", key="generate_faqs_btn", type="primary"):
            client = get_openai_client()
            if not client:
                st.error("OpenAIクライアントの取得に失敗しました。")
            else:
                with st.spinner("FAQを生成中..."):
                    count = generate_faqs_from_chunks(selected_kb, int(max_tokens), int(num_pairs), client=client)
                    refresh_search_engine(selected_kb)
                st.success(f"{count}件のFAQを生成しました。")

elif app_mode == "chatGPT":
    st.header(f"⟐ chatGPT - 現在の会話: {st.session_state.get('gpt_conversation_title', '新しい会話')}")
    
    st.sidebar.header("チャット設定")
    
    persona_details_list_chat_ui = get_persona_list()
    persona_map_chat_ui = {p['name']: p['id'] for p in persona_details_list_chat_ui}
    persona_display_names_chat_ui = [p['name'] for p in persona_details_list_chat_ui]

    current_persona_id_chat_val = st.session_state.get('persona', 'default')
    default_persona_name_display = persona_display_names_chat_ui[0] if persona_display_names_chat_ui else "標準アシスタント（リスト空）"
    current_persona_name_display_chat_val = next((p['name'] for p in persona_details_list_chat_ui if p['id'] == current_persona_id_chat_val), default_persona_name_display)
    
    selected_persona_idx_chat = 0
    if persona_display_names_chat_ui:
        if current_persona_name_display_chat_val in persona_display_names_chat_ui:
            selected_persona_idx_chat = persona_display_names_chat_ui.index(current_persona_name_display_chat_val)
    
    selected_persona_name_chat_selected_ui = st.sidebar.selectbox(
        "AIペルソナ", persona_display_names_chat_ui, 
        index=selected_persona_idx_chat,
        key="persona_chat_selectbox_main",
        help="AIの応答スタイルや性格を選択します。"
    )
    if persona_display_names_chat_ui:
        st.session_state['persona'] = persona_map_chat_ui[selected_persona_name_chat_selected_ui]

    persona_data_chat_disp = load_persona(st.session_state['persona'])
    with st.sidebar.expander(f"ペルソナ「{persona_data_chat_disp.get('name')}」の詳細を見る", expanded=False):
        st.write(f"**説明:** {persona_data_chat_disp.get('description')}")
        st.caption(f"**システムプロンプト:** {persona_data_chat_disp.get('system_prompt')[:100]}...")
    
    temp_chat_val_ui = st.sidebar.slider("温度 (創造性)", 0.0, 1.0, st.session_state.get('temperature', 0.7), 0.05, key="temp_chat_main", help="高いほど多様な応答、低いほど決まった応答。")
    st.session_state['temperature'] = temp_chat_val_ui
    
    resp_len_options_chat_ui = ["簡潔", "普通", "詳細"]
    current_resp_len_chat_val = st.session_state.get('response_length', "普通")
    resp_len_idx_chat_val = resp_len_options_chat_ui.index(current_resp_len_chat_val) if current_resp_len_chat_val in resp_len_options_chat_ui else 1
    resp_len_chat_selected_ui = st.sidebar.radio("応答の長さ", resp_len_options_chat_ui, index=resp_len_idx_chat_val, key="resp_len_chat_main", horizontal=True)
    st.session_state['response_length'] = resp_len_chat_selected_ui
    
    # メッセージ履歴の表示
    chat_container = st.container(height=500, border=False)
    with chat_container:
        for message_item in st.session_state.get('gpt_messages', []):
            with st.chat_message(message_item["role"]):
                st.markdown(message_item["content"])
    
    # ユーザー入力
    user_chat_input_val = st.chat_input(
        "メッセージを送信 (Shift+Enterで改行)", 
        key=f"chat_input_main_gpt_{st.session_state.get('gpt_conversation_id','default_conv_id')}"
    )
    
    if user_chat_input_val:
        # 1. ユーザーメッセージをセッションに追加
        st.session_state.gpt_messages.append({"role": "user", "content": user_chat_input_val})
        logger.info(f"User input received: {user_chat_input_val}")
        
        # 2. アシスタントの応答を「同じスクリプト実行サイクル内」で生成
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⟲ 考え中...")

            history_for_gpt_send = [
                msg for msg in st.session_state.gpt_messages[:-1] # 最新のユーザーメッセージを除外
                if msg["role"] in ["user", "assistant"]
            ]
            
            client_for_chat_resp = get_openai_client() 
            gpt_response_text_val = ""
            if not client_for_chat_resp:
                gpt_response_text_val = "エラー: OpenAIクライアントに接続できません。"
                placeholder.error(gpt_response_text_val)
            else:
                try:
                    gpt_response_text_val = st.session_state.chat_controller.generate_gpt_response(
                        user_input=user_chat_input_val, 
                        conversation_history=history_for_gpt_send,
                        persona=st.session_state['persona'],
                        temperature=st.session_state['temperature'],
                        response_length=st.session_state['response_length'],
                        client=client_for_chat_resp
                    )
                    placeholder.markdown(gpt_response_text_val)
                except Exception as e:
                    logger.error(f"GPT応答生成中にエラー: {e}", exc_info=True)
                    gpt_response_text_val = f"申し訳ありません、応答の生成中にエラーが発生しました。"
                    placeholder.error(gpt_response_text_val)
            
            # 3. アシスタントの応答をセッションに追加
            if gpt_response_text_val:
                 st.session_state.gpt_messages.append({"role": "assistant", "content": gpt_response_text_val})
            
        # 4. 会話タイトル生成/更新
        if len(st.session_state.gpt_messages) >= 4 and client_for_chat_resp:
            current_history_for_title_gen = [m for m in st.session_state.gpt_messages if m["role"] in ["user", "assistant"]]
            if current_history_for_title_gen:
                try:
                    new_title_val = st.session_state.chat_controller.generate_conversation_title(current_history_for_title_gen, client_for_chat_resp)
                    if new_title_val != st.session_state.get('gpt_conversation_title'):
                        st.session_state.gpt_conversation_title = new_title_val
                        logger.info(f"会話タイトルを更新: {new_title_val}")
                except Exception as e:
                    logger.error(f"会話タイトル生成エラー: {e}", exc_info=True)

        # 5. 会話保存
        save_conversation(
            st.session_state['gpt_conversation_id'],
            messages=st.session_state.gpt_messages,
            title=st.session_state.get('gpt_conversation_title', "新しい会話")
        )
        
        # 6. 全ての処理が終わった後、UIを更新するために一度だけ rerun
        st.rerun()

    # サイドバーの操作ボタン
    if st.sidebar.button("◎ 新しい会話を開始する", key="new_chat_button_main", type="secondary", use_container_width=True):
        st.session_state.gpt_messages = []
        st.session_state.gpt_conversation_id = str(uuid.uuid4())
        st.session_state.gpt_conversation_title = "新しい会話"
        st.rerun() 
    
    with st.sidebar.expander("過去の会話を管理する"):
        saved_convs_list_ui = list_conversations()
        if saved_convs_list_ui:
            conv_options_display_load = [f"{conv.get('title', '会話')} ({conv['id'][-6:]} - {conv.get('date', '')[:10]})" for conv in saved_convs_list_ui]
            conv_map_load_ids = {display_str: conv['id'] for display_str, conv in zip(conv_options_display_load, saved_convs_list_ui)}
            selected_conv_display_str_load = st.selectbox("読み込む会話を選択", conv_options_display_load, index=None, placeholder="過去の会話を選択...")
            if selected_conv_display_str_load and st.button("⟲ この会話を読み込む", key="load_chat_button_main", use_container_width=True):
                conv_id_to_load_val = conv_map_load_ids[selected_conv_display_str_load]
                loaded_msgs_val, loaded_title_val = load_conversation(conv_id_to_load_val)
                if loaded_msgs_val is not None: 
                    st.session_state.gpt_messages = loaded_msgs_val
                    st.session_state.gpt_conversation_id = conv_id_to_load_val
                    st.session_state.gpt_conversation_title = loaded_title_val 
                    st.success(f"会話「{loaded_title_val}」を読み込みました。")
                    st.rerun()
                else:
                    st.error("会話の読み込みに失敗しました。ファイルが破損している可能性があります。")
        else:
            st.write("保存された会話はまだありません。")
            
    with st.sidebar.expander("💡 chatGPTの使い方ヒント"):
        st.markdown("""
        - **会話開始**: 下の入力欄にメッセージをタイプしてEnterキーを押します。
        - **ペルソナ変更**: サイドバーでAIの性格や専門性を変更できます。
        - **応答調整**: 「温度」でAIの創造性を、「応答の長さ」で詳細度を調整できます。
        - **新規会話**: 「新しい会話を開始する」ボタンで履歴をリセットできます。
        - **過去の会話**: 「過去の会話を管理する」から以前のチャットを読み込めます。
        """)

st.markdown("---")
st.markdown('<div class="footer-text">RAGシステム統合ツール v1.1.0 - AIによる知識活用を支援</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    os.chdir(current_dir)
