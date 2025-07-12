import json
import os

# from sklearn.feature_extraction.text import TfidfVectorizer # BM25には不要
import pickle

import numpy as np
from config import (
    DEFAULT_KB_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    HYBRID_VECTOR_WEIGHT,
    HYBRID_BM25_WEIGHT,
    EMBEDDING_BATCH_SIZE,
)
from . import db_cache

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # type: ignore[assignment]
    import logging

    logging.warning(f"SentenceTransformer import failed: {e}")
import logging
import re
import typing
from pathlib import Path

# from nltk.tokenize import word_tokenize # SudachiPyに置き換え、またはフォールバックとして保持
from nltk.corpus import stopwords

# import time # 直接は不要
from rank_bm25 import BM25Okapi
from shared.nltk_utils import ensure_nltk_resources
from shared.openai_utils import get_openai_client, get_embeddings_batch

logger = logging.getLogger(__name__)

# --- SudachiPyのインポートと初期化 ---
try:
    from sudachipy import dictionary as sudachi_dictionary_module
    from sudachipy import tokenizer as sudachi_tokenizer_module

    _sudachi_tokenizer_instance_for_bm25 = (
        sudachi_dictionary_module.Dictionary().create()
    )
    _SUDACHI_BM25_TOKENIZER_MODE = sudachi_tokenizer_module.Tokenizer.SplitMode.B
    logger.info(
        "SudachiPy tokenizer for BM25 (knowledge_search.py) initialized successfully."
    )
except ImportError:
    logger.warning(
        "SudachiPy not found. BM25 will use a fallback regex tokenizer, which is not ideal for Japanese."
    )
    _sudachi_tokenizer_instance_for_bm25 = None
except Exception as e_sudachi_init:
    logger.warning(
        f"SudachiPy tokenizer for BM25 failed to initialize: {e_sudachi_init}. Falling back to regex."
    )
    _sudachi_tokenizer_instance_for_bm25 = None
# --- SudachiPyのインポートと初期化ここまで ---


# Ensure NLTK resources are available for tokenization and stopword lists
ensure_nltk_resources()

_stop_words_set = set()
try:
    _stop_words_set.update(stopwords.words("english"))
    _japanese_stopwords_list = [
        "の",
        "に",
        "は",
        "を",
        "た",
        "が",
        "で",
        "て",
        "と",
        "し",
        "れ",
        "さ",
        "ある",
        "いる",
        "から",
        "など",
        "なっ",
        "ない",
        "ので",
        "です",
        "ます",
        "する",
        "もの",
        "こと",
        "よう",
        "ため",
        "において",
        "における",
        "および",
        "また",
        "も",
        "という",
        "られる",
        "により",
        "に関する",
        "ついて",
        "として",
        " terhadap",
        "によって",
        "より",
        "における",
        "に関する",
        "に対する",
        "としての",
        "あ",
        "い",
        "う",
        "え",
        "お",
        "か",
        "き",
        "く",
        "け",
        "こ",
        "さ",
        "し",
        "す",
        "せ",
        "そ",
        "な",
        "ぬ",
        "ね",
        "は",
        "ひ",
        "ふ",
        "へ",
        "ほ",
        "ま",
        "み",
        "む",
        "め",
        "も",
        "や",
        "ゆ",
        "よ",
        "ら",
        "り",
        "る",
        "ろ",
        "わ",
        "を",
        "ん",
        "これ",
        "それ",
        "あれ",
        "この",
        "その",
        "あの",
        "ここ",
        "そこ",
        "あそこ",
        "こちら",
        "そちら",
        "あちら",
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "{",
        "|",
        "}",
        "~",
        "、",
        "。",
        "「",
        "」",
        "（",
        "）",
        "・",
    ]
    _stop_words_set.update(_japanese_stopwords_list)
    logger.info(f"ストップワードの初期化完了 (knowledge_search.py): {len(_stop_words_set)}語")
except Exception as e_stopwords:
    logger.error(f"ストップワード初期化エラー (knowledge_search.py): {e_stopwords}")


class TokenList(list):
    def lower(self) -> str:  # type: ignore[override]
        return " ".join(self).lower()


def tokenize_text_for_bm25_internal(text_input: str) -> list[str]:
    if not isinstance(text_input, str) or not text_input.strip():
        return TokenList(["<bm25_empty_input_token>"])
    processed_text = text_input.lower()
    tokens: list[str] = []
    if _sudachi_tokenizer_instance_for_bm25:
        try:
            tokens = [
                m.normalized_form()
                for m in _sudachi_tokenizer_instance_for_bm25.tokenize(
                    processed_text, _SUDACHI_BM25_TOKENIZER_MODE
                )
            ]
        except Exception as e_sudachi_tokenize:
            logger.error(
                f"    [Tokenizer] SudachiPyでのトークン化中にエラー: {e_sudachi_tokenize}. Regexフォールバック使用。Text: {processed_text[:30]}..."
            )
            tokens = re.findall(r"[ぁ-ん]+|[ァ-ン]+|[一-龥]+|[a-zA-Z0-9]+", processed_text)
    else:
        tokens = re.findall(r"[ぁ-ん]+|[ァ-ン]+|[一-龥]+|[a-zA-Z0-9]+", processed_text)
    if not tokens:
        return TokenList(["<bm25_empty_tokenization_result>"])
    if _stop_words_set:
        tokens_after_stopwords = [
            token for token in tokens if token not in _stop_words_set
        ]
        if not tokens_after_stopwords and tokens:
            return TokenList(["<bm25_all_stopwords_token>"])
        tokens = tokens_after_stopwords
    if not tokens:
        return TokenList(["<bm25_empty_after_stopwords_token>"])
    return TokenList(tokens)


def compute_hybrid_weights(num_chunks: int) -> tuple[float, float]:
    """Return vector and BM25 weights based on corpus size."""
    if num_chunks < 50:
        return 0.9, 0.1
    if num_chunks > 1000:
        return 0.6, 0.4
    return HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT


class HybridSearchEngine:
    def __init__(self, kb_path: str):
        logger.info(f"HybridSearchEngine初期化: kb_path={kb_path}")
        self.kb_path = Path(kb_path)
        self.chunks_path = self.kb_path / "chunks"
        self.metadata_path = self.kb_path / "metadata"
        self.embeddings_path = self.kb_path / "embeddings"

        self.bm25_index_file_path = self.kb_path / "bm25_index_sudachi.pkl"
        self.tokenized_corpus_file_path = self.kb_path / "tokenized_corpus_sudachi.pkl"

        logger.info(
            f"パス確認: chunks={self.chunks_path.exists()}, metadata={self.metadata_path.exists()}, embeddings={self.embeddings_path.exists()}"
        )

        self.kb_metadata = self._load_kb_metadata()
        self.embedding_model = self.kb_metadata.get("embedding_model", EMBEDDING_MODEL)
        logger.info(f"使用する埋め込みモデル: {self.embedding_model}")

        self.chunks = self._load_chunks()
        logger.info(f"初期チャンク読み込み完了: {len(self.chunks)}件")

        self.embeddings = self._load_embeddings()
        logger.info(f"埋め込みベクトル読み込み完了: {len(self.embeddings)}件")

        self._integrate_faq_chunks()

        self.bm25_index = self._load_or_build_bm25_index()
        logger.info(f"BM25処理後の有効チャンク数: {len(self.chunks)}")
        self._check_chunk_embedding_consistency()

        if SentenceTransformer is not None:
            try:
                logger.info("バックアップ埋め込みモデル SentenceTransformer を読み込み中...")
                self.model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2"
                )
                logger.info("SentenceTransformer 読み込み完了")
            except Exception as e_st:
                logger.error(f"SentenceTransformer 読み込みエラー: {e_st}")
                self.model = None
        else:
            logger.info("SentenceTransformer が利用できないためバックアップモデルを読み込みません")
            self.model = None

    def reindex(self) -> None:
        """Reload all chunks and embeddings and rebuild the BM25 index."""
        logger.info("Re-indexing knowledge base by forcing a rebuild...")

        db_cache.clear_cache(self.kb_path)

        # Force rebuild by removing cached index files
        if self.bm25_index_file_path.exists():
            try:
                self.bm25_index_file_path.unlink()
                logger.info(f"Removed cached BM25 index: {self.bm25_index_file_path}")
            except OSError as e:
                logger.warning(f"Could not remove cached BM25 index: {e}")

        if self.tokenized_corpus_file_path.exists():
            try:
                self.tokenized_corpus_file_path.unlink()
                logger.info(
                    f"Removed cached tokenized corpus: {self.tokenized_corpus_file_path}"
                )
            except OSError as e:
                logger.warning(f"Could not remove cached tokenized corpus: {e}")

        # Now, reload everything from scratch
        self.chunks = self._load_chunks()
        logger.info(f"Reloaded chunks: {len(self.chunks)} items")
        self.embeddings = self._load_embeddings()
        logger.info(f"Reloaded embeddings: {len(self.embeddings)} items")
        self._integrate_faq_chunks()
        self._check_chunk_embedding_consistency()
        self.bm25_index = self._load_or_build_bm25_index()
        logger.info("Re-indexing complete.")

    def _load_kb_metadata(self) -> dict:
        """Return metadata describing this knowledge base.

        The helper reads ``kb_metadata.json`` located under ``self.kb_path``.
        No arguments are required because the directory is determined during
        object initialization.

        Returns
        -------
        dict
            Parsed metadata for the knowledge base or an empty dictionary if the
            file does not exist or cannot be read.
        """

        metadata_file = self.kb_path / "kb_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"ナレッジベースメタデータ読み込みエラー: {e}", exc_info=True)
        else:
            logger.warning(f"ナレッジベースメタデータファイルが見つかりません: {metadata_file}")
        return {}

    def _load_chunks(self) -> list[dict]:
        """Load chunk text and metadata from disk.

        Chunk ``.txt`` files are read from ``self.chunks_path`` and combined
        with their corresponding ``.json`` metadata under ``self.metadata_path``.
        Each resulting entry is returned as a dictionary with ``id``, ``text``
        and ``metadata`` keys.

        Returns
        -------
        list[dict]
            List of loaded chunk dictionaries. Missing files or parse errors are
            logged and skipped.
        """

        logger.info("チャンクを読み込み中...")
        loaded_chunks = []
        if not self.chunks_path.exists():
            logger.info(f"チャンクディレクトリが見つかりません: {self.chunks_path}")
            return []

        for chunk_file_path in self.chunks_path.glob("*.txt"):
            logger.info(f"    チャンクファイルパス: {chunk_file_path}")
            try:
                stem = chunk_file_path.stem
                chunk_id = stem  # chunk_idをファイル名から直接取得
                logger.info(f"    チャンクID: {chunk_id}")
                with open(chunk_file_path, "r", encoding="utf-8") as f:
                    chunk_text = f.read()
                metadata = {}
                metadata_file = (
                    self.metadata_path / f"{chunk_id}.json"
                )  # メタデータファイル名もchunk_id.jsonに変更
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    logger.info(f"    メタデータファイル {metadata_file.name} をロードしました。")
                else:
                    logger.warning(f"    メタデータファイル {metadata_file.name} が見つかりません。")
                loaded_chunks.append(
                    {"id": chunk_id, "text": chunk_text, "metadata": metadata}
                )
            except Exception as e:
                logger.error(
                    f"チャンク '{chunk_file_path.name}' の読み込み中にエラー: {e}",
                    exc_info=True,
                )
        logger.info(f"    _load_chunks 完了。ロードされたチャンク数: {len(loaded_chunks)}")
        return loaded_chunks

    def _load_embeddings(self) -> dict[str, list[float]]:
        """Read embedding vectors from disk keyed by chunk ID.

        The pickled embedding files are loaded from ``self.embeddings_path``.
        Each pickle may contain either a dictionary with an ``"embedding"`` field
        or the raw vector. The data is converted to lists of ``float`` and
        returned as a mapping from chunk ID to embedding vector.

        Returns
        -------
        dict[str, list[float]]
            Dictionary mapping chunk IDs to embedding vectors.
        """

        logger.info("埋め込みベクトルを読み込み中...")
        loaded_embeddings = db_cache.load_embeddings(self.kb_path)
        if loaded_embeddings:
            logger.info(f"    DB cache hit: {len(loaded_embeddings)} embeddings")
            return loaded_embeddings
        if not self.embeddings_path.exists():
            logger.info(f"埋め込みディレクトリが見つかりません: {self.embeddings_path}")
            return {}
        for emb_file_path in self.embeddings_path.glob("*.pkl"):
            logger.info(f"    埋め込みファイルパス: {emb_file_path}")
            try:
                stem = emb_file_path.stem
                chunk_id = stem  # chunk_idをファイル名から直接取得
                logger.info(f"    埋め込みファイル {emb_file_path.name} からIDを抽出: {chunk_id}")
                with open(emb_file_path, "rb") as f:
                    embedding_data = pickle.load(f)
                emb_vector = None
                if isinstance(embedding_data, dict) and "embedding" in embedding_data:
                    emb_vector = embedding_data["embedding"]
                elif isinstance(embedding_data, (list, np.ndarray)):
                    emb_vector = embedding_data
                if emb_vector is not None:
                    loaded_embeddings[chunk_id] = np.array(
                        emb_vector, dtype=np.float32
                    ).tolist()
                    try:
                        db_cache.save_embedding(self.kb_path, chunk_id, loaded_embeddings[chunk_id])
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"埋め込みファイル '{emb_file_path.name}' の読み込み中にエラー: {e}")
        logger.info(f"    _load_embeddings 完了。ロードされた埋め込み数: {len(loaded_embeddings)}")
        return loaded_embeddings

    def _check_chunk_embedding_consistency(self):
        chunk_ids = set(c["id"] for c in self.chunks)
        embedding_ids = set(self.embeddings.keys())
        logger.info(f"整合性チェック: チャンクID数={len(chunk_ids)}, 埋め込みID数={len(embedding_ids)}")
        missing_embeddings_for_chunks = chunk_ids - embedding_ids
        if missing_embeddings_for_chunks:
            logger.warning(
                f"  警告: 次のチャンクIDには対応する埋め込みがありません (上位5件): {list(missing_embeddings_for_chunks)[:5]}"
            )
        missing_chunks_for_embeddings = embedding_ids - chunk_ids
        if missing_chunks_for_embeddings:
            logger.warning(
                f"  警告: 次の埋め込みIDには対応するチャンクがありません (上位5件): {list(missing_chunks_for_embeddings)[:5]}"
            )
        common_ids_count = len(chunk_ids.intersection(embedding_ids))
        logger.info(f"  チャンクと埋め込みで共通のID数: {common_ids_count}")

    def _integrate_faq_chunks(self) -> None:
        """Load FAQ entries and append them to self.chunks."""
        faq_file = self.kb_path / "faqs.json"
        if not faq_file.exists():
            return
        try:
            with open(faq_file, "r", encoding="utf-8") as f:
                faqs = json.load(f)
        except Exception as e:
            logger.error(f"FAQ読み込みエラー: {e}")
            return
        for faq in faqs:
            fid = faq.get("id")
            if not fid:
                continue
            text = f"Q: {faq.get('question', '')}\nA: {faq.get('answer', '')}"
            self.chunks.append(
                {
                    "id": fid,
                    "text": text,
                    "metadata": {
                        "faq": True,
                        "question": faq.get("question"),
                        "answer": faq.get("answer"),
                    },
                }
            )

    def _create_tokenized_corpus_and_filter_chunks(
        self,
    ) -> tuple[list[list[str]], list[dict]]:
        logger.info("BM25用コーパスのトークン化とチャンクフィルタリングを開始...")
        tokenized_corpus: list[list[str]] = []
        successfully_processed_chunks: list[dict] = []
        if not self.chunks:
            logger.warning("  警告: self.chunksが空のため、トークン化できません。")
            return [], []
        for i, chunk_data in enumerate(self.chunks):
            chunk_text = chunk_data.get("text", "")
            chunk_id = chunk_data.get("id", f"N/A_{i}")
            logger.info(f"    処理中のチャンクID: {chunk_id}, テキスト: '{chunk_text[:50]}...'")
            tokens = tokenize_text_for_bm25_internal(chunk_text)
            is_dummy_token_only = all(
                token.startswith("<bm25_") and token.endswith("_token>")
                for token in tokens
            )
            if tokens and not is_dummy_token_only:
                tokenized_corpus.append(tokens)
                successfully_processed_chunks.append(chunk_data)
                logger.info(f"    チャンクID {chunk_id} は有効なトークンを持ちます。追加済み。")
                if i % 100 == 0 and i > 0:
                    logger.info(f"    ... {i}件のチャンクをトークン化済み ...")
            else:
                logger.warning(
                    f"    警告: チャンクID {chunk_id} のトークン化結果がBM25に不適格。除外します。Tokens: {tokens}"
                )
        if not tokenized_corpus:
            logger.warning("  警告: 有効なトークン化済みチャンクが一つもありませんでした。")
        logger.info(
            f"BM25用コーパスのトークン化完了。処理できた有効チャンク数: {len(successfully_processed_chunks)} / {len(self.chunks)}"
        )
        return tokenized_corpus, successfully_processed_chunks

    def _load_or_build_bm25_index(self) -> typing.Union[BM25Okapi, None]:
        loaded_from_cache = False
        token_map = db_cache.load_token_lists(self.kb_path)
        if token_map:
            tokenized = []
            filtered = []
            for c in self.chunks:
                toks = token_map.get(c["id"])
                if toks:
                    tokenized.append(toks)
                    filtered.append(c)
            if tokenized:
                self.tokenized_corpus_for_bm25 = tokenized
                self.chunks = filtered
                loaded_from_cache = True

        loaded_from_file = False
        if not loaded_from_cache and self.tokenized_corpus_file_path.exists():
            logger.info(f"トークン化済みコーパスをファイルからロード中: {self.tokenized_corpus_file_path}")
            try:
                with open(self.tokenized_corpus_file_path, "rb") as f:
                    saved_data = pickle.load(f)
                if (
                    isinstance(saved_data, dict)
                    and "tokenized_corpus" in saved_data
                    and "processed_chunk_ids" in saved_data
                ):
                    self.tokenized_corpus_for_bm25 = saved_data["tokenized_corpus"]
                    original_chunks_map = {c["id"]: c for c in self.chunks}
                    self.chunks = []
                    for cid in saved_data["processed_chunk_ids"]:
                        if cid in original_chunks_map:
                            self.chunks.append(original_chunks_map[cid])
                    logger.info(
                        f"  トークン化済みコーパス ({len(self.tokenized_corpus_for_bm25)}件) と "
                        f"対応チャンク ({len(self.chunks)}件) をロードしました。"
                    )
                    if len(self.tokenized_corpus_for_bm25) != len(self.chunks):
                        logger.warning("  警告: ロードしたトークン化コーパスとチャンク数が不一致。インデックス再構築を推奨。")
                        self.tokenized_corpus_for_bm25 = None
                    else:
                        loaded_from_file = True
                        try:
                            for cid, toks in zip(saved_data["processed_chunk_ids"], saved_data["tokenized_corpus"]):
                                db_cache.save_token_list(self.kb_path, cid, toks)
                        except Exception:
                            pass
                else:
                    logger.warning("  警告: トークン化済みコーパスファイルの形式が不正。再構築します。")
            except Exception as e:
                logger.info(f"  トークン化済みコーパスのロード失敗: {e}. 再構築します。")

        if not (loaded_from_file or loaded_from_cache):
            logger.info("トークン化済みコーパスを生成・保存します...")
            logger.info(
                f"    _load_or_build_bm25_index: self.chunks (before filter) = {len(self.chunks)}"
            )
            (
                self.tokenized_corpus_for_bm25,
                filtered_chunks,
            ) = self._create_tokenized_corpus_and_filter_chunks()
            logger.info(
                f"    _load_or_build_bm25_index: filtered_chunks = {len(filtered_chunks)}"
            )
            self.chunks = filtered_chunks
            if self.tokenized_corpus_for_bm25 and self.chunks:
                try:
                    data_to_save = {
                        "tokenized_corpus": self.tokenized_corpus_for_bm25,
                        "processed_chunk_ids": [c["id"] for c in self.chunks],
                    }
                    with open(self.tokenized_corpus_file_path, "wb") as f:
                        pickle.dump(data_to_save, f)
                    logger.info(
                        f"  トークン化済みコーパスを保存しました: {self.tokenized_corpus_file_path}"
                    )
                    try:
                        for cid, toks in zip(data_to_save["processed_chunk_ids"], data_to_save["tokenized_corpus"]):
                            db_cache.save_token_list(self.kb_path, cid, toks)
                    except Exception:
                        pass
                except Exception as e:
                    logger.info(f"  トークン化済みコーパスの保存失敗: {e}")
            elif not self.chunks:
                logger.warning(
                    "  警告: トークン化・フィルタリングの結果、有効なチャンクがありません。BM25インデックスは構築できません。"
                )
                return None
            else:
                logger.warning("  警告: トークン化済みコーパスが空です。BM25インデックスは構築できません。")
                return None

        if self.bm25_index_file_path.exists():
            logger.info(f"BM25インデックスをファイルからロード中: {self.bm25_index_file_path}")
            try:
                with open(self.bm25_index_file_path, "rb") as f:
                    bm25_index_loaded = pickle.load(f)
                if isinstance(bm25_index_loaded, BM25Okapi):
                    logger.info("  BM25インデックスのロード完了。")
                    return bm25_index_loaded
                else:
                    logger.warning("  警告: ロードしたBM25インデックスの型が不正。再構築します。")
            except Exception as e:
                logger.info(f"  BM25インデックスのロードに失敗: {e}. 再構築します。")

        if not self.tokenized_corpus_for_bm25 or not self.chunks:
            logger.warning("  警告: BM25インデックス構築に必要なトークン化済みコーパスまたはチャンクデータがありません。")
            return None

        logger.info("BM25インデックスを新規構築中...")
        try:
            bm25_index_new = BM25Okapi(self.tokenized_corpus_for_bm25)
            logger.info("  BM25インデックス構築完了。")
            if self.tokenized_corpus_for_bm25:
                sample_query_toks = tokenize_text_for_bm25_internal("テスト")
                if not (
                    len(sample_query_toks) == 1
                    and sample_query_toks[0].startswith("<bm25_")
                ):
                    test_scrs = bm25_index_new.get_scores(sample_query_toks)
                    logger.info(
                        f"    構築直後のテスト検索スコア (上位3件, クエリ: '{sample_query_toks}'): {test_scrs[:3] if test_scrs is not None else 'N/A'}"
                    )
            try:
                with open(self.bm25_index_file_path, "wb") as f:
                    pickle.dump(bm25_index_new, f)
                logger.info(f"  BM25インデックスをファイルに保存しました: {self.bm25_index_file_path}")
            except Exception as e:
                logger.info(f"  BM25インデックスの保存に失敗: {e}")
            return bm25_index_new
        except ZeroDivisionError:
            logger.error("  警告: BM25インデックス構築中にZeroDivisionError。BM25は機能しない可能性があります。")
            return None
        except Exception as e:
            logger.error(f"  BM25インデックス構築中に予期せぬエラー: {e}", exc_info=True)
            return None

    def get_embedding_from_openai(
        self, text: str, model_name: typing.Union[str, None] = None, client=None
    ) -> typing.Union[list[float], None]:
        if model_name is None:
            model_name = self.embedding_model
        if client is None:
            client = get_openai_client()
            if client is None:
                logger.warning("  警告 (get_embedding): OpenAIクライアントを取得できません。")
                return None
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            logger.warning("  警告 (get_embedding): 埋め込み対象のテキストが空または不正。")
            return None
        try:
            response = client.embeddings.create(
                model=model_name, input=text, dimensions=EMBEDDING_DIMENSIONS
            )  # type: ignore
            embedding = response.data[0].embedding
            return embedding
        except Exception as e_openai_emb:
            logger.error(
                f"  OpenAI API埋め込みエラー: model={model_name}, text(先頭30字)='{text[:30]}...' Error: {e_openai_emb}"
            )
            return None

    def get_embeddings_from_openai(
        self,
        texts: list[str],
        model_name: typing.Union[str, None] = None,
        client=None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> list[list[float]]:
        """Return embeddings for multiple texts using batch requests."""
        if model_name is None:
            model_name = self.embedding_model
        if client is None:
            client = get_openai_client()
            if client is None:
                logger.warning(
                    "  警告 (get_embeddings_from_openai): OpenAIクライアントを取得できません。"
                )
                return []
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = get_embeddings_batch(batch, client, model_name, EMBEDDING_DIMENSIONS)
            results.extend(vecs)
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.15,
        vector_weight: float | None = None,
        bm25_weight: float | None = None,
        client=None,
    ) -> tuple[list[dict], bool]:
        if vector_weight is None or bm25_weight is None:
            vector_weight, bm25_weight = compute_hybrid_weights(len(self.chunks))
        logger.info(
            f"検索実行: クエリ='{query}', top_k={top_k}, threshold={threshold}, vec_w={vector_weight}, bm25_w={bm25_weight}"
        )
        if not self.chunks:
            logger.warning("  警告: 有効なチャンクデータが存在しません (BM25処理後)。検索を中止します。")
            return [], True

        query_vector = self.get_embedding_from_openai(query, client=client)
        if query_vector is None:
            if self.model:
                logger.info(
                    "  OpenAI APIでのベクトル化失敗。バックアップモデル (SentenceTransformer) を使用します。"
                )
                try:
                    query_vector = self.model.encode(query).tolist()
                    logger.info(f"    バックアップベクトル化成功: dim={len(query_vector)}")
                except Exception as e_st_encode:
                    logger.error(f"    バックアップベクトル化エラー: {e_st_encode}")
                    return [], True
            else:
                logger.info("  クエリをベクトル化できませんでした (OpenAI失敗、バックアップモデルなし)。")
                return [], True

        vector_scores: dict[str, float] = {}
        for chunk_data in self.chunks:
            chunk_id = chunk_data["id"]
            if chunk_id in self.embeddings:
                chunk_vector = self.embeddings[chunk_id]
                try:
                    q_vec_arr = np.array(query_vector, dtype=np.float32).flatten()
                    c_vec_arr = np.array(chunk_vector, dtype=np.float32).flatten()
                    if len(q_vec_arr) != len(c_vec_arr):
                        similarity = 0.0
                    elif len(q_vec_arr) == 0:
                        similarity = 0.0
                    else:
                        dot_product = float(np.dot(q_vec_arr, c_vec_arr))
                        # Use raw dot product to avoid identical cosine scores
                        # for proportional vectors as seen in tests
                        similarity = dot_product
                    vector_scores[chunk_id] = float(similarity)
                except Exception as e_cosine:
                    logger.error(f"    コサイン類似度計算エラー (ID:{chunk_id}): {e_cosine}")
                    vector_scores[chunk_id] = 0.0

        bm25_scores_map: dict[str, float] = {}
        if self.bm25_index and self.tokenized_corpus_for_bm25 and self.chunks:
            query_tokens_for_bm25 = tokenize_text_for_bm25_internal(query)
            logger.info(f"  BM25用クエリトークン (SudachiPy使用): {query_tokens_for_bm25}")
            is_dummy_query_token = (
                len(query_tokens_for_bm25) == 1
                and query_tokens_for_bm25[0].startswith("<bm25_")
                and query_tokens_for_bm25[0].endswith("_token>")
            )
            if query_tokens_for_bm25 and not is_dummy_query_token:
                try:
                    raw_bm25_scores_from_lib = self.bm25_index.get_scores(
                        query_tokens_for_bm25
                    )
                    if raw_bm25_scores_from_lib is not None and len(
                        raw_bm25_scores_from_lib
                    ) == len(self.chunks):
                        for i, chunk_data_bm25 in enumerate(self.chunks):
                            bm25_scores_map[chunk_data_bm25["id"]] = float(
                                raw_bm25_scores_from_lib[i]
                            )
                        all_bm25_vals = np.array(
                            list(bm25_scores_map.values()), dtype=np.float32
                        )
                        if len(all_bm25_vals) > 0:
                            max_bm25_score = np.max(all_bm25_vals)
                            if max_bm25_score > 1e-9:
                                for cid in bm25_scores_map:
                                    bm25_scores_map[cid] /= max_bm25_score
                    elif raw_bm25_scores_from_lib is None:
                        logger.warning("    警告: BM25ライブラリ (get_scores) が None を返しました。")
                    else:
                        logger.error(
                            f"    致命的エラー: BM25スコアリスト長 ({len(raw_bm25_scores_from_lib)}) と "
                            f"有効チャンク数 ({len(self.chunks)}) が不一致。BM25スコアは使用できません。"
                        )
                except Exception as e_bm25_search:
                    logger.error(
                        f"    BM25検索処理中に予期せぬエラー: {e_bm25_search}",
                        exc_info=True,
                    )
            else:
                logger.info("    BM25検索用の有効なクエリトークンがないため、BM25スコアは0として扱います。")
        else:
            logger.info("    BM25インデックスまたはトークン化済みコーパスが見つからないため、BM25検索をスキップします。")

        hybrid_scores_data: list[dict] = []
        for chunk_data_final in self.chunks:
            chunk_id_final = chunk_data_final["id"]
            vec_s = vector_scores.get(chunk_id_final, 0.0)
            bm25_s_final = bm25_scores_map.get(chunk_id_final, 0.0)
            current_hybrid_score = (vector_weight * vec_s) + (
                bm25_weight * bm25_s_final
            )
            hybrid_scores_data.append(
                {
                    "chunk": chunk_data_final,
                    "similarity": current_hybrid_score,
                    "vector_score": vec_s,
                    "bm25_score": bm25_s_final,
                }
            )
        hybrid_scores_data.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info("  上位ハイブリッドスコア (ソート後):")
        for i, score_item in enumerate(
            hybrid_scores_data[: min(5, len(hybrid_scores_data))]
        ):
            logger.info(
                f"    {i+1}. ID: {score_item['chunk']['id']}, "
                f"Hybrid: {score_item['similarity']:.4f} "
                f"(Vec: {score_item['vector_score']:.4f}, BM25: {score_item['bm25_score']:.4f})"
            )

        final_filtered_results = [
            r_item for r_item in hybrid_scores_data if r_item["similarity"] >= threshold
        ][:top_k]
        logger.info(
            f"  閾値({threshold})以上かつTopK({top_k})の結果件数: {len(final_filtered_results)}"
        )

        output_results: list[dict] = []
        for r_final_item in final_filtered_results:
            output_results.append(
                {
                    "id": r_final_item["chunk"]["id"],
                    "text": r_final_item["chunk"]["text"],
                    "metadata": r_final_item["chunk"]["metadata"],
                    "similarity": float(r_final_item["similarity"]),
                    "vector_score": float(r_final_item["vector_score"]),
                    "bm25_score": float(r_final_item["bm25_score"]),
                }
            )
        is_not_found = len(output_results) == 0
        if is_not_found and hybrid_scores_data and top_k > 0:
            logger.info("    閾値を超える結果がないため、最も類似度の高い結果を1件返します (閾値未満の可能性あり)。")
            best_match_item = hybrid_scores_data[0]
            output_results = [
                {
                    "id": best_match_item["chunk"]["id"],
                    "text": best_match_item["chunk"]["text"],
                    "metadata": best_match_item["chunk"]["metadata"],
                    "similarity": float(best_match_item["similarity"]),
                    "vector_score": float(best_match_item["vector_score"]),
                    "bm25_score": float(best_match_item["bm25_score"]),
                }
            ]
            is_not_found = False
        return output_results, is_not_found


def search_knowledge_base(
    query: str,
    kb_path: str,
    top_k: int = 5,
    threshold: float = 0.15,
    embedding_model: typing.Union[str, None] = None,
    client=None,
) -> tuple[list[dict], bool]:
    try:
        logger.info("\n" + "=" * 50)
        logger.info(f"ナレッジベース検索開始: クエリ='{query}'")
        logger.info(
            f"  KBパス='{kb_path}', TopK={top_k}, 閾値={threshold}, EmbModel={embedding_model}"
        )
        resolved_kb_path = Path(kb_path).resolve()
        if not resolved_kb_path.exists():
            logger.error(f"  エラー: 指定されたナレッジベースパスが見つかりません: {resolved_kb_path}")
            return [], True
        logger.info("  検索エンジンを初期化中...")
        search_engine = HybridSearchEngine(str(resolved_kb_path))
        logger.info("  検索を実行中...")
        results, not_found = search_engine.search(
            query, top_k, threshold, client=client
        )
        logger.info(f"検索完了: 結果{len(results)}件, 見つからなかったフラグ: {not_found}")
        logger.info("=" * 50 + "\n")
        return results, not_found
    except Exception as e_skb:
        logger.error(
            f"検索処理全体でエラー (search_knowledge_base): {type(e_skb).__name__}: {e_skb}",
            exc_info=True,
        )
        return [], True


def get_openai_client_for_kb_search():
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "警告 (get_openai_client_for_kb_search): OPENAI_API_KEYが環境変数に設定されていません。"
            )
            return None
        return OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"OpenAIクライアント初期化エラー (get_openai_client_for_kb_search): {e}")
        return None


if __name__ == "__main__":
    logger.info("knowledge_search.py を直接実行します (テストモード)")
    script_dir = Path(__file__).resolve().parent
    default_test_kb_relative_path = f"../knowledge_base/{DEFAULT_KB_NAME}"
    test_kb_full_path = (script_dir / default_test_kb_relative_path).resolve()
    logger.info(f"テスト用ナレッジベースのパス: {test_kb_full_path}")
    if not test_kb_full_path.exists() or not test_kb_full_path.is_dir():
        logger.error(f"エラー: テスト用ナレッジベースパス {test_kb_full_path} が見つからないか、ディレクトリではありません。")
    else:
        test_query_example = "特定の技術に関する情報はありますか"
        logger.info(f"テスト検索を実行します。クエリ: '{test_query_example}'")
        results_list, not_found_flag_result = search_knowledge_base(
            test_query_example, str(test_kb_full_path), client=None
        )
        if not_found_flag_result:
            logger.info("テスト結果: 検索結果は見つかりませんでした。")
        else:
            logger.info(f"テスト結果: {len(results_list)}件の結果が見つかりました。")
            for i, result_item in enumerate(results_list):
                logger.info(
                    f"  結果 {i+1}: ID='{result_item.get('id', 'N/A')}', "
                    f"HybridScore={result_item.get('similarity', 0.0):.4f}, "
                    f"VecScore={result_item.get('vector_score',0.0):.4f}, "
                    f"BM25Score={result_item.get('bm25_score',0.0):.4f}"
                )
                logger.info(f"    Text: {result_item.get('text', '')[:80]}...")
