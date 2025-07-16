import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import config
from shared.file_processor import FileProcessor
from shared.kb_builder import KnowledgeBuilder
from shared.openai_utils import get_openai_client

logger = logging.getLogger(__name__)

_file_processor = FileProcessor()
_kb_builder = KnowledgeBuilder(
    _file_processor,
    get_openai_client_func=get_openai_client,
    refresh_search_engine_func=None,
)


def get_embedding(text: str, client=None):
    """OpenAI埋め込みAPIを利用してテキストのベクトルを生成する。

    Args:
        text: 埋め込みを計算したいテキスト。
        client: ``openai.OpenAI`` クライアント。 ``None`` の場合は
            :func:`get_openai_client` で自動取得する。

    Returns:
        list[float] | None: 生成された埋め込みベクトル。クライアント取得や
        API 呼び出しに失敗した場合は ``None`` が返る。

    ``KnowledgeBuilder`` の内部メソッド ``_get_embedding`` を呼び出して
    埋め込みを取得する。 ``client`` を省略するとクライアントを生成し、
    取得できない場合やAPIエラー時は ``None`` を返す。
    """
    if client is None:
        client = get_openai_client()
        if client is None:
            return None
    return _kb_builder._get_embedding(text, client)


GPT4O_MODEL = "gpt-4.1"


def analyze_image_with_gpt4o(
    image_base64: str,
    filename: str,
    cad_metadata: Optional[dict] = None,
    client=None,
) -> dict:
    """GPT-4o に画像を渡して分析結果を取得する。

    Args:
        image_base64: Base64 エンコードされた画像データ。
        filename: 解析対象ファイルの名称。
        cad_metadata: CAD ファイルに付随するメタデータ。 ``None`` の場合は
            通常の画像解析として扱う。
        client: ``openai.OpenAI`` クライアント。省略時は
            :func:`get_openai_client` で取得する。

    Returns:
        dict: GPT-4o から返された JSON を解析した辞書。クライアント取得に
        失敗した場合や API エラー時は ``{"error": ...}`` を返す。

    画像とプロンプトを ``chat.completions`` API に送信し、モデルが生成した
    JSON オブジェクトを辞書として返す。 ``cad_metadata`` が指定されている
    場合はその情報を結果に統合する。
    """
    if client is None:
        client = get_openai_client()
        if client is None:
            return {"error": "OpenAIクライアントが利用できません"}

    try:
        if cad_metadata:
            cad_info = f"""
CADファイル情報:
- ファイル形式: {cad_metadata.get('file_type', 'Unknown')}
- エンティティ数: {cad_metadata.get('total_entities', 'N/A')}
- 技術仕様: {cad_metadata}
"""
            prompt = f"""
この技術図面・CADファイル（ファイル名: {filename}）を詳細に分析し、以下の情報をJSON形式で返してください：

{cad_info}

1. image_type: 図面の種類（機械図面、建築図面、回路図、組織図、3Dモデル、その他）
2. main_content: 図面の主要な内容と技術的説明（300-400文字）
3. technical_specifications: 技術仕様・寸法・材質などの詳細情報
4. detected_elements: 図面内の主要な要素・部品リスト（最大15個）
5. dimensions_info: 寸法情報や測定値（検出できる場合）
6. annotations: 注記・文字情報・記号の内容
7. drawing_standards: 図面規格・標準（JIS、ISO、ANSI等、該当する場合）
8. manufacturing_info: 製造情報・加工情報（該当する場合）
9. keywords: 技術検索用キーワード（最大20個）
10. category_tags: 専門分野タグ（機械工学、建築、電気工学等、最大10個）
11. description_for_search: 技術者向け検索結果表示用説明（100-150文字）
12. related_standards: 関連する技術標準・規格の提案
"""
        else:
            prompt = f"""
この画像（ファイル名: {filename}）を詳細に分析し、以下の情報をJSON形式で返してください：

1. image_type: 画像の種類（写真、技術図面、組織図、フローチャート、グラフ、表、地図、その他）
2. main_content: 画像の主要な内容の詳細説明（200-300文字）
3. detected_elements: 画像内の主要な要素リスト（最大10個）
4. technical_details: 技術的な詳細（寸法、規格、仕様など、該当する場合）
5. text_content: 画像内に含まれるテキスト内容（すべて正確に読み取って記載）
6. keywords: 検索に有用なキーワード（画像内容＋テキスト内容から最大20個）
7. search_terms: テキスト内容から想定される検索ワード・フレーズ（最大15個）
8. category_tags: 分類タグ（最大8個）
9. description_for_search: 検索結果表示用の簡潔な説明（80-120文字）
10. metadata_suggestions: 追加すべきメタデータの提案
11. related_topics: 画像・テキスト内容から関連しそうなトピック（最大10個）
12. document_type_hints: 文書種別の推定（報告書、マニュアル、仕様書、比較表等）

特に重要：
- text_contentには画像内のすべてのテキストを正確に読み取って記載してください
- そのテキスト内容を基に、検索で使われそうなキーワードやフレーズを多数生成してください
- 専門用語、固有名詞、数値、日付なども検索キーワードに含めてください

JSON形式で返してください。日本語で回答してください。
"""

        response = client.chat.completions.create(
            model=GPT4O_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=3000,
        )

        result = json.loads(response.choices[0].message.content)
        if cad_metadata:
            result["cad_metadata"] = cad_metadata
        return result
    except Exception as e:
        logger.error(f"GPT-4o画像解析エラー: {e}")
        return {"error": f"画像解析中にエラーが発生しました: {e}"}


# --- CLIP model utilities ---
_clip_model = None
_clip_processor = None


def load_model_and_processor() -> Tuple[object, object]:
    """Load and cache the CLIP model and processor.

    Returns the model and processor instances. They are loaded lazily from
    ``config.MULTIMODAL_MODEL`` and cached for future calls.
    """
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        from transformers import CLIPModel, CLIPProcessor

        _clip_model = CLIPModel.from_pretrained(config.MULTIMODAL_MODEL)
        _clip_processor = CLIPProcessor.from_pretrained(config.MULTIMODAL_MODEL)
    return _clip_model, _clip_processor


def get_image_embedding(image, model=None, processor=None) -> list[float]:
    """Return an embedding vector for ``image`` using the CLIP model."""

    if model is None or processor is None:
        model, processor = load_model_and_processor()

    if isinstance(image, (str, Path)):
        from PIL import Image

        img = Image.open(image)
    else:
        img = image

    inputs = processor(images=img, return_tensors="pt")
    features = model.get_image_features(**inputs)
    if hasattr(features, "detach"):
        features = features.detach()
    if hasattr(features, "cpu"):
        features = features.cpu()
    vector = features[0].tolist()
    return vector[: config.EMBEDDING_DIM]
