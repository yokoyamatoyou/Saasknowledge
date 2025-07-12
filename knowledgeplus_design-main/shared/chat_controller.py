import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.errors import GPTGenerationError, OpenAIClientError
from shared.openai_utils import get_openai_client
from shared.search_engine import HybridSearchEngine

logger = logging.getLogger(__name__)

# OpenAIクライアントの初期化
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# グローバルなクライアントインスタンスはapp.pyのget_openai_client()で管理する方針に合わせるか、
# 各関数内で必要に応じて初期化する形にする。ここでは後者で実装。

# ペルソナデータの保存ディレクトリ
#
# `PERSONA_DIR` 環境変数が指定されている場合はその値を利用し、指定が無ければ
# リポジトリ直下 ``data/personalities`` をデフォルトとする。Google
# Cloud などへの移行時に保存先を切り替えられるようにするための調整。
PERSONA_DIR = os.getenv(
    "PERSONA_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "personalities"),
)


def create_persona(
    persona_id,
    name,
    description,
    system_prompt,
    default_temperature=0.7,
    default_response_length="普通",
):
    """
    新しいペルソナを作成する関数

    Args:
        persona_id (str): ペルソナの一意識別子
        name (str): ペルソナの名前
        description (str): ペルソナの説明
        system_prompt (str): システムプロンプト
        default_temperature (float): デフォルトの温度値
        default_response_length (str): デフォルトの応答の長さ
    """
    os.makedirs(PERSONA_DIR, exist_ok=True)

    persona_data = {
        "id": persona_id,
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "default_temperature": default_temperature,
        "default_response_length": default_response_length,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(
        os.path.join(PERSONA_DIR, f"{persona_id}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(persona_data, f, ensure_ascii=False, indent=4)

    return persona_data


def get_persona_list():
    """
    利用可能なペルソナの一覧（IDと名前を含む辞書のリスト）を取得する関数

    Returns:
        list: ペルソナのIDと名前を含む辞書のリスト (例: [{"id": "default", "name": "標準アシスタント"}, ...])
    """
    if not os.path.exists(PERSONA_DIR):
        os.makedirs(PERSONA_DIR, exist_ok=True)
        create_default_personas()

    persona_files = [f for f in os.listdir(PERSONA_DIR) if f.endswith(".json")]
    persona_ids = [os.path.splitext(f)[0] for f in persona_files]

    persona_details_list = []
    for persona_id in persona_ids:
        try:
            persona_data = load_persona(persona_id)  # load_personaはIDで読み込む
            persona_details_list.append(
                {"id": persona_id, "name": persona_data.get("name", persona_id)}
            )
        except Exception as e:
            logger.error(f"Error loading persona details for {persona_id}: {e}")
            persona_details_list.append(
                {"id": persona_id, "name": persona_id}
            )  # フォールバック

    return persona_details_list


def load_persona(persona_id):
    """
    ペルソナデータを読み込む関数

    Args:
        persona_id (str): ペルソナID

    Returns:
        dict: ペルソナデータ
    """
    try:
        with open(
            os.path.join(PERSONA_DIR, f"{persona_id}.json"), "r", encoding="utf-8"
        ) as f:
            return json.load(f)
    except FileNotFoundError:
        create_default_personas()  # デフォルトがなければ作成
        default_persona_path = os.path.join(PERSONA_DIR, "default.json")
        if os.path.exists(default_persona_path):
            with open(default_persona_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:  # これでも見つからない場合（通常はありえない）
            # 最低限のデフォルトペルソナをその場で生成して返す（エラー回避）
            return {
                "id": "fallback_default",
                "name": "フォールバック標準",
                "description": "フォールバック用の標準アシスタント",
                "system_prompt": "あなたはAIアシスタントです。",
                "default_temperature": 0.7,
                "default_response_length": "普通",
            }


def create_default_personas():
    """
    デフォルトのペルソナを作成する関数
    """
    create_persona(
        persona_id="default",
        name="標準アシスタント",
        description="一般的なタスクに対応する標準的なAIアシスタント",
        system_prompt="あなたは有能なAIアシスタントです。ユーザーの質問に丁寧に、正確に、そして詳細に回答してください。",
        default_temperature=0.7,
        default_response_length="普通",
    )
    create_persona(
        persona_id="expert",
        name="専門家",
        description="専門的な知識を持つアドバイザー",
        system_prompt="あなたは様々な分野の専門的な知識を持つAIアドバイザーです。ユーザーの質問に対して、専門的な観点から詳細かつ正確な情報を提供してください。複雑な概念も分かりやすく説明し、必要に応じて例を挙げて解説してください。",
        default_temperature=0.5,
        default_response_length="詳細",
    )
    create_persona(
        persona_id="creative",
        name="クリエイティブライター",
        description="創造的な文章を生成するライター",
        system_prompt="あなたは創造性豊かなAIライターです。ユーザーのリクエストに基づいて、独創的で魅力的な文章、ストーリー、アイデアを生成してください。常に新鮮で想像力に富んだ内容を心がけ、読者を引き込む表現を使ってください。",
        default_temperature=0.9,
        default_response_length="詳細",
    )
    create_persona(
        persona_id="coach",
        name="コーチ",
        description="励ましと実用的なアドバイスを提供するコーチ",
        system_prompt="あなたは前向きで実用的なアドバイスを提供するAIコーチです。ユーザーの目標達成や課題解決を支援するために、明確なステップや戦略を提案してください。常に励ましの言葉を添え、ユーザーのモチベーションを高められるよう心がけてください。",
        default_temperature=0.7,
        default_response_length="普通",
    )


class ChatController:
    """Manage conversations and perform RAG-based response generation."""

    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine

    @staticmethod
    def _get_openai_client_internal():  # Renamed and made static
        return get_openai_client()

    def generate_gpt_response(
        self,
        user_input,
        conversation_history=None,
        persona="default",
        temperature=None,
        response_length=None,
        client=None,
    ):
        """
        GPTを使用して応答を生成する関数（新しいOpenAI API v1.0.0+に対応）

        Args:
            user_input (str): ユーザーの入力テキスト
            conversation_history (list): 会話履歴
            persona (str): 使用するペルソナのID
            temperature (float): 温度パラメータ（創造性の度合い）
            response_length (str): 応答の長さ指定
            client (OpenAI, optional): OpenAIクライアントインスタンス。Noneの場合、内部で初期化。

        Yields:
            str: 応答の一部テキスト

        Raises:
            OpenAIClientError: OpenAIクライアントの初期化に失敗した場合
            GPTGenerationError: 応答生成中にエラーが発生した場合
        """
        if client is None:
            client = self._get_openai_client_internal()  # Call static method
            if client is None:
                raise OpenAIClientError("OpenAIクライアントの初期化に失敗しました。APIキーを確認してください。")

        if conversation_history is None:
            conversation_history = []

        persona_data = load_persona(persona)

        if temperature is None:
            temperature = persona_data["default_temperature"]
        if response_length is None:
            response_length = persona_data["default_response_length"]

        max_tokens = {"簡潔": 600, "普通": 1200, "詳細": 2400}.get(response_length, 1200)
        system_prompt = persona_data["system_prompt"]

        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            messages.append(msg)
        messages.append({"role": "user", "content": user_input})

        try:
            stream = client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",  # app.pyのGPT4_MINI_MODELと合わせる
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:  # pragma: no cover - network errors may vary
            logger.error("GPT response generation failed: %s", e, exc_info=True)
            raise GPTGenerationError(f"応答生成中に予期しないエラーが発生しました: {e}") from e

    def generate_conversation_title(self, conversation_history, client=None):
        """
        会話履歴から簡潔なタイトルを生成する関数。

        Args:
            conversation_history (list): 会話履歴のリスト [{"role": "user/assistant", "content": "..."}]
            client (OpenAI, optional): OpenAIクライアントインスタンス。Noneの場合、内部で初期化。

        Returns:
            str: 生成された会話タイトル
        """
        if client is None:
            client = self._get_openai_client_internal()  # Call static method
            if client is None:
                return "会話タイトル生成エラー"

        if not conversation_history:
            return "新しい会話"

        # システムプロンプトを除外し、ユーザーとアシスタントのメッセージのみを使用
        user_assistant_history = [
            msg for msg in conversation_history if msg["role"] in ["user", "assistant"]
        ]

        # 最新の数メッセージをコンテキストとして使用 (例: 最新10メッセージ)
        history_text = "\n".join(
            [
                f"{msg['role']}: {msg['content'][:200]}"
                for msg in user_assistant_history[-10:]
            ]
        )  # 各メッセージの先頭200文字

        if not history_text.strip():
            return "会話の開始"

        try:
            from config import TITLE_GENERATION_MODEL

            response = client.chat.completions.create(
                model=TITLE_GENERATION_MODEL,  # configurable lightweight model
                messages=[
                    {
                        "role": "system",
                        "content": "以下の会話のやり取りに基づいて、この会話全体を表す簡潔なタイトルを日本語で5～10単語程度で生成してください。",
                    },
                    {
                        "role": "user",
                        "content": f"会話の抜粋:\n{history_text}\n\nこの会話のタイトルは？",
                    },
                ],
                temperature=0.2,  # タイトルなので決定的に
                max_tokens=30,  # 短いタイトルを期待
            )
            title = response.choices[0].message.content.strip()
            # 不要な引用符や接頭辞を取り除く
            title = (
                title.replace("「", "")
                .replace("」", "")
                .replace("『", "")
                .replace("』", "")
            )
            title = title.replace("タイトル:", "").replace("会話のタイトル:", "").strip()
            if not title:
                return "会話"  # フォールバック
            return title
        except Exception as e:
            logger.error(f"会話タイトル生成中にエラーが発生しました: {e}")
            return "会話"  # エラー時のフォールバックタイトル

    def generate_response(self, *args, **kwargs):
        return self.generate_gpt_response(
            *args, **kwargs
        )  # Changed to self.generate_gpt_response

    def generate_response_with_rag(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.15,
        persona: str = "default",
        temperature: Optional[float] = None,
        response_length: Optional[str] = None,
        client=None,
    ) -> str:
        """Search the knowledge base and answer using retrieved context."""
        results, _ = self.search_engine.search(
            query, top_k=top_k, threshold=threshold, client=client
        )
        context = "\n".join(r.get("text", "") for r in results)
        prompt = f"次の情報を参考にユーザーの質問に答えてください:\n{context}\n\n質問:{query}"
        return self.generate_gpt_response(  # Changed to self.generate_gpt_response
            prompt,
            conversation_history=[],
            persona=persona,
            temperature=temperature,
            response_length=response_length,
            client=client,
        )
