import logging
from typing import Optional

from openai import OpenAI

from .upload_utils import ensure_openai_key

logger = logging.getLogger(__name__)

# Lightweight model for prompt advice generation
PROMPT_ADVICE_MODEL = "gpt-4o-mini"


def generate_prompt_advice(
    user_prompt: str, client: Optional[OpenAI] = None
) -> Optional[str]:
    """Return improved prompt suggestions using a lightweight GPT model."""
    if not user_prompt:
        return None
    if client is None:
        try:
            api_key = ensure_openai_key()
            client = OpenAI(api_key=api_key)
        except Exception as e:  # pragma: no cover - env misconfig
            logger.error(f"OpenAI client init failed: {e}")
            return None

    try:
        resp = client.chat.completions.create(
            model=PROMPT_ADVICE_MODEL,
            messages=[
                {"role": "system", "content": "ユーザープロンプトを明確にするアドバイスを日本語で箇条書きで返してください。"},
                {
                    "role": "user",
                    "content": f"以下のプロンプトを改善してください:\n\n---\n{user_prompt}\n---",
                },
            ],
            temperature=0.0,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:  # pragma: no cover - API failure
        logger.error(f"Prompt advice generation error: {e}", exc_info=True)
        return None
