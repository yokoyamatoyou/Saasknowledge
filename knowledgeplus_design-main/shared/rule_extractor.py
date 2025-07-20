import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from openai import OpenAI

from .openai_utils import get_openai_client

logger = logging.getLogger(__name__)


def extract_rules(text: str, client: Optional["OpenAI"] = None) -> List[Dict[str, Any]]:
    """Extract structured business rules from ``text`` using GPT-4.

    Parameters
    ----------
    text : str
        Source text that may contain rule descriptions.
    client : OpenAI, optional
        OpenAI client to use. If ``None`` the function attempts to create one.

    Returns
    -------
    List[Dict[str, Any]]
        Extracted rule objects or an empty list on failure.
    """
    if not text:
        return []
    if client is None:
        client = get_openai_client()
        if client is None:
            return []

    prompt = (
        "以下のテキストからビジネスルールを抽出し、JSON一覧で返してください。\n"
        f"テキスト:\n{text}\n\n"
        "出力例:\n"
        "[{'rule_id':'r1','rule_type':'承認権限','rule_content':'100万円以下は課長承認',"
        "'conditions':['金額<=1000000'],"
        "'values':{'threshold_amount':1000000,'approver':'課長'},"
        "'confidence':0.9}]"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "ビジネスルール抽出の専門家"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("rules", [])  # type: ignore[return-value]
    except Exception as e:  # pragma: no cover - network issues
        logger.error("Rule extraction failed: %s", e)
    return []
