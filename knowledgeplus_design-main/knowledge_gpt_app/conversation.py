import os
import json
import uuid
from datetime import datetime
import openai
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 会話保存ディレクトリ
CONVERSATION_DIR = Path("data/conversations")

def ensure_directories():
    """必要なディレクトリを確保"""
    CONVERSATION_DIR.mkdir(parents=True, exist_ok=True)

def save_conversation(conversation_id, title=None, history=None, messages=None):
    """
    会話を保存
    
    Args:
        conversation_id: 会話ID
        title: 会話のタイトル（指定がなければ保持）
        history: 会話履歴リスト
        messages: 表示用メッセージリスト
    
    Returns:
        成功したかどうか
    """
    ensure_directories()
    
    conversation_file = CONVERSATION_DIR / f"{conversation_id}.json"
    
    # 既存のデータを読み込む
    existing_data = {}
    if conversation_file.exists():
        with open(conversation_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    
    # 新しいデータで更新
    data = {
        "id": conversation_id,
        "title": title if title is not None else existing_data.get("title", "無題の会話"),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "history": history if history is not None else existing_data.get("history", []),
        "messages": messages if messages is not None else existing_data.get("messages", [])
    }
    
    # ファイルに保存
    with open(conversation_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return True

def load_conversation(conversation_id):
    """
    会話を読み込み
    
    Args:
        conversation_id: 会話ID
    
    Returns:
        会話データ
    """
    ensure_directories()
    
    conversation_file = CONVERSATION_DIR / f"{conversation_id}.json"
    
    if not conversation_file.exists():
        return {
            "id": conversation_id,
            "title": "無題の会話",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": [],
            "messages": []
        }
    
    with open(conversation_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def list_conversations():
    """
    保存された会話の一覧を取得
    
    Returns:
        会話情報のリスト
    """
    ensure_directories()
    
    conversations = []
    for conversation_file in CONVERSATION_DIR.glob("*.json"):
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversations.append({
                    "id": data.get("id", conversation_file.stem),
                    "title": data.get("title", "無題の会話"),
                    "date": data.get("date", "不明な日時")
                })
        except Exception as e:
            logger.error(f"会話ファイル読み込みエラー {conversation_file}: {e}")
    
    # 日付の新しい順にソート
    conversations.sort(key=lambda x: x["date"], reverse=True)
    
    return conversations

def auto_generate_title(messages):
    """
    GPT-4.1-miniを使用して会話の題名を自動生成
    
    Args:
        messages: 会話メッセージのリスト
    
    Returns:
        自動生成された題名
    """
    if not messages:
        return "無題の会話"
    
    try:
        # 会話の最初の数ターンを取得
        sample = messages[:min(3, len(messages))]
        
        # 題名生成クエリ
        prompt = "以下の会話の内容を表す簡潔な題名（30文字以内）を生成してください:\n\n"
        
        for msg in sample:
            role_text = "ユーザー" if msg["role"] == "user" else "アシスタント"
            prompt += f"{role_text}: {msg['content'][:100]}...\n"
        
        # GPT-4.1-miniでタイトル生成
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini-2025-04-14",  # この日付はサンプルです
            messages=[
                {"role": "system", "content": "あなたは会話の題名を生成するアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=15
        )
        
        title = response.choices[0].message["content"].strip().strip('"')
        
        # 題名が長すぎる場合は切り詰める
        if len(title) > 30:
            title = title[:27] + "..."
        
        return title
    
    except Exception as e:
        logger.error(f"タイトル生成エラー: {e}")
        # ユーザーの最初のメッセージから題名を生成（フォールバック）
        first_message = messages[0]["content"] if messages else "会話"
        if len(first_message) > 30:
            first_message = first_message[:27] + "..."
        return first_message
