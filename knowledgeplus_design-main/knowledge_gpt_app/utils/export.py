import os
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
import textwrap
import re

# エクスポートディレクトリ
EXPORT_DIR = Path("data/exports")

def ensure_directories():
    """必要なディレクトリを確保"""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text_for_pdf(text):
    """PDFに適した形式にテキストを整形"""
    # 制御文字を削除
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # マークダウンの見出しを普通のテキストに変換
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    # コードブロックを整形
    text = re.sub(r'```.*?\n(.*?)```', r'\1', text, flags=re.DOTALL)
    return text

def export_conversation_to_pdf(conversation_id, messages):
    """
    会話をPDFにエクスポート
    
    Args:
        conversation_id: 会話ID
        messages: 会話メッセージのリスト
    
    Returns:
        PDFファイルのパス
    """
    ensure_directories()
    
    # 現在の日時
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ファイル名
    filename = f"conversation_{conversation_id}_{now}.pdf"
    filepath = EXPORT_DIR / filename
    
    # PDFの作成
    pdf = FPDF()
    pdf.add_page()

    # フォント設定
    font_path = Path("ipaexg.ttf")
    if not font_path.exists():
        raise FileNotFoundError(
            "ipaexg.ttf not found. Download IPAexGothic from https://moji.or.jp/ipafont/ and place it in the repository root."
        )
    pdf.add_font('IPAGothic', '', str(font_path), uni=True)
    pdf.set_font('IPAGothic', '', 10)
    
    # タイトル
    pdf.set_font('IPAGothic', '', 16)
    pdf.cell(0, 10, 'ナレッジGPT 会話エクスポート', 0, 1, 'C')
    
    # 日時
    pdf.set_font('IPAGothic', '', 10)
    pdf.cell(0, 10, f'エクスポート日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    
    pdf.ln(5)
    
    # 会話内容
    for message in messages:
        role = "ユーザー" if message["role"] == "user" else "アシスタント"
        
        # 役割を表示
        pdf.set_font('IPAGothic', '', 12)
        if role == "ユーザー":
            pdf.set_fill_color(230, 247, 255)  # 薄い青
        else:
            pdf.set_fill_color(240, 240, 240)  # 薄いグレー
            
        pdf.cell(0, 8, f'{role}:', 0, 1, 'L', 1)
        
        # メッセージ内容
        pdf.set_font('IPAGothic', '', 10)
        content = clean_text_for_pdf(message["content"])
        
        # 長いテキストを折り返す
        wrapped_text = textwrap.fill(content, width=80)
        
        # 段落を処理
        paragraphs = wrapped_text.split('\n')
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph)
        
        # 検索結果の表示（ある場合）
        if "search_results" in message:
            pdf.ln(2)
            pdf.set_font('IPAGothic', '', 9)
            pdf.cell(0, 6, '検索結果:', 0, 1, 'L')
            
            for i, result in enumerate(message["search_results"]):
                pdf.set_font('IPAGothic', '', 8)
                pdf.cell(0, 5, f'結果 {i+1} - 類似度: {result["similarity"]:.2f}', 0, 1, 'L')
                
                if "metadata" in result and "source" in result["metadata"]:
                    pdf.cell(0, 5, f'出典: {result["metadata"]["source"]}', 0, 1, 'L')
                
                # 検索結果のテキスト（短く切り詰める）
                text = result["text"]
                if len(text) > 200:
                    text = text[:197] + "..."
                pdf.multi_cell(0, 5, text)
                pdf.ln(1)
        
        pdf.ln(5)
    
    # PDFを保存
    pdf.output(str(filepath))
    
    return str(filepath)

def export_conversation_to_text(conversation_id, messages):
    """
    会話をテキストファイルにエクスポート
    
    Args:
        conversation_id: 会話ID
        messages: 会話メッセージのリスト
    
    Returns:
        テキストファイルのパス
    """
    ensure_directories()
    
    # 現在の日時
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ファイル名
    filename = f"conversation_{conversation_id}_{now}.txt"
    filepath = EXPORT_DIR / filename
    
    # テキストファイルに会話を書き込み
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("====================================\n")
        f.write("ナレッジGPT 会話エクスポート\n")
        f.write(f"エクスポート日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("====================================\n\n")
        
        for message in messages:
            role = "ユーザー" if message["role"] == "user" else "アシスタント"
            
            f.write(f"【{role}】\n")
            f.write(f"{message['content']}\n\n")
            
            # 検索結果の表示（ある場合）
            if "search_results" in message:
                f.write("---【検索結果】---\n")
                
                for i, result in enumerate(message["search_results"]):
                    f.write(f"結果 {i+1} - 類似度: {result['similarity']:.2f}\n")
                    
                    if "metadata" in result and "source" in result["metadata"]:
                        f.write(f"出典: {result['metadata']['source']}\n")
                    
                    f.write(f"{result['text']}\n\n")
                
                f.write("------------------\n")
            
            f.write("\n" + "-" * 50 + "\n\n")
    
    return str(filepath)
