import PyInstaller.__main__
import os
import shutil
from pathlib import Path
import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

# 実行ファイル名
EXE_NAME = "ナレッジGPT"

def build_executable():
    """Streamlitアプリケーションを実行ファイル化"""
    logger.info("====== ナレッジGPTアプリケーションをEXE化します ======")
    
    # 必要なディレクトリを作成
    build_dir = Path("build")
    dist_dir = Path("dist")
    
    build_dir.mkdir(exist_ok=True)
    dist_dir.mkdir(exist_ok=True)
    
    # PyInstallerの設定
    pyinstaller_args = [
        "app.py",                           # メインスクリプト
        "--name", EXE_NAME,                 # 出力ファイル名
        "--onefile",                        # 単一ファイル化
        "--windowed",                       # コンソールを表示しない
        "--icon=resources/app_icon.ico",    # アイコン設定（resources/app_icon.icoが必要）
        "--add-data=ipaexg.ttf;.",          # 日本語フォントを含める
        "--add-data=resources;resources",   # リソースディレクトリを含める
        "--add-data=data;data",             # データディレクトリを含める
        "--hidden-import=sklearn.metrics",  # 必要な非明示的インポート
        "--hidden-import=sentence_transformers",
        "--hidden-import=nltk",
        "--hidden-import=rank_bm25",
        "--hidden-import=streamlit",
        "--hidden-import=openai",
        "--hidden-import=fpdf",
    ]
    
    logger.info("PyInstallerを実行中...")
    PyInstaller.__main__.run(pyinstaller_args)
    
    logger.info("ランチャースクリプトを作成中...")
    create_launcher()
    
    logger.info("====== ビルド完了 ======")
    logger.info(f"実行ファイル: dist/{EXE_NAME}.exe")

def create_launcher():
    """
    アプリケーション起動用のバッチファイルを作成
    (環境変数を設定してからアプリを起動)
    """
    launcher_path = Path("dist") / f"{EXE_NAME}_起動.bat"
    
    batch_content = f"""@echo off
echo ナレッジGPTアプリケーションを起動しています...

REM OpenAI APIキーを設定（必要に応じて編集してください）
set OPENAI_API_KEY=your_api_key_here

REM アプリケーション起動
start "" "{EXE_NAME}.exe"
"""
    
    with open(launcher_path, 'w', encoding='utf-8') as f:
        f.write(batch_content)
    
    logger.info(f"ランチャー作成完了: {launcher_path}")

def check_dependencies():
    """必要な依存パッケージをチェック"""
    required_packages = [
        "streamlit",
        "openai",
        "scikit-learn",
        "sentence-transformers",
        "nltk",
        "rank_bm25",
        "fpdf",
        "pyinstaller",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning("以下の必要なパッケージがインストールされていません:")
        for pkg in missing_packages:
            logger.info(f"  - {pkg}")
        
        install = input("これらのパッケージをインストールしますか？ (y/n): ")
        if install.lower() == 'y':
            for pkg in missing_packages:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            logger.info("依存パッケージのインストールが完了しました。")
        else:
            logger.warning("依存パッケージをインストールしないとビルドできません。")
            return False
    
    return True

def check_resources():
    """必要なリソースファイルをチェック"""
    # リソースディレクトリ
    resources_dir = Path("resources")
    resources_dir.mkdir(exist_ok=True)
    
    # アイコンファイルをチェック
    icon_path = resources_dir / "app_icon.ico"
    if not icon_path.exists():
        logger.warning("アプリケーションアイコンが見つかりません。デフォルトアイコンを作成します。")
        create_default_icon(icon_path)
    
    # フォントファイルをチェック
    font_path = Path("ipaexg.ttf")
    if not font_path.exists():
        logger.warning("日本語フォントファイル（ipaexg.ttf）が見つかりません。")
        logger.warning("以下のURLからダウンロードしてプロジェクトルートに配置してください:")
        logger.warning("https://moji.or.jp/ipafont/")
        return False
    
    return True

def create_default_icon(icon_path):
    """デフォルトのアイコンファイルを作成（ダミー）"""
    # 実際には何もしないが、本来はデフォルトアイコンを生成するコードが入る
    logger.warning("注意: アイコンファイルが作成されていません。Windowsのデフォルトアイコンが使用されます。")

def main():
    """メイン実行関数"""
    logger.info("ナレッジGPTアプリケーションのEXE化ツール")
    
    # 依存パッケージをチェック
    if not check_dependencies():
        logger.error("依存パッケージエラーのため、ビルドを中止します。")
        return
    
    # リソースファイルをチェック
    if not check_resources():
        logger.error("リソースファイルエラーのため、ビルドを中止します。")
        return
    
    # 実行ファイルをビルド
    build_executable()

if __name__ == "__main__":
    main()
