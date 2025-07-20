# Saasknowledge 日本語ガイド

このリポジトリは、Streamlit を用いたナレッジベース構築・検索・チャットアプリの試作版です。ファイルをアップロードし、OpenAI の埋め込みを活用した検索やチャットが行えます。

## セットアップ手順

1. Python 環境を準備し、リポジトリのルートで以下を実行して依存パッケージをインストールします。
   ```bash
   scripts/install_light.sh
   scripts/install_extra.sh  # 必要に応じて heavy パッケージを追加
   ```
   FAQ 生成では Web からテキストを取得するため `requests` と
   `beautifulsoup4` が必要です。これらは `requirements-light.txt` に含まれています。
2. OpenAI API キーを環境変数 `OPENAI_API_KEY` に設定します。`.env.example` をコピーしても構いません。
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

## アプリの起動

次のコマンドでアプリを起動します。
```bash
./run_app.sh          # Linux/macOS
# または
run_app.bat           # Windows
```

起動後、ブラウザから表示されるインターフェースで PDF、Word、画像、CAD などのファイルをアップロードし、ナレッジベースを構築できます。チャット画面ではアップロードしたドキュメントの内容を参照しながら質問が可能です。

デフォルトではサイドバーは表示された状態で起動します。折りたたんだ状態で開始したい場合は起動前に次の環境変数を設定してください。

```bash
export SIDEBAR_DEFAULT_VISIBLE=false
```

サイドバーの状態は `chat_history/sidebar_state.json` に保存されます。再表示されたくない場合は、このファイルを削除するか `SIDEBAR_DEFAULT_VISIBLE=false` を設定して起動してください。デフォルト値は `true` です。

## 保存先ディレクトリの変更

デフォルトではリポジトリのルートにある `knowledge_base/` フォルダにファイルやメタデータが保存されます。保存場所を変更する場合は以下の環境変数を設定してください。

- `KNOWLEDGE_BASE_DIR` – ナレッジベースの保存先
- `CHAT_HISTORY_DIR` – チャット履歴の保存先
- `PERSONA_DIR` – ペルソナ設定の保存先

## FAQ作成

`generate_faq.py` を実行すると、ナレッジベースから FAQ を生成できます。
`--source` オプションでテキストや URL を指定すると、その内容から Q&A を作成します。

```bash
python knowledgeplus_design-main/generate_faq.py my_kb --source "https://example.com"
```

GPT‑4.1 mini を使用し、温度は生成した質問数に応じて 0 から 0.8 まで
0.01 ずつ上昇します。

## テスト

自動テストは `pytest -q` で実行できます。開発前に `scripts/install_light.sh` を実行して必要な依存パッケージを導入してください。
テストのみを実行したい場合は、`numpy`、`requests`、`PyMuPDF` など必須ライブラリだけをインストールする `scripts/install_tests.sh` を使います。CI でもこのスクリプトを実行してから `pytest` を呼び出しています。

## 参考

より詳細な設計や統合方針については `knowledgeplus_design-main/docs` 以下のドキュメントを参照してください。
