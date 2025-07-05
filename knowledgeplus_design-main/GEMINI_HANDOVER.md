# Gemini CLI 作業引き継ぎ書

## 1. 現在の進捗

### 「Task 1-3: 検索＆対話エンジンの部品化」の進捗

*   `shared/chat_controller.py` のクラス化とメソッド化が完了しました。
*   `knowledge_gpt_app/app.py` および `unified_app.py` における `ChatController` の利用への移行が完了しました。

### テスト環境のセットアップとエラー修正

これまでの作業で発生した以下のテストエラーについて修正を行いました。

*   **循環参照の問題:** `shared/kb_builder.py` と `mm_kb_builder/app.py` 間の循環参照を解決しました。`KnowledgeBuilder` の初期化時に必要な関数を引数として渡すように変更しました。
*   **`pytesseract` のインポートエラー:** `knowledge_gpt_app/app.py` で `pytesseract` が常にインポートされるように修正しました。
*   **`streamlit.experimental_rerun` の非推奨化:** `ui_modules/thumbnail_editor.py` および関連テストで `st.experimental_rerun()` を `st.rerun()` に置き換えました。
*   **`st.session_state` のモック問題:** `tests/test_unified_app.py` において、`st.session_state` を辞書のように振る舞い、かつ属性アクセスも可能なオブジェクトとしてモックするように修正しました。
*   **`test_reindex.py` のインデントエラー:** `test_reindex.py` のインデントがずれていた問題を修正しました。
*   **`test_reindex.py` のテストデータ更新:** ストップワードとして扱われない単語
    "apple", "banana", "orange" を使用するよう変更し、検索エンジンのテストが正しく
    動作することを確認しました。
*   **`streamlit` モジュール不足の解決:** `ModuleNotFoundError: No module named 'streamlit'`
    でテストが失敗したため、まず `scripts/install_light.sh` を実行して基本パッケージを
    インストールし、続けて `scripts/install_extra.sh` で重いライブラリを追加する手順を
    記載しました。初回起動時のアンケートを抑制するには、以下
    を一度実行しておくと自動テレメトリー確認のプロンプトを回避できます。

    ```bash
    mkdir -p ~/.streamlit
    printf "[browser]\ngatherUsageStats = false\n" > ~/.streamlit/config.toml
    ```

## 2. 現在の課題

特に共有すべき大きな課題はありません。

## 3. 今後の作業

### その他の作業

*   上記テストが全てパスした後、`shared/search_engine.py` に追加したデバッグログ（`print` ステートメント）を削除してください。
*   最終的なコードのクリーンアップ（不要なコメントや一時的なコードの削除）。
*   プロジェクト固有のビルド、リンティング、型チェックコマンドを実行し、コード品質と標準への準拠を確認してください。

### PyMuPDF移行について

- `pdf2image` に依存するコードを PyMuPDF に置き換え、poppler なしで PDF の画像化ができるようにしてください。
- 詳細手順は `docs/pymupdf_migration.md` を参照のこと。
---
**このドキュメントは、Gemini CLI との会話のコンテキストを維持するために作成されました。次の会話を開始する際は、このドキュメントを参照するように指示してください。**
