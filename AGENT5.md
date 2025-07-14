Codexへの指示プロンプト

\# 役割  
あなたは、優秀なフロントエンドエンジニアです。既存のPython StreamlitアプリケーションのUI/UXを、ユーザーからのフィードバックに基づき、より洗練されたモダンなデザインにリファクタリングする役割を担います。  
\# 背景  
現在のStreamlitアプリケーションには、以下のUI/UXに関する課題があります。

1. **レイアウト**: ウィンドウサイズにUIが収まらず、左下部分が切れて表示される。  
2. **コンポーネントのサイズ**: 「モデル選択」「Temperature」などの設定項目や、アクションボタンの横幅が広すぎる。  
3. **ファイルアップロード**: ファイルアップロードのUIが場所を取りすぎているため、よりコンパクトなリスト表示形式にしたい。  
4. **配色**: 現在の青を基調としたボタンの色を、よりモダンで落ち着いた配色に変更したい。

幸い、これらの問題を解決するための設計指針がすでに存在します。unified\_app.pyを起点とする新しいUI構造と、ui\_modules/static/theme.cssのカスタムテーマを活用して、以下の具体的な修正を実施してください。

**\# 実行タスク**

### **1\. UI要素の横幅調整**

st.columnsを利用して、ボタンや設定項目のレイアウトを調整し、横幅を縮小してください。

対象ファイル: ui\_modules/search\_ui.py など、ボタンが配置されている各UIモジュール  
修正例: ui\_modules/search\_ui.py

Python

\# 修正前  
\# col1, col2, \_ \= st.columns(\[1, 1, 4\])

\# 修正後: ボタンが配置される列の比率を小さくする  
col1, col2, \_ \= st.columns(\[0.15, 0.15, 0.7\]) 

if col1.button("検索", type\="primary"):  
    \# ...  
if col2.button("クリア"):  
    \# ...

この修正を、アプリケーション内の他のボタンレイアウトにも適用してください。

---

### **2\. ボタンの配色変更**

theme.cssファイルを編集し、プライマリボタンの配色を現在の青色からモダンなグレー系に変更してください。

対象ファイル: ui\_modules/static/theme.css  
修正内容: 以下のCSSスタイルを適用してください。

CSS

/\* 既存の .stButton \> button\[kind="primary"\] スタイルを以下に置き換える \*/

.stButton \> button\[kind="primary"\] {  
    background-color: \#5f6368; /\* モダンなグレー \*/  
    color: \#ffffff;  
    border: none;  
    border-radius: 6px;  
    padding: 0.75rem 1.5rem;  
    font-weight: 600;  
    transition: all 0.3s ease;  
}

.stButton \> button\[kind="primary"\]:hover {  
    background-color: \#3c4043; /\* ホバー時により濃いグレーに \*/  
    transform: translateY(-2px);  
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);  
}

---

### **3\. レイアウトの再編成**

unified\_app.pyの全体的なレイアウト構成に倣い、以下の点を確実に実装してください。

* 設定項目の集約:  
  「モデル」「Temperature」などの各種設定は、ui\_modules/sidebar.py内で定義されているサイドバーのst.expander内にすべて移動させ、メイン画面のスペースを確保してください。  
* ファイルアップロードUIの改善:  
  ui\_modules/management\_ui.py内で、ファイルアップロード機能をst.file\_uploaderを用いて実装し、アップロードされたファイルはデータフレームやリスト形式で表示するようにしてください。メイン画面の大きなアップロード枠は不要です。

\# 最終的な目標  
以上のタスクを完了させることで、初期報告にあったレイアウト崩れや視覚的な問題を解消し、ユーザーにとって直感的で使いやすい、モダンなUIを持つアプリケーションを完成させてください。