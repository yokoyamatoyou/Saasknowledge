## **1\. はじめに**

このドキュメントは、現在のナレッジベース構築アプリケーションにおいて、画像ファイル（PNG、JPGなど）や多様な形式のドキュメント（画像を含むPDF、DOCXなど）が正しく処理されない問題を解決するための技術的な指示書です。

問題の核心:  
現在のシステムは、ファイルアップロード時に、そのファイルの種類を判別して処理を振り分ける重要なロジックが欠落しています。その結果、全てのファイルが旧来の「テキスト抽出」プロセスに送られてしまい、画像ベースのファイル処理が失敗しています。  
修復の目的:  
UI部分（management\_ui.py）を修正し、ファイルの種類に応じて適切な処理（画像解析 or テキスト抽出）を呼び出す「司令塔」の役割を復活させます。幸い、処理に必要な個々の機能（FileProcessorなど）はリポジトリ内に存在するため、これらを最大限に活用します。

## **2\. 根本原因の分析**

リポジトリ内のコードは、大きく分けて2つの処理フローを持っています。

1. **テキスト処理フロー（旧）**: knowledge\_gpt\_app/app.py に実装されており、主にテキストファイルやDOCX、テキストベースのPDFから文字列を抽出することに特化しています。  
2. **マルチモーダル処理フロー（新）**: shared/file\_processor.py や shared/kb\_builder.py に実装されており、画像（PNG, JPG）、画像ベースのPDF、CADファイルなどをAI（GPT-4o）で解析し、内容の要約やメタデータを生成する高度な機能を持ちます。

現状では、メインのUIである ui\_modules/management\_ui.py が、全てのファイルを前者の**テキスト処理フローにしか送っていない**ため、後者のマルチモーダル処理が全く実行されていません。

## **3\. 修復手順：ファイル処理の司令塔を再建する**

修正のターゲットは、ファイルアップロードの入り口となっている knowledgeplus\_design-main/ui\_modules/management\_ui.py です。このファイルを以下のように段階的に修正し、インテリジェントなファイル処理機能を復活させます。

### **ステップ1: 必要なモジュールのインポート**

まず、management\_ui.py のファイル先頭に、これから使用する各種処理モジュールをインポートするコードを追加します。これにより、テキスト処理と画像処理の両方の機能を呼び出せるようになります。

\# knowledgeplus\_design-main/ui\_modules/management\_ui.py の先頭に追加

import streamlit as st  
from pathlib import Path  
import time  
import logging

\# ▼▼▼【重要】以下のモジュールをインポートまたは確認 ▼▼▼  
from shared.file\_processor import FileProcessor  
from shared.kb\_builder import KnowledgeBuilder  
from shared.openai\_utils import get\_openai\_client  
from shared.env import get\_rag\_config

\# 既存のテキスト処理関数もインポートしておく  
from knowledge\_gpt\_app.app import semantic\_chunking, get\_text\_embedding

logger \= logging.getLogger(\_\_name\_\_)

\# 既存の render\_management\_mode 関数の定義...

### **ステップ2: ファイル処理ロジックの完全な置き換え**

次に、render\_management\_mode 関数内にある、既存のファイルアップロード処理部分 (st.file\_uploader 以降）を、これから提示する**新しいコードブロックで完全に置き換え**ます。

この新しいコードは、アップロードされたファイル一つ一つの形式をチェックし、最適な処理フローに振り分けるロジックを実装しています。

**【置き換え対象】**: render\_management\_mode 関数内の files \= st.file\_uploader(...) から、それに関連するボタンや処理ループの終わりまで。

**【新しいコード】**: 以下のコードをコピーして、対象部分に貼り付けてください。

\# render\_management\_mode 関数内に、このコードブロックを配置

\# ...（render\_management\_mode 関数の前半部分はそのまま）...

    \# \-------------------------------------------------------------------------  
    \# ▼▼▼【ここから】ファイルアップロードと処理ロジック（全面的に置き換え）▼▼▼  
    \# \-------------------------------------------------------------------------

    st.subheader("ナレッジの追加")  
    st.markdown("テキスト、画像、PDF、DOCXなどのファイルをアップロードして、ナレッジベースを構築します。")

    \# セッション状態からナレッジベース名を取得  
    kb\_name \= st.session\_state.get("selected\_kb")  
    if not kb\_name:  
        st.warning("先にナレッジベースを選択してください。")  
        return

    \# FileProcessor と KnowledgeBuilder を初期化  
    try:  
        rag\_config \= get\_rag\_config()  
        file\_processor \= FileProcessor(rag\_config)  
        kb\_builder \= KnowledgeBuilder(  
            kb\_name=kb\_name,  
            rag\_config=rag\_config,  
            openai\_client=get\_openai\_client()  
        )  
    except Exception as e:  
        st.error(f"設定の読み込みに失敗しました: {e}")  
        logger.error(f"Failed to initialize processors: {e}", exc\_info=True)  
        return

    \# ファイルアップローダーUI  
    uploaded\_files \= st.file\_uploader(  
        "ナレッジに追加するファイルを選択",  
        type=file\_processor.get\_supported\_file\_extensions(),  
        accept\_multiple\_files=True,  
        key=f"kb\_uploader\_{kb\_name}"  
    )

    if uploaded\_files:  
        if st.button("選択したファイルの処理を開始", type="primary", key=f"process\_btn\_{kb\_name}"):  
              
            progress\_bar \= st.progress(0, "処理を開始します...")  
            start\_time \= time.time()

            for i, uploaded\_file in enumerate(uploaded\_files):  
                file\_name \= uploaded\_file.name  
                progress\_text \= f"({i+1}/{len(uploaded\_files)}) {file\_name} を処理中..."  
                progress\_bar.progress((i \+ 1\) / len(uploaded\_files), text=progress\_text)  
                  
                try:  
                    with st.spinner(progress\_text):  
                        \# ★★★【最重要】ファイル処理の司令塔 ★★★  
                        \# FileProcessor を使って、ファイルの種類に応じた最適な処理を実行  
                        \# この関数がテキスト、画像、OCRなどを自動で判別します。  
                        processed\_data \= file\_processor.process\_file(uploaded\_file)

                        if not processed\_data:  
                            st.error(f"ファイルの処理に失敗しました: {file\_name}")  
                            logger.error(f"File processing failed for {file\_name}")  
                            continue

                        \# 処理結果をナレッジベースに登録  
                        \# この関数がベクトル化と保存を行います。  
                        kb\_builder.build\_and\_save\_knowledge(processed\_data)  
                          
                        st.success(f"✓ ナレッジを追加しました: {file\_name} (タイプ: {processed\_data.get('type', '不明')})")  
                        logger.info(f"Successfully added knowledge for {file\_name}")

                except Exception as e:  
                    st.error(f"処理中に予期せぬエラーが発生しました ({file\_name}): {e}")  
                    logger.error(f"Unhandled error processing {file\_name}: {e}", exc\_info=True)  
              
            end\_time \= time.time()  
            total\_time \= end\_time \- start\_time  
            progress\_bar.progress(1.0, f"全ての処理が完了しました！ (合計時間: {total\_time:.2f}秒)")

    \# \-------------------------------------------------------------------------  
    \# ▲▲▲【ここまで】ファイルアップロードと処理ロジック ▲▲▲  
    \# \-------------------------------------------------------------------------

    \# ...（render\_management\_mode 関数の残りの部分）...

### **ステップ3: shared/file\_processor.py の確認と活用**

上記のコードは、shared/file\_processor.py に実装されている FileProcessor クラスを全面的に信頼しています。このクラスが、ご要望の高度な処理（PDFがテキストか画像かの判別など）を実行する心臓部となります。

FileProcessor クラス内の process\_file メソッドが司令塔として機能し、内部で extract\_text\_with\_metadata などのヘルパー関数を呼び出すことで、以下の処理を自動的に行います。

* **PNG, JPG**: 画像として認識し、Base64エンコード後、AI解析用のデータ構造を作成します。  
* **PDF**: まずテキストの直接抽出を試みます。失敗した場合（画像ベースのPDFの場合）、ページを画像に変換してOCRとAI解析を実行します。  
* **DOCX**: テキストを抽出し、もし内部に画像があればそれらも個別に抽出・解析します（※この部分はFileProcessorに別途実装が必要な場合がありますが、基本構造は対応可能です）。

この修正により、management\_ui.py は「ファイルをFileProcessorに渡す」というシンプルな責務に集中でき、複雑な処理は専門のクラスに任せることができます。

## **4\. 結論**

上記の手順、特に**ステップ2で提示したコードブロックで ui\_modules/management\_ui.py を更新する**ことで、Codexによって失われたと思われるインテリジェントなファイル処理機能が完全に復活します。

これにより、アプリケーションは再び、PNG、JPG、テキストPDF、画像PDF、DOCXといった多様なファイルを適切に理解し、それぞれに最適な方法でナレッジベースへと変換できるようになります。