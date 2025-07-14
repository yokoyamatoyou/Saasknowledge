**Codex向け UIリファクタリング指示書**

**\# 概要**

このプロジェクトは、Streamlitで構築されたWebアプリケーションのUIリファクタリングを目的としています。現在のメイン画面の右側に意図しない大きな空白スペースが存在しており、これを削除して、より洗練されたレイアウトに修正してください。

**\# 背景と目的**

現状のアプリケーションでは、st.columnsによるレイアウト設定が原因で、メインコンテンツの右側に大きな余白が生まれてしまっています。このスペースは機能的に何かの役割を持つものではなく、デザイン上の不要な要素です。  
今回のリファクタリングの目的は、この不要なスペースを完全に削除し、コンテンツ（特に検索ボタン周り）の配置を最適化することです。  
**\# 対象ファイル**

リファクタリングの対象となるファイルは以下の通りです。

* knowledgeplus\_design-main/ui\_modules/search\_ui.py

**\# 修正箇所の特定**

対象ファイル内の display\_search\_ui 関数に問題のコードが存在します。

**問題のコードブロック:**

Python

def display\_search\_ui():  
    """検索UIを表示する"""  
    \# （...中略...）  
      
    \# ボタンのレイアウト  
    col1, col2, \_ \= st.columns(\[1, 1, 4\])  
    with col1:  
        if st.button("🔍 検索実行", key="search\_button"):  
            \# （...検索処理...）  
    with col2:  
        if st.button("クリア", key="clear\_button"):  
            \# （...クリア処理...）

    \# （...以下略...）

問題点:  
st.columns(\[1, 1, 4\]) という記述により、画面が3つの列に分割されています。最初の2つの列（比率1, 1）に「検索実行」ボタンと「クリア」ボタンが配置されていますが、3番目の列（比率4）は何も配置されずに空白となっています。これが、画面右側に大きな不要スペースを生み出している原因です。  
**\# 実行すべきタスク**

以下の手順に従って、コードをリファクタリングしてください。

**\#\# ステップ1: 不要な列の削除**

st.columns の定義を修正し、不要な3番目の列を削除します。ボタンは2つなので、2つの列を作成するように変更します。

**修正前:**

Python

col1, col2, \_ \= st.columns(\[1, 1, 4\])

**修正後:**

Python

col1, col2 \= st.columns(2)

*ヒント: st.columns(2) は st.columns(\[1, 1\]) と等価であり、2つの同じ幅の列を作成します。これにより、2つのボタンが画面の左半分にきれいに並びます。*

**\#\# ステップ2: 修正後のコード全体像**

以下が display\_search\_ui 関数の修正後の理想的な状態です。このコードに書き換えてください。

Python

\# knowledgeplus\_design-main/ui\_modules/search\_ui.py

import streamlit as st  
from services.search\_service import search\_and\_display  
from state\_management.session\_state import get\_session\_state

def display\_search\_ui():  
    """検索UIを表示する"""  
    state \= get\_session\_state()

    \# 検索クエリ入力  
    st.text\_input("検索クエリを入力してください", key="query\_input")

    \# 検索モードの選択  
    st.selectbox(  
        "検索モードを選択してください",  
        \["ナレッジ検索", "Web検索"\],  
        key="search\_mode"  
    )

    \# ボタンのレイアウトを修正  
    col1, col2 \= st.columns(2)  \# \<-- ★この行を修正  
    with col1:  
        if st.button("🔍 検索実行", key="search\_button"):  
            search\_and\_display()  
    with col2:  
        if st.button("クリア", key="clear\_button"):  
            state.query\_input \= ""  
            state.search\_results \= None  
            st.experimental\_rerun()

    \# 検索結果の表示エリア  
    st.markdown("---")  
    st.write("\#\#\# 検索結果")  
    if state.search\_results:  
        st.write(state.search\_results)  
    else:  
        st.info("ここに検索結果が表示されます。")

**\# 期待される結果**

* アプリケーションのメイン画面右側にあった大きな空白スペースがなくなる。  
* 「検索実行」ボタンと「クリア」ボタンが、画面の左側に隣接してきれいに配置される。  
* 全体的なレイアウトが引き締まり、より直感的に操作できるUIになる。

**\# 指示は以上です。**

この指示書に基づいて、UIのリファクタリングを実行してください。