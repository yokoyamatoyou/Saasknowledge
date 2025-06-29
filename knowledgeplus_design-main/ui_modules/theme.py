"""Shared UI theming utilities."""


def apply_intel_theme(st):
    """Inject Intel themed CSS into a Streamlit app."""
    st.markdown(
        """
    <style>
    /* インポートフォント（メイリオを優先） */
    * {
        font-family: 'Meiryo', 'メイリオ', 'Hiragino Sans', 'ヒラギノ角ゴシック', sans-serif !important;
    }

    /* インテルカラーパレット */
    :root {
        --intel-blue: #0071c5;
        --intel-dark-blue: #003c71;
        --intel-light-blue: #00c7fd;
        --intel-white: #ffffff;
        --intel-light-gray: #f5f5f5;
        --intel-gray: #e5e5e5;
        --intel-dark-gray: #666666;
    }

    /* メインコンテナ */
    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    }

    /* フォントと背景色 */
    html, body, [class*="st-"] {
        background-color: #FFFFFF;
        color: #3C4043;
    }

    /* ヘッダー */
    h1 {
        color: var(--intel-dark-blue) !important;
        font-weight: 600 !important;
        font-size: 2.5rem !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        padding: 1.5rem !important;
        background: linear-gradient(90deg, var(--intel-white) 0%, var(--intel-light-gray) 50%, var(--intel-white) 100%);
        border: 2px solid var(--intel-blue);
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 113, 197, 0.1);
    }

    /* サブヘッダー */
    h2, h3 {
        color: var(--intel-blue) !important;
        font-weight: 500 !important;
        border-bottom: 2px solid var(--intel-light-gray);
        padding-bottom: 0.5rem !important;
        margin-bottom: 1rem !important;
    }

    /* サイドバー */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--intel-dark-blue) 0%, var(--intel-blue) 100%);
    }

    .css-1d391kg .block-container {
        padding-top: 1rem;
    }

    .sidebar .sidebar-content {
        background: var(--intel-dark-blue);
    }

    /* サイドバー内のテキスト */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3,
    .css-1d391kg .stMarkdown, .css-1d391kg label {
        color: var(--intel-white) !important;
    }

    /* タブコンテナ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--intel-light-gray);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid var(--intel-gray);
    }

    /* 非アクティブタブ */
    .stTabs [data-baseweb="tab"] {
        background: var(--intel-white);
        color: var(--intel-dark-gray);
        border: 1px solid var(--intel-gray);
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    /* アクティブタブ */
    .stTabs [aria-selected="true"] {
        background: var(--intel-blue) !important;
        color: var(--intel-white) !important;
        border: 1px solid var(--intel-blue) !important;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0, 113, 197, 0.3);
    }

    /* ホバー時タブ */
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--intel-light-blue);
        color: var(--intel-white);
        transform: translateY(-1px);
    }

    /* プライマリボタン */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--intel-blue) 0%, var(--intel-dark-blue) 100%);
        color: var(--intel-white);
        border: none;
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 113, 197, 0.2);
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, var(--intel-dark-blue) 0%, var(--intel-blue) 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 113, 197, 0.3);
    }

    /* セカンダリボタン */
    .stButton > button[kind="secondary"] {
        background: var(--intel-white);
        color: var(--intel-blue);
        border: 2px solid var(--intel-blue);
        border-radius: 6px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button[kind="secondary"]:hover {
        background: var(--intel-blue);
        color: var(--intel-white);
        transform: translateY(-1px);
    }

    /* 通常ボタン */
    .stButton > button {
        background: var(--intel-light-gray);
        color: var(--intel-dark-blue);
        border: 1px solid var(--intel-gray);
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: var(--intel-gray);
        border-color: var(--intel-blue);
    }

    /* 入力フィールド */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 2px solid var(--intel-gray);
        border-radius: 6px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--intel-blue);
        box-shadow: 0 0 0 3px rgba(0, 113, 197, 0.1);
    }

    /* スライダー */
    .stSlider > div > div > div > div {
        background: var(--intel-blue);
    }

    /* チェックボックス・ラジオボタン */
    .stCheckbox > label > div:first-child, .stRadio > label > div:first-child {
        border-color: var(--intel-blue);
    }

    .stCheckbox > label > div[data-checked="true"],
    .stRadio > label > div[data-checked="true"] {
        background-color: var(--intel-blue);
    }

    /* エキスパンダー */
    .streamlit-expanderHeader {
        background: var(--intel-light-gray);
        border: 1px solid var(--intel-gray);
        border-radius: 6px;
        color: var(--intel-dark-blue);
        font-weight: 500;
    }

    /* アラート・通知 */
    .stAlert {
        border-radius: 6px;
        border-left: 4px solid var(--intel-blue);
    }

    .stSuccess {
        background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left-color: var(--intel-light-blue);
    }

    .stInfo {
        background: linear-gradient(90deg, #eff6ff 0%, #dbeafe 100%);
        border-left-color: var(--intel-blue);
    }

    .stWarning {
        background: linear-gradient(90deg, #fffbeb 0%, #fef3c7 100%);
        border-left-color: #f59e0b;
    }

    .stError {
        background: linear-gradient(90deg, #fef2f2 0%, #fecaca 100%);
        border-left-color: #ef4444;
    }

    /* プログレスバー */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--intel-light-blue) 0%, var(--intel-blue) 100%);
    }

    /* データフレーム */
    .dataframe {
        border: 1px solid var(--intel-gray);
        border-radius: 6px;
    }

    .dataframe th {
        background: var(--intel-blue);
        color: var(--intel-white);
        font-weight: 600;
    }

    .dataframe td:hover {
        background: var(--intel-light-gray);
    }

    /* チャットメッセージ */
    .stChatMessage {
        border-radius: 8px;
        border: 1px solid var(--intel-gray);
        margin: 0.5rem 0;
    }

    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, var(--intel-blue) 0%, var(--intel-dark-blue) 100%);
        color: var(--intel-white);
    }

    .stChatMessage[data-testid="assistant"] {
        background: var(--intel-light-gray);
        color: var(--intel-dark-blue);
    }

    /* ステータスメッセージ */
    .stStatus {
        border-radius: 6px;
        border: 1px solid var(--intel-blue);
    }

    /* フッター */
    .footer-text {
        text-align: center;
        color: var(--intel-dark-gray);
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid var(--intel-gray);
        background: var(--intel-light-gray);
        border-radius: 6px;
    }

    /* モード選択のスタイル */
    .mode-header {
        background: linear-gradient(135deg, var(--intel-blue) 0%, var(--intel-light-blue) 100%);
        color: var(--intel-white);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 113, 197, 0.2);
    }

    /* カードスタイル */
    .info-card {
        background: var(--intel-white);
        border: 1px solid var(--intel-gray);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }

    .info-card:hover {
        box-shadow: 0 4px 16px rgba(0, 113, 197, 0.1);
        transform: translateY(-2px);
    }

    /* スピナー・ローディング */
    .stSpinner {
        color: var(--intel-blue);
    }

    /* ドキュメントカード */
    .doc-card {
        border: 1px solid #dfe1e5;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,.3);
        font-size: 10pt;
    }

    /* Search input styling */
    [data-testid="stTextInput"] input {
        border-color: #dfe1e5;
        border-radius: 24px;
        padding: 10px 20px;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 1px #1a73e8;
    }

    /* Button styling */
    [data-testid="stButton"] button {
        background-color: #1a73e8;
        color: #FFFFFF;
        border-radius: 4px;
        border: none;
        font-size: 10pt;
    }

    /* 縦線区切り */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--intel-blue) 50%, transparent 100%);
        margin: 2rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
