# Internal AI Tool 設計書

## 1. 概要
- Streamlit ベースの社内向け AI ツール。チャット補助、Web検索、画像生成、RAG 管理、コード改修支援を単一アプリで提供する。
- OpenAI Responses API / Images API / Vector Store API と連携し、リアルタイムのテキスト生成とナレッジ活用を実現。
- UI はタブとサイドバーで構成され、用途ごとに機能を切り替える。コード改修ツールは専用画面として提供。

## 2. 実行環境・依存関係
- Python 3.10 以上を想定。
- 主要ライブラリ
  - `streamlit` : Web UI 構築。`@st.cache_resource` / `@st.cache_data` で API クライアントやデータをキャッシュ。
  - `openai` : OpenAI SDK。Responses API, Images API, Vector Stores API を使用。
  - `python-dotenv` : `.env` から環境変数を読込む。
  - その他標準ライブラリ: `os`, `time`, `json`, `base64`, `difflib`, `datetime`, `typing`。
- 必須環境変数
  - `OPENAI_API_KEY` : OpenAI API 認証キー。
  - `DEFAULT_MODEL` (任意) : 既定のチャットモデルを指定。未設定時は `gpt-5-mini`。

## 3. 構成要素

### 3.1 環境設定
- `load_dotenv(override=True)` で `.env` をロード。
- `get_client()` : `@st.cache_resource` でラップされた OpenAI クライアントファクトリ。

### 3.2 モデル識別子ユーティリティ
- `FRIENDLY_TO_OPENAI` / `OPENAI_TO_FRIENDLY` : 表示名と実際のモデル ID を双方向にマップ。
- `resolve_model_for_api(name: str)` : UI から渡されるモデル名を OpenAI SDK 用 ID に正規化。
- `normalize_model_for_ui(name: str)` : 環境変数や API から得た ID を UI 表示用名称に変換。

### 3.3 セッションステート初期化
- `init_session_state()` : チャット履歴、各タブの進行状態、RAG 設定、コード改修ツール専用のステートを初期化。
  - `messages`, `is_generating`, `model`, `use_rag`, `selected_vector_store_ids`, `download_format`, `newline`, `prev_response_id` などの基本情報。
  - Web検索 (`web_is_generating`, `web_query`, `last_web_error`)、画像生成 (`img_is_generating`, `img_prompt`, `img_size`, `last_image_error`) の管理フラグ。
  - コード改修ツール用 (`code_fix_*`) に、モデル選択・プロンプト・実行状況・差分・ログ・追い質問履歴などを保持。

### 3.4 フォーマット変換ユーティリティ
- `nl_join()` : LF/CRLF の正規化。
- `to_markdown()`, `to_txt()`, `to_json()` : チャット履歴を各フォーマットでダウンロード可能に整形。
- コード改修ログ用にも `codefix_log_to_json/markdown/txt()` を提供。

### 3.5 キャッシュされたデータアクセス
- `list_vector_stores()`, `list_vector_store_files()` : OpenAI Vector Store のリストを取得し、`@st.cache_data` でキャッシュ。更新操作後は `.clear()` でキャッシュを無効化。

### 3.6 RAG 管理アクション
- `rag_create_vector_store(name)`, `rag_rename_vector_store(vs_id, new_name)`, `rag_delete_vector_store(vs_id)` : Vector Store の CRUD。
- `rag_upload_files(vs_id, uploaded_files)` : Streamlit のアップロードデータを OpenAI Vector Store に登録。
- `rag_delete_file(vs_id, file_id)` : Vector Store のファイル削除。

### 3.7 応答生成／ツール連携ロジック
- `stream_response_text(prompt, model, use_rag, vector_store_ids, previous_response_id)`
  - Responses API のストリーミングを利用し、UI に逐次テキストを送るジェネレータ。
  - RAG を有効化した場合 `file_search` ツールを含める。
  - 最終応答テキストと `response_id` をジェネレータ属性として格納。
- `stream_web_search(query)` : `web_search` ツールを呼び出し、検索結果テキストをストリーム。
- 画像生成 (`ui_image_generation()` 内) では `client.images.generate()` を呼び出し、Base64 を復号してダウンロード可能なリンクを表示。

### 3.8 チャット処理
- `handle_user_message(user_text)`
  - ユーザ発話を履歴に追加し、`stream_response_text()` でアシスタント応答を取得。
  - 生成中フラグ・エラー管理、RAG 情報の保存、レスポンス ID の連結管理を行う。

### 3.9 UI コンポーネント
- `sidebar_settings()` : サイドバーに RAG 設定とツールモード切替 (`AIツール` / `コード改修ツール`) を表示。
- `ui_ai_chat()` : チャットタブ本体。履歴の描画、入力欄固定のカスタム CSS/JS、ダウンロード、履歴クリアを提供。
- `ui_applied_tools()` : コード改修ツールの実処理。
  - Step1: 与えられた設計書差分抽出。
  - Step2: コード修正。差分・ログ保存、ダウンロード、追い質問機能、実行履歴ダウンロードを提供。
- `ui_code_fix_followups()` : 追い質問チャットの送受信と履歴管理。
- `ui_code_fix_full_log_download()` : コード改修ログの一括ダウンロード。
- `ui_web_search()` : Web検索タブ。`st.status` で進捗表示。
- `ui_image_generation()` : 画像生成タブ。結果画像の保存リンクを提示。
- `ui_rag_management()` : Vector Store 管理タブ。作成・選択・名称変更・削除・ファイル一覧・アップロードを提供。
- `ui_settings()` : 設定タブ。モデル・ダウンロード形式・改行コードの選択と現在値表示。

### 3.10 エントリポイント
- `main()`
  - `st.set_page_config` → `init_session_state()` → API キー確認。
  - サイドバー表示後、ツールモードに応じて画面を切替。
  - `AIツール` モードでは 5 タブ構成 (`AIチャット`, `Web検索`, `画像生成`, `RAG管理`, `設定`) を表示し、`設定` タブを最初に評価して他タブで最新状態を利用。
  - `コード改修ツール` モードでは `ui_applied_tools()` のみをレンダリング。

## 4. 処理フロー
```mermaid
graph TD
    A[アプリ起動] --> B[main() 実行]
    B --> C[init_session_state()]
    C --> D{OPENAI_API_KEY 有無}
    D -- 未設定 --> E[警告表示]
    D -- 設定済 --> F[サイドバー sidebar_settings()]
    F --> G{tool_mode}
    G -- "コード改修ツール" --> H[ui_applied_tools()]
    G -- "AIツール" --> I[タブ生成]
    I --> J[設定タブ: ui_settings()]
    I --> K[チャットタブ: ui_ai_chat()]
    K --> L[ユーザ入力]
    L --> M[handle_user_message()]
    M --> N[stream_response_text()]
    N --> O[OpenAI Responses API]
    I --> P[Web検索タブ: ui_web_search()]
    P --> Q[stream_web_search() -> OpenAI web_search]
    I --> R[画像生成タブ: ui_image_generation()]
    R --> S[OpenAI Images API]
    I --> T[RAG管理タブ: ui_rag_management()]
    T --> U[list_vector_stores()/files() -> OpenAI Vector Store API]
    H --> V[Step1/Step2 Responses API 呼出]
    H --> W[追い質問: ui_code_fix_followups()]
    W --> X[Responses API 再呼出]
```

## 5. 状態管理
- すべてのタブで `st.session_state` を共有し、生成中フラグで UI 操作を制御。
- チャット履歴やダウンロード形式などはセッション間で保持。
- コード改修ツールは複数段階の実行結果（差分・ログ・追い質問）をセッション内に保存し、`st.download_button` でエクスポート可能。

## 6. 外部 API と入出力
- Responses API
  - 入力: `model`, `input`（system/user メッセージ配列）, 任意で `tools`, `previous_response_id`。
  - 出力: ストリーミングテキスト、`final_text`, `id`。
- Web Search ツール
  - 入力: `web_search` ツールタイプ。
  - 出力: ストリーミングテキスト。
- Images API
  - 入力: `model="gpt-image-1"`, `prompt`, `size`, `output_format="png"`, `n`, `quality`。
  - 出力: Base64 画像または URL。
- Vector Store API
  - Store 作成/更新/削除、ファイルのアップロード/削除、一覧取得。

## 7. エラーハンドリング
- 例外は `try-except` で捕捉し、`st.session_state.last_error` 等に格納して UI に反映。
- コード改修ツールでは API 応答欠落時に `RuntimeError` を送出し、ユーザにプロンプト調整を促す。
- Web検索・画像生成・RAG 操作でも例外を捕捉し、`st.error` / `st.warning` で通知。

## 8. 拡張のポイント
- モデル一覧 (`MODEL_OPTIONS`) を更新することで簡単に新モデルへ対応可能。
- Vector Store 操作は `.clear()` を通じてキャッシュを制御。別ストレージ連携にも応用しやすい構造。
- セッションステートを統一的に初期化しているため、新タブや機能を追加する際は `init_session_state()` に専用ステートを追記すればよい。

