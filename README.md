# Internal_AI_tool
Internal AI Tool is a Streamlit app for AI chat with RAG and vector store management.

## 環境要件
- Python: 3.11+
- ライブラリ:
  - openai==1.76.0
  - streamlit==1.33.0
  - python-dotenv==1.1.1

## 環境変数 (.env)
- `OPENAI_API_KEY` 必須
- `DEFAULT_MODEL` 任意（既定: `gpt5-mini`）

## セットアップ
1. 依存ライブラリをインストール
   - `pip install -r requirements.txt`
2. `.env` を用意し `OPENAI_API_KEY` を設定
3. 起動
   - `streamlit run app.py`

## 機能
- AIチャット: OpenAI Responses API で応答生成（ストリーミング対応）
- RAG: Responses API の `file_search` ツール対応、ベクターストア複数選択
- モデル切替: 既定は `gpt5-mini`、選択: `gpt5` / `gpt5-mini` / `gpt5-nano` / `o4-mini`
- 入力固定: `st.chat_input` により画面下部固定
- 生成中制御: 生成中は追加入力とダウンロードを無効化
- ダウンロード: チャット履歴を `md` / `json` / `txt` で保存、LF/CRLF 選択
- RAG管理: ベクターストアの作成・名称変更・削除、ファイル追加・削除
- 一覧更新: ベクターストア/ファイル一覧の手動リフレッシュボタン
- 設定: モデル、RAG使用、ストア選択、保存形式、改行コード（サイドバー下部）
- Web検索: Built-in Web Search ツールを利用（モデル固定: gpt-5-mini）
- 画像生成: 画像生成APIで画像を生成（モデル固定: gpt-5-nano 表示／内部は image モデルを使用）

## レイアウト
- サイドバー: 「AIツール」ヘッダー + RAG利用切替、ストア選択（複数可）
- メイン: タブで「AIチャット」「Web検索」「画像生成」「RAG管理」「設定」
  - AIチャット: チャット履歴表示、固定入力欄（画面最下部）、履歴ダウンロード/クリア
  - Web検索: クエリ入力→検索→結果表示（ストリーミング）
  - 画像生成: プロンプトとサイズを指定して生成
  - 設定: モデル選択、ダウンロードのファイル種別と改行コード

## 非機能
- 高速化: `st.session_state` と `st.cache_*` によるキャッシュ
- 手動リフレッシュ: 一覧の「再読込」ボタンでキャッシュ刷新
- エラー表示: 直前のエラーをUI上に保持表示

## 注意
- RAGのベクターストア/ファイル操作は OpenAI Vector Stores API を利用しています。
- `file_search` ツールは `tools=[{"type": "file_search", "vector_store_ids": ["..."]}]` 形式で利用します。
- モデル名は設定タブでフレンドリ名を選択（内部でOpenAI実モデルにマップ）。
  - `gpt5` → `gpt-4o`
  - `gpt5-mini` → `gpt-4o-mini`
  - `gpt5-nano` → `gpt-4o-mini`
  - `o4-mini` → `o4-mini`
  - `DEFAULT_MODEL` はフレンドリ名または実モデルIDのどちらでも指定可。
