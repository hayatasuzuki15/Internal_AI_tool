# Internal_AI_tool
Internal AI Tool は、Streamlit 製の社内向け AI ツールです。チャット、Web 検索、画像生成、RAG 管理に加え、コード改修ツールを備えています。

## 動作環境
- Python: 3.11+
- 依存パッケージ:
  - openai==1.76.0
  - streamlit==1.33.0
  - python-dotenv==1.1.1

## 環境変数 (.env)
- `OPENAI_API_KEY`: OpenAI API キー（必須）
- `DEFAULT_MODEL`: 既定のモデル ID またはフレンドリ名。省略時は `gpt-5-mini`

フレンドリ名と実モデル ID の対応:
- `gpt-5` → `gpt-5`
- `gpt-5-mini` → `gpt-5-mini`
- `gpt-5-nano` → `gpt-5-nano`
- `o4-mini` → `o4-mini`

`DEFAULT_MODEL` には上記いずれか、または実モデル ID を指定できます。

## セットアップと起動
1. 依存パッケージをインストール
   - `pip install -r requirements.txt`
2. `.env` を用意（`OPENAI_API_KEY` を設定）
3. 起動
   - `streamlit run app.py`

## 全体構成（サイドバー）
- ツール選択: `AIツール` / `コード改修ツール`

---

## AIツール（タブ構成）

### AIチャット
- OpenAI Responses API でのチャット
- RAG 設定（Expander 内）
  - `file_search` ツールを使い、選択した Vector Store を参照
- 下部固定の入力欄（履歴が上に積み上がる UI）
- チャット履歴のダウンロード: `md` / `json` / `txt`（改行コード: `CRLF` / `LF`）
- 履歴クリアボタン

### Web検索
- Built-in `web_search` ツールを使用
- モデルは実質 `gpt-5-mini` 固定

### 画像生成
- OpenAI Images API（`gpt-image-1`）で画像生成
- 入力: プロンプト、サイズ（`1024x1024` / `1024x1536` / `1536x1024` / `auto`）
- 出力: 画像の表示（複数の場合はグリッド表示）

### RAG管理
- Vector Store の一覧/作成/名称変更/削除
- ファイルの一覧/追加/削除

### 設定
- モデル選択（AIチャット用）: `gpt-5` / `gpt-5-mini` / `gpt-5-nano` / `o4-mini`
- ダウンロード設定（AIチャット履歴）: ファイルタイプ（`md` / `json` / `txt`）、改行コード（`CRLF` / `LF`）
- 現在の設定の確認（JSON 表示）

---

## コード改修ツール
2 タブ構成（`実行` / `設定`）。Responses API を 2 段階で使用します。

### 実行タブ
- 入力（ファイル 3 つ）
  - INPUT1: 改修対象のソースコード（拡張子: `py` / `sql`）
  - INPUT2: 設計書（改修前、拡張子: `txt` / `md`）
  - INPUT3: 設計書（改修後、拡張子: `txt` / `md`）
- 一括実行
  1) INPUT2/INPUT3 の差分要約を生成（Step1）
  2) Step1 の要約に基づき、INPUT1 のコードを改修（Step2）
  3) 差分表示（unified diff）と修正後コードの表示/ダウンロード
- ダウンロード
  - 修正後コードは `modified_{元ファイル名}` でダウンロード
  - 拡張子は INPUT1 に合わせて `py` / `sql`

### 設定タブ
- モデル選択（コード改修ツール用）: `gpt-5` / `gpt-5-mini` / `gpt-5-nano` / `o4-mini`
- プロンプト編集
  - Step1（設計差分抽出）用テンプレート（編集可能）
  - Step2（コード改修）用テンプレート（編集可能）

---

## 補足
- キャッシュや状態
  - `st.session_state` と `st.cache_*` を適切に利用
- エラー表示
  - 直近のエラーは各画面上部に表示（`st.info` / `st.error`）
- モデル名の表記
  - 画像生成は UI 上の表記に関わらず `gpt-image-1` を使用

## 既知の制限
- 大きなファイルや長文入力はトークン制限により要約・分割が必要になる場合があります。
- Web検索の結果はモデル応答に依存し、出典リンク等は応答内容により変化します。
