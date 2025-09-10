import os
import io
import json
import time
import base64
import difflib
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from openai.lib.streaming.responses import (
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from streamlit.runtime import exists as runtime_exists

# -----------------------------
# Env & client setup
# -----------------------------
load_dotenv(override=True)


@st.cache_resource(show_spinner=False)
def get_client() -> OpenAI:
    # OpenAI() picks up OPENAI_API_KEY from environment
    return OpenAI()


# -----------------------------
# Constants & helpers
# -----------------------------
DEFAULT_MODEL_RAW = os.getenv("DEFAULT_MODEL", "gpt-5-mini")

# Friendly name to real model ID mapping
FRIENDLY_TO_OPENAI = {
    "gpt-5": "gpt-5",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-5-nano": "gpt-5-nano",
    "o4-mini": "o4-mini",
}
OPENAI_TO_FRIENDLY = {v: k for k, v in FRIENDLY_TO_OPENAI.items()}

MODEL_OPTIONS = list(FRIENDLY_TO_OPENAI.keys())


def resolve_model_for_api(name: str) -> str:
    # Convert UI-friendly name to real model id; pass through known ids
    if name in FRIENDLY_TO_OPENAI:
        return FRIENDLY_TO_OPENAI[name]
    # If a real model id is provided, use it as-is
    if name in OPENAI_TO_FRIENDLY or name.startswith("gpt-") or name.startswith("o"):
        return name
    # Fallback
    return FRIENDLY_TO_OPENAI["gpt-5-mini"]


def normalize_model_for_ui(name: str) -> str:
    # Convert real model id from env to friendly option for UI
    if name in MODEL_OPTIONS:
        return name
    if name in OPENAI_TO_FRIENDLY:
        return OPENAI_TO_FRIENDLY[name]
    # common direct ids
    if name in ("gpt-5", "gpt-5-mini", "o4-mini"):
        return OPENAI_TO_FRIENDLY.get(name, "gpt-5-mini")
    return "gpt-5-mini"


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False
    if "last_error" not in st.session_state:
        st.session_state.last_error: Optional[str] = None
    if "model" not in st.session_state:
        st.session_state.model = normalize_model_for_ui(DEFAULT_MODEL_RAW)
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = False
    if "selected_vector_store_ids" not in st.session_state:
        st.session_state.selected_vector_store_ids: List[str] = []
    if "download_format" not in st.session_state:
        st.session_state.download_format = "md"  # md | json | txt
    if "newline" not in st.session_state:
        st.session_state.newline = "CRLF"  # LF | CRLF
    if "prev_response_id" not in st.session_state:
        st.session_state.prev_response_id: Optional[str] = None
    if "web_is_generating" not in st.session_state:
        st.session_state.web_is_generating = False
    if "img_is_generating" not in st.session_state:
        st.session_state.img_is_generating = False
    if "web_query" not in st.session_state:
        st.session_state.web_query = ""
    if "img_prompt" not in st.session_state:
        st.session_state.img_prompt = ""
    if "img_size" not in st.session_state:
        st.session_state.img_size = "1024x1024"
    if "last_web_error" not in st.session_state:
        st.session_state.last_web_error = None
    if "last_image_error" not in st.session_state:
        st.session_state.last_image_error = None
    # Tool mode (AIツール / 応用ツール)
    if "tool_mode" not in st.session_state:
        st.session_state.tool_mode = "AIツール"
    # 応用ツール用のモック結果テキスト
    if "applied_result_text" not in st.session_state:
        st.session_state.applied_result_text = ""

    # コード改修ツール用ステート
    if "code_fix_model" not in st.session_state:
        st.session_state.code_fix_model = normalize_model_for_ui(DEFAULT_MODEL_RAW)
    if "code_fix_prompt_step1" not in st.session_state:
        st.session_state.code_fix_prompt_step1 = (
            "あなたはシステム設計レビューの専門家です。\n"
            "与えられた『改修前の設計書』と『改修後の設計書』の差分を日本語で簡潔に要約してください。\n"
            "出力は以下のセクションを含む箇条書きにしてください:\n"
            "- 追加された仕様\n- 変更された仕様\n- 削除された仕様\n- 影響範囲（関数/クラス/テーブル/入出力）\n"
            "不明な点があれば推測せず『不明』と明記してください。"
        )
    if "code_fix_prompt_step2" not in st.session_state:
        st.session_state.code_fix_prompt_step2 = (
            "あなたは熟練のソフトウェアエンジニアです。\n"
            "以下の『変更点の要約』に基づいて、与えられたソースコード全体を改修してください。\n"
            "出力はコードのみ（ファイル全体）を返し、説明文やコードブロック記号は含めないでください。\n"
            "指定がない箇所は変更せず、仕様が曖昧な場合はTODOコメントを残してください。"
        )
    if "code_fix_is_running" not in st.session_state:
        st.session_state.code_fix_is_running = False
    if "code_fix_summary" not in st.session_state:
        st.session_state.code_fix_summary = ""
    if "code_fix_modified_code" not in st.session_state:
        st.session_state.code_fix_modified_code = ""
    if "code_fix_diff" not in st.session_state:
        st.session_state.code_fix_diff = ""
    if "code_fix_last_error" not in st.session_state:
        st.session_state.code_fix_last_error = None
    # Follow-up Q&A (for code-fix tool)
    if "code_fix_followup_messages" not in st.session_state:
        st.session_state.code_fix_followup_messages: List[Dict[str, Any]] = []
    if "code_fix_followup_is_running" not in st.session_state:
        st.session_state.code_fix_followup_is_running = False
    if "code_fix_followup_last_error" not in st.session_state:
        st.session_state.code_fix_followup_last_error = None
    # Full log of code-fix API calls
    if "code_fix_log" not in st.session_state:
        st.session_state.code_fix_log: List[Dict[str, Any]] = []


def nl_join(s: str, newline: str) -> str:
    # Normalize to LF, then convert if CRLF requested
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s if newline == "LF" else s.replace("\n", "\r\n")


def to_markdown(messages: List[Dict[str, Any]], newline: str) -> bytes:
    lines: List[str] = ["# Chat Transcript"]
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", ""))
        ts = m.get("ts")
        ts_str = f" ({ts})" if ts else ""
        lines.append(f"\n## {role.capitalize()}{ts_str}\n")
        lines.append(content)
    body = "\n".join(lines)
    return nl_join(body, newline).encode("utf-8")


def to_txt(messages: List[Dict[str, Any]], newline: str) -> bytes:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "").capitalize()
        content = str(m.get("content", ""))
        parts.append(f"{role}:\n{content}\n")
    body = "\n".join(parts)
    return nl_join(body, newline).encode("utf-8")


def to_json(messages: List[Dict[str, Any]], newline: str) -> bytes:
    # Ensure strings only for JSON safety
    safe_msgs = [
        {"role": m.get("role", ""), "content": str(m.get("content", "")), "ts": m.get("ts")}
        for m in messages
    ]
    body = json.dumps(safe_msgs, ensure_ascii=False, indent=2)
    return nl_join(body, newline).encode("utf-8")


# -----------------------------
# Code-fix: history serializers
# -----------------------------
def codefix_log_to_json(entries: List[Dict[str, Any]], newline: str) -> bytes:
    def _stringify(v: Any) -> Any:
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        try:
            return str(v)
        except Exception:
            return ""

    safe = []
    for e in entries:
        safe.append({
            "stage": _stringify(e.get("stage")),
            "ts": _stringify(e.get("ts")),
            "model": _stringify(e.get("model")),
            "request": {
                "system": _stringify((e.get("request") or {}).get("system")),
                "user": _stringify((e.get("request") or {}).get("user")),
                "context_code": _stringify((e.get("request") or {}).get("context_code")),
            },
            "response": _stringify(e.get("response")),
            "error": _stringify(e.get("error")),
        })
    body = json.dumps(safe, ensure_ascii=False, indent=2)
    return nl_join(body, newline).encode("utf-8")


def codefix_log_to_markdown(entries: List[Dict[str, Any]], newline: str) -> bytes:
    lines: List[str] = ["# Code Fix Tool History"]
    for i, e in enumerate(entries, start=1):
        stage = str(e.get("stage", ""))
        ts = str(e.get("ts", ""))
        model = str(e.get("model", ""))
        req = e.get("request", {}) or {}
        system = str(req.get("system", ""))
        user = str(req.get("user", ""))
        ctx = str(req.get("context_code", ""))
        resp = str(e.get("response", ""))
        err = str(e.get("error", ""))

        lines.append(f"\n## {i}. {stage}")
        meta = []
        if ts:
            meta.append(f"Time: {ts}")
        if model:
            meta.append(f"Model: {model}")
        if meta:
            lines.append("- " + " | ".join(meta))

        lines.append("\n### Request")
        if system:
            lines.append("System:\n")
            lines.append(system)
        if user:
            lines.append("\nUser:\n")
            lines.append(user)
        if ctx:
            lines.append("\nContext Code:\n")
            lines.append("```\n" + ctx + "\n```")

        if resp:
            lines.append("\n### Response\n")
            lines.append("```\n" + resp + "\n```")
        if err:
            lines.append("\n### Error\n")
            lines.append(err)

    body = "\n".join(lines)
    return nl_join(body, newline).encode("utf-8")


def codefix_log_to_txt(entries: List[Dict[str, Any]], newline: str) -> bytes:
    parts: List[str] = []
    for i, e in enumerate(entries, start=1):
        parts.append(f"[{i}] {e.get('stage','')}")
        parts.append(f"Time: {e.get('ts','')} | Model: {e.get('model','')}")
        req = e.get("request", {}) or {}
        if req.get("system"):
            parts.append("System:\n" + str(req.get("system")))
        if req.get("user"):
            parts.append("User:\n" + str(req.get("user")))
        if req.get("context_code"):
            parts.append("Context Code:\n" + str(req.get("context_code")))
        if e.get("response"):
            parts.append("Response:\n" + str(e.get("response")))
        if e.get("error"):
            parts.append("Error:\n" + str(e.get("error")))
        parts.append("")
    body = "\n".join(parts)
    return nl_join(body, newline).encode("utf-8")
# -----------------------------
# Cached data accessors
# -----------------------------
@st.cache_data(show_spinner=False)
def list_vector_stores(limit: int = 100) -> List[Dict[str, Any]]:
    client = get_client()
    page = client.vector_stores.list(limit=limit)
    # Convert to simple list of dicts
    items = []
    for vs in page.data:
        items.append({
            "id": vs.id,
            "name": vs.name or vs.id,
            "created_at": vs.created_at,
        })
    return items


@st.cache_data(show_spinner=False)
def list_vector_store_files(vector_store_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    client = get_client()
    page = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=limit)
    files = []
    for f in page.data:
        files.append({
            "id": f.id,
            "status": f.status,
            "created_at": f.created_at,
            "last_error": getattr(f, "last_error", None),
            "usage_bytes": getattr(f, "usage_bytes", None),
            "attributes": getattr(f, "attributes", None),
        })
    return files


# -----------------------------
# RAG management actions
# -----------------------------
def rag_create_vector_store(name: str) -> str:
    client = get_client()
    vs = client.vector_stores.create(name=name)
    list_vector_stores.clear()  # refresh cache
    return vs.id


def rag_rename_vector_store(vs_id: str, new_name: str) -> None:
    client = get_client()
    client.vector_stores.update(vs_id, name=new_name)
    list_vector_stores.clear()


def rag_delete_vector_store(vs_id: str) -> None:
    client = get_client()
    client.vector_stores.delete(vs_id)
    list_vector_stores.clear()


def rag_upload_files(vs_id: str, uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    client = get_client()
    new_ids: List[str] = []
    for uf in uploaded_files:
        try:
            file_tuple = (uf.name, uf.getvalue())  # accepted by SDK
            vs_file = client.vector_stores.files.upload(vector_store_id=vs_id, file=file_tuple)
            new_ids.append(vs_file.id)
        except Exception as e:
            st.session_state.last_error = f"ファイル追加に失敗: {uf.name}: {e}"
    list_vector_store_files.clear()
    return new_ids


def rag_delete_file(vs_id: str, file_id: str) -> None:
    client = get_client()
    client.vector_stores.files.delete(vector_store_id=vs_id, file_id=file_id)
    list_vector_store_files.clear()


# -----------------------------
# Chat logic using Responses API
# -----------------------------
def stream_response_text(
    prompt: str,
    model: str,
    use_rag: bool,
    vector_store_ids: List[str],
    previous_response_id: Optional[str] = None,
):
    client = get_client()

    tools: List[Dict[str, Any]] = []
    if use_rag and vector_store_ids:
        tools.append({
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
        })

    # Stream using helper to accumulate text deltas
    with client.responses.stream(
        model=resolve_model_for_api(model),
        input=[{"role": "user", "content": prompt}],
        tools=tools,
        include=(
            ["file_search_call.results"] if use_rag and vector_store_ids else None
        ),
        previous_response_id=previous_response_id,
    ) as stream:
        # Yield text delta chunks for st.write_stream
        for event in stream:
            if isinstance(event, ResponseTextDeltaEvent):
                yield event.delta
            elif isinstance(event, ResponseTextDoneEvent):
                # Can also inspect event.parsed if structured outputs are used
                pass

        final = stream.get_final_response()
        # Return final text and response id via attribute on the generator
        stream.final_text = final.output_text  # type: ignore[attr-defined]
        stream.response_id = final.id  # type: ignore[attr-defined]


# -----------------------------
# Web Search (built-in tool)
# -----------------------------
def stream_web_search(query: str):
    client = get_client()
    with client.responses.stream(
        model=resolve_model_for_api("gpt-5-mini"),
        input=[{"role": "user", "content": query}],
        tools=[{"type": "web_search"}],
    ) as stream:
        for event in stream:
            if isinstance(event, ResponseTextDeltaEvent):
                yield event.delta
            elif isinstance(event, ResponseTextDoneEvent):
                # text finished
                pass
            elif getattr(event, "type", "").startswith("response.web_search_call."):
                # searching / in_progress / completed — optionally show progress
                pass

        final = stream.get_final_response()
        stream.final_text = final.output_text  # type: ignore[attr-defined]


def handle_user_message(user_text: str) -> None:
    st.session_state.messages.append({
        "role": "user",
        "content": user_text,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    st.session_state.is_generating = True
    st.session_state.last_error = None

    # Placeholder for assistant stream
    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            gen = stream_response_text(
                prompt=user_text,
                model=st.session_state.model,
                use_rag=st.session_state.use_rag,
                vector_store_ids=st.session_state.selected_vector_store_ids,
                previous_response_id=st.session_state.prev_response_id,
            )

            # Stream to the UI
            full_text = st.write_stream(gen)  # returns concatenated text

            # Fallback in case write_stream returns None for some Streamlit versions
            if full_text is None:
                try:
                    # Access attrs set in generator context
                    full_text = getattr(gen, "final_text", "")  # type: ignore[attr-defined]
                except Exception:
                    full_text = ""

            # Track conversation state via previous_response_id
            try:
                st.session_state.prev_response_id = getattr(gen, "response_id", None)  # type: ignore[attr-defined]
            except Exception:
                pass

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text or "",
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            st.session_state.last_error = f"応答生成に失敗しました: {e}"
            placeholder.error(st.session_state.last_error)
        finally:
            st.session_state.is_generating = False
            # After finishing generation, trigger a rerun so all widgets
            # (like download) see the latest messages without extra clicks.
            st.rerun()


# -----------------------------
# UI Components
# -----------------------------
def sidebar_settings() -> None:
    st.sidebar.markdown("### ツール選択")
    options = ["AIツール", "コード改修ツール"]
    default_index = options.index(st.session_state.tool_mode) if st.session_state.get("tool_mode") in options else 0
    # ウィジェットが同じ key を管理するため、ここで session_state に再代入しない
    st.sidebar.radio("カテゴリー", options=options, index=default_index, key="tool_mode")
    # st.sidebar.markdown("### AIチャット設定")
    # モデル選択は設定タブへ移動、RAG設定はAIチャットタブに移動

    # ダウンロード設定・ボタン・履歴クリアは別画面へ移動


def ui_ai_chat() -> None:
    st.subheader("AIチャット")
    # RAG 設定（AIチャット画面で制御） — 画面上部に配置
    with st.expander("RAG設定", expanded=False):
        st.toggle(
            "RAG（file_search）を使用",
            key="use_rag",
            disabled=st.session_state.is_generating,
            help="Responses APIのfile_searchでベクターストアを検索に利用",
        )
        if st.session_state.use_rag:
            # 手動更新
            if st.button("ストア一覧を再読込", disabled=st.session_state.is_generating, key="refresh_vs_in_chat_top"):
                list_vector_stores.clear()

            stores = list_vector_stores()
            store_name_by_id = {s["id"]: s["name"] for s in stores}
            options = [s["id"] for s in stores]
            format_label = lambda _id: f"{store_name_by_id.get(_id, _id)} ({_id})"

            st.multiselect(
                "使用するベクターストア（複数可）",
                options=options,
                default=[_id for _id in st.session_state.selected_vector_store_ids if _id in options],
                format_func=format_label,
                key="selected_vector_store_ids",
                disabled=st.session_state.is_generating,
            )
    # Fix chat input to bottom of the screen
    st.markdown(
        """
        <style>
        /* 入力欄の高さぶんだけ本文下に余白を確保（重なり防止） */
        section.main > div { padding-bottom: calc(var(--chat-h, 100px) + 12px) !important; }

        /* 画面下に固定。サイドバー幅ぶんを左にオフセットしてメイン領域いっぱい */
        .stChatInput {
        position: fixed;
        bottom: 0;
        left: var(--sbw, 0px);
        width: calc(100vw - var(--sbw, 0px));
        z-index: 9999;
        background: var(--background-color, #fff);
        border-top: 1px solid rgba(0,0,0,0.08);
        margin: 0; padding-top: 8px;
        }
        /* Streamlit の最大幅制約を解除して横一杯に */
        .stChatInput > div { max-width: none !important; }

        /* モバイル等でサイドバーがオーバーレイ表示の時は全幅固定 */
        @media (max-width: 767.98px){
        .stChatInput { left: 0 !important; width: 100vw !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # --- JS（CSS の直後に追加。サイドバー幅と入力高さを監視して CSS 変数に反映）---
    components.html("""
    <script>
    (() => {
    const doc = window.parent.document;
    const root = doc.documentElement;

    function updateSidebarWidth() {
        const sb = doc.querySelector('[data-testid="stSidebar"]');
        // オーバーレイで重なるケースを避けるため、開いている時だけ幅を使う
        const open = sb && sb.getAttribute('aria-expanded') === 'true';
        const w = open ? (sb.offsetWidth || 0) : 0;
        root.style.setProperty('--sbw', w + 'px');
    }

    function updateChatHeight() {
        const el = doc.querySelector('.stChatInput');
        const h = el ? el.offsetHeight : 0;
        root.style.setProperty('--chat-h', (h || 100) + 'px');
    }

    // 初期反映
    updateSidebarWidth();
    updateChatHeight();

    // 監視（サイドバー開閉・リサイズ／入力欄の高さ変化／ウィンドウリサイズ）
    const RO = window.parent.ResizeObserver || ResizeObserver;
    const ro = new RO(() => { updateSidebarWidth(); updateChatHeight(); });

    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (sb) ro.observe(sb);

    const chat = doc.querySelector('.stChatInput');
    if (chat) ro.observe(chat);

    ro.observe(doc.body);
    window.parent.addEventListener('resize', () => { updateSidebarWidth(); updateChatHeight(); }, { passive: true });
    })();
    </script>
    """, height=0, width=0)

    if st.session_state.last_error:
        st.info(f"直前のエラー: {st.session_state.last_error}")

    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m.get("role", "assistant")):
            st.markdown(str(m.get("content", "")))

    # Bottom-anchored input
    prompt = st.chat_input(
        "メッセージを入力",
        disabled=st.session_state.is_generating,
    )
    if prompt and not st.session_state.is_generating:
        with st.chat_message("user"):
            st.markdown(prompt)
        handle_user_message(prompt)

    # Toolbar above the fixed input
    col1, col2 = st.columns([2, 1])
    with col1:
        disabled = st.session_state.is_generating or len(st.session_state.messages) == 0
        # 設定ウィジェットの値はスクリプト開始時に session_state に反映されるため
        # Chat 側は settings_* を優先して参照し、一走遅れ問題を避ける
        fmt = st.session_state.get("settings_download_format", st.session_state.download_format)
        nl = st.session_state.get("settings_newline", st.session_state.newline)
        # prepare data and mime per format
        if fmt == "md":
            data = to_markdown(st.session_state.messages, nl)
            mime = "text/markdown"
        elif fmt == "json":
            data = to_json(st.session_state.messages, nl)
            mime = "application/json"
        else:
            data = to_txt(st.session_state.messages, nl)
            mime = "text/plain"
        # ダウンロードボタンのキーを設定に連動させ、設定変更直後の初回ダウンロードでも反映されるようにする
        download_key = f"chat_download_btn_{fmt}_{nl}"
        st.download_button(
            label="チャット履歴をダウンロード",
            data=data if not st.session_state.is_generating else b'',
            file_name=f"chat_{int(time.time())}.{fmt}",
            mime=mime,
            disabled=disabled,
            key=download_key,
        )
    with col2:
        if st.button("履歴クリア", disabled=st.session_state.is_generating, key="clear_history_btn"):
            st.session_state.messages.clear()
            st.session_state.prev_response_id = None


def ui_applied_tools() -> None:
    """コード改修ツール（実装）。"""
    st.subheader("コード改修ツール")

    # Helpers
    def _read_uploaded_text(uf) -> str:
        if not uf:
            return ""
        try:
            return uf.getvalue().decode("utf-8")
        except Exception:
            return uf.getvalue().decode("utf-8", errors="ignore")

    def _strip_code_fences(s: str) -> str:
        t = s.strip()
        if t.startswith("```") and t.endswith("```"):
            t = t[3:]
            if "\n" in t:
                t = t.split("\n", 1)[1]
            t = t.rsplit("```", 1)[0] if t.endswith("```") else t
        return t

    def _unified_diff(src: str, dst: str, src_name: str) -> str:
        to_name = f"modified_{src_name}" if src_name else "modified_code"
        diff = difflib.unified_diff(
            src.splitlines(),
            dst.splitlines(),
            fromfile=src_name or "original",
            tofile=to_name,
            lineterm="",
        )
        return "\n".join(diff)

    tabs = st.tabs(["実行", "設定"])

    # 設定タブ
    with tabs[1]:
        st.markdown("#### モデル設定")
        st.selectbox(
            "モデル",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state.code_fix_model)
            if st.session_state.code_fix_model in MODEL_OPTIONS
            else MODEL_OPTIONS.index(normalize_model_for_ui(DEFAULT_MODEL_RAW)),
            key="code_fix_model",
        )

        st.markdown("#### プロンプト設定（Step1: 設計差分抽出）")
        st.text_area(
            "Step1 プロンプト",
            value=st.session_state.code_fix_prompt_step1,
            height=180,
            key="code_fix_prompt_step1",
        )

        st.markdown("#### プロンプト設定（Step2: コード改修）")
        st.text_area(
            "Step2 プロンプト",
            value=st.session_state.code_fix_prompt_step2,
            height=220,
            key="code_fix_prompt_step2",
        )

    # 実行タブ
    with tabs[0]:
        if st.session_state.code_fix_last_error:
            st.info(f"エラー: {st.session_state.code_fix_last_error}")

        c1, c2, c3 = st.columns(3)
        with c1:
            f_code = st.file_uploader("入力ファイル（コード）", type=["py", "sql"], key="code_fix_file_code")
        with c2:
            f_before = st.file_uploader("設計書（改修前）", type=["txt", "md"], key="code_fix_file_before")
        with c3:
            f_after = st.file_uploader("設計書（改修後）", type=["txt", "md"], key="code_fix_file_after")

        run = st.button("一括実行", disabled=st.session_state.code_fix_is_running)

        if run:
            if not (f_code and f_before and f_after):
                st.warning("3つの入力ファイルすべてを指定してください。")
            else:
                st.session_state.code_fix_is_running = True
                st.session_state.code_fix_last_error = None
                # Start fresh log for this run
                st.session_state.code_fix_log = []
                try:
                    # 読み込み
                    code_text = _read_uploaded_text(f_code)
                    before_text = _read_uploaded_text(f_before)
                    after_text = _read_uploaded_text(f_after)

                    # Step1: 設計差分抽出
                    client = get_client()
                    step1_input = [
                        {"role": "system", "content": st.session_state.code_fix_prompt_step1},
                        {
                            "role": "user",
                            "content": (
                                "[改修前の設計書]\n" + before_text + "\n\n" +
                                "[改修後の設計書]\n" + after_text
                            ),
                        },
                    ]
                    resp1 = client.responses.create(
                        model=resolve_model_for_api(st.session_state.code_fix_model),
                        input=step1_input,
                    )
                    summary = getattr(resp1, "output_text", None) or ""
                    st.session_state.code_fix_summary = summary
                    # Log: 設計書比較（Step1）
                    try:
                        st.session_state.code_fix_log.append(
                            {
                                "stage": "設計書比較",
                                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "model": resolve_model_for_api(st.session_state.code_fix_model),
                                "request": {
                                    "system": st.session_state.code_fix_prompt_step1,
                                    "user": step1_input[1]["content"],
                                },
                                "response": summary,
                            }
                        )
                    except Exception:
                        pass

                    # Step2: コード改修
                    step2_input = [
                        {"role": "system", "content": st.session_state.code_fix_prompt_step2},
                        {
                            "role": "user",
                            "content": (
                                "[変更点の要約]\n" + summary + "\n\n" +
                                "[元のコード]\n" + code_text
                            ),
                        },
                    ]
                    resp2 = client.responses.create(
                        model=resolve_model_for_api(st.session_state.code_fix_model),
                        input=step2_input,
                    )
                    modified = getattr(resp2, "output_text", None) or ""
                    modified = _strip_code_fences(modified)
                    st.session_state.code_fix_modified_code = modified
                    # Log: コード修正（Step2）
                    try:
                        st.session_state.code_fix_log.append(
                            {
                                "stage": "コード修正",
                                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "model": resolve_model_for_api(st.session_state.code_fix_model),
                                "request": {
                                    "system": st.session_state.code_fix_prompt_step2,
                                    "user": step2_input[1]["content"],
                                    "context_code": code_text,
                                },
                                "response": modified,
                            }
                        )
                    except Exception:
                        pass

                    # Diff
                    diff_text = _unified_diff(code_text, modified, f_code.name)
                    st.session_state.code_fix_diff = diff_text
                    # Reset follow-up history on fresh run
                    st.session_state.code_fix_followup_messages = []
                except Exception as e:
                    st.session_state.code_fix_last_error = str(e)
                finally:
                    st.session_state.code_fix_is_running = False

        # 結果表示
        if st.session_state.get("code_fix_log"):
            ui_code_fix_full_log_download()
        if st.session_state.code_fix_summary:
            st.markdown("### 変更点の要約")
            st.text_area("要約", value=st.session_state.code_fix_summary, height=200, key="code_fix_summary_view")

        if st.session_state.code_fix_diff:
            st.markdown("### コード差分")
            st.code(st.session_state.code_fix_diff, language="diff")

        if st.session_state.code_fix_modified_code:
            st.markdown("### 修正後コード")
            ext = "py" if (f_code and f_code.name.lower().endswith(".py")) else "sql"
            lang = "python" if ext == "py" else "sql"
            st.code(st.session_state.code_fix_modified_code, language=lang)

            # Download
            base_name = f_code.name if f_code else ("code.py" if ext == "py" else "code.sql")
            out_name = f"modified_{base_name}"
            st.download_button(
                label="修正後コードをダウンロード",
                data=st.session_state.code_fix_modified_code.encode("utf-8"),
                file_name=out_name,
                mime="text/plain",
                key="code_fix_download_btn",
            )
            # Follow-up editor
            ui_code_fix_followups(f_code)

def ui_code_fix_full_log_download() -> None:
    """Render a download button for the entire code-fix history (Step1/Step2/follow-ups)."""
    entries = st.session_state.get("code_fix_log", []) or []
    fmt = st.session_state.get("settings_download_format", st.session_state.download_format)
    nl = st.session_state.get("settings_newline", st.session_state.newline)
    if fmt == "md":
        data = codefix_log_to_markdown(entries, nl)
        mime = "text/markdown"
    elif fmt == "json":
        data = codefix_log_to_json(entries, nl)
        mime = "application/json"
    else:
        data = codefix_log_to_txt(entries, nl)
        mime = "text/plain"
    st.markdown("---")
    st.markdown("### コード改修ツールの全履歴ダウンロード")
    st.download_button(
        label="全履歴をダウンロード",
        data=data if entries else b"",
        file_name=f"codefix_full_history_{int(time.time())}.{fmt}",
        mime=mime,
        disabled=len(entries) == 0,
        key=f"codefix_full_log_{fmt}_{nl}",
    )


def ui_code_fix_followups(f_code) -> None:
    """Render follow-up Q&A section for code-fix tool."""

    def _strip_code_fences(s: str) -> str:
        t = s.strip()
        if t.startswith("```") and t.endswith("```"):
            t = t[3:]
            if "\n" in t:
                t = t.split("\n", 1)[1]
            t = t.rsplit("```", 1)[0] if t.endswith("```") else t
        return t

    def _unified_diff(src: str, dst: str, src_name: str) -> str:
        to_name = f"modified_{src_name}" if src_name else "modified_code"
        diff = difflib.unified_diff(
            src.splitlines(),
            dst.splitlines(),
            fromfile=src_name or "original",
            tofile=to_name,
            lineterm="",
        )
        return "\n".join(diff)
    if not st.session_state.code_fix_modified_code:
        return

    st.markdown("---")
    st.markdown("### 追い質問（コードへの追加修正）")

    # Render follow-up history
    for m in st.session_state.code_fix_followup_messages:
        role = m.get("role", "assistant")
        content = str(m.get("content", ""))
        with st.chat_message(role):
            st.markdown(content)

    # Download follow-up history using global format settings
    fmt = st.session_state.get("settings_download_format", st.session_state.download_format)
    nl = st.session_state.get("settings_newline", st.session_state.newline)
    if fmt == "md":
        log_bytes = to_markdown(st.session_state.code_fix_followup_messages, nl)
        mime = "text/markdown"
    elif fmt == "json":
        log_bytes = to_json(st.session_state.code_fix_followup_messages, nl)
        mime = "application/json"
    else:
        log_bytes = to_txt(st.session_state.code_fix_followup_messages, nl)
        mime = "text/plain"
    st.download_button(
        label="追い質問履歴をダウンロード",
        data=log_bytes if st.session_state.code_fix_followup_messages else b'',
        file_name=f"codefix_followups_{int(time.time())}.{fmt}",
        mime=mime,
        disabled=len(st.session_state.code_fix_followup_messages) == 0,
        key=f"codefix_log_download_{fmt}_{nl}",
    )

    # Follow-up input
    follow_text = st.chat_input(
        "この出力コードに対する追加の要望や修正指示を入力",
        disabled=st.session_state.code_fix_is_running or st.session_state.code_fix_followup_is_running,
    )

    if follow_text and not st.session_state.code_fix_followup_is_running:
        st.session_state.code_fix_followup_messages.append({
            "role": "user",
            "content": follow_text,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        st.session_state.code_fix_followup_is_running = True
        st.session_state.code_fix_followup_last_error = None

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                client = get_client()
                system_prompt = (
                    (st.session_state.code_fix_prompt_step2 or "").strip()
                    + "\n\n# 出力ルール\n"
                      "- 必ず修正後の完全なコードのみを出力する\n"
                      "- 解説・前置き・コードフェンス( ``` )は出力しない\n"
                )
                user_payload = (
                    "[現在のコード]\n" + st.session_state.code_fix_modified_code + "\n\n"
                    + "[ユーザーの要望]\n" + follow_text
                )
                resp = client.responses.create(
                    model=resolve_model_for_api(st.session_state.code_fix_model),
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                )
                new_code = getattr(resp, "output_text", None) or ""
                new_code = _strip_code_fences(new_code)

                if not new_code.strip():
                    raise RuntimeError("追い質問の応答にコードが含まれていません。プロンプトを見直してください。")

                prev_code = st.session_state.code_fix_modified_code
                diff_text = _unified_diff(prev_code, new_code, f_code.name if f_code else "code")
                st.session_state.code_fix_modified_code = new_code
                st.session_state.code_fix_diff = diff_text

                st.session_state.code_fix_followup_messages.append({
                    "role": "assistant",
                    "content": new_code,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

                # Log: 追いチャット
                try:
                    st.session_state.code_fix_log.append(
                        {
                            "stage": "追いチャット",
                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "model": resolve_model_for_api(st.session_state.code_fix_model),
                            "request": {
                                "system": system_prompt,
                                "user": follow_text,
                                "context_code": prev_code,
                            },
                            "response": new_code,
                        }
                    )
                except Exception:
                    pass

                placeholder.markdown("(修正後コードを反映しました)")
            except Exception as e:
                st.session_state.code_fix_followup_last_error = str(e)
                placeholder.error(st.session_state.code_fix_followup_last_error)
            finally:
                st.session_state.code_fix_followup_is_running = False
                st.rerun()


def ui_web_search() -> None:
    st.subheader("Web検索（gpt-5-mini固定）")
    if st.session_state.last_web_error:
        st.info(f"直前のエラー: {st.session_state.last_web_error}")

    query = st.text_area("検索クエリ", placeholder="調べたい内容を入力", key="web_query")
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if not st.session_state.web_is_generating:
            if st.button("検索", disabled=not query, key="web_search_btn"):
                st.session_state.web_is_generating = True
                st.session_state.last_web_error = None
                st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
        else:
            st.button("検索中...", disabled=True, key="web_search_btn_disabled")

    if st.session_state.web_is_generating:
        try:
            with st.status("検索中...", expanded=True):
                gen = stream_web_search(st.session_state.web_query)
                result = st.write_stream(gen)
                if result is None:
                    try:
                        result = getattr(gen, "final_text", "")  # type: ignore[attr-defined]
                    except Exception:
                        result = ""
            st.markdown("### 結果")
            st.write(result or "(空)")
        except Exception as e:
            st.session_state.last_web_error = f"Web検索に失敗しました: {e}"
            st.error(st.session_state.last_web_error)
        finally:
            st.session_state.web_is_generating = False


def ui_image_generation() -> None:
    st.subheader("画像生成（gpt-image-1固定）")
    if st.session_state.last_image_error:
        st.info(f"直前のエラー: {st.session_state.last_image_error}")

    prompt = st.text_area("プロンプト", placeholder="生成したい画像の説明文を入力", key="img_prompt")
    # Supported by gpt-image-1: 1024x1024, 1024x1536 (portrait), 1536x1024 (landscape), or auto
    size = st.selectbox("サイズ", ["1024x1024", "1024x1536", "1536x1024", "auto"], index=0, key="img_size")
    if not st.session_state.img_is_generating:
        st.button("生成", disabled=not prompt, key="img_generate_btn")
        # If clicked, set flag and rerun
        if st.session_state.get("img_generate_btn"):
            st.session_state.img_is_generating = True
            st.session_state.last_image_error = None
            st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()
    else:
        st.button("生成中...", disabled=True, key="img_generate_btn_disabled")

    if st.session_state.img_is_generating:
        try:
            client = get_client()
            resp = client.images.generate(
                model="gpt-image-1",
                prompt=st.session_state.img_prompt,
                size=st.session_state.img_size,
                output_format="png",
                n=1,
                quality="auto",
            )
            images = []
            data_list = getattr(resp, "data", None) or []
            for d in data_list:
                b64 = getattr(d, "b64_json", None)
                url = getattr(d, "url", None)
                if b64:
                    img_bytes = base64.b64decode(b64)
                    images.append(("b64", img_bytes))
                elif url:
                    images.append(("url", url))

            if not images:
                st.warning("画像を取得できませんでした。応答を表示します。")
                try:
                    st.write(resp)
                except Exception:
                    st.write("(no response contents)")
            else:
                per_row = 3
                for i, (kind, payload) in enumerate(images):
                    if i % per_row == 0:
                        cols = st.columns(min(per_row, len(images) - i))
                    with cols[i % per_row]:
                        if kind == "b64":
                            st.image(io.BytesIO(payload), caption=f"生成画像 {i+1}")
                        else:
                            st.image(payload, caption=f"生成画像 {i+1}")
        except Exception as e:
            st.session_state.last_image_error = f"画像生成に失敗しました: {e}"
            st.error(st.session_state.last_image_error)
        finally:
            st.session_state.img_is_generating = False


def ui_rag_management() -> None:
    st.subheader("RAG管理（ベクターストア / ファイル）")

    # Manual refresh buttons
    c1, c2 = st.columns(2)
    with c1:
        if st.button("ベクターストア一覧を再読込"):
            list_vector_stores.clear()
    with c2:
        st.write("")

    # List stores
    stores = list_vector_stores()
    if not stores:
        st.caption("ベクターストアがありません。作成してください。")

    # Create store
    with st.expander("ベクターストアの作成", expanded=False):
        name = st.text_input("名称", key="create_vs_name")
        if st.button("作成", disabled=not name):
            try:
                vs_id = rag_create_vector_store(name)
                st.success(f"作成しました: {name} ({vs_id})")
            except Exception as e:
                st.session_state.last_error = f"ベクターストア作成に失敗: {e}"
                st.error(st.session_state.last_error)

    # Select a store to manage files
    st.markdown("---")
    st.markdown("#### ベクターストア一覧")
    if stores:
        vs_labels = {s["id"]: f"{s['name']} ({s['id']})" for s in stores}
        selected_vs_id = st.selectbox(
            "選択",
            options=[s["id"] for s in stores],
            format_func=lambda _id: vs_labels.get(_id, _id),
        )

        # Rename / Delete actions
        col_a, col_b = st.columns([3, 1])
        with col_a:
            new_name = st.text_input("名称変更", value=vs_labels[selected_vs_id].split(" (")[0])
        with col_b:
            if st.button("名称変更", use_container_width=True):
                try:
                    rag_rename_vector_store(selected_vs_id, new_name)
                    st.success("名称を更新しました")
                except Exception as e:
                    st.session_state.last_error = f"名称変更に失敗: {e}"
                    st.error(st.session_state.last_error)

        if st.button("このベクターストアを削除", type="secondary"):
            try:
                rag_delete_vector_store(selected_vs_id)
                st.warning("削除しました")
            except Exception as e:
                st.session_state.last_error = f"削除に失敗: {e}"
                st.error(st.session_state.last_error)

        st.markdown("---")
        st.markdown("#### ファイル管理")

        # Manual refresh for files
        if st.button("ファイル一覧を再読込"):
            list_vector_store_files.clear()

        files = list_vector_store_files(selected_vs_id)
        if files:
            for f in files:
                cols = st.columns([5, 2, 2, 2])
                with cols[0]:
                    st.caption(f"{f['id']}")
                with cols[1]:
                    st.write(f"状態: {f.get('status', '-')}")
                with cols[2]:
                    err = f.get("last_error")
                    if err:
                        st.error("エラーあり")
                with cols[3]:
                    if st.button("削除", key=f"del_{f['id']}"):
                        try:
                            rag_delete_file(selected_vs_id, f["id"])
                            st.success("削除しました")
                        except Exception as e:
                            st.session_state.last_error = f"ファイル削除に失敗: {e}"
                            st.error(st.session_state.last_error)
        else:
            st.caption("ファイルはありません。")

        st.markdown("---")
        st.markdown("#### ファイル追加")
        uploads = st.file_uploader("ファイルを追加", accept_multiple_files=True)
        if uploads:
            if st.button("アップロード"):
                ids = rag_upload_files(selected_vs_id, uploads)
                if ids:
                    st.success(f"{len(ids)} 件のファイルを追加しました")


def ui_settings() -> None:
    st.subheader("設定")

    st.markdown("#### モデル選択")
    st.session_state.model = st.selectbox(
        "モデル",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.model)
        if st.session_state.model in MODEL_OPTIONS
        else MODEL_OPTIONS.index(normalize_model_for_ui(DEFAULT_MODEL_RAW)),
        help="既定は gpt-5-mini。選択: gpt-5 / gpt-5-mini / gpt-5-nano / o4-mini",
        key="settings_model",
    )

    st.markdown("#### ダウンロード設定")
    st.session_state.download_format = st.selectbox("ファイルタイプ", ["md", "json", "txt"], key="settings_download_format")
    st.session_state.newline = st.selectbox("改行コード", ["CRLF", "LF"], key="settings_newline")

    st.markdown("---")
    st.markdown("現在の設定:")
    st.json(
        {
            "model": st.session_state.model,
            "use_rag": st.session_state.use_rag,
            "vector_store_ids": st.session_state.selected_vector_store_ids,
            "download_format": st.session_state.download_format,
            "newline": st.session_state.newline,
        }
    )


# -----------------------------
# Main entry
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Internal AI Tool", layout="wide")
    init_session_state()

    # Quick check for API key presence to help first-time setup
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY が未設定です。.env を確認してください。")

    st.sidebar.header("AIツール")
    sidebar_settings()
    # コード改修ツールが選択されている場合は専用画面を表示
    if st.session_state.get("tool_mode") == "コード改修ツール":
        ui_applied_tools()
        return

    tabs = st.tabs(["AIチャット", "Web検索", "画像生成", "RAG管理", "設定"])
    # 設定タブを先に評価して、他タブで最新設定が使われるようにする
    with tabs[4]:
        ui_settings()

    with tabs[0]:
        ui_ai_chat()
    with tabs[1]:
        ui_web_search()
    with tabs[2]:
        ui_image_generation()
    with tabs[3]:
        ui_rag_management()


if __name__ == "__main__":
    if not runtime_exists():
        print("このアプリは 'streamlit run app.py' で実行してください。")
    else:
        main()
