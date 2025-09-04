import os
import io
import json
import time
import base64
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
        st.session_state.newline = "LF"  # LF | CRLF
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
        model=resolve_model_for_api("gpt5-mini"),
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
    pass
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
        fmt = st.session_state.download_format
        if fmt == "md":
            data = to_markdown(st.session_state.messages, st.session_state.newline)
        elif fmt == "json":
            data = to_json(st.session_state.messages, st.session_state.newline)
        else:
            data = to_txt(st.session_state.messages, st.session_state.newline)
        st.download_button(
            label="チャット履歴をダウンロード",
            data=data if not st.session_state.is_generating else None,
            file_name=f"chat_{int(time.time())}.{fmt}",
            mime="text/plain",
            disabled=disabled,
            key="chat_download_btn",
        )
    with col2:
        if st.button("履歴クリア", disabled=st.session_state.is_generating, key="clear_history_btn"):
            st.session_state.messages.clear()
            st.session_state.prev_response_id = None


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
    st.subheader("画像生成（gpt-5-nano固定）")
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
        help="既定は gpt5-mini。選択: gpt5 / gpt5-mini / gpt5-nano / o4-mini",
        key="settings_model",
    )

    st.markdown("#### ダウンロード設定")
    st.session_state.download_format = st.selectbox("ファイルタイプ", ["md", "json", "txt"], key="settings_download_format")
    st.session_state.newline = st.selectbox("改行コード", ["LF", "CRLF"], key="settings_newline")

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

    st.sidebar.header("AIツール")
    sidebar_settings()

    tabs = st.tabs(["AIチャット", "Web検索", "画像生成", "RAG管理", "設定"])
    with tabs[0]:
        ui_ai_chat()
    with tabs[1]:
        ui_web_search()
    with tabs[2]:
        ui_image_generation()
    with tabs[3]:
        ui_rag_management()

    with tabs[4]:
        ui_settings()


if __name__ == "__main__":
    main()
