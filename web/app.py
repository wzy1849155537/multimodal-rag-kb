"""RAG Knowledge Base — Streamlit Web UI."""

import json, os, sys, uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
_PROJECT_ROOT = Path(__file__).parent.parent

# --- API Key ---
if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
if not os.environ.get("SILICONFLOW_API_KEY"):
    env_file = _PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SILICONFLOW_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val != "your-api-key-here":
                        os.environ["SILICONFLOW_API_KEY"] = val
                    break
os.environ.setdefault("OPENAI_API_KEY", os.environ["SILICONFLOW_API_KEY"])
os.environ.setdefault("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")

import streamlit as st

st.set_page_config(page_title="知识库问答", page_icon="📚", layout="wide",
                   initial_sidebar_state="collapsed")

# ================================================================
# CSS
# ================================================================
st.markdown("""<style>
    .main-header { font-size: 1.3rem; font-weight: 600; color: #1a1a1a; padding-bottom: 0.3rem; }
    .source-card { background: #F1F5F9; padding: 0.4rem 0.7rem; border-radius: 6px;
        margin: 0.2rem 0; font-size: 0.8rem; border-left: 3px solid #3B82F6; }
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .history-item {
        padding: 0.4rem 0.6rem; border-radius: 4px; margin: 0.15rem 0;
        cursor: pointer; font-size: 0.82rem; border-bottom: 1px solid #f0f0f0;
        transition: background 0.15s;
    }
    .history-item:hover { background: #EEF2FF; }
    .history-time { font-size: 0.7rem; color: #aaa; float: right; }
    .panel-title { font-size: 0.95rem; font-weight: 600; color: #333; margin-bottom: 0.5rem; }
    .stat-badge { display: inline-block; background: #EEF2FF; color: #3B82F6;
        padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.75rem; margin-right: 0.3rem; }
</style>""", unsafe_allow_html=True)

# ================================================================
# Multi-Conversation Chat History
# ================================================================
HISTORY_FILE = _PROJECT_ROOT / "data" / "chat_history.json"

def load_conversations() -> dict:
    """Load all conversations. Returns {conv_id: {name, messages, created_at}}."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_conversations(convs: dict):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(convs, f, ensure_ascii=False, default=str)

def new_conversation(name: str = "") -> tuple:
    """Create a new conversation. Returns (conv_id, conv_dict)."""
    cid = uuid.uuid4().hex[:8]
    now = datetime.now().strftime("%m/%d %H:%M")
    conv = {
        "name": name or f"对话 {now}",
        "created_at": now,
        "messages": [],
    }
    return cid, conv

# ================================================================
# Pipeline
# ================================================================
def get_pipeline():
    from src.pipeline import RAGPipeline
    p = RAGPipeline()
    p.config._data["embedder"] = "siliconflow"
    p.config._data["indexer"] = "dense"
    p.config._data["llm"] = {
        "default": "siliconflow",
        "providers": {"siliconflow": {
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": os.environ.get("SILICONFLOW_API_KEY", ""),
            "model": "Qwen/Qwen2.5-32B-Instruct",
        }},
    }
    p.config._data["generation"] = {
        "llm_model": "Qwen/Qwen2.5-32B-Instruct",
        "llm_temperature": 0.3, "max_context_tokens": 4096,
    }
    p._initialized = False
    return p

# ================================================================
# Main
# ================================================================
def main():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline()

    # --- Load conversations ---
    if "conversations" not in st.session_state:
        st.session_state.conversations = load_conversations()
    if "current_conv" not in st.session_state:
        # Auto-create first conversation if none exist
        convs = st.session_state.conversations
        if not convs:
            cid, conv = new_conversation()
            convs[cid] = conv
            save_conversations(convs)
        st.session_state.current_conv = list(convs.keys())[-1]

    pipeline = st.session_state.pipeline
    convs = st.session_state.conversations
    cid = st.session_state.current_conv
    msgs = convs[cid]["messages"]

    # ============================================================
    # TOP BAR
    # ============================================================
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        st.markdown('<div class="main-header">📚 知识库问答</div>', unsafe_allow_html=True)
    with c3:
        with st.popover("📎 上传知识", use_container_width=True):
            uploaded_files = st.file_uploader(
                "选择文件",
                type=["pdf", "md", "txt", "png", "jpg", "mp4", "avi", "mov", "mkv"],
                accept_multiple_files=True, label_visibility="collapsed",
            )
            if uploaded_files and st.button("加入知识库", type="primary", use_container_width=True):
                import tempfile
                total = 0
                with tempfile.TemporaryDirectory() as tmpdir:
                    for uploaded in uploaded_files:
                        tmp_path = Path(tmpdir) / uploaded.name
                        tmp_path.write_bytes(uploaded.read())
                        try:
                            count = pipeline.ingest_file(tmp_path)
                            total += count
                        except Exception as e:
                            st.error(f"{uploaded.name}: {e}")
                if total > 0:
                    st.success(f"已添加 {total} 条")
                    st.rerun()
            try:
                stats = pipeline.get_stats()
                st.caption(f"知识库: {stats.get('total_chunks', 0)} 条片段")
            except Exception:
                pass
            if st.button("🗑 清空知识库", use_container_width=True):
                pipeline.clear_index()
                st.session_state.pipeline = get_pipeline()
                st.success("知识库已清空")
                st.rerun()

    # ============================================================
    # SIDEBAR: Settings
    # ============================================================
    with st.sidebar:
        st.markdown("### ⚙️ 设置")
        top_k = st.slider("搜索范围", 1, 20, 5, help="检索片段数量")

    # ============================================================
    # MAIN LAYOUT: Conversation list (left) + Chat (right)
    # ============================================================
    left, right = st.columns([1, 3])

    # --- LEFT: Conversation List ---
    with left:
        st.markdown('<div class="panel-title">💬 对话列表</div>', unsafe_allow_html=True)

        # New conversation button
        new_name = st.text_input("", placeholder="新对话名称（可选）", label_visibility="collapsed")
        col_a, col_b = st.columns([2, 1])
        if col_a.button("➕ 新建对话", use_container_width=True):
            nid, nconv = new_conversation(new_name.strip() if new_name.strip() else "")
            convs[nid] = nconv
            save_conversations(convs)
            st.session_state.current_conv = nid
            st.rerun()

        st.divider()

        # List conversations
        sorted_convs = sorted(convs.items(), key=lambda x: x[1].get("created_at", ""), reverse=True)

        for conv_id, conv_data in sorted_convs:
            name = conv_data.get("name", "未命名")
            msg_count = len(conv_data.get("messages", []))
            is_active = (conv_id == cid)

            # Active conversation highlighted
            style = "background: #EEF2FF; border-left: 3px solid #3B82F6;" if is_active else ""

            c_left, c_right = st.columns([4, 1])
            with c_left:
                if st.button(
                    f"{'📌 ' if is_active else ''}{name}",
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.current_conv = conv_id
                    st.rerun()
            with c_right:
                if len(convs) > 1 and st.button("✕", key=f"del_{conv_id}", help="删除对话"):
                    del convs[conv_id]
                    save_conversations(convs)
                    if cid == conv_id:
                        st.session_state.current_conv = list(convs.keys())[-1] if convs else ""
                    st.rerun()

            st.caption(f"   {msg_count} 条消息 · {conv_data.get('created_at', '')}")

    # --- RIGHT: Chat area ---
    with right:
        # Show current conversation name
        st.caption(f"当前对话：{convs[cid].get('name', '')}")

        for msg in msgs:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 来源"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>[{i}] {src["doc_name"]}</b> '
                                f'<span style="color:#888">({src["score"]:.2f})</span>'
                                f'<br>{src["content_snippet"][:250]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

        if question := st.chat_input("输入问题，基于知识库回答..."):
            now = datetime.now().strftime("%H:%M")
            msgs.append({"role": "user", "content": question, "time": now})
            # Auto-name conversation from first question
            if len(msgs) <= 2 and convs[cid]["name"].startswith("对话 "):
                convs[cid]["name"] = question[:20] + ("..." if len(question) > 20 else "")
            save_conversations(convs)

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner(""):
                    try:
                        answer = pipeline.query(question, top_k=top_k)
                        st.markdown(answer.answer)
                        if answer.sources:
                            with st.expander("📎 来源"):
                                for i, src in enumerate(answer.sources, 1):
                                    st.markdown(
                                        f'<div class="source-card">'
                                        f'<b>[{i}] {src["doc_name"]}</b> '
                                        f'<span style="color:#888">({src["score"]:.2f})</span>'
                                        f'<br>{src["content_snippet"][:250]}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                        msgs.append({
                            "role": "assistant", "content": answer.answer,
                            "sources": answer.sources,
                            "time": datetime.now().strftime("%H:%M"),
                        })
                        save_conversations(convs)
                    except Exception as e:
                        st.error(f"查询失败: {e}")


if __name__ == "__main__":
    main()
