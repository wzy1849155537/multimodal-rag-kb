"""RAG Knowledge Base — Streamlit Web UI."""

import json, os, sys
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
# Chat History
# ================================================================
HISTORY_FILE = _PROJECT_ROOT / "data" / "chat_history.json"

def load_history() -> list:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_history(messages: list):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, default=str)

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
    if "messages" not in st.session_state:
        st.session_state.messages = load_history()

    pipeline = st.session_state.pipeline

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
                st.success("知识库已清空（对话记录保留）")
                st.rerun()

    # ============================================================
    # SIDEBAR: Settings only
    # ============================================================
    with st.sidebar:
        st.markdown("### ⚙️ 设置")
        top_k = st.slider("搜索范围", 1, 20, 5, help="检索片段数量")
        if st.button("🆕 新对话", use_container_width=True):
            st.session_state.messages = []
            save_history([])
            st.rerun()

    # ============================================================
    # MAIN LAYOUT: History panel (left) + Chat (right)
    # ============================================================
    left, right = st.columns([1, 3])

    # --- LEFT: Chat History ---
    with left:
        st.markdown('<div class="panel-title">💬 对话记录</div>', unsafe_allow_html=True)

        msgs = st.session_state.messages
        # Build Q-A pairs
        pairs = []
        i = 0
        while i < len(msgs):
            if msgs[i]["role"] == "user":
                q = msgs[i]["content"]
                a = msgs[i + 1]["content"] if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant" else ""
                t = msgs[i].get("time", "")
                pairs.append({"q": q, "a": a, "t": t, "idx": i})
                i += 2
            else:
                i += 1

        if not pairs:
            st.caption("暂无对话")

        for p in reversed(pairs[-30:]):
            q_short = p["q"][:28] + ("..." if len(p["q"]) > 28 else "")
            with st.expander(f"{q_short}", expanded=False):
                st.caption(f"**问：** {p['q']}")
                st.caption(f"**答：** {p['a'][:200]}")
                st.caption(f"_{p['t']}_")

    # --- RIGHT: Chat area ---
    with right:
        for msg in st.session_state.messages:
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
            st.session_state.messages.append({"role": "user", "content": question, "time": now})
            save_history(st.session_state.messages)

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
                        st.session_state.messages.append({
                            "role": "assistant", "content": answer.answer,
                            "sources": answer.sources,
                            "time": datetime.now().strftime("%H:%M"),
                        })
                        save_history(st.session_state.messages)
                    except Exception as e:
                        st.error(f"查询失败: {e}")


if __name__ == "__main__":
    main()
