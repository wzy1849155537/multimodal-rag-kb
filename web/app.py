"""RAG Knowledge Base — Streamlit Web UI (Professional Edition)."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- API Key ---
if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("SILICONFLOW_API_KEY", ""))
os.environ.setdefault("OPENAI_API_BASE", "https://api.siliconflow.cn/v1")

import streamlit as st

# ================================================================
# Page Config
# ================================================================
st.set_page_config(
    page_title="知识库问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# Custom CSS — Professional look
# ================================================================
st.markdown("""
<style>
    /* Clean header */
    .main-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a1a;
        padding-bottom: 0.5rem;
    }
    /* Chat bubbles */
    .chat-row {
        display: flex;
        margin: 0.5rem 0;
    }
    .chat-bubble-user {
        background: #EEF2FF;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 4px 12px;
    }
    .chat-bubble-assistant {
        background: #F8FAFC;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 12px 4px;
        border: 1px solid #E2E8F0;
    }
    /* Source card */
    .source-card {
        background: #F1F5F9;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.82rem;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #FAFBFC;
    }
    /* Hide Streamlit junk */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ================================================================
# Chat History Persistence
# ================================================================
HISTORY_FILE = Path(__file__).parent.parent / "data" / "chat_history.json"


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
        "providers": {
            "siliconflow": {
                "api_base": "https://api.siliconflow.cn/v1",
                "api_key": os.environ.get("SILICONFLOW_API_KEY", ""),
                "model": "Qwen/Qwen2.5-32B-Instruct",
            }
        },
    }
    p.config._data["generation"] = {
        "llm_model": "Qwen/Qwen2.5-32B-Instruct",
        "llm_temperature": 0.3,
        "max_context_tokens": 4096,
    }
    p._initialized = False
    return p


# ================================================================
# Main UI
# ================================================================
def main():
    # --- Header ---
    st.markdown(
        '<div class="main-header">📚 知识库问答系统</div>',
        unsafe_allow_html=True,
    )

    # --- Init state ---
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = load_history()

    pipeline = st.session_state.pipeline

    # ================================================================
    # Sidebar
    # ================================================================
    with st.sidebar:
        st.markdown("### 📁 知识库管理")

        uploaded_files = st.file_uploader(
            "上传知识文件",
            type=["pdf", "md", "txt", "png", "jpg", "mp4", "avi", "mov", "mkv"],
            accept_multiple_files=True,
            help="支持 PDF、Markdown、TXT、图片、视频（视频将自动提取语音转文字）",
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        indexed = False
        if col1.button("📥 添加到知识库", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("请先选择文件")
            else:
                import tempfile
                total = 0
                progress = st.progress(0, text="解析中...")
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, uploaded in enumerate(uploaded_files):
                        tmp_path = Path(tmpdir) / uploaded.name
                        tmp_path.write_bytes(uploaded.read())
                        progress.progress(
                            (i + 1) / len(uploaded_files),
                            text=f"处理中: {uploaded.name}..."
                        )
                        try:
                            count = pipeline.ingest_file(tmp_path)
                            total += count
                            st.toast(f"✓ {uploaded.name}", icon="✅")
                        except Exception as e:
                            st.toast(f"✗ {uploaded.name}", icon="❌")
                progress.progress(1.0, text="完成!")
                indexed = True
                st.success(f"已添加 {total} 条知识片段")

        if col2.button("🗑 清空知识库", use_container_width=True):
            pipeline.clear_index()
            st.session_state.pipeline = get_pipeline()
            st.session_state.messages = []
            save_history([])
            st.success("已清空")
            st.rerun()

        st.divider()

        # Stats
        try:
            stats = pipeline.get_stats()
            col_a, col_b = st.columns(2)
            col_a.metric("知识片段", stats.get("total_chunks", 0))
            col_b.metric("文档数", len(uploaded_files) if uploaded_files else "-")
        except Exception:
            pass

        st.divider()
        st.markdown("### ⚙️ 设置")
        top_k = st.slider("搜索范围", 1, 20, 5, help="越大覆盖越全，但可能引入噪音")

        st.divider()
        if st.button("💾 导出聊天记录", use_container_width=True):
            save_history(st.session_state.messages)
            st.success("已保存")

    # ================================================================
    # Chat Area
    # ================================================================
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 查看来源"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<div class="source-card">'
                            f'<b>[{i}] {src["doc_name"]}</b> '
                            f'<span style="color:#888">相关度 {src["score"]:.2f}</span>'
                            f'<br>{src["content_snippet"][:250]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Chat input
    if question := st.chat_input("输入你的问题，基于知识库回答..."):
        now = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user", "content": question, "time": now,
        })
        save_history(st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    answer = pipeline.query(question, top_k=top_k)
                    st.markdown(answer.answer)

                    if answer.sources:
                        with st.expander("📎 查看来源"):
                            for i, src in enumerate(answer.sources, 1):
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<b>[{i}] {src["doc_name"]}</b> '
                                    f'<span style="color:#888">相关度 {src["score"]:.2f}</span>'
                                    f'<br>{src["content_snippet"][:250]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer.answer,
                        "sources": answer.sources,
                        "time": datetime.now().strftime("%H:%M"),
                    })
                    save_history(st.session_state.messages)
                except Exception as e:
                    st.error(f"查询失败: {e}")

    # Reload after indexing
    if indexed:
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
