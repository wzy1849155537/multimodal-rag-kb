"""Streamlit web UI for the RAG Knowledge Base."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- MUST set API key BEFORE importing pipeline ---
if not os.environ.get("SILICONFLOW_API_KEY"):
    os.environ["SILICONFLOW_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("SILICONFLOW_API_KEY", ""))
os.environ["OPENAI_API_BASE"] = "https://api.siliconflow.cn/v1"

import streamlit as st

st.set_page_config(
    page_title="RAG 知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_pipeline():
    """Get a fresh pipeline with correct config each time."""
    from src.pipeline import RAGPipeline
    p = RAGPipeline()
    # Force SiliconFlow config
    p.config._data["embedder"] = "siliconflow"
    p.config._data["indexer"] = "dense"
    p.config._data["llm"] = {
        "default": "siliconflow",
        "providers": {
            "siliconflow": {
                "api_base": "https://api.siliconflow.cn/v1",
                "api_key": os.environ["SILICONFLOW_API_KEY"],
                "model": "Qwen/Qwen2.5-32B-Instruct",
            }
        },
    }
    p.config._data["generation"] = {
        "llm_model": "Qwen/Qwen2.5-7B-Instruct",
        "llm_temperature": 0.3,
        "max_context_tokens": 4096,
    }
    p._initialized = False
    return p


def main():
    st.title("📚 多模态 RAG 知识库问答系统")
    st.caption("支持 PDF、Markdown、TXT — 上传文档，即刻问答 | API: SiliconFlow")

    # Initialize pipeline in session state
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = get_pipeline()

    pipeline = st.session_state.pipeline

    # Sidebar
    with st.sidebar:
        st.header("📁 文档管理")

        uploaded_files = st.file_uploader(
            "上传文档",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
            help="支持 PDF、Markdown、TXT 文件",
        )

        col1, col2 = st.columns(2)

        if col1.button("🔨 索引上传文件", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("请先上传文件")
            else:
                import tempfile
                total = 0
                progress = st.progress(0, text="解析中...")
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, uploaded in enumerate(uploaded_files):
                        tmp_path = Path(tmpdir) / uploaded.name
                        tmp_path.write_bytes(uploaded.read())
                        progress.progress(
                            (i) / len(uploaded_files),
                            text=f"索引中: {uploaded.name}..."
                        )
                        try:
                            count = pipeline.ingest_file(tmp_path)
                            total += count
                            st.success(f"✓ {uploaded.name}: {count} chunks")
                        except Exception as e:
                            st.error(f"✗ {uploaded.name}: {e}")
                progress.progress(1.0, text="完成!")
                st.info(f"共索引 {total} 个 chunk")

        if col2.button("🔄 清空索引", use_container_width=True):
            pipeline.clear_index()
            st.session_state.pipeline = get_pipeline()
            st.session_state.pop("messages", None)
            st.success("索引已清空")
            st.rerun()

        st.divider()

        if st.button("📊 查看索引统计"):
            try:
                stats = pipeline.get_stats()
                st.metric("总 Chunk 数", stats.get("total_chunks", 0))
                st.caption(f"存储: {stats.get('persist_directory', 'N/A')}")
            except Exception as e:
                st.error(f"获取统计失败: {e}")

        st.divider()
        st.header("⚙️ 设置")
        top_k = st.slider("检索数量 (Top-K)", 1, 20, 5)
        st.caption("检索 Chunk 越多覆盖越全，但噪音也越多")

    # Main chat area
    st.header("💬 知识库问答")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 参考来源"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.caption(
                            f"[{i}] **{src['doc_name']}** "
                            f"(相关度: {src['score']:.3f})"
                        )
                        st.text(src["content_snippet"][:300])

    # Chat input
    if question := st.chat_input("请输入你的问题..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("检索中..."):
                try:
                    answer = pipeline.query(question, top_k=top_k)
                    st.markdown(answer.answer)
                    st.caption(
                        f"⏱ {answer.latency_ms:.0f}ms | "
                        f"置信度: {answer.confidence:.2f}"
                    )

                    if answer.sources:
                        with st.expander("📎 参考来源"):
                            for i, src in enumerate(answer.sources, 1):
                                st.caption(
                                    f"[{i}] **{src['doc_name']}** "
                                    f"(相关度: {src['score']:.3f})"
                                )
                                st.text(src["content_snippet"][:300])

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer.answer,
                        "sources": answer.sources,
                    })
                except Exception as e:
                    st.error(f"查询失败: {e}")


if __name__ == "__main__":
    main()
