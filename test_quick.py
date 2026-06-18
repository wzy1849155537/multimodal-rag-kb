"""Quick test with SiliconFlow API for both embedding and LLM."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# SiliconFlow config
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
API_BASE = "https://api.siliconflow.cn/v1"

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = API_BASE

print("=" * 50)
print("  测试 SiliconFlow API 连接...")
print("=" * 50)

# Test embedding API
print("\n[1] 测试 Embedding API...")
from src.embedders.openai_embedder import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_base=API_BASE,
    api_key=API_KEY,
    model_name="BAAI/bge-m3",
)
vec = embedder.embed_query("你好世界")
print(f"  Embedding dim: {len(vec)}, first 5: {vec[:5]}")

# Test embedding batch
vecs = embedder.embed_texts(["测试文本1", "测试文本2"])
print(f"  Batch: {len(vecs)} vectors, dim={len(vecs[0])}")

# Test LLM API
print("\n[2] 测试 LLM API...")
from openai import OpenAI
client = OpenAI(base_url=API_BASE, api_key=API_KEY)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "你好，请用一句话介绍自己"}],
    max_tokens=100,
)
print(f"  Model: {response.model}")
print(f"  Response: {response.choices[0].message.content}")

print("\n[3] 测试完整 RAG Pipeline...")
from src.pipeline import RAGPipeline

p = RAGPipeline()

# Override config with SiliconFlow
# Config is loaded from YAML files - set keys that pipeline uses
p.config._data["embedder"] = "openai"
p.config._data["indexer"] = "dense"  # Use dense only for quick test
p.config._data["embedding"] = {
    "models": {
        "openai": {
            "model_name": "BAAI/bge-m3",
            "dimension": 1024,
            "api_base": API_BASE,
            "api_key": API_KEY,
        }
    }
}

# Also ensure LLM uses SiliconFlow
p.config._data["llm"] = {
    "default": "siliconflow",
    "providers": {
        "siliconflow": {
            "api_base": API_BASE,
            "api_key": API_KEY,
            "model": "Qwen/Qwen2.5-7B-Instruct",
        }
    }
}
p.config._data["generator"] = "rag"
p.config._data["generation"] = {
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "llm_temperature": 0.3,
    "max_context_tokens": 4096,
}

# Re-initialize pipeline with new config
p._initialized = False
p.initialize()

# Ingest test file
test_file = Path("test_data/test.md")
print(f"\n  索引文档: {test_file}")
n = p.ingest_file(test_file)
print(f"  已索引 {n} 个 chunk")

# Query
print("\n  问答测试: '什么是RAG'")
answer = p.query("什么是RAG", top_k=5)
print(f"\n{'─'*50}")
print(f"  Q: 什么是RAG")
print(f"  A: {answer.answer}")
print(f"{'─'*50}")
print(f"  延迟: {answer.latency_ms:.0f}ms")
print(f"  来源数: {len(answer.sources)}")
for s in answer.sources:
    print(f"    [{s['doc_name']}] {s['content_snippet'][:80]}...")

print("\n" + "=" * 50)
print("  测试通过!")
print("=" * 50)
