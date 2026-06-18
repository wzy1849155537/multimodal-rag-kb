#!/usr/bin/env python
"""Demo: 多模态 RAG 知识库问答系统 完整演示"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# API Key
os.environ.setdefault("SILICONFLOW_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"
SEP = "=" * 60

def header(text):
    print(f"\n{BOLD}{CYAN}{SEP}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{SEP}{RESET}\n")

def step(n, text):
    print(f"{BOLD}{YELLOW}[{n}/5] {text}{RESET}")

# ================================================================
print(f"\n{BOLD}{'='*60}{RESET}")
print(f"{BOLD}  多模态 RAG 知识库问答系统 — 完整演示{RESET}")
print(f"{BOLD}  API: SiliconFlow | LLM: Qwen2.5-7B | Embed: BGE-M3{RESET}")
print(f"{BOLD}{'='*60}{RESET}")

# ================================================================
step(1, "准备测试文档...")

demo_dir = Path("demo_docs")
demo_dir.mkdir(exist_ok=True)

# Create PDF-like content
(demo_dir / "人工智能导论.md").write_text("""# 人工智能导论

## 什么是人工智能
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。

## 机器学习
机器学习是AI的核心技术之一。它使计算机能够从数据中学习，而无需显式编程。

### 监督学习
监督学习使用带标签的数据训练模型。常见算法包括：
- 线性回归
- 决策树
- 支持向量机（SVM）
- 神经网络

### 无监督学习
无监督学习从无标签数据中发现模式，如聚类和降维。

## 深度学习
深度学习使用多层神经网络处理复杂任务。CNN用于图像识别，RNN和Transformer用于自然语言处理。

## 大语言模型
2023年以来，GPT-4、Claude、文心一言、通义千问等大语言模型迅速发展。
这些模型基于Transformer架构，通过海量文本预训练获得强大的语言理解和生成能力。
RAG（检索增强生成）技术将大模型与外部知识库结合，有效缓解了幻觉问题。
""", encoding="utf-8")

(demo_dir / "Python入门笔记.md").write_text("""# Python 编程入门

## Python 简介
Python是一种高级解释型编程语言，由Guido van Rossum于1991年创建。
其设计哲学强调代码的可读性和简洁的语法。

## 基础语法
```python
# 变量和数据类型
name = "世界"
age = 30
print(f"你好，{name}！")

# 列表推导式
squares = [x**2 for x in range(10)]
```

## 常用库
- **NumPy**: 数值计算，多维数组操作
- **Pandas**: 数据分析，DataFrame
- **LangChain**: 大语言模型应用开发框架
- **ChromaDB**: 向量数据库，用于语义搜索

## 在AI领域的应用
Python是AI开发的主流语言。PyTorch和TensorFlow等深度学习框架都基于Python。
LangChain框架让开发者能快速构建RAG应用，结合向量数据库实现知识库问答。
""", encoding="utf-8")

(demo_dir / "公司介绍.txt").write_text("""公司简介

星海科技有限公司成立于2020年，总部位于北京中关村。
公司专注于企业级AI解决方案，主要产品包括：
1. 智能客服系统 - 基于大语言模型的7x24小时客服
2. RAG知识库平台 - 帮助企业构建内部知识问答系统
3. 文档智能处理 - OCR识别、文档结构化

公司现有员工200余人，其中研发团队占比60%。
2024年获得A轮融资5000万元。

核心价值观：创新、务实、协作、共赢
""", encoding="utf-8")

print(f"  {GREEN}[OK]{RESET} 创建了 3 份测试文档:")
print(f"    - 人工智能导论.md (AI理论知识)")
print(f"    - Python入门笔记.md (编程技术)")
print(f"    - 公司介绍.txt (企业信息)")

# ================================================================
step(2, "索引文档...")

from src.pipeline import RAGPipeline

p = RAGPipeline()

p.config._data["embedder"] = "siliconflow"
p.config._data["indexer"] = "dense"
p.config._data["llm"] = {
    "default": "siliconflow",
    "providers": {
        "siliconflow": {
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": os.environ["SILICONFLOW_API_KEY"],
            "model": "Qwen/Qwen2.5-7B-Instruct",
        }
    }
}
p.config._data["generation"] = {
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "llm_temperature": 0.3,
    "max_context_tokens": 4096,
}
p._initialized = False

for f in sorted(demo_dir.iterdir()):
    if f.is_file():
        n = p.ingest_file(f)
        print(f"  {GREEN}[OK]{RESET} {f.name}: {n} chunks")

stats = p.get_stats()
print(f"\n  {BOLD}索引统计:{RESET} 共 {stats['total_chunks']} 个 chunks")

# ================================================================
step(3, "知识问答测试...")

questions = [
    "什么是RAG技术？",
    "Python在AI开发中有什么优势？",
    "星海科技的主要产品有哪些？",
    "什么是监督学习和无监督学习？",
]

for q in questions:
    print(f"\n  {BOLD}>> Q:{RESET} {q}")
    answer = p.query(q, top_k=5)
    # Decode answer
    print(f"  {GREEN}>> A:{RESET} {answer.answer}")
    print(f"  {YELLOW}>> {answer.latency_ms:.0f}ms{RESET}  |  ", end="")
    print(f"来源: ", end="")
    for s in answer.sources[:3]:
        print(f"[{s['doc_name']}]", end=" ")
    print()

# ================================================================
step(4, "检索精度验证...")

print(f"\n  验证：问题明确指向文档内容，答案是否引用了正确的来源？")
test_q = "大语言模型基于什么架构？"
answer = p.query(test_q, top_k=5)
print(f"\n  {BOLD}>> Q:{RESET} {test_q}")
print(f"  {GREEN}>> A:{RESET} {answer.answer}")
print(f"  {YELLOW}期望答案包含:{RESET} Transformer")
print(f"  {YELLOW}期望来源:{RESET} 人工智能导论.md")

# ================================================================
step(5, "系统状态...")

stats = p.get_stats()
print(f"\n  {GREEN}[OK]{RESET} 索引文档数: 3")
print(f"  {GREEN}[OK]{RESET} 总Chunks: {stats['total_chunks']}")
print(f"  {GREEN}[OK]{RESET} 存储位置: {stats.get('persist_directory', 'N/A')}")
print(f"  {GREEN}[OK]{RESET} 嵌入模型: BAAI/bge-m3 (SiliconFlow API)")
print(f"  {GREEN}[OK]{RESET} 生成模型: Qwen/Qwen2.5-7B-Instruct")

# ================================================================
print(f"\n{BOLD}{CYAN}{SEP}{RESET}")
print(f"{BOLD}{CYAN}  演示完成！{RESET}")
print(f"{BOLD}{CYAN}{SEP}{RESET}")

print(f"""
  {BOLD}接下来你可以:{RESET}

  1. 放自己的文档到 demo_docs/ 目录
     python -m cli.main ingest demo_docs/

  2. 命令行问答
     setup_env.bat
     python -m cli.main query "你的问题" -v

  3. 启动 Web 界面
     python -m cli.main serve
""")
