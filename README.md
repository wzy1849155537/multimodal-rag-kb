# 📚 多模态 RAG 知识库问答系统

面向个人的多模态 RAG（检索增强生成）知识库问答系统。支持 PDF、Markdown、图片等常见知识源，围绕**离线索引**、**在线检索**、**效果测评**、**增量索引与缓存加速**完成全链路工程优化。

## ✨ 功能特性

- 📄 **多模态文档解析**：PDF（含扫描件 VLM-OCR）、Markdown、TXT、图片
- 🔍 **混合检索**：BM25 稀疏 + ChromaDB 稠密向量，RRF 融合
- 🧠 **智能问答**：Query 改写 → 问题路由 → 多路召回 → Rerank → 低置信度二次检索
- 📊 **效果测评**：RAGAS + MRR + Hit@k + 网格搜索实验
- 🔄 **增量索引**：SHA256 哈希变更检测，支持新增/修改/删除热更新
- ⚡ **分层缓存**：QA 语义缓存 + 索引结构缓存
- 🏭 **工厂模式**：所有模块可插拔配置，pipeline.yaml 一键切换
- 🖥️ **桌面应用**：PyWebView 原生 Windows 窗口，可打包为独立 exe

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# Windows
set SILICONFLOW_API_KEY=你的Key

# Linux/Mac
export SILICONFLOW_API_KEY=你的Key
```

> 免费获取 Key：[SiliconFlow 控制台](https://cloud.siliconflow.cn/account/ak)

### 3. 索引文档

```bash
# 索引单个文件
python -m cli.main ingest document.pdf

# 索引整个目录
python -m cli.main ingest ./my_docs/

# 查看索引统计
python -m cli.main stats
```

### 4. 开始问答

```bash
# 命令行问答
python -m cli.main query "文档的主要内容是什么？" -v

# 启动 Web 界面
python -m cli.main serve

# 启动桌面版（原生窗口）
python desktop_app.py
```

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────┐
│                    CLI / Web UI / 桌面版              │
├─────────────────────────────────────────────────────┤
│  离线索引                    │  在线检索              │
│  ┌──────────┐               │  ┌──────────────┐     │
│  │ Parser   │ PDF/MD/Img    │  │ Query Rewrite│     │
│  ├──────────┤               │  ├──────────────┤     │
│  │ Chunker  │ Recursive     │  │ Router       │     │
│  ├──────────┤               │  ├──────────────┤     │
│  │ Embedder │ BGE-M3/OpenAI │  │ Multi-Recall │     │
│  ├──────────┤               │  ├──────────────┤     │
│  │ Index    │ Chroma + BM25 │  │ Rerank       │     │
│  └──────────┘               │  ├──────────────┤     │
│                              │  │ Verify       │     │
│  增量同步                    │  ├──────────────┤     │
│  ┌──────────────────┐       │  │ Generate     │     │
│  │ HashManager      │       │  └──────────────┘     │
│  │ ChangeDetector   │       │                        │
│  │ SyncEngine       │       │  缓存层                │
│  └──────────────────┘       │  ┌──────────────┐     │
│                              │  │ QA Cache     │     │
│  测评                       │  │ Index Cache  │     │
│  ┌──────────────────┐       │  └──────────────┘     │
│  │ RAGAS + MRR/Hit  │       │                        │
│  │ Grid Search      │       │                        │
│  └──────────────────┘       │                        │
└─────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
multimodal-rag-kb/
├── config/                         # YAML 配置
│   ├── default.yaml                # 主配置（路径、参数）
│   ├── models.yaml                 # 模型配置（LLM/Embedding）
│   └── pipeline.yaml               # 模块选择（工厂模式切换）
├── src/
│   ├── core/                       # ABC 抽象接口（11个）
│   │   ├── parser.py               #   BaseParser
│   │   ├── chunker.py              #   BaseChunker
│   │   ├── embedder.py             #   BaseEmbedder
│   │   ├── indexer.py              #   BaseIndexer
│   │   ├── query_rewriter.py       #   BaseQueryRewriter
│   │   ├── router.py               #   BaseRouter
│   │   ├── retriever.py            #   BaseRetriever
│   │   ├── reranker.py             #   BaseReranker
│   │   ├── verifier.py             #   BaseVerifier
│   │   ├── generator.py            #   BaseGenerator
│   │   └── schemas.py              #   共享数据类
│   ├── parsers/                    # 文档解析器
│   │   ├── pdf_parser.py           #   PDF（含扫描件 VLM-OCR）
│   │   ├── markdown_parser.py      #   Markdown
│   │   ├── text_parser.py          #   纯文本
│   │   ├── image_parser.py         #   图片
│   │   └── ocr_vlm.py              #   OCR + VLM 管线
│   ├── chunkers/                   # 切分策略
│   ├── embedders/                  # 嵌入模型
│   ├── index/                      # 索引管理
│   │   ├── vector_store.py         #   ChromaDB
│   │   ├── bm25_index.py           #   BM25
│   │   └── hybrid_index.py         #   混合索引
│   ├── retrieval/                  # 检索管线
│   │   ├── query_rewriting/        #   查询改写
│   │   ├── routing/                #   问题路由
│   │   ├── recall/                 #   多路召回 + RRF 融合
│   │   ├── rerank/                 #   重排序
│   │   └── verification/           #   置信度校验
│   ├── generation/                 # 答案生成
│   ├── evaluation/                 # RAGAS 测评
│   ├── cache/                      # 缓存层
│   ├── incremental/                # 增量索引
│   ├── utils/                      # 工具模块
│   ├── pipeline.py                 # 核心编排器
│   ├── registry.py                 # 工厂注册中心
│   └── config.py                   # 配置加载器
├── cli/main.py                     # CLI 入口
├── web/app.py                      # Streamlit Web UI
├── desktop_app.py                  # 桌面应用启动器
├── build_desktop.py                # 打包脚本
└── requirements.txt
```

## ⚙️ 配置说明

### 切换嵌入模型

编辑 `config/pipeline.yaml`：
```yaml
embedder: "siliconflow"  # siliconflow | bge-m3 | bge-small | openai
```

### 切换索引模式

```yaml
indexer: "hybrid"  # dense | hybrid
```

### 调整检索参数

编辑 `config/default.yaml`：
```yaml
retrieval:
  top_k_dense: 20      # 稠密召回数量
  top_k_sparse: 20     # 稀疏召回数量
  top_k_fusion: 15     # 融合后数量
  bm25_weight: 0.3     # BM25 权重 (0=仅稠密, 1=仅稀疏)
```

## 🔧 可用模型

| 类型 | 模型 | 说明 |
|------|------|------|
| LLM | Qwen/Qwen2.5-32B-Instruct | 指令遵循强，推荐 |
| LLM | Qwen/Qwen3-8B | 速度快 |
| LLM | deepseek-ai/DeepSeek-V3 | 综合能力强 |
| Embedding | BAAI/bge-m3 | 1024维，中英双语 |
| Vision | Qwen/Qwen3-VL-8B-Instruct | 扫描件 OCR |

## 📦 打包为桌面应用

```bash
python build_desktop.py
# 输出: dist/RAG知识库问答系统.exe (~300MB)
```

## 📄 License

MIT
