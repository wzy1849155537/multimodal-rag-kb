"""Quick setup check: verify all imports and basic functionality."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

errors = []

def check(name, code):
    try:
        exec(code)
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append((name, str(e)))
        return False

print("=" * 50)
print("  检查导入...")
print("=" * 50)

check("schemas", "from src.core.schemas import RawDocument, Chunk, SearchResult, GeneratedAnswer, IndexStats")
check("registry", "from src.registry import ModuleRegistry")
check("config", "from src.config import ConfigLoader")
check("pdf_parser", "from src.parsers.pdf_parser import PDFParser")
check("markdown_parser", "from src.parsers.markdown_parser import MarkdownParser")
check("text_parser", "from src.parsers.text_parser import TextParser")
check("chunker", "from src.chunkers.recursive_chunker import RecursiveChunker")
check("embedder", "from src.embedders.huggingface_embedder import HuggingFaceEmbedder")
check("vector_store", "from src.index.vector_store import ChromaVectorStore")
check("bm25", "from src.index.bm25_index import BM25Index")
check("hybrid", "from src.index.hybrid_index import HybridIndex")
check("fusion", "from src.retrieval.recall.fusion import reciprocal_rank_fusion")
check("reranker", "from src.retrieval.rerank.cross_encoder_reranker import CrossEncoderReranker")
check("rewriter", "from src.retrieval.query_rewriting.llm_rewriter import LLMRewriter")
check("verifier", "from src.retrieval.verification.confidence import ConfidenceVerifier")
check("generator", "from src.generation.rag_generator import RAGGenerator")
check("hash_mgr", "from src.incremental.hash_manager import HashManager")
check("change_detector", "from src.incremental.change_detector import ChangeDetector")
check("qa_cache", "from src.cache.qa_cache import QACache")
check("evaluator", "from src.evaluation.evaluator import RAGEvaluator")
check("pipeline", "from src.pipeline import RAGPipeline; print('    (pipeline imports all submodules)')")

print()
print("=" * 50)
if errors:
    print(f"  {len(errors)} 个导入失败:")
    for name, err in errors:
        print(f"    - {name}: {err}")
else:
    print("  全部导入成功!")
print("=" * 50)

# Quick test: create test doc and parse it
if not errors:
    print()
    print("=" * 50)
    print("  功能测试: 创建测试文档并解析...")
    print("=" * 50)

    # Create a test markdown file
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    test_file = test_dir / "test.md"
    test_file.write_text(
        "# 测试文档\n\n"
        "这是一个测试文档。\n\n"
        "## 第一节\n"
        "RAG（检索增强生成）是一种结合检索和生成的技术。\n\n"
        "## 第二节\n"
        "向量数据库用于存储和检索高维向量。\n",
        encoding="utf-8"
    )
    print(f"  创建测试文件: {test_file}")

    # Test parser
    try:
        from src.parsers.markdown_parser import MarkdownParser
        from src.chunkers.recursive_chunker import RecursiveChunker

        parser = MarkdownParser()
        doc = parser.parse(test_file)
        print(f"  [OK] 解析成功: {len(doc.content)} 字符, doc_id={doc.doc_id}")

        # Test chunker
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)
        print(f"  [OK] 切分成功: {len(chunks)} 个 chunk")
        for c in chunks:
            print(f"    [{c.chunk_id}] {c.content[:60]}...")
        print()

        print("=" * 50)
        print("  基础功能验证通过!")
        print("=" * 50)
        print()
        print("  下一步:")
        print("    1. 设置 LLM API Key: set OPENAI_API_KEY=sk-xxx")
        print("      或: set DASHSCOPE_API_KEY=sk-xxx (通义千问)")
        print("    2. 索引文档: python -m cli.main ingest ./test_data/")
        print("    3. 提问: python -m cli.main query \"什么是RAG\" -v")
    except Exception as e:
        print(f"  [FAIL] 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
