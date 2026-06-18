#!/usr/bin/env python
"""
Build the RAG Knowledge Base as a standalone Windows desktop app (.exe).

Usage:
    python build_desktop.py

Output:
    dist/rag-kb-desktop.exe    (~300MB single executable)
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
DIST = PROJECT_ROOT / "dist"
NAME = "RAG知识库问答系统"


def clean():
    for d in ["build", "dist"]:
        path = PROJECT_ROOT / d
        if path.exists():
            shutil.rmtree(path)
    print("[OK] Cleaned old build artifacts")


def build():
    print(f"Building {NAME}.exe ...")
    print("This will take 3-5 minutes on first run...\n")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",                          # No console window
        "--name", NAME,
        "--add-data", f"{PROJECT_ROOT / 'config'};config",
        "--add-data", f"{PROJECT_ROOT / 'web'};web",
        "--add-data", f"{PROJECT_ROOT / 'src'};src",
        "--hidden-import", "src",
        "--hidden-import", "src.core",
        "--hidden-import", "src.core.schemas",
        "--hidden-import", "src.parsers",
        "--hidden-import", "src.parsers.pdf_parser",
        "--hidden-import", "src.parsers.markdown_parser",
        "--hidden-import", "src.parsers.text_parser",
        "--hidden-import", "src.chunkers",
        "--hidden-import", "src.chunkers.recursive_chunker",
        "--hidden-import", "src.embedders",
        "--hidden-import", "src.embedders.openai_embedder",
        "--hidden-import", "src.index",
        "--hidden-import", "src.index.vector_store",
        "--hidden-import", "src.generation",
        "--hidden-import", "src.generation.rag_generator",
        "--hidden-import", "src.generation.prompt_templates",
        "--hidden-import", "src.pipeline",
        "--hidden-import", "src.registry",
        "--hidden-import", "src.config",
        "--hidden-import", "src.utils",
        "--hidden-import", "src.incremental",
        "--hidden-import", "src.cache",
        "--hidden-import", "src.evaluation",
        "--hidden-import", "src.retrieval",
        "--hidden-import", "src.retrieval.recall",
        "--hidden-import", "src.retrieval.recall.fusion",
        "--hidden-import", "src.retrieval.recall.multi_recall",
        "--hidden-import", "src.retrieval.rerank",
        "--hidden-import", "src.retrieval.rerank.cross_encoder_reranker",
        "--hidden-import", "src.retrieval.query_rewriting",
        "--hidden-import", "src.retrieval.query_rewriting.llm_rewriter",
        "--hidden-import", "src.retrieval.routing",
        "--hidden-import", "src.retrieval.verification",
        "--hidden-import", "src.retrieval.verification.confidence",
        # Core dependencies
        "--hidden-import", "streamlit",
        "--hidden-import", "streamlit.web.bootstrap",
        "--hidden-import", "webview",
        "--hidden-import", "langchain",
        "--hidden-import", "langchain_text_splitters",
        "--hidden-import", "chromadb",
        "--hidden-import", "openai",
        "--hidden-import", "typer",
        "--hidden-import", "yaml",
        "--hidden-import", "loguru",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "fitz",
        "--hidden-import", "tqdm",
        "--hidden-import", "numpy",
        "--hidden-import", "rank_bm25",
        # Exclude heavy unused modules
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "tensorflow",
        "--exclude-module", "pandas",
        "--exclude-module", "scipy",
        "--exclude-module", "matplotlib",
        "--exclude-module", "paddleocr",
        "--exclude-module", "FlagEmbedding",
        "--exclude-module", "sentence_transformers",
        "--exclude-module", "transformers",
        "--exclude-module", "tokenizers",
        "--exclude-module", "accelerate",
        "--exclude-module", "datasets",
        "--exclude-module", "ragas",
        "--exclude-module", "jupyter",
        "--exclude-module", "ipykernel",
        "--exclude-module", "tkinter",
        str(PROJECT_ROOT / "desktop_app.py"),
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("\n[FAIL] Build failed!")
        sys.exit(1)

    # Copy config next to exe
    config_dest = DIST / "config"
    if config_dest.exists():
        shutil.rmtree(config_dest)
    shutil.copytree(PROJECT_ROOT / "config", config_dest)

    # Show result
    exe = DIST / f"{NAME}.exe"
    size_mb = exe.stat().st_size / (1024 * 1024) if exe.exists() else 0
    print(f"\n{'='*50}")
    print(f"  Build successful!")
    print(f"  Output: {exe}")
    print(f"  Size: {size_mb:.0f} MB")
    print(f"{'='*50}")
    print(f"\n  To use on desktop:")
    print(f"    1. Copy '{NAME}.exe' to Desktop")
    print(f"    2. Copy 'config/' folder next to it")
    print(f"    3. Double-click to launch")


if __name__ == "__main__":
    clean()
    build()
