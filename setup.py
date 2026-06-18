"""Package setup for multimodal-rag-kb."""

from setuptools import setup, find_packages

setup(
    name="multimodal-rag-kb",
    version="0.1.0",
    description="多模态 RAG 知识库问答系统",
    author="",
    packages=find_packages(include=["src", "src.*", "cli", "cli.*"]),
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-community>=0.3.0",
        "chromadb>=0.5.0",
        "rank-bm25>=0.2.2",
        "sentence-transformers>=3.0.0",
        "PyMuPDF>=1.24.0",
        "Pillow>=10.0.0",
        "openai>=1.40.0",
        "typer[all]>=0.12.0",
        "streamlit>=1.35.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "tqdm>=4.66.0",
        "numpy>=1.26.0",
    ],
    extras_require={
        "full": [
            "FlagEmbedding>=1.3.0",
            "paddleocr>=2.8.0",
            "ragas>=0.2.0",
            "omegaconf>=2.3.0",
            "diskcache>=5.6.0",
            "tiktoken>=0.7.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-kb=cli.main:app",
        ],
    },
    # Include config files in the package
    package_data={
        "": ["*.yaml", "*.yml"],
    },
    include_package_data=True,
)
