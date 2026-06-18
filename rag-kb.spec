# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for rag-kb executable.

Build with:
    pyinstaller rag-kb.spec

Output will be at: dist/rag-kb.exe
"""

import sys
from pathlib import Path

# --- Project paths ---
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
CLI_DIR = PROJECT_ROOT / "cli"
CONFIG_DIR = PROJECT_ROOT / "config"

# --- Hidden imports that PyInstaller can't detect ---
hiddenimports = [
    # LangChain internals
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain.text_splitter",
    # ChromaDB internals
    "chromadb",
    "chromadb.config",
    "chromadb.db",
    "chromadb.db.mixins",
    "chromadb.utils.embedding_functions",
    "chromadb.api",
    "chromadb.api.types",
    # sentence-transformers
    "sentence_transformers",
    "sentence_transformers.models",
    # PyMuPDF
    "fitz",
    # Streamlit (for `serve` command)
    "streamlit",
    "streamlit.web.bootstrap",
    "streamlit.runtime",
    "streamlit.runtime.scriptrunner",
    # YAML
    "yaml",
    # Logging
    "loguru",
    # Additional
    "tqdm",
    "numpy",
    "PIL",
    "PIL.Image",
    "openai",
    "typer",
    "rich",
    "click",
    "rank_bm25",
    "json",
    "hashlib",
    "sqlite3",
    "uuid",
    "re",
    "pathlib",
]

# --- Collect config files as external data ---
datas = [
    (str(CONFIG_DIR / "default.yaml"), "config"),
    (str(CONFIG_DIR / "models.yaml"), "config"),
    (str(CONFIG_DIR / "pipeline.yaml"), "config"),
]

a = Analysis(
    [str(CLI_DIR / "main.py")],
    pathex=[str(PROJECT_ROOT), str(SRC_DIR.parent)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "pandas",
        "torch",
        "torchvision",
        "tensorflow",
        "keras",
        "jupyter",
        "notebook",
        "IPython",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=None,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="rag-kb",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # 可替换为自定义图标路径
)
