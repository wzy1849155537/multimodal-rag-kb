#!/usr/bin/env python
"""
One-click build script: packages rag-kb into a single executable.

Usage:
    python build_exe.py          # Build CLI exe only (~200MB)
    python build_exe.py --lite   # Build lite version without Streamlit (~100MB)

Output:
    dist/rag-kb.exe              # Single executable
    dist/config/                 # Config files (copy alongside exe)

Requirements:
    pip install pyinstaller
"""

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
DIST_DIR = PROJECT_ROOT / "dist"


def clean():
    """Remove previous build artifacts."""
    for d in ["build", "dist", "__pycache__"]:
        path = PROJECT_ROOT / d
        if path.exists():
            shutil.rmtree(path)
    for spec in PROJECT_ROOT.glob("*.spec"):
        if spec.name != "rag-kb.spec":
            spec.unlink()
    print("✓ Cleaned build artifacts")


def build(lite: bool = False):
    """Run PyInstaller build."""
    spec_file = PROJECT_ROOT / "rag-kb.spec"

    # Base command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
    ]

    if lite:
        # Lite: CLI only, no Streamlit bundled
        entry = PROJECT_ROOT / "cli" / "main.py"
        cmd += [
            "--onefile",
            "--name", "rag-kb",
            "--add-data", f"{PROJECT_ROOT / 'config' / 'default.yaml'};config",
            "--add-data", f"{PROJECT_ROOT / 'config' / 'models.yaml'};config",
            "--add-data", f"{PROJECT_ROOT / 'config' / 'pipeline.yaml'};config",
            "--hidden-import", "src",
            "--hidden-import", "src.core",
            "--hidden-import", "src.parsers",
            "--hidden-import", "src.chunkers",
            "--hidden-import", "src.embedders",
            "--hidden-import", "src.index",
            "--hidden-import", "src.generation",
            "--hidden-import", "src.pipeline",
            "--hidden-import", "src.registry",
            "--hidden-import", "src.config",
            "--hidden-import", "src.utils",
            "--hidden-import", "langchain",
            "--hidden-import", "langchain.text_splitter",
            "--hidden-import", "chromadb",
            "--hidden-import", "sentence_transformers",
            "--hidden-import", "fitz",
            "--hidden-import", "yaml",
            "--hidden-import", "loguru",
            "--hidden-import", "openai",
            "--hidden-import", "typer",
            "--hidden-import", "rank_bm25",
            "--hidden-import", "PIL",
            "--hidden-import", "PIL.Image",
            "--exclude-module", "streamlit",
            "--exclude-module", "paddleocr",
            "--exclude-module", "FlagEmbedding",
            "--exclude-module", "ragas",
            "--exclude-module", "torch",
            "--exclude-module", "matplotlib",
            "--exclude-module", "pandas",
            str(entry),
        ]
    else:
        # Full: CLI + Streamlit
        cmd += [
            str(spec_file),
        ]

    print(f"Building {'lite' if lite else 'full'} version...")
    print(f"Command: {' '.join(cmd[:5])} ...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("✗ Build failed!")
        sys.exit(1)

    # Copy config files next to exe
    config_dest = DIST_DIR / "config"
    if config_dest.exists():
        shutil.rmtree(config_dest)
    shutil.copytree(PROJECT_ROOT / "config", config_dest)
    print(f"✓ Config files copied to {config_dest}")

    # Show result
    exe_path = DIST_DIR / "rag-kb.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n{'='*50}")
        print(f"  Build successful!")
        print(f"  Output: {exe_path}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"{'='*50}")
        print(f"\n  Usage:")
        print(f"    rag-kb.exe ingest ./docs/     # Index documents")
        print(f"    rag-kb.exe query \"问题\"      # Ask a question")
        print(f"    rag-kb.exe stats              # View index stats")


def main():
    lite = "--lite" in sys.argv

    clean()
    build(lite=lite)

    # Also clean up build dir (keep only dist)
    build_dir = PROJECT_ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)


if __name__ == "__main__":
    main()
