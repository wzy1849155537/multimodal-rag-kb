"""File I/O and hashing utilities."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_files(
    directory: Path,
    extensions: Optional[List[str]] = None,
    recursive: bool = True,
) -> List[Path]:
    """Find all files with given extensions in a directory."""
    if not directory.exists():
        return []

    pattern = "**/*" if recursive else "*"
    files = []
    for path in directory.glob(pattern):
        if path.is_file():
            if extensions is None or path.suffix.lower() in extensions:
                files.append(path)
    return sorted(files)


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute the hash of a file."""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_hashes(
    directory: Path,
    extensions: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Compute hashes for all files in a directory. Returns {rel_path: hash}."""
    files = find_files(directory, extensions=extensions)
    result = {}
    for f in files:
        rel_path = str(f.relative_to(directory))
        result[rel_path] = compute_file_hash(f)
    return result


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Get basic file metadata."""
    stat = file_path.stat()
    return {
        "name": file_path.name,
        "stem": file_path.stem,
        "suffix": file_path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": stat.st_mtime,
    }


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file. Returns empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save data as JSON file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
