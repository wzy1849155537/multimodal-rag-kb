"""Chinese text processing utilities."""

import re
import hashlib
from typing import List


_WHITESPACE_RE = re.compile(r"\s+")
_BOILERPLATE_PATTERNS = [
    re.compile(r"第\s*\d+\s*页\s*(共\s*\d+\s*页)?", re.IGNORECASE),
    re.compile(r"Page\s+\d+\s*(of\s+\d+)?", re.IGNORECASE),
    re.compile(r"Copyright\s+©.*", re.IGNORECASE),
    re.compile(r"All\s+Rights\s+Reserved", re.IGNORECASE),
]
_CHINESE_CHAR_RE = re.compile(r"[一-鿿]")


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace to single spaces and strip."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def is_mostly_chinese(text: str, threshold: float = 0.3) -> bool:
    """Check if text contains a significant proportion of Chinese characters."""
    if not text:
        return False
    chinese_chars = len(_CHINESE_CHAR_RE.findall(text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return False
    return (chinese_chars / total_chars) >= threshold


def truncate_by_tokens(text: str, max_tokens: int = 512) -> str:
    """Roughly truncate text by token count (4 chars ~= 1 token for Chinese)."""
    char_limit = max_tokens * 3  # conservative estimate
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "..."


def clean_chunk_text(text: str) -> str:
    """Clean a chunk: normalize whitespace, remove boilerplate."""
    text = normalize_whitespace(text)
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return normalize_whitespace(text)


def chunk_id_from(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{doc_id}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
