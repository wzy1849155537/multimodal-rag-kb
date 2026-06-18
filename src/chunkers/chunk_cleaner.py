"""Chunk post-processing: dedup, clean, metadata binding."""

import hashlib
from typing import List

from src.core.schemas import Chunk
from src.utils.text_utils import clean_chunk_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkCleaner:
    """Post-process chunks: deduplicate, clean, enrich metadata."""

    @staticmethod
    def clean(chunks: List[Chunk]) -> List[Chunk]:
        """Clean and deduplicate chunks."""
        seen_hashes = set()
        cleaned = []

        for chunk in chunks:
            # Normalize text
            chunk.content = clean_chunk_text(chunk.content)

            # Skip empty chunks
            if len(chunk.content) < 10:
                continue

            # Deduplicate by content hash
            content_hash = hashlib.md5(
                chunk.content.encode("utf-8")
            ).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            chunk.metadata["content_hash"] = content_hash
            chunk.metadata["char_count"] = len(chunk.content)
            cleaned.append(chunk)

        removed = len(chunks) - len(cleaned)
        if removed > 0:
            logger.info(f"ChunkCleaner removed {removed} duplicate/empty chunks")
        return cleaned
