"""BM25 sparse index using rank_bm25 with JSON persistence."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.schemas import Chunk, SearchResult, SearchSource
from src.utils.file_utils import ensure_dir, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Chinese-aware tokenization (character + word boundary aware)
_TOKEN_RE = re.compile(r"[一-鿿]+|[a-zA-Z0-9]+|[^\s]")


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25, handling Chinese characters."""
    tokens = _TOKEN_RE.findall(text.lower())
    # Split Chinese sequences into individual characters for BM25
    result = []
    for token in tokens:
        if re.match(r"^[一-鿿]+$", token):
            result.extend(token)  # Split Chinese word into characters
            result.append(token)  # Also keep the full word
        else:
            result.append(token)
    return result


class BM25Index:
    """BM25 sparse index with persistence."""

    def __init__(self, persist_path: Optional[Path] = None):
        self._persist_path = persist_path
        self._chunks: Dict[str, Chunk] = {}  # chunk_id -> chunk
        self._corpus: List[str] = []        # tokenized texts
        self._chunk_ids: List[str] = []     # parallel to corpus
        self._bm25: Optional[BM25Okapi] = None

        if persist_path and persist_path.exists():
            self._load()

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the BM25 index."""
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            tokens = tokenize(chunk.content)
            self._corpus.append(" ".join(tokens))
            self._chunk_ids.append(chunk.chunk_id)

        # Rebuild BM25
        tokenized_corpus = [doc.split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index updated: {len(self._chunk_ids)} documents")

        if self._persist_path:
            self._save()

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document."""
        deleted = 0
        new_chunks = {}
        for cid, chunk in self._chunks.items():
            if chunk.doc_id != doc_id:
                new_chunks[cid] = chunk
            else:
                deleted += 1

        if deleted > 0:
            self._chunks = new_chunks
            # Rebuild corpus without deleted chunks
            self._corpus = []
            self._chunk_ids = []
            for cid in new_chunks:
                chunk = new_chunks[cid]
                tokens = tokenize(chunk.content)
                self._corpus.append(" ".join(tokens))
                self._chunk_ids.append(cid)
            self._bm25 = BM25Okapi([doc.split() for doc in self._corpus])

            if self._persist_path:
                self._save()
            logger.info(f"BM25: deleted {deleted} chunks for doc {doc_id}")
        return deleted

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Search BM25 index with a text query."""
        if not self._bm25 or not self._chunk_ids:
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        if len(scores) == 0:
            return []

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        max_score = float(scores.max()) if len(scores) > 0 else 1.0

        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks.get(chunk_id)
            if chunk:
                normalized_score = float(scores[idx] / max_score) if max_score > 0 else 0.0
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    score=normalized_score,
                    source=SearchSource.SPARSE,
                    metadata=chunk.metadata.copy(),
                ))

        return results[:top_k]

    def _save(self) -> None:
        """Persist BM25 state to JSON."""
        if not self._persist_path:
            return
        ensure_dir(self._persist_path.parent)
        data = {
            "chunk_ids": self._chunk_ids,
            "corpus": self._corpus,
            "chunks": {
                cid: {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "content": c.content,
                    "chunk_index": c.chunk_index,
                    "metadata": c.metadata,
                }
                for cid, c in self._chunks.items()
            },
        }
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.debug(f"BM25 persisted to {self._persist_path}")

    def _load(self) -> bool:
        """Load BM25 state from JSON."""
        try:
            data = load_json(self._persist_path)
            self._chunk_ids = data.get("chunk_ids", [])
            self._corpus = data.get("corpus", [])

            from src.core.schemas import Chunk
            for cid, cdata in data.get("chunks", {}).items():
                self._chunks[cid] = Chunk(
                    chunk_id=cdata["chunk_id"],
                    doc_id=cdata["doc_id"],
                    content=cdata["content"],
                    chunk_index=cdata.get("chunk_index", 0),
                    metadata=cdata.get("metadata", {}),
                )

            if self._corpus:
                self._bm25 = BM25Okapi([doc.split() for doc in self._corpus])
                logger.info(f"BM25 loaded: {len(self._chunk_ids)} documents")
                return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 from {self._persist_path}: {e}")
        return False

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self._chunk_ids),
            "persist_path": str(self._persist_path) if self._persist_path else None,
        }
