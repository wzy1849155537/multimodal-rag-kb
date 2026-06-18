"""BaseIndexer ABC for vector + sparse index operations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .schemas import Chunk, SearchResult


class BaseIndexer(ABC):
    """Pluggable indexer. Manages storage and search of Chunks."""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index."""
        ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        ...

    @abstractmethod
    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Dense vector similarity search."""
        ...

    @abstractmethod
    def search_sparse(
        self,
        query_text: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Sparse (keyword/BM25) search."""
        ...

    @abstractmethod
    def get_stats(self) -> Dict:
        """Return index statistics."""
        ...
