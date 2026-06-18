"""BaseReranker ABC for re-ranking search results."""

from abc import ABC, abstractmethod
from typing import List

from .schemas import SearchResult


class BaseReranker(ABC):
    """Pluggable reranker. Re-scores candidates for precision."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Re-rank candidates and return top_k results."""
        ...
