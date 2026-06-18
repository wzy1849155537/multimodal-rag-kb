"""No-op reranker (returns results as-is)."""

from typing import List

from src.core.reranker import BaseReranker
from src.core.schemas import SearchResult
from src.registry import ModuleRegistry


@ModuleRegistry.rerankers.register("noop")
class NoOpReranker(BaseReranker):
    """Passes results through without re-ranking."""

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        return candidates[:top_k]
