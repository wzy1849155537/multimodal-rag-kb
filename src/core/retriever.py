"""BaseRetriever ABC for multi-path recall orchestration."""

from abc import ABC, abstractmethod
from typing import List

from .schemas import SearchResult, RouteDecision


class BaseRetriever(ABC):
    """Pluggable retriever. Executes multi-path recall and fusion."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        route: RouteDecision,
    ) -> List[SearchResult]:
        """Execute retrieval and return fused results."""
        ...
