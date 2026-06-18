"""Multi-path recall orchestrator."""

from typing import List, Optional

from src.core.retriever import BaseRetriever
from src.core.schemas import SearchResult, RouteDecision
from src.index.hybrid_index import HybridIndex
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.retrievers.register("multi")
class MultiRecallRetriever(BaseRetriever):
    """Multi-path recall: dense + sparse, then RRF fusion."""

    def __init__(self, index: HybridIndex):
        self._index = index

    def retrieve(
        self,
        query: str,
        query_embedding: List[float],
        route: RouteDecision,
    ) -> List[SearchResult]:
        return self._index.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            top_k_dense=route.top_k,
            top_k_sparse=route.top_k,
            top_k_fusion=route.top_k,
            bm25_weight=route.bm25_weight,
            filter_metadata=route.metadata_filter,
        )
