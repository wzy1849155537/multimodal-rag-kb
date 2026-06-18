"""Hybrid index: orchestrates ChromaDB dense + BM25 sparse indices."""

from pathlib import Path
from typing import Dict, List, Optional

from src.core.indexer import BaseIndexer
from src.core.schemas import Chunk, SearchResult, SearchSource
from src.index.vector_store import ChromaVectorStore
from src.index.bm25_index import BM25Index
from src.registry import ModuleRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@ModuleRegistry.indexers.register("hybrid")
class HybridIndex(BaseIndexer):
    """Dense (ChromaDB) + Sparse (BM25) hybrid index."""

    def __init__(
        self,
        persist_directory: str = "./data/index/chroma",
        bm25_persist_path: str = "./data/index/bm25/bm25_index.json",
        collection_name: str = "rag_kb_default",
        distance_metric: str = "cosine",
    ):
        self._dense = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            distance_metric=distance_metric,
        )
        self._sparse = BM25Index(
            persist_path=Path(bm25_persist_path)
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to both indices."""
        # Add to dense
        chunks_with_embeddings = [c for c in chunks if c.embedding]
        if chunks_with_embeddings:
            self._dense.add_chunks(chunks_with_embeddings)

        # Add to sparse (all chunks, even without embedding)
        self._sparse.add_chunks(chunks)
        logger.info(
            f"Hybrid: added {len(chunks_with_embeddings)} dense, "
            f"{len(chunks)} sparse chunks"
        )

    def delete_by_doc_id(self, doc_id: str) -> int:
        dense_deleted = self._dense.delete_by_doc_id(doc_id)
        sparse_deleted = self._sparse.delete_by_doc_id(doc_id)
        return max(dense_deleted, sparse_deleted)

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        return self._dense.search_dense(query_embedding, top_k, filter_metadata)

    def search_sparse(
        self,
        query_text: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        return self._sparse.search(query_text, top_k)

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        top_k_fusion: int = 15,
        bm25_weight: float = 0.3,
        filter_metadata: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Execute both dense and sparse search, then fuse results."""
        from src.retrieval.recall.fusion import reciprocal_rank_fusion

        # Parallel retrieval
        dense_results = self._dense.search_dense(
            query_embedding, top_k_dense, filter_metadata
        )
        sparse_results = self._sparse.search(
            query_text, top_k_sparse
        )

        # Fuse
        fused = reciprocal_rank_fusion(
            dense_results, sparse_results,
            k=60,
            dense_weight=1.0 - bm25_weight,
            sparse_weight=bm25_weight,
        )

        # Sort by fused score and take top-k
        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k_fusion]

    def get_stats(self) -> Dict:
        dense_stats = self._dense.get_stats()
        sparse_stats = self._sparse.get_stats()
        return {**dense_stats, "bm25_documents": sparse_stats.get("total_documents", 0)}
