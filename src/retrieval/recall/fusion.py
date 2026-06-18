"""Reciprocal Rank Fusion (RRF) and weighted score fusion."""

from typing import Dict, List

from src.core.schemas import SearchResult, SearchSource


def reciprocal_rank_fusion(
    dense_results: List[SearchResult],
    sparse_results: List[SearchResult],
    k: int = 60,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[SearchResult]:
    """Merge dense and sparse results using Reciprocal Rank Fusion.

    RRF score = SUM(weight / (k + rank))
    This avoids the problem of normalizing scores from different similarity spaces.
    """
    # Build score map: chunk_id -> accumulated RRF score
    scores: Dict[str, float] = {}
    chunks: Dict[str, SearchResult] = {}

    # Dense results
    for rank, r in enumerate(dense_results, 1):
        rrf = dense_weight / (k + rank)
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + rrf
        if r.chunk_id not in chunks:
            chunks[r.chunk_id] = r

    # Sparse results
    for rank, r in enumerate(sparse_results, 1):
        rrf = sparse_weight / (k + rank)
        scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + rrf
        if r.chunk_id not in chunks:
            chunks[r.chunk_id] = r

    # Build fused results
    fused = []
    for chunk_id, rrf_score in scores.items():
        chunk = chunks[chunk_id]
        chunk.score = rrf_score
        chunk.source = SearchSource.HYBRID
        fused.append(chunk)

    return fused


def weighted_score_fusion(
    dense_results: List[SearchResult],
    sparse_results: List[SearchResult],
    dense_weight: float = 0.7,
) -> List[SearchResult]:
    """Merge using weighted score combination (requires normalized scores)."""
    merged: Dict[str, SearchResult] = {}

    for r in dense_results:
        r.score *= dense_weight
        r.source = SearchSource.HYBRID
        merged[r.chunk_id] = r

    for r in sparse_results:
        r.score *= (1.0 - dense_weight)
        if r.chunk_id in merged:
            merged[r.chunk_id].score += r.score
        else:
            r.source = SearchSource.HYBRID
            merged[r.chunk_id] = r

    return sorted(merged.values(), key=lambda x: x.score, reverse=True)
