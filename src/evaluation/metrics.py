"""Custom retrieval metrics: MRR, Hit@k, NDCG."""

from typing import List
import math


def mean_reciprocal_rank(
    queries: List[str],
    relevant_doc_ids: List[List[str]],
    retrieved_results: List[List[str]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Args:
        queries: List of query strings (for logging).
        relevant_doc_ids: For each query, list of relevant chunk IDs.
        retrieved_results: For each query, ordered list of retrieved chunk IDs.

    Returns:
        MRR score (0.0 to 1.0).
    """
    if not queries:
        return 0.0

    reciprocal_ranks = []
    for i, (relevant, retrieved) in enumerate(zip(relevant_doc_ids, retrieved_results)):
        for rank, chunk_id in enumerate(retrieved, 1):
            if chunk_id in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def hit_at_k(
    queries: List[str],
    relevant_doc_ids: List[List[str]],
    retrieved_results: List[List[str]],
    k: int = 5,
) -> float:
    """Compute Hit@k: fraction of queries with at least one relevant result in top-k.

    Args:
        queries: List of query strings.
        relevant_doc_ids: For each query, list of relevant chunk IDs.
        retrieved_results: For each query, ordered list of retrieved chunk IDs.
        k: Number of top results to check.

    Returns:
        Hit@k score (0.0 to 1.0).
    """
    if not queries:
        return 0.0

    hits = 0
    for relevant, retrieved in zip(relevant_doc_ids, retrieved_results):
        top_k = set(retrieved[:k])
        if any(r in top_k for r in relevant):
            hits += 1

    return hits / len(queries)


def ndcg_at_k(
    queries: List[str],
    relevance_scores: List[List[float]],
    retrieved_results: List[List[str]],
    k: int = 10,
) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        queries: List of query strings.
        relevance_scores: For each query, dict mapping chunk_id -> relevance (0-1).
        retrieved_results: For each query, ordered list of retrieved chunk IDs.
        k: Cutoff rank.

    Returns:
        NDCG@k score (0.0 to 1.0).
    """
    if not queries:
        return 0.0

    def dcg(scores: List[float]) -> float:
        return sum(
            score / math.log2(i + 2)  # i+2 because i is 0-indexed
            for i, score in enumerate(scores[:k])
        )

    ndcg_scores = []
    for i, retrieved in enumerate(retrieved_results):
        # Get relevance scores for retrieved items
        rel_dict = {}
        if i < len(relevance_scores):
            if isinstance(relevance_scores[i], dict):
                rel_dict = relevance_scores[i]
            elif isinstance(relevance_scores[i], list):
                for j, score in enumerate(relevance_scores[i]):
                    if j < len(retrieved):
                        rel_dict[retrieved[j]] = score

        retrieved_scores = [rel_dict.get(cid, 0.0) for cid in retrieved[:k]]
        ideal_scores = sorted(rel_dict.values(), reverse=True)[:k]

        actual_dcg = dcg(retrieved_scores)
        ideal_dcg = dcg(ideal_scores)

        if ideal_dcg == 0:
            ndcg_scores.append(1.0 if actual_dcg == 0 else 0.0)
        else:
            ndcg_scores.append(actual_dcg / ideal_dcg)

    return sum(ndcg_scores) / len(ndcg_scores)
