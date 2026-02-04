"""
Retrieval Fusion Module for Medical-Agentic-RAG

This module implements fusion algorithms for combining dense and sparse retrieval results,
including Reciprocal Rank Fusion (RRF) and weighted score fusion.
"""

from typing import List, Tuple, Dict


def _normalize_scores(results: List[Tuple[str, float]]) -> Dict[str, float]:
    """
    Min-max normalize scores to [0, 1] range.

    Args:
        results: List of (chunk_id, score) tuples

    Returns:
        Dictionary mapping chunk_id to normalized score
    """
    if not results:
        return {}

    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # All scores are the same, return uniform scores
        return {chunk_id: 1.0 for chunk_id, _ in results}

    normalized = {}
    for chunk_id, score in results:
        normalized[chunk_id] = (score - min_score) / (max_score - min_score)

    return normalized


def _get_rank_map(results: List[Tuple[str, float]]) -> Dict[str, int]:
    """
    Convert results list to rank dictionary.
    Rank starts at 1 for the first (highest scored) result.

    Args:
        results: List of (chunk_id, score) tuples, sorted by score desc

    Returns:
        Dictionary mapping chunk_id to rank (1-indexed)
    """
    rank_map = {}
    for rank, (chunk_id, _) in enumerate(results, start=1):
        rank_map[chunk_id] = rank
    return rank_map


def rrf_fuse(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: float = 60.0,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm for combining retrieval results.

    RRF formula: score = sum(1 / (k + rank)) for each list where the chunk appears

    This fusion method is effective because:
    - It only uses ranks, not raw scores (different scales don't matter)
    - Items appearing in multiple lists get boosted
    - The constant k (typically 60) dampens the impact of high ranks

    Args:
        dense_results: Results from dense vector retrieval, sorted by score desc.
                       Format: [(chunk_id, score), ...]
        sparse_results: Results from BM25 retrieval, sorted by score desc.
                        Format: [(chunk_id, score), ...]
        k: RRF constant (typically 60). Higher values reduce the impact of ranking differences.
        top_k: Number of top results to return.

    Returns:
        List of (chunk_id, rrf_score) tuples sorted by RRF score descending.

    Example:
        >>> dense = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        >>> sparse = [("doc2", 0.95), ("doc1", 0.6), ("doc4", 0.5)]
        >>> fused = rrf_fuse(dense, sparse, k=60, top_k=3)
        >>> # doc1 and doc2 rank high as they appear in both lists
    """
    # Get rank mappings for both result lists
    dense_ranks = _get_rank_map(dense_results)
    sparse_ranks = _get_rank_map(sparse_results)

    # Collect all unique chunk IDs
    all_chunk_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

    # Calculate RRF score for each chunk
    rrf_scores = {}
    for chunk_id in all_chunk_ids:
        score = 0.0

        # Add contribution from dense results if present
        if chunk_id in dense_ranks:
            rank = dense_ranks[chunk_id]
            score += 1.0 / (k + rank)

        # Add contribution from sparse results if present
        if chunk_id in sparse_ranks:
            rank = sparse_ranks[chunk_id]
            score += 1.0 / (k + rank)

        rrf_scores[chunk_id] = score

    # Sort by RRF score descending and return top_k
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def weighted_fuse(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Weighted fusion using normalized scores.

    This method normalizes scores from each retrieval method to [0, 1] range,
    then combines them using weighted sum.

    Args:
        dense_results: Results from dense vector retrieval, sorted by score desc.
                       Format: [(chunk_id, score), ...]
        sparse_results: Results from BM25 retrieval, sorted by score desc.
                        Format: [(chunk_id, score), ...]
        dense_weight: Weight for dense retrieval scores (default 0.5)
        sparse_weight: Weight for sparse retrieval scores (default 0.5)
        top_k: Number of top results to return.

    Returns:
        List of (chunk_id, fused_score) tuples sorted by fused score descending.

    Example:
        >>> dense = [("doc1", 0.9), ("doc2", 0.8)]
        >>> sparse = [("doc2", 0.95), ("doc3", 0.7)]
        >>> fused = weighted_fuse(dense, sparse, dense_weight=0.6, sparse_weight=0.4)
    """
    # Normalize scores from both lists
    dense_normalized = _normalize_scores(dense_results)
    sparse_normalized = _normalize_scores(sparse_results)

    # Collect all unique chunk IDs
    all_chunk_ids = set(dense_normalized.keys()) | set(sparse_normalized.keys())

    # Calculate weighted fusion score for each chunk
    fused_scores = {}
    for chunk_id in all_chunk_ids:
        score = 0.0

        if chunk_id in dense_normalized:
            score += dense_weight * dense_normalized[chunk_id]

        if chunk_id in sparse_normalized:
            score += sparse_weight * sparse_normalized[chunk_id]

        fused_scores[chunk_id] = score

    # Sort by fused score descending and return top_k
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def reciprocal_fuse(
    *result_lists: List[Tuple[str, float]],
    k: float = 60.0,
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Generalized RRF for fusing multiple retrieval result lists (2 or more).

    Args:
        *result_lists: Variable number of result lists, each sorted by score desc.
                       Each list: [(chunk_id, score), ...]
        k: RRF constant (typically 60)
        top_k: Number of top results to return

    Returns:
        List of (chunk_id, rrf_score) tuples sorted by RRF score descending.

    Example:
        >>> dense = [("doc1", 0.9), ("doc2", 0.8)]
        >>> sparse = [("doc2", 0.95), ("doc3", 0.7)]
        >>> semantic = [("doc1", 0.85), ("doc3", 0.75)]
        >>> fused = reciprocal_fuse(dense, sparse, semantic, k=60, top_k=5)
    """
    # Get rank mappings for all result lists
    all_ranks = [_get_rank_map(results) for results in result_lists]

    # Collect all unique chunk IDs
    all_chunk_ids = set()
    for rank_map in all_ranks:
        all_chunk_ids.update(rank_map.keys())

    # Calculate RRF score for each chunk across all lists
    rrf_scores = {}
    for chunk_id in all_chunk_ids:
        score = 0.0

        for rank_map in all_ranks:
            if chunk_id in rank_map:
                rank = rank_map[chunk_id]
                score += 1.0 / (k + rank)

        rrf_scores[chunk_id] = score

    # Sort by RRF score descending and return top_k
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
