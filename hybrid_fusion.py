"""
Reciprocal Rank Fusion (RRF) and score fusion utilities.

RRF operates at the RANKING level, not the score level. This avoids the need
for score normalization between incompatible scoring systems (e.g., cosine
similarity 0-1 vs exponential decay causal scores).

Key insight: RRF uses only rank positions, not raw scores:
    RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

The k=60 parameter is empirically validated across Elasticsearch, OpenSearch,
and Azure AI Search. Lower k values (5-20) emphasize top ranks more aggressively.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class FusedResult:
    """Result with scores from each retrieval source."""
    doc_id: str
    rrf_score: float
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    content: str = ""
    rank_contributions: Dict[str, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"FusedResult(doc_id='{self.doc_id}', rrf={self.rrf_score:.4f}, "
            f"semantic={self.semantic_score:.3f}, bm25={self.bm25_score:.3f}, "
            f"graph={self.graph_score:.3f})"
        )


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    RRF Formula: RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

    Args:
        ranked_lists: List of [(doc_id, score), ...] for each retrieval method.
                      Each inner list should be sorted by score descending.
        k: Smoothing constant (60 is standard, validated empirically).
           Lower k (5-20) emphasizes top ranks more aggressively.
        weights: Optional per-method weights. Defaults to equal weights (1.0).

    Returns:
        List of (doc_id, rrf_score) sorted by RRF score descending.

    Example:
        >>> semantic = [("col_a", 0.9), ("col_b", 0.8), ("col_c", 0.7)]
        >>> bm25 = [("col_b", 12.5), ("col_a", 10.2), ("col_d", 8.0)]
        >>> fused = reciprocal_rank_fusion([semantic, bm25], k=60)
    """
    if not ranked_lists:
        return []

    weights = weights or [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"Weights length ({len(weights)}) must match ranked_lists ({len(ranked_lists)})")

    rrf_scores: Dict[str, float] = defaultdict(float)

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, (doc_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += weight / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def three_signal_rrf(
    semantic_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    graph_results: List[Tuple[str, float]],
    semantic_weight: float = 1.0,
    bm25_weight: float = 0.8,
    graph_weight: float = 1.5,
    k: int = 60
) -> List[FusedResult]:
    """
    Fuse semantic, BM25, and causal graph signals at ranking level.

    Graph weight is typically higher for precision-focused retrieval
    as it enforces causal structure (avoiding leakage columns).

    Args:
        semantic_results: [(column, similarity_score), ...] sorted descending
        bm25_results: [(column, bm25_score), ...] sorted descending
        graph_results: [(column, causal_score), ...] sorted descending
        semantic_weight: Weight for semantic signal (default: 1.0)
        bm25_weight: Weight for BM25 signal (default: 0.8)
        graph_weight: Weight for graph/causal signal (default: 1.5, higher for precision)
        k: RRF smoothing constant (default: 60)

    Returns:
        List of FusedResult objects sorted by RRF score descending.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    raw_scores: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {'semantic': 0.0, 'bm25': 0.0, 'graph': 0.0}
    )
    rank_contributions: Dict[str, Dict[str, int]] = defaultdict(dict)

    # Collect raw scores for analysis
    for doc_id, score in semantic_results:
        raw_scores[doc_id]['semantic'] = score
    for doc_id, score in bm25_results:
        raw_scores[doc_id]['bm25'] = score
    for doc_id, score in graph_results:
        raw_scores[doc_id]['graph'] = score

    # Calculate RRF contributions
    for rank, (doc_id, _) in enumerate(semantic_results, 1):
        rrf_scores[doc_id] += semantic_weight / (k + rank)
        rank_contributions[doc_id]['semantic'] = rank

    for rank, (doc_id, _) in enumerate(bm25_results, 1):
        rrf_scores[doc_id] += bm25_weight / (k + rank)
        rank_contributions[doc_id]['bm25'] = rank

    for rank, (doc_id, _) in enumerate(graph_results, 1):
        rrf_scores[doc_id] += graph_weight / (k + rank)
        rank_contributions[doc_id]['graph'] = rank

    # Sort and return with all scores for analysis
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        FusedResult(
            doc_id=doc_id,
            rrf_score=score,
            semantic_score=raw_scores[doc_id]['semantic'],
            bm25_score=raw_scores[doc_id]['bm25'],
            graph_score=raw_scores[doc_id]['graph'],
            rank_contributions=dict(rank_contributions[doc_id])
        )
        for doc_id, score in sorted_results
    ]


def two_signal_rrf(
    primary_results: List[Tuple[str, float]],
    secondary_results: List[Tuple[str, float]],
    primary_weight: float = 1.0,
    secondary_weight: float = 1.0,
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Simple two-signal RRF fusion.

    Args:
        primary_results: First ranked list [(doc_id, score), ...]
        secondary_results: Second ranked list [(doc_id, score), ...]
        primary_weight: Weight for primary signal
        secondary_weight: Weight for secondary signal
        k: RRF smoothing constant

    Returns:
        Fused ranked list [(doc_id, rrf_score), ...]
    """
    return reciprocal_rank_fusion(
        [primary_results, secondary_results],
        k=k,
        weights=[primary_weight, secondary_weight]
    )


def weighted_score_fusion(
    score_dicts: List[Dict[str, float]],
    weights: Optional[List[float]] = None,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Alternative: Linear combination of normalized scores.

    Unlike RRF, this operates on actual scores (requires normalization).
    Use when you want to preserve score magnitude information.

    Args:
        score_dicts: List of {doc_id: score} dicts from each method
        weights: Per-method weights (defaults to equal)
        normalize: Whether to min-max normalize scores (recommended)

    Returns:
        Dict of {doc_id: combined_score}
    """
    if not score_dicts:
        return {}

    weights = weights or [1.0] * len(score_dicts)
    if len(weights) != len(score_dicts):
        raise ValueError("Weights length must match score_dicts")

    # Optionally normalize each score dict to [0, 1]
    normalized_dicts = []
    for scores in score_dicts:
        if not scores:
            normalized_dicts.append({})
            continue

        if normalize:
            values = list(scores.values())
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val
            if range_val == 0:
                # All same value - normalize to 1.0
                normalized = {k: 1.0 for k in scores}
            else:
                normalized = {k: (v - min_val) / range_val for k, v in scores.items()}
            normalized_dicts.append(normalized)
        else:
            normalized_dicts.append(scores)

    # Combine with weights
    combined: Dict[str, float] = defaultdict(float)
    for weight, scores in zip(weights, normalized_dicts):
        for doc_id, score in scores.items():
            combined[doc_id] += weight * score

    # Normalize combined scores
    total_weight = sum(weights)
    return {k: v / total_weight for k, v in combined.items()}


def filter_by_score(
    ranked_list: List[Tuple[str, float]],
    threshold: float = 0.0,
    top_k: Optional[int] = None
) -> List[Tuple[str, float]]:
    """
    Filter ranked results by score threshold and/or top-k.

    Args:
        ranked_list: [(doc_id, score), ...] sorted descending
        threshold: Minimum score to include
        top_k: Maximum number of results (applied after threshold)

    Returns:
        Filtered list
    """
    filtered = [(doc_id, score) for doc_id, score in ranked_list if score >= threshold]
    if top_k is not None:
        filtered = filtered[:top_k]
    return filtered


def analyze_fusion_contributions(
    fused_results: List[FusedResult],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Analyze which signals contributed most to the top-k results.

    Useful for debugging and tuning fusion weights.

    Returns:
        Dict with contribution statistics
    """
    top_results = fused_results[:top_k]

    if not top_results:
        return {'top_k': 0, 'signals': {}}

    signal_counts = {'semantic': 0, 'bm25': 0, 'graph': 0}
    exclusive_counts = {'semantic': 0, 'bm25': 0, 'graph': 0}

    for result in top_results:
        has_semantic = result.semantic_score > 0
        has_bm25 = result.bm25_score > 0
        has_graph = result.graph_score > 0

        if has_semantic:
            signal_counts['semantic'] += 1
        if has_bm25:
            signal_counts['bm25'] += 1
        if has_graph:
            signal_counts['graph'] += 1

        # Count exclusive contributions (only one signal found this)
        if has_semantic and not has_bm25 and not has_graph:
            exclusive_counts['semantic'] += 1
        if has_bm25 and not has_semantic and not has_graph:
            exclusive_counts['bm25'] += 1
        if has_graph and not has_semantic and not has_bm25:
            exclusive_counts['graph'] += 1

    return {
        'top_k': len(top_results),
        'signal_counts': signal_counts,
        'exclusive_counts': exclusive_counts,
        'avg_rrf_score': sum(r.rrf_score for r in top_results) / len(top_results),
        'avg_semantic': sum(r.semantic_score for r in top_results) / len(top_results),
        'avg_bm25': sum(r.bm25_score for r in top_results) / len(top_results),
        'avg_graph': sum(r.graph_score for r in top_results) / len(top_results),
    }
