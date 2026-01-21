"""
Cross-encoder reranking for precision improvement.

Cross-encoders score query-document pairs jointly, achieving +14-25% NDCG
improvement over bi-encoders. The key insight: use bi-encoders for fast
top-50 retrieval, then cross-encoders for precise top-k reranking.

Model selection for column names:
- Recommended: cross-encoder/ms-marco-MiniLM-L6-v2 (best speed/accuracy, 1800 docs/sec)
- Maximum accuracy: cross-encoder/ms-marco-MiniLM-L12-v2
- Multilingual schemas: BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import torch
from sentence_transformers import CrossEncoder


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranking."""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    batch_size: int = 32
    max_length: int = 256
    device: Optional[str] = None  # Auto-detect if None


@dataclass
class RerankResult:
    """Result from reranking with score details."""
    column: str
    cross_encoder_score: float
    original_score: float = 0.0
    original_rank: int = 0
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"RerankResult(column='{self.column}', "
            f"ce_score={self.cross_encoder_score:.4f}, "
            f"orig_rank={self.original_rank})"
        )


class SchemaReranker:
    """
    Cross-encoder reranker for schema/column selection.

    Provides precise relevance scoring by encoding query and column together,
    capturing fine-grained semantic relationships that bi-encoders miss.
    """

    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize the cross-encoder reranker.

        Args:
            config: Reranker configuration (uses defaults if None)
        """
        self.config = config or RerankerConfig()

        # Auto-detect device
        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        # Load cross-encoder model
        self.model = CrossEncoder(
            self.config.model_name,
            max_length=self.config.max_length,
            device=self.device
        )

    def format_column(
        self,
        column: str,
        description: Optional[str] = None,
        data_type: Optional[str] = None,
        table: Optional[str] = None,
    ) -> str:
        """
        Format column information for cross-encoder input.

        Structured format works best for cross-encoders.

        Args:
            column: Column name
            description: Optional column description
            data_type: Optional data type
            table: Optional table name

        Returns:
            Formatted string for cross-encoder
        """
        parts = []
        if table:
            parts.append(f"Table: {table}")
        parts.append(f"Column: {column}")
        if data_type:
            parts.append(f"Type: {data_type}")
        if description:
            parts.append(f"Description: {description}")
        return "\n".join(parts)

    def rerank(
        self,
        query: str,
        columns: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None,
        original_scores: Optional[Dict[str, float]] = None,
    ) -> List[RerankResult]:
        """
        Rerank columns using cross-encoder relevance scores.

        Args:
            query: The search query
            columns: List of column names to rerank
            descriptions: Optional dict mapping columns to descriptions
            top_k: Number of results to return (None = all)
            original_scores: Optional original scores for tracking

        Returns:
            List of RerankResult sorted by cross-encoder score descending
        """
        if not columns:
            return []

        descriptions = descriptions or {}
        original_scores = original_scores or {}

        # Create (query, column_text) pairs
        pairs = []
        for col in columns:
            col_text = self.format_column(col, descriptions.get(col))
            pairs.append((query, col_text))

        # Batch prediction with cross-encoder
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=len(pairs) > 100
        )

        # Build results with original rank tracking
        results = []
        for i, (col, score) in enumerate(zip(columns, scores)):
            results.append(RerankResult(
                column=col,
                cross_encoder_score=float(score),
                original_score=original_scores.get(col, 0.0),
                original_rank=i + 1,
                description=descriptions.get(col, ""),
            ))

        # Sort by cross-encoder score
        results.sort(key=lambda x: x.cross_encoder_score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    def rerank_with_threshold(
        self,
        query: str,
        columns: List[str],
        descriptions: Optional[Dict[str, str]] = None,
        threshold: float = 0.0,
        min_results: int = 1,
        max_results: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank and filter by score threshold.

        Useful for dynamic k selection based on relevance confidence.

        Args:
            query: The search query
            columns: List of column names to rerank
            descriptions: Optional column descriptions
            threshold: Minimum cross-encoder score to include
            min_results: Minimum number of results to return
            max_results: Maximum number of results to return

        Returns:
            Filtered list of RerankResult
        """
        results = self.rerank(query, columns, descriptions)

        # Filter by threshold
        filtered = [r for r in results if r.cross_encoder_score >= threshold]

        # Ensure min_results
        if len(filtered) < min_results:
            filtered = results[:min_results]

        # Cap at max_results
        return filtered[:max_results]


class TwoStageRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Fast bi-encoder retrieval (top-N candidates)
    Stage 2: Precise cross-encoder reranking (top-K final)

    This achieves both speed (bi-encoder) and accuracy (cross-encoder).
    """

    def __init__(
        self,
        bi_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        stage1_k: int = 50,
        stage2_k: int = 10,
    ):
        """
        Initialize two-stage retriever.

        Args:
            bi_encoder_model: Model for stage 1 retrieval
            cross_encoder_model: Model for stage 2 reranking
            stage1_k: Number of candidates from stage 1
            stage2_k: Number of final results from stage 2
        """
        from sentence_transformers import SentenceTransformer

        self.stage1_k = stage1_k
        self.stage2_k = stage2_k

        # Stage 1: Bi-encoder for fast retrieval
        self.bi_encoder = SentenceTransformer(bi_encoder_model)

        # Stage 2: Cross-encoder for precise reranking
        self.reranker = SchemaReranker(RerankerConfig(model_name=cross_encoder_model))

        # Storage
        self.columns: List[str] = []
        self.descriptions: Dict[str, str] = {}
        self.embeddings = None

    def index(
        self,
        columns: List[str],
        descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Index columns for retrieval.

        Args:
            columns: List of column names
            descriptions: Optional column descriptions
        """
        self.columns = columns
        self.descriptions = descriptions or {}

        # Create text representations
        texts = []
        for col in columns:
            if col in self.descriptions:
                texts.append(f"{col}: {self.descriptions[col]}")
            else:
                texts.append(col)

        # Compute embeddings
        self.embeddings = self.bi_encoder.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )

    def retrieve(
        self,
        query: str,
        stage1_k: Optional[int] = None,
        stage2_k: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Two-stage retrieval: bi-encoder → cross-encoder.

        Args:
            query: Search query
            stage1_k: Override for stage 1 candidates
            stage2_k: Override for stage 2 final results

        Returns:
            List of RerankResult sorted by cross-encoder score
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call index() first.")

        stage1_k = stage1_k or self.stage1_k
        stage2_k = stage2_k or self.stage2_k

        # Stage 1: Fast bi-encoder retrieval
        from sentence_transformers import util

        query_emb = self.bi_encoder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        cos_scores = util.cos_sim(query_emb, self.embeddings)[0]
        top_indices = cos_scores.argsort(descending=True)[:stage1_k]

        # Get stage 1 candidates with scores
        candidates = [self.columns[i] for i in top_indices.cpu().numpy()]
        stage1_scores = {
            self.columns[i]: float(cos_scores[i])
            for i in top_indices.cpu().numpy()
        }

        # Stage 2: Precise cross-encoder reranking
        results = self.reranker.rerank(
            query=query,
            columns=candidates,
            descriptions=self.descriptions,
            top_k=stage2_k,
            original_scores=stage1_scores,
        )

        return results


def rerank_columns(
    query: str,
    columns: List[str],
    descriptions: Optional[Dict[str, str]] = None,
    top_k: int = 10,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
) -> List[Tuple[str, float]]:
    """
    Convenience function for one-off reranking.

    Args:
        query: Search query
        columns: Columns to rerank
        descriptions: Optional column descriptions
        top_k: Number of results
        model_name: Cross-encoder model

    Returns:
        List of (column, score) tuples
    """
    reranker = SchemaReranker(RerankerConfig(model_name=model_name))
    results = reranker.rerank(query, columns, descriptions, top_k=top_k)
    return [(r.column, r.cross_encoder_score) for r in results]


def analyze_rerank_changes(
    original: List[str],
    reranked: List[RerankResult],
) -> Dict[str, Any]:
    """
    Analyze how reranking changed the order.

    Useful for debugging and understanding cross-encoder behavior.

    Args:
        original: Original column order
        reranked: Reranked results

    Returns:
        Analysis dict with rank changes and statistics
    """
    original_ranks = {col: i + 1 for i, col in enumerate(original)}

    promoted = []  # Moved up significantly
    demoted = []   # Moved down significantly
    stable = []    # Stayed roughly same

    rank_changes = []

    for new_rank, result in enumerate(reranked, 1):
        old_rank = original_ranks.get(result.column, len(original) + 1)
        change = old_rank - new_rank  # Positive = promoted
        rank_changes.append(change)

        if change >= 3:
            promoted.append((result.column, old_rank, new_rank))
        elif change <= -3:
            demoted.append((result.column, old_rank, new_rank))
        else:
            stable.append((result.column, old_rank, new_rank))

    return {
        "promoted": promoted,
        "demoted": demoted,
        "stable": stable,
        "avg_rank_change": sum(rank_changes) / len(rank_changes) if rank_changes else 0,
        "max_promotion": max(rank_changes) if rank_changes else 0,
        "max_demotion": min(rank_changes) if rank_changes else 0,
    }
