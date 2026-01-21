"""
Threshold-based hybrid retrieval.

Core idea: keep semantic columns only when similarity is high enough,
then fill the remaining slots with causally related columns.

This module contains both the original ThresholdHybridRetriever and the
new RRFHybridRetriever which uses Reciprocal Rank Fusion for combining
semantic, BM25, and causal graph signals at the ranking level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Tuple, Set, Optional, Literal

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer

from markov_blanket import (
    MarkovBlanketScorer,
    get_graph_relevance_scores,
    CausalMode,
)
from hybrid_fusion import (
    three_signal_rrf,
    FusedResult,
    analyze_fusion_contributions,
)
from bm25_retriever import (
    BM25ColumnRetriever,
    SimpleBM25,
    HAS_BM25,
)
from reranker import SchemaReranker, RerankerConfig, RerankResult
from prompt_optimizer import reorder_for_attention


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of a and vector b."""
    b_norm = np.linalg.norm(b)
    a_norm = np.linalg.norm(a, axis=1)
    # Avoid division by zero
    denom = (a_norm * b_norm)
    denom[denom == 0] = 1.0
    return np.dot(a, b) / denom


@dataclass
class SemanticResult:
    scores: Dict[str, float]
    mentioned: Set[str]


class SemanticScorer:
    """Semantic similarity scorer with optional abbreviation expansions."""

    EXPANSIONS = {
        "alt": "ALT alanine transaminase liver enzyme",
        "ast": "AST aspartate transaminase liver enzyme",
        "ggtp": "GGTP gamma glutamyl transpeptidase",
        "inr": "INR international normalized ratio blood clotting",
        "esr": "ESR erythrocyte sedimentation rate inflammation",
        "pbc": "PBC primary biliary cholangitis liver disease",
        "hbsag": "HBsAg hepatitis B surface antigen",
        "hcv_anti": "anti-HCV hepatitis C antibody",
        "hbc_anti": "anti-HBc hepatitis B core antibody",
        "hbeag": "HBeAg hepatitis B e antigen",
        "hbsag_anti": "anti-HBs hepatitis B surface antibody",
        "ama": "AMA antimitochondrial antibodies autoimmune",
        "le_cells": "LE cells lupus erythematosus",
        "vh_amn": "viral hepatitis history",
        "pain_ruq": "pain right upper quadrant",
        "pressure_ruq": "pressure right upper quadrant",
    }

    def __init__(self, columns: List[str], model_name: str = "all-MiniLM-L6-v2"):
        self.columns = columns
        self.model = SentenceTransformer(model_name)

        # Raw and expanded embeddings
        self.raw_embeddings = self.model.encode(columns)
        expanded = [self._expand(col) for col in columns]
        self.expanded_embeddings = self.model.encode(expanded)

    def _expand(self, name: str) -> str:
        key = name.lower()
        if key in self.EXPANSIONS:
            return self.EXPANSIONS[key]
        return name.replace("_", " ")

    def _find_mentioned(self, query: str) -> Set[str]:
        q = query.lower()
        mentioned = set()
        for col in self.columns:
            c = col.lower()
            if c in q:
                # word boundary check
                import re
                if re.search(r"\b" + re.escape(c) + r"\b", q):
                    mentioned.add(col)
        return mentioned

    def score(self, query: str) -> SemanticResult:
        q_emb = self.model.encode(query)
        raw_sims = _cosine_sim_matrix(self.raw_embeddings, q_emb)
        exp_sims = _cosine_sim_matrix(self.expanded_embeddings, q_emb)
        sims = np.maximum(raw_sims, exp_sims)
        scores = {col: float(sim) for col, sim in zip(self.columns, sims)}
        mentioned = self._find_mentioned(query)
        return SemanticResult(scores=scores, mentioned=mentioned)


class SemanticOnlyRetriever:
    """Pure semantic baseline (top-k by similarity, with mention guarantee)."""

    def __init__(self, columns: List[str], model_name: str = "all-MiniLM-L6-v2"):
        self.columns = columns
        self.scorer = SemanticScorer(columns, model_name=model_name)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        res = self.scorer.score(query)
        scores = res.scores
        mentioned = res.mentioned

        sorted_cols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        mentioned_sorted = [col for col, _ in sorted_cols if col in mentioned]
        others_sorted = [col for col, _ in sorted_cols if col not in mentioned]

        result = []
        result.extend(mentioned_sorted)
        for col in others_sorted:
            if len(result) >= k:
                break
            result.append(col)

        return result[:k]


class CausalScorer:
    """Simple causal scorer using k-hop neighborhood in an undirected graph."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    def score(self, targets: Iterable[str], max_depth: int = 2) -> Dict[str, float]:
        scores: Dict[str, float] = {node: 0.0 for node in self.graph.nodes}
        for t in targets:
            if t not in self.graph:
                continue
            # BFS up to max_depth
            queue = [(t, 0)]
            visited = {t}
            while queue:
                node, depth = queue.pop(0)
                if depth > 0:
                    # Exponential decay by distance
                    score = 1.0 / (2 ** (depth - 1))
                    if score > scores.get(node, 0.0):
                        scores[node] = score
                if depth >= max_depth:
                    continue
                for nei in self.graph.neighbors(node):
                    if nei not in visited:
                        visited.add(nei)
                        queue.append((nei, depth + 1))
        return scores


@dataclass
class ThresholdHybridConfig:
    k: int = 8
    semantic_threshold: float = 0.5
    max_depth: int = 2
    n_targets: int = 1
    max_semantic: Optional[int] = None


class ThresholdHybridRetriever:
    """Threshold hybrid: keep semantic above threshold, then fill with causal."""

    def __init__(self, columns: List[str], graph: nx.Graph, model_name: str = "all-MiniLM-L6-v2"):
        self.columns = columns
        self.semantic = SemanticScorer(columns, model_name=model_name)
        self.causal = CausalScorer(graph)

    def retrieve(self, query: str, config: ThresholdHybridConfig) -> List[str]:
        sem_res = self.semantic.score(query)
        sem_scores = sem_res.scores
        mentioned = sem_res.mentioned

        # Sort semantic scores descending
        sem_sorted = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)

        # Select semantic columns above threshold OR explicitly mentioned
        semantic_selected = []
        for col, score in sem_sorted:
            if score >= config.semantic_threshold or col in mentioned:
                semantic_selected.append(col)

        # Optional cap for semantic selections
        if config.max_semantic is not None:
            semantic_selected = semantic_selected[: config.max_semantic]

        # If semantic already fills k, return top-k semantic
        if len(semantic_selected) >= config.k:
            return semantic_selected[: config.k]

        # Choose causal targets: mentioned columns if any, else top-n semantic
        if mentioned:
            targets = list(mentioned)
        else:
            targets = [col for col, _ in sem_sorted[: config.n_targets]]

        causal_scores = self.causal.score(targets, max_depth=config.max_depth)
        causal_sorted = sorted(causal_scores.items(), key=lambda x: x[1], reverse=True)

        # Fill remaining slots with causal columns not already selected
        result = list(dict.fromkeys(semantic_selected))
        for col, score in causal_sorted:
            if len(result) >= config.k:
                break
            if col in result:
                continue
            # Skip zero-score nodes to avoid noise
            if score <= 0.0:
                continue
            result.append(col)

        # If still not enough, pad with remaining semantic by score
        if len(result) < config.k:
            for col, _ in sem_sorted:
                if len(result) >= config.k:
                    break
                if col not in result:
                    result.append(col)

        return result[: config.k]


# =============================================================================
# NEW: RRF-based Hybrid Retrieval with Markov Blanket
# =============================================================================


@dataclass
class RRFHybridConfig:
    """Configuration for RRF-based hybrid retrieval (precision-optimized defaults)."""
    k: int = 10
    semantic_weight: float = 1.0
    bm25_weight: float = 0.8
    graph_weight: float = 1.5  # Higher weight prioritizes causal structure
    rrf_k: int = 60  # RRF smoothing constant
    n_targets: int = 1  # Number of top semantic columns for causal seeds
    causal_mode: CausalMode = 'parents_only'  # 'parents_only' | 'ancestors' | 'full_mb'
    max_causal_depth: int = 2  # For 'ancestors' mode
    exclude_effects: bool = True  # Filter out known effects (leakage prevention)
    use_bm25: bool = True  # Whether to use BM25 signal


@dataclass
class RRFRetrievalResult:
    """Detailed result from RRF hybrid retrieval."""
    columns: List[str]
    fused_results: List[FusedResult]
    target_columns: List[str]
    analysis: Dict = field(default_factory=dict)

    @property
    def top_k(self) -> List[str]:
        return self.columns


class RRFHybridRetriever:
    """
    RRF-based hybrid retrieval combining semantic, BM25, and causal graph signals.

    Key differences from ThresholdHybridRetriever:
    1. Uses RRF for rank-level fusion (no score normalization needed)
    2. Uses Markov Blanket instead of undirected BFS (preserves causal direction)
    3. Supports BM25 for lexical matching
    4. Higher graph weight for precision-focused retrieval
    """

    def __init__(
        self,
        columns: List[str],
        graph: nx.DiGraph,
        model_name: str = "all-MiniLM-L6-v2",
        column_descriptions: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize RRF hybrid retriever.

        Args:
            columns: List of column names
            graph: Directed causal graph (edges go from cause to effect)
            model_name: Sentence transformer model for semantic similarity
            column_descriptions: Optional dict of column descriptions for BM25
        """
        self.columns = columns
        self.graph = graph
        self.column_descriptions = column_descriptions or {}

        # Semantic scorer (reuse existing)
        self.semantic = SemanticScorer(columns, model_name=model_name)

        # Markov Blanket scorer (new - preserves causal direction)
        self.mb_scorer = MarkovBlanketScorer(graph)

        # BM25 retriever (new - lexical matching)
        self.bm25: Optional[BM25ColumnRetriever] = None
        if HAS_BM25:
            self.bm25 = BM25ColumnRetriever()
            self.bm25.index_columns(columns, column_descriptions)
        else:
            # Fallback to simple BM25
            self.bm25 = SimpleBM25()
            self.bm25.index_columns(columns)

    def retrieve(
        self,
        query: str,
        config: Optional[RRFHybridConfig] = None,
    ) -> List[str]:
        """
        Retrieve columns using RRF fusion of semantic, BM25, and causal signals.

        Args:
            query: Natural language query
            config: RRF hybrid configuration (uses defaults if not provided)

        Returns:
            List of column names, top-k by RRF score
        """
        result = self.retrieve_detailed(query, config)
        return result.columns

    def retrieve_detailed(
        self,
        query: str,
        config: Optional[RRFHybridConfig] = None,
    ) -> RRFRetrievalResult:
        """
        Retrieve with detailed results including all scores and analysis.

        Args:
            query: Natural language query
            config: RRF hybrid configuration

        Returns:
            RRFRetrievalResult with full details
        """
        config = config or RRFHybridConfig()

        # Step 1: Get semantic scores and mentioned columns
        sem_res = self.semantic.score(query)
        sem_scores = sem_res.scores
        mentioned = sem_res.mentioned

        # Sort semantic scores descending
        semantic_ranked = sorted(sem_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 2: Get BM25 scores
        if config.use_bm25 and self.bm25 is not None:
            bm25_scores = self.bm25.get_all_scores(query)
            bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            bm25_ranked = [(col, 0.0) for col in self.columns]

        # Step 3: Determine causal targets
        # Priority: explicitly mentioned columns, else top semantic columns
        if mentioned:
            targets = list(mentioned)
        else:
            targets = [col for col, _ in semantic_ranked[:config.n_targets]]

        # Step 4: Get causal/graph scores using Markov Blanket
        graph_scores: Dict[str, float] = {}
        for target in targets:
            if target not in self.graph:
                continue

            target_scores = self.mb_scorer.score(
                targets=[target],
                mode=config.causal_mode,
                max_depth=config.max_causal_depth,
            )

            # Optionally filter out effects (leakage prevention)
            if config.exclude_effects:
                leakage_cols = self.mb_scorer.get_leakage_columns(target)
                target_scores = {
                    k: v for k, v in target_scores.items()
                    if k not in leakage_cols
                }

            # Merge scores, keeping maximum
            for col, score in target_scores.items():
                if score > graph_scores.get(col, 0.0):
                    graph_scores[col] = score

        graph_ranked = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 5: RRF fusion
        fused = three_signal_rrf(
            semantic_results=semantic_ranked,
            bm25_results=bm25_ranked,
            graph_results=graph_ranked,
            semantic_weight=config.semantic_weight,
            bm25_weight=config.bm25_weight if config.use_bm25 else 0.0,
            graph_weight=config.graph_weight,
            k=config.rrf_k,
        )

        # Step 6: Ensure mentioned columns are included (with priority)
        result_columns = []
        mentioned_in_results = [f for f in fused if f.doc_id in mentioned]
        others_in_results = [f for f in fused if f.doc_id not in mentioned]

        # Add mentioned first (preserving their RRF order)
        for f in mentioned_in_results:
            result_columns.append(f.doc_id)

        # Add others up to k
        for f in others_in_results:
            if len(result_columns) >= config.k:
                break
            result_columns.append(f.doc_id)

        # Analysis for debugging
        analysis = analyze_fusion_contributions(fused, top_k=config.k)

        return RRFRetrievalResult(
            columns=result_columns[:config.k],
            fused_results=fused[:config.k * 2],  # Keep more for analysis
            target_columns=targets,
            analysis=analysis,
        )


class EnhancedSemanticScorer(SemanticScorer):
    """
    Enhanced semantic scorer with optional BGE embedding model support.

    BGE (bge-base-en-v1.5) achieves 53.25 nDCG@10 vs 43.8 for all-MiniLM-L6-v2.
    """

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        columns: List[str],
        model_name: str = "BAAI/bge-base-en-v1.5",
        use_query_prefix: bool = True,
    ):
        """
        Initialize with configurable embedding model.

        Args:
            columns: List of column names
            model_name: Sentence transformer model name
            use_query_prefix: Whether to use BGE query prefix (for BGE models)
        """
        self.use_query_prefix = use_query_prefix and "bge" in model_name.lower()
        super().__init__(columns, model_name=model_name)

    def score(self, query: str) -> SemanticResult:
        """Score with optional query prefix for BGE models."""
        if self.use_query_prefix:
            query = self.BGE_QUERY_PREFIX + query

        return super().score(query)


# =============================================================================
# Phase 3-4: RRF with Cross-Encoder Reranking and Attention Optimization
# =============================================================================


@dataclass
class RRFRerankedConfig:
    """Configuration for RRF hybrid retrieval with cross-encoder reranking."""
    # Stage 1: RRF retrieval
    stage1_k: int = 30  # Candidates from RRF fusion
    semantic_weight: float = 1.0
    bm25_weight: float = 0.8
    graph_weight: float = 1.5
    rrf_k: int = 60
    n_targets: int = 1
    causal_mode: CausalMode = 'parents_only'
    max_causal_depth: int = 2
    exclude_effects: bool = True
    use_bm25: bool = True

    # Stage 2: Cross-encoder reranking
    final_k: int = 10  # Final results after reranking
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    use_reranking: bool = True

    # Stage 3: Attention optimization
    use_attention_ordering: bool = True


@dataclass
class RRFRerankedResult:
    """Detailed result from RRF + reranked retrieval."""
    columns: List[str]
    reranked_results: List[RerankResult]
    stage1_columns: List[str]
    target_columns: List[str]
    analysis: Dict = field(default_factory=dict)
    attention_ordered: bool = False

    @property
    def top_k(self) -> List[str]:
        return self.columns


class RRFRerankedRetriever:
    """
    Three-stage retrieval pipeline:
    Stage 1: RRF hybrid retrieval (semantic + BM25 + Markov Blanket) → top-30
    Stage 2: Cross-encoder reranking → top-10
    Stage 3: Attention-optimized ordering for LLM

    This achieves both high recall (RRF) and high precision (cross-encoder),
    while optimizing for LLM attention patterns.
    """

    def __init__(
        self,
        columns: List[str],
        graph: nx.DiGraph,
        model_name: str = "all-MiniLM-L6-v2",
        column_descriptions: Optional[Dict[str, str]] = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        """
        Initialize three-stage retriever.

        Args:
            columns: List of column names
            graph: Directed causal graph
            model_name: Bi-encoder model for semantic similarity
            column_descriptions: Optional column descriptions
            reranker_model: Cross-encoder model for reranking
        """
        self.columns = columns
        self.graph = graph
        self.column_descriptions = column_descriptions or {}

        # Stage 1: RRF hybrid retriever
        self.rrf_retriever = RRFHybridRetriever(
            columns=columns,
            graph=graph,
            model_name=model_name,
            column_descriptions=column_descriptions,
        )

        # Stage 2: Cross-encoder reranker
        self.reranker = SchemaReranker(RerankerConfig(model_name=reranker_model))

    def retrieve(
        self,
        query: str,
        config: Optional[RRFRerankedConfig] = None,
    ) -> List[str]:
        """
        Three-stage retrieval: RRF → Cross-encoder → Attention ordering.

        Args:
            query: Natural language query
            config: Configuration (uses defaults if None)

        Returns:
            List of column names optimized for LLM attention
        """
        result = self.retrieve_detailed(query, config)
        return result.columns

    def retrieve_detailed(
        self,
        query: str,
        config: Optional[RRFRerankedConfig] = None,
    ) -> RRFRerankedResult:
        """
        Retrieve with detailed results from all stages.

        Args:
            query: Natural language query
            config: Configuration

        Returns:
            RRFRerankedResult with full details
        """
        config = config or RRFRerankedConfig()

        # Stage 1: RRF hybrid retrieval (high recall)
        rrf_config = RRFHybridConfig(
            k=config.stage1_k,
            semantic_weight=config.semantic_weight,
            bm25_weight=config.bm25_weight,
            graph_weight=config.graph_weight,
            rrf_k=config.rrf_k,
            n_targets=config.n_targets,
            causal_mode=config.causal_mode,
            max_causal_depth=config.max_causal_depth,
            exclude_effects=config.exclude_effects,
            use_bm25=config.use_bm25,
        )

        stage1_result = self.rrf_retriever.retrieve_detailed(query, rrf_config)
        stage1_columns = stage1_result.columns

        # Stage 2: Cross-encoder reranking (high precision)
        if config.use_reranking and len(stage1_columns) > config.final_k:
            # Get original scores for tracking
            original_scores = {
                r.doc_id: r.rrf_score
                for r in stage1_result.fused_results
                if r.doc_id in stage1_columns
            }

            reranked = self.reranker.rerank(
                query=query,
                columns=stage1_columns,
                descriptions=self.column_descriptions,
                top_k=config.final_k,
                original_scores=original_scores,
            )
            final_columns = [r.column for r in reranked]
            final_scores = [r.cross_encoder_score for r in reranked]
        else:
            # Skip reranking if not enough candidates
            reranked = [
                RerankResult(
                    column=col,
                    cross_encoder_score=0.0,
                    original_rank=i + 1
                )
                for i, col in enumerate(stage1_columns[:config.final_k])
            ]
            final_columns = stage1_columns[:config.final_k]
            final_scores = [1.0 - i * 0.1 for i in range(len(final_columns))]

        # Stage 3: Attention-optimized ordering
        if config.use_attention_ordering and len(final_columns) > 2:
            ordered_columns = reorder_for_attention(final_columns, final_scores)
            attention_ordered = True
        else:
            ordered_columns = final_columns
            attention_ordered = False

        return RRFRerankedResult(
            columns=ordered_columns,
            reranked_results=reranked,
            stage1_columns=stage1_columns,
            target_columns=stage1_result.target_columns,
            analysis=stage1_result.analysis,
            attention_ordered=attention_ordered,
        )
