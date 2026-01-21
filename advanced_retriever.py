"""
Advanced Retriever combining HyDE, ColBERT, and CRAG techniques.

This module creates a unified retrieval pipeline that:
1. Uses HyDE for query expansion (hypothetical document generation)
2. Uses ColBERT/bi-encoder for late-interaction retrieval
3. Uses CRAG for self-correcting retrieval with confidence evaluation
4. Optionally integrates Markov blanket causal scoring

The goal is to create a competitive approach to CausalRAG.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np

# HyDE imports
from hyde_retriever import HyDERetriever, HyDEConfig, HyDEScorer, HyDEResult

# ColBERT imports
from colbert_retriever import ColBERTRetriever, ColBERTHybridRetriever, ColBERTConfig

# CRAG imports
from crag_evaluator import (
    CRAGRetriever, CRAGConfig, CRAGResult,
    RetrievalConfidenceEvaluator, QueryRefiner,
    create_crag_retriever
)

# Markov blanket imports
from markov_blanket import MarkovBlanketScorer

# BM25 imports
from bm25_retriever import BM25ColumnRetriever, create_bm25_retriever

# Reranker imports
try:
    from reranker import CrossEncoderReranker, RerankerConfig
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class AdvancedRetrieverConfig:
    """Configuration for the Advanced Retriever pipeline."""

    # HyDE settings
    use_hyde: bool = True
    hyde_n_hypotheses: int = 3
    hyde_aggregation: str = "mean"
    hyde_weight: float = 0.6

    # ColBERT/retrieval settings
    use_colbert: bool = True
    colbert_weight: float = 0.4

    # BM25 settings
    use_bm25: bool = True
    bm25_weight: float = 0.3

    # CRAG settings
    use_crag: bool = True
    crag_high_threshold: float = 0.7
    crag_low_threshold: float = 0.4
    crag_max_retries: int = 2

    # Markov blanket settings
    use_markov_blanket: bool = False
    markov_weight: float = 0.2
    markov_max_depth: int = 2

    # Reranking settings
    use_reranker: bool = False
    reranker_top_n: int = 20

    # Fusion settings
    rrf_k: int = 60

    # Output settings
    k: int = 10


@dataclass
class AdvancedRetrievalResult:
    """Result from the Advanced Retriever."""
    columns: List[str]
    scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Component results
    hyde_result: Optional[HyDEResult] = None
    crag_result: Optional[CRAGResult] = None

    # Debug info
    component_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fusion_weights: Dict[str, float] = field(default_factory=dict)


class AdvancedRetriever:
    """
    Advanced retriever combining multiple techniques:
    - HyDE for query expansion
    - ColBERT/bi-encoder for semantic retrieval
    - BM25 for lexical retrieval
    - CRAG for self-correction
    - Markov blanket for causal scoring
    - Cross-encoder reranking
    """

    def __init__(
        self,
        columns: List[str],
        descriptions: Dict[str, str],
        graph: Optional[Any] = None,
        llm: Optional[Any] = None,
        config: Optional[AdvancedRetrieverConfig] = None,
    ):
        """
        Initialize the Advanced Retriever.

        Args:
            columns: List of column names
            descriptions: Column descriptions
            graph: NetworkX graph for causal scoring (optional)
            llm: LLM for HyDE and CRAG (optional)
            config: Configuration options
        """
        self.columns = columns
        self.descriptions = descriptions
        self.graph = graph
        self.llm = llm
        self.config = config or AdvancedRetrieverConfig()

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all retrieval components."""

        # HyDE
        if self.config.use_hyde and self.llm:
            logger.info("Initializing HyDE retriever...")
            hyde_config = HyDEConfig(
                n_hypotheses=self.config.hyde_n_hypotheses,
                aggregation=self.config.hyde_aggregation,
                hypothesis_weight=self.config.hyde_weight,
            )
            self.hyde_retriever = HyDERetriever(
                columns=self.columns,
                column_descriptions=self.descriptions,
                llm=self.llm,
                config=hyde_config,
            )
        else:
            self.hyde_retriever = None

        # ColBERT
        if self.config.use_colbert:
            logger.info("Initializing ColBERT retriever...")
            try:
                self.colbert_retriever = ColBERTRetriever(
                    columns=self.columns,
                    column_descriptions=self.descriptions,
                )
            except Exception as e:
                logger.warning(f"ColBERT init failed: {e}, will use fallback")
                self.colbert_retriever = None
        else:
            self.colbert_retriever = None

        # BM25
        if self.config.use_bm25:
            logger.info("Initializing BM25 retriever...")
            try:
                self.bm25_retriever = BM25ColumnRetriever()
                self.bm25_retriever.index_columns(self.columns, self.descriptions)
            except ImportError as e:
                logger.warning(f"BM25 not available: {e}")
                self.bm25_retriever = None
        else:
            self.bm25_retriever = None

        # Markov blanket
        if self.config.use_markov_blanket and self.graph:
            logger.info("Initializing Markov blanket scorer...")
            self.mb_scorer = MarkovBlanketScorer(self.graph)
        else:
            self.mb_scorer = None

        # Reranker
        if self.config.use_reranker and RERANKER_AVAILABLE:
            logger.info("Initializing cross-encoder reranker...")
            self.reranker = CrossEncoderReranker()
        else:
            self.reranker = None

        logger.info("Advanced retriever initialized")

    def _rrf_fusion(
        self,
        rankings: Dict[str, List[str]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion for combining multiple rankings.

        Args:
            rankings: Dict of {method_name: [ranked_columns]}
            weights: Optional weights for each method

        Returns:
            Dict of column -> RRF score
        """
        k = self.config.rrf_k
        weights = weights or {m: 1.0 for m in rankings}

        scores = {}
        for method, ranked in rankings.items():
            weight = weights.get(method, 1.0)
            for rank, col in enumerate(ranked):
                rrf_score = weight / (k + rank + 1)
                scores[col] = scores.get(col, 0) + rrf_score

        return scores

    def _weighted_fusion(
        self,
        score_dicts: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Weighted fusion of multiple score dictionaries.

        Args:
            score_dicts: Dict of {method_name: {column: score}}
            weights: Weights for each method

        Returns:
            Dict of column -> weighted score
        """
        # Normalize each score dict to [0, 1]
        normalized = {}
        for method, scores in score_dicts.items():
            if not scores:
                continue
            max_score = max(scores.values()) if scores.values() else 1
            min_score = min(scores.values()) if scores.values() else 0
            range_score = max_score - min_score if max_score > min_score else 1
            normalized[method] = {
                col: (s - min_score) / range_score
                for col, s in scores.items()
            }

        # Combine with weights
        combined = {}
        total_weight = sum(weights.get(m, 0) for m in normalized)

        for method, scores in normalized.items():
            weight = weights.get(method, 1.0) / total_weight if total_weight > 0 else 0
            for col, score in scores.items():
                combined[col] = combined.get(col, 0) + weight * score

        return combined

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[str]:
        """
        Retrieve top-k columns for the query.

        Args:
            query: Natural language query
            k: Number of results (defaults to config.k)

        Returns:
            List of column names
        """
        result = self.retrieve_detailed(query, k)
        return result.columns

    def retrieve_detailed(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> AdvancedRetrievalResult:
        """
        Retrieve columns with detailed scoring information.

        Args:
            query: Natural language query
            k: Number of results

        Returns:
            AdvancedRetrievalResult with columns, scores, and metadata
        """
        k = k or self.config.k
        component_scores = {}
        rankings = {}
        hyde_result = None

        # Step 1: HyDE query expansion
        expanded_query = query
        if self.hyde_retriever:
            logger.info("Running HyDE query expansion...")
            cols, scores, hyde_result = self.hyde_retriever.retrieve_detailed(query, k=k*2)
            component_scores["hyde"] = scores
            rankings["hyde"] = cols

            # Use hypotheses to enhance query
            if hyde_result and hyde_result.hypotheses:
                expanded_query = f"{query}. {' '.join(hyde_result.hypotheses[:2])}"

        # Step 2: ColBERT/bi-encoder retrieval
        if self.colbert_retriever:
            logger.info("Running ColBERT retrieval...")
            cols, scores, colbert_result = self.colbert_retriever.retrieve_detailed(expanded_query, k=k*2)
            component_scores["colbert"] = scores
            rankings["colbert"] = cols

        # Step 3: BM25 retrieval
        if self.bm25_retriever:
            logger.info("Running BM25 retrieval...")
            bm25_scores = self.bm25_retriever.get_all_scores(expanded_query)
            component_scores["bm25"] = bm25_scores
            sorted_bm25 = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)
            rankings["bm25"] = sorted_bm25

        # Step 4: Markov blanket scoring (if graph available)
        if self.mb_scorer:
            logger.info("Computing Markov blanket scores...")
            # Find target columns from query
            targets = self._extract_targets(query, rankings)
            if targets:
                mb_scores = self.mb_scorer.score(
                    targets=targets,
                    mode='ancestors',
                    max_depth=self.config.markov_max_depth
                )
                component_scores["markov"] = mb_scores
                sorted_mb = sorted(mb_scores.keys(), key=lambda x: mb_scores[x], reverse=True)
                rankings["markov"] = sorted_mb

        # Step 5: Fuse scores
        logger.info("Fusing scores...")
        weights = {
            "hyde": self.config.hyde_weight if self.hyde_retriever else 0,
            "colbert": self.config.colbert_weight if self.colbert_retriever else 0,
            "bm25": self.config.bm25_weight if self.bm25_retriever else 0,
            "markov": self.config.markov_weight if self.mb_scorer else 0,
        }

        # Use RRF fusion
        rrf_scores = self._rrf_fusion(rankings, weights)

        # Sort by fused score
        sorted_cols = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Step 6: Reranking (optional)
        if self.reranker and len(sorted_cols) > 0:
            logger.info("Reranking with cross-encoder...")
            top_candidates = sorted_cols[:self.config.reranker_top_n]
            reranked = self.reranker.rerank(
                query, top_candidates, self.descriptions
            )
            sorted_cols = reranked + [c for c in sorted_cols if c not in reranked]

        # Get top-k
        final_cols = sorted_cols[:k]
        final_scores = {c: rrf_scores.get(c, 0) for c in final_cols}

        return AdvancedRetrievalResult(
            columns=final_cols,
            scores=final_scores,
            metadata={
                "query": query,
                "expanded_query": expanded_query,
                "components_used": list(component_scores.keys()),
            },
            hyde_result=hyde_result,
            component_scores=component_scores,
            fusion_weights=weights,
        )

    def _extract_targets(
        self,
        query: str,
        rankings: Dict[str, List[str]],
    ) -> List[str]:
        """
        Extract target columns for Markov blanket scoring.
        Uses top columns from other retrievers as targets.
        """
        targets = set()

        # Take top-1 from each ranking
        for method, ranked in rankings.items():
            if ranked:
                targets.add(ranked[0])

        return list(targets)[:3]  # Max 3 targets


class CRAGAdvancedRetriever:
    """
    CRAG wrapper around the Advanced Retriever.
    Adds self-correcting capability with confidence evaluation.
    """

    def __init__(
        self,
        base_retriever: AdvancedRetriever,
        config: Optional[CRAGConfig] = None,
    ):
        """
        Initialize CRAG wrapper.

        Args:
            base_retriever: The advanced retriever to wrap
            config: CRAG configuration
        """
        self.base_retriever = base_retriever
        self.config = config or CRAGConfig()

        # Create CRAG retriever
        self.crag = create_crag_retriever(
            retrieve_fn=self._base_retrieve,
            column_descriptions=base_retriever.descriptions,
            llm=base_retriever.llm,
            max_retries=self.config.max_retries,
        )

    def _base_retrieve(
        self,
        query: str,
        k: int,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Base retrieval function for CRAG."""
        result = self.base_retriever.retrieve_detailed(query, k)
        return result.columns, result.scores

    def retrieve(self, query: str, k: int = 10) -> List[str]:
        """Retrieve with CRAG self-correction."""
        result = self.retrieve_detailed(query, k)
        return result.columns

    def retrieve_detailed(
        self,
        query: str,
        k: int = 10,
    ) -> AdvancedRetrievalResult:
        """
        Retrieve with CRAG and detailed results.
        """
        crag_result = self.crag.retrieve_detailed(query, k)

        return AdvancedRetrievalResult(
            columns=crag_result.columns,
            scores=crag_result.scores,
            metadata={
                "crag_confidence": crag_result.confidence.level,
                "crag_score": crag_result.confidence.score,
                "crag_attempts": crag_result.attempts,
                "query_history": crag_result.query_history,
            },
            crag_result=crag_result,
        )


def create_advanced_retriever(
    columns: List[str],
    descriptions: Dict[str, str],
    graph: Optional[Any] = None,
    llm: Optional[Any] = None,
    use_hyde: bool = True,
    use_colbert: bool = True,
    use_bm25: bool = True,
    use_crag: bool = True,
    use_markov: bool = False,
    use_reranker: bool = False,
) -> AdvancedRetriever:
    """
    Factory function to create an Advanced Retriever.

    Args:
        columns: List of column names
        descriptions: Column descriptions
        graph: NetworkX graph (optional)
        llm: LLM instance (optional)
        use_hyde: Enable HyDE
        use_colbert: Enable ColBERT
        use_bm25: Enable BM25
        use_crag: Enable CRAG
        use_markov: Enable Markov blanket
        use_reranker: Enable reranker

    Returns:
        Configured AdvancedRetriever
    """
    config = AdvancedRetrieverConfig(
        use_hyde=use_hyde and llm is not None,
        use_colbert=use_colbert,
        use_bm25=use_bm25,
        use_crag=use_crag,
        use_markov_blanket=use_markov and graph is not None,
        use_reranker=use_reranker and RERANKER_AVAILABLE,
    )

    retriever = AdvancedRetriever(
        columns=columns,
        descriptions=descriptions,
        graph=graph,
        llm=llm,
        config=config,
    )

    if use_crag:
        return CRAGAdvancedRetriever(retriever)

    return retriever


# Quick test
if __name__ == "__main__":
    import json
    import networkx as nx
    import pandas as pd

    # Load data
    print("Loading data...")
    df = pd.read_csv("data/HEPAR_simulated_patients.csv")
    columns = list(df.columns)

    with open("data/hepar_graph.json") as f:
        graph_data = json.load(f)
    graph = nx.DiGraph()
    graph.add_nodes_from(graph_data["nodes"])
    graph.add_edges_from(graph_data["edges"])

    from rag_llamaindex import HEPAR_COLUMN_MEANINGS
    descriptions = {col: HEPAR_COLUMN_MEANINGS.get(col, col) for col in columns}

    # Initialize LLM
    print("Loading LLM...")
    from llm_huggingface import HuggingFaceLLM, LLMConfig
    llm_config = LLMConfig(model_name="Qwen/Qwen2.5-3B-Instruct")
    llm = HuggingFaceLLM(llm_config)

    # Create retriever (without CRAG for simple test)
    print("Creating advanced retriever...")
    config = AdvancedRetrieverConfig(
        use_hyde=True,
        use_colbert=True,
        use_bm25=True,
        use_crag=False,
        use_markov_blanket=True,
    )
    retriever = AdvancedRetriever(
        columns=columns,
        descriptions=descriptions,
        graph=graph,
        llm=llm,
        config=config,
    )

    # Test query
    query = "Which features are related to liver damage in a patient with hepatitis?"
    print(f"\nQuery: {query}")

    result = retriever.retrieve_detailed(query, k=10)

    print(f"\nRetrieved columns: {result.columns}")
    print(f"Components used: {result.metadata.get('components_used', [])}")
    print(f"Expanded query: {result.metadata.get('expanded_query', '')[:100]}...")

    print("\nComponent scores (top 5 each):")
    for comp, scores in result.component_scores.items():
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  {comp}: {top5}")
