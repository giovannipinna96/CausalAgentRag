"""
ColBERT (Contextualized Late Interaction over BERT) retriever for column selection.

ColBERT uses late interaction: instead of compressing query and document to single vectors,
it preserves token-level representations and computes MaxSim (maximum similarity per query token).

This captures fine-grained semantic relationships that bi-encoders miss:
- Partial matches: "liver enzyme" matches "alt" (alanine aminotransferase)
- Synonym handling: "hepatic" matches "liver"
- Abbreviation expansion: "AST" matches "aspartate transaminase"

Reference: "ColBERT: Efficient and Effective Passage Search" (Khattab & Zaharia, SIGIR 2020)
"""

from __future__ import annotations

# ============================================================================
# Langchain Compatibility Shim for RAGatouille
# ============================================================================
# RAGatouille 0.0.9.x imports from langchain.retrievers which doesn't exist
# in langchain v1.2+. These modules were moved to langchain_classic.
# This shim patches sys.modules to redirect imports.
import sys
try:
    import langchain_classic.retrievers
    import langchain_classic.retrievers.document_compressors
    import langchain_classic.retrievers.document_compressors.base
    import langchain
    sys.modules['langchain.retrievers'] = langchain_classic.retrievers
    sys.modules['langchain.retrievers.document_compressors'] = langchain_classic.retrievers.document_compressors
    sys.modules['langchain.retrievers.document_compressors.base'] = langchain_classic.retrievers.document_compressors.base
except ImportError:
    pass  # langchain_classic not installed, RAGatouille will use fallback
# ============================================================================

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Index storage location
INDEX_DIR = Path(__file__).parent / ".colbert_index"


@dataclass
class ColBERTConfig:
    """Configuration for ColBERT retrieval."""
    model_name: str = "colbert-ir/colbertv2.0"
    index_name: str = "hepar_columns"
    use_gpu: bool = True
    max_document_length: int = 128
    split_documents: bool = False  # Don't split short column descriptions
    use_cache: bool = True  # Cache the index


@dataclass
class ColBERTResult:
    """Result from ColBERT scoring."""
    scores: Dict[str, float]  # column -> ColBERT score
    rankings: List[Tuple[str, float]]  # Sorted (column, score) pairs


class ColBERTScorer:
    """
    ColBERT late-interaction scorer for columns.

    Uses RAGatouille for easy ColBERT integration. The key insight:
    - Bi-encoders: query_emb · doc_emb (single vectors)
    - ColBERT: Σ max(query_token · doc_tokens) (token-level interaction)

    This MaxSim operation allows partial matching and is more robust
    to vocabulary mismatch.
    """

    def __init__(
        self,
        columns: List[str],
        column_descriptions: Optional[Dict[str, str]] = None,
        config: Optional[ColBERTConfig] = None,
    ):
        """
        Initialize ColBERT scorer.

        Args:
            columns: List of column names
            column_descriptions: Optional descriptions for each column
            config: ColBERT configuration
        """
        self.columns = columns
        self.descriptions = column_descriptions or {}
        self.config = config or ColBERTConfig()
        self.rag = None
        self._indexed = False

        # Prepare document texts
        self._prepare_documents()

        # Try to load or build index
        self._init_model()

    def _prepare_documents(self) -> None:
        """Prepare column documents for indexing."""
        self.documents = []
        self.doc_ids = []

        for col in self.columns:
            # Create document text
            if col in self.descriptions:
                doc_text = f"{col}: {self.descriptions[col]}"
            else:
                # Expand column name
                expanded = col.replace("_", " ")
                doc_text = f"{col}: {expanded}"

            self.documents.append(doc_text)
            self.doc_ids.append(col)

        logger.info(f"Prepared {len(self.documents)} documents for ColBERT indexing")

    def _init_model(self) -> None:
        """Initialize RAGatouille model and index."""
        try:
            from ragatouille import RAGPretrainedModel

            index_path = INDEX_DIR / self.config.index_name

            # Check if index exists and caching is enabled
            if self.config.use_cache and index_path.exists():
                logger.info(f"Loading existing ColBERT index from {index_path}")
                try:
                    self.rag = RAGPretrainedModel.from_index(str(index_path))
                    self._indexed = True
                    logger.info("ColBERT index loaded successfully")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}, rebuilding...")

            # Create new model and index
            logger.info(f"Initializing ColBERT model: {self.config.model_name}")
            self.rag = RAGPretrainedModel.from_pretrained(self.config.model_name)

            # Build index
            self._build_index()

        except ImportError as e:
            logger.error(f"RAGatouille not available: {e}")
            logger.info("Falling back to simple semantic scoring")
            self.rag = None
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Initialize fallback semantic scorer when ColBERT unavailable."""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        logger.info("Using fallback semantic scorer (bi-encoder)")
        self.fallback_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.fallback_embeddings = self.fallback_model.encode(
            self.documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def _build_index(self) -> None:
        """Build ColBERT index over documents."""
        if self.rag is None:
            return

        logger.info(f"Building ColBERT index for {len(self.documents)} documents...")

        # Create index directory
        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        try:
            # Index documents
            self.rag.index(
                collection=self.documents,
                document_ids=self.doc_ids,
                index_name=self.config.index_name,
                max_document_length=self.config.max_document_length,
                split_documents=self.config.split_documents,
            )
            self._indexed = True
            logger.info("ColBERT index built successfully")
        except Exception as e:
            logger.error(f"Failed to build ColBERT index: {e}")
            self._init_fallback()

    def score(self, query: str, k: int = None) -> ColBERTResult:
        """
        Score all columns using ColBERT MaxSim.

        Args:
            query: User query
            k: Number of results (None = all columns)

        Returns:
            ColBERTResult with scores and rankings
        """
        k = k or len(self.columns)

        if self.rag is not None and self._indexed:
            return self._score_colbert(query, k)
        else:
            return self._score_fallback(query, k)

    def _score_colbert(self, query: str, k: int) -> ColBERTResult:
        """Score using ColBERT."""
        try:
            # Search with ColBERT
            results = self.rag.search(query=query, k=min(k, len(self.columns)))

            # Build scores dict
            scores = {}
            rankings = []

            for result in results:
                doc_id = result.get("document_id") or result.get("content", "").split(":")[0]
                score = result.get("score", 0.0)

                # Map back to column name
                if doc_id in self.columns:
                    scores[doc_id] = score
                    rankings.append((doc_id, score))

            # Fill in missing columns with 0 score
            for col in self.columns:
                if col not in scores:
                    scores[col] = 0.0

            return ColBERTResult(scores=scores, rankings=rankings)

        except Exception as e:
            logger.error(f"ColBERT search failed: {e}, using fallback")
            return self._score_fallback(query, k)

    def _score_fallback(self, query: str, k: int) -> ColBERTResult:
        """Score using fallback bi-encoder."""
        import numpy as np

        query_emb = self.fallback_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = np.dot(self.fallback_embeddings, query_emb)

        # Build results
        scores = {col: float(sim) for col, sim in zip(self.columns, similarities)}

        # Sort for rankings
        rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return ColBERTResult(scores=scores, rankings=rankings)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search and return top-k columns.

        Args:
            query: User query
            k: Number of results

        Returns:
            List of (column, score) tuples
        """
        result = self.score(query, k)
        return result.rankings[:k]


class ColBERTRetriever:
    """
    ColBERT-based column retrieval.

    Wraps ColBERTScorer to provide a simple retrieve interface.
    """

    def __init__(
        self,
        columns: List[str],
        column_descriptions: Optional[Dict[str, str]] = None,
        config: Optional[ColBERTConfig] = None,
    ):
        """
        Initialize ColBERT retriever.

        Args:
            columns: List of column names
            column_descriptions: Optional column descriptions
            config: ColBERT configuration
        """
        self.config = config or ColBERTConfig()
        self.scorer = ColBERTScorer(
            columns=columns,
            column_descriptions=column_descriptions,
            config=self.config,
        )
        self.columns = columns

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> List[str]:
        """
        Retrieve top-k columns using ColBERT.

        Args:
            query: User query
            k: Number of columns to retrieve

        Returns:
            List of column names
        """
        result = self.scorer.score(query, k)
        return [col for col, _ in result.rankings[:k]]

    def retrieve_detailed(
        self,
        query: str,
        k: int = 10,
    ) -> Tuple[List[str], Dict[str, float], ColBERTResult]:
        """
        Retrieve with full details.

        Returns:
            Tuple of (columns, scores, colbert_result)
        """
        result = self.scorer.score(query, k)
        top_cols = [col for col, _ in result.rankings[:k]]
        top_scores = {col: result.scores[col] for col in top_cols}

        return top_cols, top_scores, result


class ColBERTHybridRetriever:
    """
    Hybrid retriever combining ColBERT with BM25.

    Uses RRF fusion at rank level to combine:
    - ColBERT: Semantic + token-level matching
    - BM25: Lexical/keyword matching
    """

    def __init__(
        self,
        columns: List[str],
        column_descriptions: Optional[Dict[str, str]] = None,
        colbert_config: Optional[ColBERTConfig] = None,
        colbert_weight: float = 1.0,
        bm25_weight: float = 0.8,
    ):
        """
        Initialize hybrid retriever.

        Args:
            columns: Column names
            column_descriptions: Optional descriptions
            colbert_config: ColBERT configuration
            colbert_weight: Weight for ColBERT in RRF
            bm25_weight: Weight for BM25 in RRF
        """
        self.columns = columns
        self.descriptions = column_descriptions or {}
        self.colbert_weight = colbert_weight
        self.bm25_weight = bm25_weight

        # Initialize ColBERT
        self.colbert = ColBERTScorer(
            columns=columns,
            column_descriptions=column_descriptions,
            config=colbert_config,
        )

        # Initialize BM25
        self._init_bm25()

    def _init_bm25(self) -> None:
        """Initialize BM25 retriever."""
        try:
            from bm25_retriever import BM25ColumnRetriever
            self.bm25 = BM25ColumnRetriever()
            self.bm25.index_columns(self.columns, self.descriptions)
        except ImportError:
            logger.warning("BM25ColumnRetriever not available, using ColBERT only")
            self.bm25 = None

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> List[str]:
        """
        Retrieve using ColBERT + BM25 hybrid.

        Args:
            query: User query
            k: Number of results

        Returns:
            List of column names
        """
        cols, _, _ = self.retrieve_detailed(query, k)
        return cols

    def retrieve_detailed(
        self,
        query: str,
        k: int = 10,
    ) -> Tuple[List[str], Dict[str, float], Dict]:
        """
        Retrieve with full details including fusion analysis.

        Returns:
            Tuple of (columns, scores, analysis_dict)
        """
        from hybrid_fusion import reciprocal_rank_fusion

        # Get ColBERT scores
        colbert_result = self.colbert.score(query, k * 2)
        colbert_ranked = colbert_result.rankings

        # Get BM25 scores if available
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_all_scores(query)
            bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            bm25_ranked = []

        # RRF fusion
        if bm25_ranked:
            fused = reciprocal_rank_fusion(
                [colbert_ranked, bm25_ranked],
                weights=[self.colbert_weight, self.bm25_weight],
                k=60
            )
        else:
            fused = colbert_ranked

        # Extract top-k
        top_cols = [doc_id for doc_id, _ in fused[:k]]
        top_scores = {doc_id: score for doc_id, score in fused[:k]}

        analysis = {
            "colbert_top5": colbert_ranked[:5],
            "bm25_top5": bm25_ranked[:5] if bm25_ranked else [],
            "fusion_method": "rrf",
        }

        return top_cols, top_scores, analysis


def create_colbert_retriever(
    columns: List[str],
    descriptions: Optional[Dict[str, str]] = None,
    use_hybrid: bool = True,
) -> ColBERTRetriever | ColBERTHybridRetriever:
    """
    Convenience function to create a ColBERT retriever.

    Args:
        columns: Column names
        descriptions: Optional column descriptions
        use_hybrid: Whether to use hybrid ColBERT + BM25

    Returns:
        Configured retriever
    """
    config = ColBERTConfig()

    if use_hybrid:
        return ColBERTHybridRetriever(
            columns=columns,
            column_descriptions=descriptions,
            colbert_config=config,
        )
    else:
        return ColBERTRetriever(
            columns=columns,
            column_descriptions=descriptions,
            config=config,
        )
