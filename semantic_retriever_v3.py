"""
semantic_retriever_v3.py

Semantic schema retrieval over column-nodes using dense + BM25 fusion.
Adapted from causal_rag_hepar_v3.

Features:
- Dense retriever: VectorIndexRetriever with HuggingFace embeddings
- Sparse retriever: BM25Retriever
- Fusion: weighted sum after min-max normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    from llama_index.retrievers.bm25 import BM25Retriever
    HAS_BM25 = True
except ImportError:
    BM25Retriever = None
    HAS_BM25 = False


def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores to [0, 1] range."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (float(v) - lo) / (hi - lo) for k, v in scores.items()}


@dataclass
class SemanticSchemaRetriever:
    """
    Semantic retriever for schema nodes with dense + BM25 fusion.

    This is the RAG_SEMANTIC baseline from v3.

    Args:
        nodes: List of TextNode objects representing columns
        embed_model: HuggingFace embedding model name
        vector_top_k: Number of candidates from dense retrieval
        bm25_top_k: Number of candidates from BM25 retrieval
        alpha_dense: Weight for dense scores (1-alpha for BM25)
    """
    nodes: Sequence[BaseNode]
    embed_model: str = "BAAI/bge-small-en-v1.5"
    vector_top_k: int = 24
    bm25_top_k: int = 24
    alpha_dense: float = 0.6

    def __post_init__(self):
        # Set up embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model)
        Settings.llm = None  # retrieval only

        # Build vector index
        self.index = VectorStoreIndex(list(self.nodes))
        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.vector_top_k
        )

        # Build BM25 index
        self.bm25_retriever = None
        if HAS_BM25 and BM25Retriever is not None and self.bm25_top_k > 0:
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=list(self.nodes),
                similarity_top_k=self.bm25_top_k
            )

    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        """
        Retrieve top-k column nodes using fused dense + BM25 scores.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of NodeWithScore ordered by fused score
        """
        scores = self.retrieve_scores(query)
        items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        id2node = {n.node_id: n for n in self.nodes}
        out: List[NodeWithScore] = []
        for node_id, s in items:
            node = id2node.get(node_id)
            if node is None:
                continue
            out.append(NodeWithScore(node=node, score=float(s)))
        return out

    def retrieve_scores(self, query: str) -> Dict[str, float]:
        """
        Return fused semantic scores for all candidate columns.

        Scores are meaningful only for candidates returned by the retrievers.
        Columns not in results will be absent from the dict.
        """
        # Dense retrieval
        dense = self.vector_retriever.retrieve(query)
        dense_scores = {n.node.node_id: float(n.score or 0.0) for n in dense}

        # Sparse (BM25) retrieval
        sparse_scores: Dict[str, float] = {}
        if self.bm25_retriever is not None:
            sparse = self.bm25_retriever.retrieve(query)
            sparse_scores = {n.node.node_id: float(n.score or 0.0) for n in sparse}

        # Normalize each channel then fuse
        dense_n = _minmax(dense_scores)
        sparse_n = _minmax(sparse_scores)

        fused: Dict[str, float] = {}
        keys = set(dense_n) | set(sparse_n)
        for k in keys:
            fused[k] = (
                self.alpha_dense * dense_n.get(k, 0.0) +
                (1.0 - self.alpha_dense) * sparse_n.get(k, 0.0)
            )
        return fused

    def get_column_names(self, results: List[NodeWithScore]) -> List[str]:
        """Extract column names from retrieval results."""
        return [r.node.node_id for r in results]


def demo_semantic_retriever():
    """Demo: run semantic retriever on HEPAR data."""
    import pandas as pd
    from schema_v3 import build_column_infos, build_schema_nodes, HEPAR_SYNONYMS

    print("=" * 60)
    print("SemanticSchemaRetriever (v3) Demo")
    print("=" * 60)

    # Load data and build schema
    df = pd.read_csv("data/HEPAR_simulated_patients.csv")
    print(f"\nLoaded {len(df)} records with {len(df.columns)} columns")

    infos = build_column_infos(df, synonyms=HEPAR_SYNONYMS)
    nodes, col2node = build_schema_nodes(infos)
    print(f"Built {len(nodes)} schema nodes")

    # Initialize retriever
    print("\nInitializing SemanticSchemaRetriever...")
    print(f"  - Embedding model: BAAI/bge-small-en-v1.5")
    print(f"  - BM25 enabled: {HAS_BM25}")
    print(f"  - Alpha (dense weight): 0.6")

    retriever = SemanticSchemaRetriever(
        nodes=nodes,
        embed_model="BAAI/bge-small-en-v1.5",
        vector_top_k=24,
        bm25_top_k=24,
        alpha_dense=0.6,
    )

    # Test queries
    test_queries = [
        "What columns do I need to analyze liver enzyme levels?",
        "Find patients with viral hepatitis B infection markers",
        "Which variables are relevant for predicting cirrhosis?",
        "I want to analyze the effect of alcoholism on liver disease",
        "Select columns for early screening of liver problems",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("-" * 60)

        results = retriever.retrieve(query, top_k=8)

        print(f"Top-8 columns (fused dense + BM25):")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.node.node_id:20s} (score: {r.score:.4f})")

    print(f"\n{'=' * 60}")
    print("Demo complete!")


if __name__ == "__main__":
    demo_semantic_retriever()
