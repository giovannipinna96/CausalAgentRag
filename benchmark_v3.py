#!/usr/bin/env python3
"""
Benchmark v3: Column selection evaluation across multiple methods.

Includes both original methods and new RRF-based hybrid retrieval with
Markov Blanket for precision-focused column selection.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx
import pandas as pd
import requests
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# New imports for RRF hybrid retrieval
from retrievers import (
    RRFHybridRetriever,
    RRFHybridConfig,
    RRFRetrievalResult,
)
from markov_blanket import (
    MarkovBlanketScorer,
    get_causal_descendants,
)
from llm_huggingface import HuggingFaceLLM, LLMConfig

# Imports for reranking and prompt optimization
from reranker import SchemaReranker, RerankerConfig, RerankResult
from prompt_optimizer import (
    reorder_for_attention,
    build_optimized_prompt,
)

# Imports for HyDE retrieval
from hyde_retriever import HyDERetriever, HyDEConfig, HyDEScorer

# Imports for ColBERT retrieval
from colbert_retriever import ColBERTRetriever, ColBERTHybridRetriever, ColBERTConfig

# Imports for CRAG (Corrective RAG)
from crag_evaluator import CRAGRetriever, CRAGConfig, create_crag_retriever

# Imports for Advanced Retriever (combined HyDE + ColBERT + CRAG + optional Markov)
from advanced_retriever import (
    AdvancedRetriever, AdvancedRetrieverConfig, CRAGAdvancedRetriever,
    create_advanced_retriever
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = DATA_DIR / "HEPAR_simulated_patients.csv"
EDGE_CSV = DATA_DIR / "HEPAR_arcs.csv"
BENCH_JSON = DATA_DIR / "causal_rag_benchmark_v3.json"
DATASET_MD = DATA_DIR / "HEPAR2_DATASET.md"

# LLM configuration
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-3B-Instruct")
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "false").lower() == "true"

# Global HuggingFace LLM instance
_hf_llm: Optional[HuggingFaceLLM] = None

K_RETRIEVE = int(os.environ.get("K_RETRIEVE", "10"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.5"))
CAUSAL_MAX_HOPS = int(os.environ.get("CAUSAL_MAX_HOPS", "2"))
RUN_QUERY_LIMIT = int(os.environ.get("RUN_QUERY_LIMIT", "0"))


EXPANSIONS = {
    "alt": "ALT (alanine aminotransferase)",
    "ast": "AST (aspartate aminotransferase)",
    "ggtp": "GGTP (gamma-glutamyl transferase)",
    "esr": "ESR (erythrocyte sedimentation rate)",
    "inr": "INR (international normalized ratio)",
    "pbc": "PBC (primary biliary cholangitis/cirrhosis)",
    "hbsag": "HBsAg (hepatitis B surface antigen)",
    "hbsag_anti": "Anti-HBs (HBsAg antibody)",
    "hbc_anti": "Anti-HBc (hepatitis B core antibody)",
    "hcv_anti": "Anti-HCV (hepatitis C antibody)",
    "hbeag": "HBeAg (hepatitis B e antigen)",
    "ama": "AMA (anti-mitochondrial antibodies)",
    "le_cells": "LE cells (lupus erythematosus cells)",
}


@dataclass
class MethodConfig:
    key: str
    label: str
    mode: str  # simple, description, graph, rag_semantic, rag_hybrid


METHODS = [
    MethodConfig("simple", "Simple", "simple"),
    MethodConfig("description", "Descriptions", "description"),
    MethodConfig("graph", "Graph", "graph"),
    MethodConfig("rag_semantic", "Semantic RAG", "rag_semantic"),
    MethodConfig("rag_hybrid", "Hybrid RAG", "rag_hybrid"),
    # New RRF-based methods with Markov Blanket
    MethodConfig("rrf_hybrid", "RRF Hybrid", "rrf_hybrid"),
    MethodConfig("rrf_markov", "RRF Markov", "rrf_markov"),
    # Cross-encoder reranked methods
    MethodConfig("rrf_reranked", "RRF Reranked", "rrf_reranked"),
    MethodConfig("rag_reranked", "RAG Reranked", "rag_reranked"),
    # HyDE methods
    MethodConfig("hyde_only", "HyDE Only", "hyde_only"),
    MethodConfig("hyde_bm25", "HyDE + BM25", "hyde_bm25"),
    # ColBERT methods
    MethodConfig("colbert_only", "ColBERT Only", "colbert_only"),
    MethodConfig("colbert_bm25", "ColBERT + BM25", "colbert_bm25"),
    # CRAG methods (self-correcting)
    MethodConfig("crag_semantic", "CRAG Semantic", "crag_semantic"),
    MethodConfig("crag_rrf", "CRAG RRF", "crag_rrf"),
    # Advanced methods (combined pipeline)
    MethodConfig("advanced_full", "Advanced (HyDE+ColBERT+BM25+CRAG)", "advanced_full"),
    MethodConfig("advanced_markov", "Advanced + Markov", "advanced_markov"),
]

# Select which methods to run (set via environment variable, comma-separated)
RUN_METHODS = os.environ.get("RUN_METHODS", "").strip()
if RUN_METHODS:
    method_keys = [m.strip() for m in RUN_METHODS.split(",")]
    METHODS = [m for m in METHODS if m.key in method_keys]


# -----------------------------
# Data loading
# -----------------------------


def load_benchmark(path: Path) -> Tuple[List[Dict], Dict]:
    data = json.loads(path.read_text())
    return data["queries"], data.get("meta", {})


def load_data() -> Tuple[pd.DataFrame, nx.DiGraph]:
    df = pd.read_csv(DATA_CSV)
    edges = pd.read_csv(EDGE_CSV)
    graph = nx.DiGraph()
    graph.add_edges_from(edges[["from", "to"]].itertuples(index=False, name=None))
    return df, graph


def parse_category_map(md_path: Path) -> Dict[str, str]:
    if not md_path.exists():
        return {}
    lines = md_path.read_text().splitlines()
    categories: Dict[str, str] = {}
    for i, line in enumerate(lines):
        if line.startswith("### "):
            category = line[4:].split("(")[0].strip()
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                items = [x.strip() for x in lines[j].split(",") if x.strip()]
                for item in items:
                    categories[item] = category
    return categories


# -----------------------------
# Column descriptions
# -----------------------------


def prettify_name(col: str) -> str:
    if col in EXPANSIONS:
        return EXPANSIONS[col]
    lower = col.lower()
    if lower in EXPANSIONS:
        return EXPANSIONS[lower]
    name = col.replace("_", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return name


def infer_col_type(values: List[str]) -> str:
    value_set = set(values)
    if value_set <= {"present", "absent"}:
        return "binary"
    if value_set <= {"male", "female"}:
        return "binary"
    if all(re.match(r"^[a-z]+\d+_\d+$", v) or re.match(r"^a\d+_\d+$", v) for v in value_set):
        return "range"
    return "categorical"


def describe_column(col: str, values: List[str], category: str | None) -> str:
    pretty = prettify_name(col)
    col_type = infer_col_type(values)
    cat_desc = f"{category} variable" if category else "Clinical variable"
    if col_type == "binary":
        type_desc = "Binary indicator (present/absent)."
    elif col_type == "range":
        type_desc = "Ordinal range-coded value."
    else:
        type_desc = "Categorical status."
    return f"{cat_desc}: {pretty}. {type_desc}"


def build_descriptions(df: pd.DataFrame, category_map: Dict[str, str]) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    for col in df.columns:
        values = df[col].astype(str).unique().tolist()
        descriptions[col] = describe_column(col, values, category_map.get(col))
    return descriptions


# -----------------------------
# Retrieval
# -----------------------------


def build_documents(descriptions: Dict[str, str]) -> List[Document]:
    docs = []
    for col, desc in descriptions.items():
        text = f"Column: {col}. Description: {desc}"
        docs.append(Document(text=text, metadata={"column": col}))
    return docs


def build_retriever(docs: List[Document]):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_documents(docs)
    return index.as_retriever(similarity_top_k=K_RETRIEVE * 2)


def semantic_retrieve(query: str, retriever) -> Tuple[List[str], Dict[str, float]]:
    nodes = retriever.retrieve(query)
    nodes_sorted = sorted(nodes, key=lambda n: (n.score or 0.0), reverse=True)
    cols: List[str] = []
    scores: Dict[str, float] = {}
    for n in nodes_sorted:
        col = n.node.metadata.get("column")
        if col and col not in cols:
            cols.append(col)
            scores[col] = float(n.score or 0.0)
        if len(cols) >= K_RETRIEVE:
            break
    return cols, scores


# Global RRF retriever instance (initialized in main)
_rrf_retriever: Optional[RRFHybridRetriever] = None

# Global cross-encoder reranker instance (initialized in main)
_reranker: Optional[SchemaReranker] = None

# Global HyDE retriever instance (initialized in main)
_hyde_retriever: Optional[HyDERetriever] = None

# Global ColBERT retriever instances (initialized in main)
_colbert_retriever: Optional[ColBERTRetriever] = None
_colbert_hybrid_retriever: Optional[ColBERTHybridRetriever] = None

# Global CRAG retriever instances (initialized in main)
_crag_semantic_retriever: Optional[CRAGRetriever] = None
_crag_rrf_retriever: Optional[CRAGRetriever] = None

# Global Advanced retriever instances (initialized in main)
_advanced_retriever: Optional[AdvancedRetriever] = None
_advanced_markov_retriever: Optional[AdvancedRetriever] = None


def init_reranker() -> SchemaReranker:
    """Initialize the cross-encoder reranker."""
    global _reranker
    _reranker = SchemaReranker(RerankerConfig(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        batch_size=32,
    ))
    return _reranker


def init_hyde_retriever(
    columns: List[str],
    descriptions: Dict[str, str],
    llm: Optional[HuggingFaceLLM] = None,
) -> HyDERetriever:
    """Initialize the HyDE retriever."""
    global _hyde_retriever
    config = HyDEConfig(n_hypotheses=3, hypothesis_weight=0.6)
    _hyde_retriever = HyDERetriever(
        columns=columns,
        column_descriptions=descriptions,
        llm=llm,
        config=config,
    )
    return _hyde_retriever


def hyde_retrieve(
    query: str,
    k: int = K_RETRIEVE,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Retrieve columns using HyDE.

    Args:
        query: Natural language query
        k: Number of results to return

    Returns:
        Tuple of (columns, scores)
    """
    if _hyde_retriever is None:
        raise RuntimeError("HyDE retriever not initialized. Call init_hyde_retriever() first.")

    cols, scores, _ = _hyde_retriever.retrieve_detailed(query, k)
    return cols, scores


def init_colbert_retriever(
    columns: List[str],
    descriptions: Dict[str, str],
) -> None:
    """Initialize ColBERT retrievers."""
    global _colbert_retriever, _colbert_hybrid_retriever

    config = ColBERTConfig(use_cache=True)

    # Initialize ColBERT only retriever
    _colbert_retriever = ColBERTRetriever(
        columns=columns,
        column_descriptions=descriptions,
        config=config,
    )

    # Initialize ColBERT + BM25 hybrid retriever
    _colbert_hybrid_retriever = ColBERTHybridRetriever(
        columns=columns,
        column_descriptions=descriptions,
        colbert_config=config,
    )


def colbert_retrieve(
    query: str,
    k: int = K_RETRIEVE,
    use_hybrid: bool = False,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Retrieve columns using ColBERT.

    Args:
        query: Natural language query
        k: Number of results to return
        use_hybrid: Whether to use ColBERT + BM25 hybrid

    Returns:
        Tuple of (columns, scores)
    """
    if use_hybrid:
        if _colbert_hybrid_retriever is None:
            raise RuntimeError("ColBERT hybrid retriever not initialized.")
        cols, scores, _ = _colbert_hybrid_retriever.retrieve_detailed(query, k)
    else:
        if _colbert_retriever is None:
            raise RuntimeError("ColBERT retriever not initialized.")
        cols, scores, _ = _colbert_retriever.retrieve_detailed(query, k)

    return cols, scores


def init_crag_retrievers(
    descriptions: Dict[str, str],
    llm: Optional[HuggingFaceLLM] = None,
) -> None:
    """Initialize CRAG retrievers wrapping semantic and RRF base retrievers."""
    global _crag_semantic_retriever, _crag_rrf_retriever

    config = CRAGConfig(max_retries=2)

    # CRAG wrapping semantic retrieval
    def semantic_retrieve_fn(query: str, k: int) -> Tuple[List[str], Dict[str, float]]:
        # Use the global retriever for semantic retrieval
        cols, scores = semantic_retrieve(query, retriever)  # retriever is the llamaindex one
        return cols, scores

    # For now, we'll initialize CRAG retrievers in the main loop
    # since they need access to the retriever object
    pass


def crag_retrieve(
    query: str,
    k: int = K_RETRIEVE,
    mode: str = "semantic",
    retriever = None,
    descriptions: Dict[str, str] = None,
    llm: Optional[HuggingFaceLLM] = None,
) -> Tuple[List[str], Dict[str, float], Dict]:
    """
    Retrieve columns using CRAG (self-correcting retrieval).

    Args:
        query: Natural language query
        k: Number of results to return
        mode: 'semantic' or 'rrf'
        retriever: The base retriever (llamaindex retriever for semantic)
        descriptions: Column descriptions
        llm: Optional LLM for query refinement

    Returns:
        Tuple of (columns, scores, crag_metadata)
    """
    descriptions = descriptions or {}

    # Define base retrieve function based on mode
    if mode == "rrf":
        def base_retrieve_fn(q: str, k: int) -> Tuple[List[str], Dict[str, float]]:
            cols, scores, _ = rrf_retrieve(q, mode="rrf_hybrid")
            return cols[:k], {c: scores.get(c, 0) for c in cols[:k]}
    else:  # semantic
        def base_retrieve_fn(q: str, k: int) -> Tuple[List[str], Dict[str, float]]:
            cols, scores = semantic_retrieve(q, retriever)
            return cols[:k], {c: scores.get(c, 0) for c in cols[:k]}

    # Create CRAG retriever
    crag = create_crag_retriever(
        retrieve_fn=base_retrieve_fn,
        column_descriptions=descriptions,
        llm=llm,
        max_retries=2,
    )

    # Retrieve with CRAG
    result = crag.retrieve_detailed(query, k)

    # Build metadata
    metadata = {
        "confidence_level": result.confidence.level,
        "confidence_score": result.confidence.score,
        "attempts": result.attempts,
        "query_history": result.query_history,
        "metrics": result.confidence.metrics,
    }

    return result.columns, result.scores, metadata


def init_rrf_retriever(
    columns: List[str],
    graph: nx.DiGraph,
    descriptions: Dict[str, str],
) -> RRFHybridRetriever:
    """Initialize the RRF hybrid retriever."""
    global _rrf_retriever
    _rrf_retriever = RRFHybridRetriever(
        columns=columns,
        graph=graph,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        column_descriptions=descriptions,
    )
    return _rrf_retriever


def init_advanced_retriever(
    columns: List[str],
    graph: nx.DiGraph,
    descriptions: Dict[str, str],
    llm: Optional[HuggingFaceLLM] = None,
    use_markov: bool = False,
) -> AdvancedRetriever:
    """Initialize the Advanced retriever (HyDE + ColBERT + BM25 + optional Markov)."""
    global _advanced_retriever, _advanced_markov_retriever

    # Create config
    config = AdvancedRetrieverConfig(
        use_hyde=True,
        use_colbert=True,
        use_bm25=True,
        use_crag=True,  # CRAG wrapper for self-correction
        use_markov_blanket=use_markov,
        k=K_RETRIEVE,
    )

    if use_markov:
        _advanced_markov_retriever = CRAGAdvancedRetriever(
            AdvancedRetriever(
                columns=columns,
                descriptions=descriptions,
                graph=graph,
                llm=llm,
                config=config,
            )
        )
        return _advanced_markov_retriever
    else:
        config.use_markov_blanket = False
        _advanced_retriever = CRAGAdvancedRetriever(
            AdvancedRetriever(
                columns=columns,
                descriptions=descriptions,
                graph=graph,
                llm=llm,
                config=config,
            )
        )
        return _advanced_retriever


def advanced_retrieve(
    query: str,
    k: int = K_RETRIEVE,
    use_markov: bool = False,
) -> Tuple[List[str], Dict[str, float], Dict]:
    """
    Retrieve columns using the Advanced retriever pipeline.

    Args:
        query: Natural language query
        k: Number of results
        use_markov: Whether to use Markov blanket scoring

    Returns:
        Tuple of (columns, scores, metadata)
    """
    retriever = _advanced_markov_retriever if use_markov else _advanced_retriever
    if retriever is None:
        raise RuntimeError("Advanced retriever not initialized. Call init_advanced_retriever() first.")

    result = retriever.retrieve_detailed(query, k)

    return result.columns, result.scores, result.metadata


def rrf_retrieve(
    query: str,
    mode: str = "rrf_hybrid",
) -> Tuple[List[str], Dict[str, float], Dict]:
    """
    Retrieve columns using RRF hybrid method.

    Args:
        query: Natural language query
        mode: 'rrf_hybrid' (full RRF) or 'rrf_markov' (Markov Blanket only)

    Returns:
        Tuple of (columns, scores, analysis_dict)
    """
    if _rrf_retriever is None:
        raise RuntimeError("RRF retriever not initialized. Call init_rrf_retriever() first.")

    if mode == "rrf_markov":
        # Markov Blanket only mode (no BM25, higher graph weight)
        config = RRFHybridConfig(
            k=K_RETRIEVE,
            semantic_weight=0.5,
            bm25_weight=0.0,
            graph_weight=2.0,
            causal_mode='parents_only',
            exclude_effects=True,
            use_bm25=False,
        )
    else:
        # Full RRF hybrid mode
        config = RRFHybridConfig(
            k=K_RETRIEVE,
            semantic_weight=1.0,
            bm25_weight=0.8,
            graph_weight=1.5,
            causal_mode='parents_only',
            exclude_effects=True,
            use_bm25=True,
        )

    result = _rrf_retriever.retrieve_detailed(query, config)

    # Extract scores from fused results
    scores = {r.doc_id: r.rrf_score for r in result.fused_results if r.doc_id in result.columns}

    return result.columns, scores, result.analysis


def rerank_columns(
    query: str,
    columns: List[str],
    descriptions: Dict[str, str],
    top_k: int = K_RETRIEVE,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Rerank columns using cross-encoder.

    Args:
        query: Natural language query
        columns: Candidate columns to rerank
        descriptions: Column descriptions
        top_k: Number of results to return

    Returns:
        Tuple of (reranked_columns, scores)
    """
    if _reranker is None:
        raise RuntimeError("Reranker not initialized. Call init_reranker() first.")

    results = _reranker.rerank(
        query=query,
        columns=columns,
        descriptions=descriptions,
        top_k=top_k,
    )

    reranked_cols = [r.column for r in results]
    scores = {r.column: r.cross_encoder_score for r in results}

    return reranked_cols, scores


def compute_leakage_metrics(
    found_cols: List[str],
    gold_cols: List[str],
    leakage_cols: List[str],
) -> Dict[str, float]:
    """
    Compute metrics related to data leakage.

    Args:
        found_cols: Columns selected by the method
        gold_cols: Ground truth valid columns
        leakage_cols: Columns that would cause data leakage

    Returns:
        Dict with leakage metrics
    """
    found_set = set(found_cols)
    leakage_set = set(leakage_cols)

    # How many leakage columns were selected
    leakage_selected = found_set & leakage_set
    leakage_count = len(leakage_selected)

    # Leakage rate: proportion of selected columns that are leakage
    leakage_rate = leakage_count / len(found_set) if found_set else 0.0

    # Leakage avoidance: proportion of leakage columns avoided
    leakage_avoidance = 1.0 - (leakage_count / len(leakage_set)) if leakage_set else 1.0

    return {
        "leakage_count": leakage_count,
        "leakage_rate": leakage_rate,
        "leakage_avoidance": leakage_avoidance,
        "leakage_columns_selected": list(leakage_selected),
    }


def causal_expand(
    base_cols: List[str],
    base_scores: Dict[str, float],
    graph: nx.DiGraph,
    all_cols: List[str],
    max_hops: int = CAUSAL_MAX_HOPS,
) -> Tuple[List[str], Dict[str, float]]:
    selected: List[str] = []
    selected_scores: Dict[str, float] = {}

    for col in base_cols:
        if base_scores.get(col, 0.0) >= SIM_THRESHOLD:
            selected.append(col)
            selected_scores[col] = base_scores[col]

    frontier = set(selected)
    visited = set(selected)

    for hop in range(max_hops):
        if len(selected) >= K_RETRIEVE:
            break
        next_frontier = set()
        for col in frontier:
            if col not in graph:
                continue
            neighbors = set(graph.predecessors(col)) | set(graph.successors(col))
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in all_cols:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)
        decay = 0.03 * (hop + 1)
        for neighbor in sorted(next_frontier):
            if len(selected) >= K_RETRIEVE:
                break
            best_score = 0.0
            for col in frontier:
                if col in graph and (
                    neighbor in graph.predecessors(col)
                    or neighbor in graph.successors(col)
                ):
                    best_score = max(best_score, selected_scores.get(col, 0.0))
            selected.append(neighbor)
            selected_scores[neighbor] = max(0.0, best_score - decay)
        frontier = next_frontier

    if len(selected) < K_RETRIEVE:
        for col in base_cols:
            if col not in selected:
                selected.append(col)
                selected_scores[col] = base_scores.get(col, 0.0)
            if len(selected) >= K_RETRIEVE:
                break

    return selected[:K_RETRIEVE], selected_scores


# -----------------------------
# LLM interaction
# -----------------------------


def init_hf_llm() -> HuggingFaceLLM:
    """Initialize the HuggingFace LLM."""
    global _hf_llm
    if _hf_llm is None:
        config = LLMConfig(
            model_name=HF_MODEL,
            temperature=0.1,
            max_new_tokens=512,
            use_quantization=USE_QUANTIZATION,
            quantization_bits=4 if USE_QUANTIZATION else 4,
        )
        _hf_llm = HuggingFaceLLM(config)
    return _hf_llm


def call_llm(prompt: str, max_tokens: int = 512) -> Dict[str, object]:
    """Call the HuggingFace LLM with the given prompt."""
    if _hf_llm is None:
        raise RuntimeError("LLM not initialized. Call init_hf_llm() first.")

    response = _hf_llm.generate(prompt)

    # Estimate token counts (approximate)
    prompt_tokens = len(prompt.split()) * 1.3  # rough estimate
    output_tokens = len(response.split()) * 1.3

    return {
        "response": response,
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
    }


def build_prompt(
    query: str,
    columns: List[str],
    descriptions: Dict[str, str] | None = None,
    graph_edges: List[Dict[str, str]] | None = None,
) -> str:
    header = (
        "You are selecting dataset columns needed to answer the user's question.\n"
        "Rules:\n"
        "- Return a JSON array of column names.\n"
        "- Use ONLY column names from the provided list.\n"
        "- Include all necessary columns; exclude unrelated columns.\n"
        "- For prediction tasks, avoid data leakage (do not include effects of the target).\n"
        "\n"
    )

    if descriptions is None:
        col_block = "\n".join([f"- {c}" for c in columns])
    else:
        col_block = "\n".join([f"- {c}: {descriptions[c]}" for c in columns])

    graph_block = ""
    if graph_edges is not None:
        graph_block = "\nCausal graph (JSON edges):\n" + json.dumps(graph_edges, indent=2)

    return (
        f"{header}"
        f"Question: {query}\n\n"
        f"Columns:\n{col_block}\n"
        f"{graph_block}\n\n"
        "Return ONLY the JSON array."
    )


# -----------------------------
# Parsing and metrics
# -----------------------------


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def parse_column_list(text: str, valid_cols: List[str]) -> List[str]:
    valid_set = set(valid_cols)
    lower_map = {c.lower(): c for c in valid_cols}

    # Try JSON array
    array = None
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            array = json.loads(text[start : end + 1])
    except Exception:
        array = None

    if isinstance(array, list):
        cols: List[str] = []
        for item in array:
            if not isinstance(item, str):
                continue
            raw = item.strip()
            if raw in valid_set:
                cols.append(raw)
            else:
                key = raw.lower()
                if key in lower_map:
                    cols.append(lower_map[key])
                else:
                    cols.append(raw)
        return _dedupe_keep_order(cols)

    # Fallback: match valid columns by substring
    low_text = text.lower()
    found = [c for c in valid_cols if c.lower() in low_text]
    if found:
        return _dedupe_keep_order(found)

    # Fallback: split tokens
    tokens = re.split(r"[^A-Za-z0-9_]+", text)
    cols = []
    for tok in tokens:
        if not tok:
            continue
        if tok in valid_set:
            cols.append(tok)
        else:
            key = tok.lower()
            if key in lower_map:
                cols.append(lower_map[key])
    return _dedupe_keep_order(cols)


def classify(found: List[str], gold: List[str]) -> str:
    found_set = set(found)
    gold_set = set(gold)
    if gold_set.issubset(found_set):
        if found_set == gold_set:
            return "correct"
        return "partial"
    return "wrong"


def compute_metrics(found: List[str], gold: List[str]) -> Dict[str, float]:
    found_set = set(found)
    gold_set = set(gold)
    inter = len(found_set & gold_set)
    union = len(found_set | gold_set)
    precision = inter / len(found_set) if found_set else 0.0
    recall = inter / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    jaccard = inter / union if union else 1.0
    exact = 1.0 if found_set == gold_set else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact": exact,
    }


def render_report(summary: Dict) -> str:
    total = summary["total_queries"]
    lines = []
    lines.append(f"Risultati della Valutazione (Benchmark v3 - {total} queries)")
    lines.append("")
    lines.append("Classificazione delle Risposte")
    lines.append("")
    lines.append("| Modo | Corrette (Esatte) | Parziali (tutte gold + extra) | Sbagliate (manca almeno 1 gold) |")
    lines.append("|------|--------------------|-----------------------------|--------------------------------|")
    for method in METHODS:
        stats = summary["by_method"][method.key]["classification"]
        correct = stats["correct"]
        partial = stats["partial"]
        wrong = stats["wrong"]
        lines.append(
            f"| {method.label} | {correct}/{total} ({correct/total*100:.1f}%) | "
            f"{partial}/{total} ({partial/total*100:.1f}%) | "
            f"{wrong}/{total} ({wrong/total*100:.1f}%) |"
        )
    lines.append("")
    lines.append("Metriche Aggregate")
    lines.append("")
    lines.append("| Modo | Precision | Recall | F1 Score | Jaccard | Exact Match Rate |")
    lines.append("|------|-----------|--------|----------|---------|------------------|")
    for method in METHODS:
        metrics = summary["by_method"][method.key]["metrics"]
        lines.append(
            "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {jaccard:.4f} | {exact:.4f} |".format(
                label=method.label,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                jaccard=metrics["jaccard"],
                exact=metrics["exact"],
            )
        )

    # Add leakage metrics table if available
    has_leakage = any(
        summary["by_method"][m.key].get("leakage_metrics") is not None
        for m in METHODS
    )
    if has_leakage:
        lines.append("")
        lines.append("Leakage Metrics (Data Leakage Avoidance)")
        lines.append("")
        lines.append("| Modo | Avg Leakage Count | Avg Leakage Rate | Leakage Avoidance |")
        lines.append("|------|-------------------|------------------|-------------------|")
        for method in METHODS:
            leakage = summary["by_method"][method.key].get("leakage_metrics")
            if leakage:
                lines.append(
                    "| {label} | {count:.2f} | {rate:.4f} | {avoid:.4f} |".format(
                        label=method.label,
                        count=leakage["avg_leakage_count"],
                        rate=leakage["avg_leakage_rate"],
                        avoid=leakage["avg_leakage_avoidance"],
                    )
                )
            else:
                lines.append(f"| {method.label} | N/A | N/A | N/A |")

    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    queries, meta = load_benchmark(BENCH_JSON)
    if RUN_QUERY_LIMIT > 0:
        queries = queries[:RUN_QUERY_LIMIT]

    df, graph = load_data()
    all_cols = list(df.columns)

    category_map = parse_category_map(DATASET_MD)
    descriptions = build_descriptions(df, category_map)

    docs = build_documents(descriptions)
    retriever = build_retriever(docs)

    # Initialize RRF retriever for new methods
    has_rrf_methods = any(m.mode in {"rrf_hybrid", "rrf_markov", "rrf_reranked", "crag_rrf"} for m in METHODS)
    if has_rrf_methods:
        print("Initializing RRF hybrid retriever...")
        init_rrf_retriever(all_cols, graph, descriptions)
        print("RRF retriever initialized.")

    # Initialize cross-encoder reranker for reranked methods
    has_reranked_methods = any(m.mode in {"rrf_reranked", "rag_reranked"} for m in METHODS)
    if has_reranked_methods:
        print("Initializing cross-encoder reranker...")
        init_reranker()
        print("Reranker initialized.")

    # Initialize HuggingFace LLM
    print("Initializing HuggingFace LLM...")
    init_hf_llm()
    print(f"LLM initialized: {HF_MODEL}")

    # Initialize HyDE retriever for HyDE methods (needs LLM)
    has_hyde_methods = any(m.mode in {"hyde_only", "hyde_bm25"} for m in METHODS)
    if has_hyde_methods:
        print("Initializing HyDE retriever...")
        init_hyde_retriever(all_cols, descriptions, _hf_llm)
        print("HyDE retriever initialized.")

    # Initialize ColBERT retriever for ColBERT methods
    has_colbert_methods = any(m.mode in {"colbert_only", "colbert_bm25"} for m in METHODS)
    if has_colbert_methods:
        print("Initializing ColBERT retriever...")
        init_colbert_retriever(all_cols, descriptions)
        print("ColBERT retriever initialized.")

    # Initialize Advanced retriever for advanced methods
    has_advanced_methods = any(m.mode in {"advanced_full", "advanced_markov"} for m in METHODS)
    if has_advanced_methods:
        print("Initializing Advanced retriever...")
        init_advanced_retriever(all_cols, graph, descriptions, _hf_llm, use_markov=False)
        if any(m.mode == "advanced_markov" for m in METHODS):
            init_advanced_retriever(all_cols, graph, descriptions, _hf_llm, use_markov=True)
        print("Advanced retriever initialized.")

    edges = pd.read_csv(EDGE_CSV).to_dict(orient="records")

    results = {
        "meta": {
            "benchmark": str(BENCH_JSON),
            "dataset": str(DATA_CSV),
            "llm_model": HF_MODEL,
            "k_retrieve": K_RETRIEVE,
            "sim_threshold": SIM_THRESHOLD,
            "causal_max_hops": CAUSAL_MAX_HOPS,
            "n_queries": len(queries),
            "methods": [m.label for m in METHODS],
        },
        "queries": [],
        "summary": {},
    }

    by_method = {
        m.key: {
            "classification": {"correct": 0, "partial": 0, "wrong": 0},
            "metrics_list": [],
        }
        for m in METHODS
    }

    total_q = len(queries)
    total_m = len(METHODS)
    for qi, q in enumerate(queries, start=1):
        qid = q.get("id")
        query = q.get("query")
        gold = q.get("gold_columns", [])

        print(f"\n{'='*60}")
        print(f"Query {qi}/{total_q} - {qid}: {query[:90]}{'...' if len(query) > 90 else ''}")
        print(f"Gold columns: {gold}")
        print(f"{'='*60}")

        q_entry = {
            "id": qid,
            "query": query,
            "gold_columns": gold,
            "methods": {},
        }

        # Get leakage columns from benchmark if available
        leakage_cols = q.get("leakage_columns", [])

        for mi, method in enumerate(METHODS, start=1):
            print(f"[{qi}/{total_q}] Method {mi}/{total_m}: {method.label}")
            # Select candidate columns for prompt
            candidate_cols = all_cols
            retrieved_cols = None
            retrieval_scores = None
            rrf_analysis = None

            if method.mode in {"rag_semantic", "rag_hybrid"}:
                base_cols, base_scores = semantic_retrieve(query, retriever)
                if method.mode == "rag_hybrid":
                    candidate_cols, retrieval_scores = causal_expand(
                        base_cols, base_scores, graph, all_cols, CAUSAL_MAX_HOPS
                    )
                else:
                    candidate_cols, retrieval_scores = base_cols, base_scores
                retrieved_cols = candidate_cols

            elif method.mode in {"rrf_hybrid", "rrf_markov"}:
                # New RRF-based retrieval
                retrieved_cols, retrieval_scores, rrf_analysis = rrf_retrieve(
                    query, mode=method.mode
                )
                candidate_cols = retrieved_cols

            elif method.mode == "rrf_reranked":
                # RRF retrieval + cross-encoder reranking
                # Stage 1: Get more candidates from RRF
                rrf_cols, rrf_scores, rrf_analysis = rrf_retrieve(query, mode="rrf_hybrid")
                # Stage 2: Rerank with cross-encoder
                candidate_cols, retrieval_scores = rerank_columns(
                    query, rrf_cols, descriptions, top_k=K_RETRIEVE
                )
                retrieved_cols = candidate_cols

            elif method.mode == "rag_reranked":
                # Semantic RAG + cross-encoder reranking
                # Stage 1: Get candidates from semantic retrieval
                base_cols, base_scores = semantic_retrieve(query, retriever)
                # Stage 2: Rerank with cross-encoder
                candidate_cols, retrieval_scores = rerank_columns(
                    query, base_cols, descriptions, top_k=K_RETRIEVE
                )
                retrieved_cols = candidate_cols

            elif method.mode == "hyde_only":
                # HyDE retrieval only
                candidate_cols, retrieval_scores = hyde_retrieve(query, K_RETRIEVE)
                retrieved_cols = candidate_cols

            elif method.mode == "hyde_bm25":
                # HyDE + BM25 fusion via simple RRF
                hyde_cols, hyde_scores = hyde_retrieve(query, K_RETRIEVE * 2)
                # Get BM25 scores from RRF retriever's BM25 component
                if _rrf_retriever is not None:
                    bm25_scores = _rrf_retriever.bm25.get_all_scores(query)
                else:
                    bm25_scores = {}
                # Simple RRF fusion of HyDE and BM25
                from hybrid_fusion import reciprocal_rank_fusion
                hyde_ranked = sorted(hyde_scores.items(), key=lambda x: x[1], reverse=True)
                bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
                fused = reciprocal_rank_fusion(
                    [hyde_ranked, bm25_ranked],
                    weights=[1.0, 0.8],
                    k=60
                )
                candidate_cols = [doc_id for doc_id, _ in fused[:K_RETRIEVE]]
                retrieval_scores = {doc_id: score for doc_id, score in fused[:K_RETRIEVE]}
                retrieved_cols = candidate_cols

            elif method.mode == "colbert_only":
                # ColBERT retrieval only
                candidate_cols, retrieval_scores = colbert_retrieve(query, K_RETRIEVE, use_hybrid=False)
                retrieved_cols = candidate_cols

            elif method.mode == "colbert_bm25":
                # ColBERT + BM25 hybrid retrieval
                candidate_cols, retrieval_scores = colbert_retrieve(query, K_RETRIEVE, use_hybrid=True)
                retrieved_cols = candidate_cols

            elif method.mode == "crag_semantic":
                # CRAG wrapping semantic retrieval
                candidate_cols, retrieval_scores, crag_meta = crag_retrieve(
                    query, K_RETRIEVE, mode="semantic",
                    retriever=retriever, descriptions=descriptions, llm=_hf_llm
                )
                retrieved_cols = candidate_cols
                rrf_analysis = crag_meta  # Store CRAG metadata

            elif method.mode == "crag_rrf":
                # CRAG wrapping RRF retrieval
                candidate_cols, retrieval_scores, crag_meta = crag_retrieve(
                    query, K_RETRIEVE, mode="rrf",
                    retriever=retriever, descriptions=descriptions, llm=_hf_llm
                )
                retrieved_cols = candidate_cols
                rrf_analysis = crag_meta  # Store CRAG metadata

            elif method.mode == "advanced_full":
                # Advanced retriever: HyDE + ColBERT + BM25 + CRAG (no Markov)
                candidate_cols, retrieval_scores, adv_meta = advanced_retrieve(
                    query, K_RETRIEVE, use_markov=False
                )
                retrieved_cols = candidate_cols
                rrf_analysis = adv_meta  # Store advanced metadata

            elif method.mode == "advanced_markov":
                # Advanced retriever with Markov blanket
                candidate_cols, retrieval_scores, adv_meta = advanced_retrieve(
                    query, K_RETRIEVE, use_markov=True
                )
                retrieved_cols = candidate_cols
                rrf_analysis = adv_meta  # Store advanced metadata

            # Use attention-optimized prompt for reranked methods
            use_optimized_prompt = method.mode in {"rrf_reranked", "rag_reranked"}

            if use_optimized_prompt and retrieval_scores:
                # Build prompt with attention ordering
                scores_list = [retrieval_scores.get(c, 0.0) for c in candidate_cols]
                prompt = build_optimized_prompt(
                    query=query,
                    columns=candidate_cols,
                    scores=scores_list,
                    descriptions=descriptions,
                    use_attention_ordering=True,
                )
            else:
                prompt = build_prompt(
                    query=query,
                    columns=candidate_cols,
                    descriptions=descriptions if method.mode != "simple" else None,
                    graph_edges=edges if method.mode == "graph" else None,
                )

            llm = call_llm(prompt)
            response_text = llm["response"].strip()
            found_cols = parse_column_list(response_text, all_cols)

            classification = classify(found_cols, gold)
            metrics = compute_metrics(found_cols, gold)

            # Compute leakage metrics if leakage columns are available
            leakage_metrics = None
            if leakage_cols:
                leakage_metrics = compute_leakage_metrics(found_cols, gold, leakage_cols)
                leakage_info = f" | leakage {leakage_metrics['leakage_count']}/{len(leakage_cols)}"
            else:
                leakage_info = ""

            print(
                f"  -> {classification.upper()} | found {len(found_cols)} cols "
                f"| gold hit {len(set(found_cols) & set(gold))}/{len(gold)}{leakage_info}"
            )

            by_method[method.key]["classification"][classification] += 1
            by_method[method.key]["metrics_list"].append(metrics)

            # Track leakage metrics for aggregation
            if leakage_metrics and "leakage_list" not in by_method[method.key]:
                by_method[method.key]["leakage_list"] = []
            if leakage_metrics:
                by_method[method.key]["leakage_list"].append(leakage_metrics)

            q_entry["methods"][method.key] = {
                "prompt_columns": candidate_cols,
                "retrieved_columns": retrieved_cols,
                "retrieval_scores": retrieval_scores,
                "rrf_analysis": rrf_analysis,
                "response": response_text,
                "found_columns": found_cols,
                "classification": classification,
                "metrics": metrics,
                "leakage_metrics": leakage_metrics,
                "tokens": {
                    "prompt": llm["prompt_tokens"],
                    "output": llm["output_tokens"],
                },
            }

        results["queries"].append(q_entry)

    # Aggregate metrics
    summary = {
        "total_queries": len(queries),
        "by_method": {},
    }
    for method in METHODS:
        metrics_list = by_method[method.key]["metrics_list"]
        if metrics_list:
            avg = {
                "precision": sum(m["precision"] for m in metrics_list) / len(metrics_list),
                "recall": sum(m["recall"] for m in metrics_list) / len(metrics_list),
                "f1": sum(m["f1"] for m in metrics_list) / len(metrics_list),
                "jaccard": sum(m["jaccard"] for m in metrics_list) / len(metrics_list),
                "exact": sum(m["exact"] for m in metrics_list) / len(metrics_list),
            }
        else:
            avg = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0, "exact": 0.0}

        # Aggregate leakage metrics if available
        leakage_avg = None
        leakage_list = by_method[method.key].get("leakage_list", [])
        if leakage_list:
            leakage_avg = {
                "avg_leakage_count": sum(m["leakage_count"] for m in leakage_list) / len(leakage_list),
                "avg_leakage_rate": sum(m["leakage_rate"] for m in leakage_list) / len(leakage_list),
                "avg_leakage_avoidance": sum(m["leakage_avoidance"] for m in leakage_list) / len(leakage_list),
            }

        summary["by_method"][method.key] = {
            "classification": by_method[method.key]["classification"],
            "metrics": avg,
            "leakage_metrics": leakage_avg,
        }

    results["summary"] = summary

    report = render_report(summary)

    # Write outputs
    out_json = OUT_DIR / "benchmark_v3_results.json"
    out_report = OUT_DIR / "benchmark_v3_report.txt"
    out_json.write_text(json.dumps(results, indent=2))
    out_report.write_text(report + "\n")

    print(report)
    print("")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_report}")


if __name__ == "__main__":
    main()
