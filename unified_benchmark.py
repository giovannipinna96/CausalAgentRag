"""
unified_benchmark.py

Unified benchmark runner for CausalAgentRag.
Evaluates all retrieval methods on the same task set using schema-level metrics.

Usage:
    uv run unified_benchmark.py                     # Run all methods
    uv run unified_benchmark.py --methods RAG_HYBRID RAG_SEMANTIC  # Specific methods
    uv run unified_benchmark.py --budget 10         # Custom budget
    uv run unified_benchmark.py --no-llm            # Skip LLM methods (faster)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
import time

import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path(__file__).parent / "data"
BENCHMARK_PATH = DATA_DIR / "causal_rag_benchmark_v3.json"
PATIENTS_PATH = DATA_DIR / "HEPAR_simulated_patients.csv"
ARCS_PATH = DATA_DIR / "HEPAR_arcs.csv"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    id: str
    query: str
    gold_columns: List[str]
    leakage_columns: List[str]
    target: Optional[str] = None
    reasoning: str = ""
    trap_severity: str = "none"


@dataclass
class TaskResult:
    """Results for a single task-method pair."""
    task_id: str
    method: str
    query: str
    selected_columns: List[str]
    precision: float
    recall: float
    f1: float
    leakage_rate: float
    label: str  # "correct", "partial", "wrong"
    runtime_ms: float = 0.0


def load_benchmark(path: Path) -> List[BenchmarkTask]:
    """Load benchmark tasks from JSON."""
    with open(path, "r") as f:
        data = json.load(f)

    tasks = []
    for q in data["queries"]:
        tasks.append(BenchmarkTask(
            id=q["id"],
            query=q["query"],
            gold_columns=q["gold_columns"],
            leakage_columns=q.get("leakage_columns", []),
            target=q.get("target"),
            reasoning=q.get("reasoning", ""),
            trap_severity=q.get("trap_severity", "none"),
        ))
    return tasks


def compute_metrics(
    retrieved: List[str],
    gold: List[str],
    leakage: List[str],
) -> Dict[str, float]:
    """Compute schema-level metrics."""
    retrieved_set = set(retrieved)
    gold_set = set(gold)
    leakage_set = set(leakage)

    # Precision and recall
    hits = retrieved_set & gold_set
    precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(hits) / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Leakage rate
    leaked = retrieved_set & leakage_set
    leakage_rate = len(leaked) / len(retrieved_set) if retrieved_set else 0.0

    # Label assignment
    if gold_set <= retrieved_set and len(leaked) == 0:
        label = "correct"
    elif len(hits) > 0 and len(leaked) == 0:
        label = "partial"
    else:
        label = "wrong"

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "leakage_rate": leakage_rate,
        "label": label,
    }


# =============================================================================
# Method Implementations
# =============================================================================

class MethodRegistry:
    """Registry of all benchmark methods."""

    def __init__(self):
        self.methods: Dict[str, Callable] = {}
        self._initialized: Dict[str, bool] = {}
        self._resources: Dict[str, object] = {}

    def register(self, name: str, fn: Callable):
        self.methods[name] = fn

    def list_methods(self) -> List[str]:
        return list(self.methods.keys())

    def run(self, name: str, query: str, budget: int, **kwargs) -> List[str]:
        if name not in self.methods:
            raise ValueError(f"Unknown method: {name}")
        return self.methods[name](query, budget, **kwargs)


REGISTRY = MethodRegistry()


# --- v3 Methods ---

def _get_v3_semantic_retriever(nodes, cols):
    """Lazy init for v3 semantic retriever."""
    from semantic_retriever_v3 import SemanticSchemaRetriever
    return SemanticSchemaRetriever(
        nodes=nodes,
        embed_model="BAAI/bge-small-en-v1.5",
        vector_top_k=24,
        bm25_top_k=24,
        alpha_dense=0.6,
    )


def _get_v3_hybrid_retriever(nodes, semantic, graph):
    """Lazy init for v3 hybrid retriever."""
    from causal_hybrid_retriever_v3 import CausalHybridSchemaRetriever
    return CausalHybridSchemaRetriever(
        nodes=nodes,
        semantic=semantic,
        graph=graph,
        alpha_semantic=0.55,
        alpha_graph=0.45,
        enable_ppr=True,
        enable_backdoor=True,
        enable_parent_heuristic=True,
    )


def run_rag_semantic(query: str, budget: int, **kwargs) -> List[str]:
    """v3 RAG_SEMANTIC: Dense + BM25 fusion."""
    semantic = kwargs["semantic_retriever"]
    results = semantic.retrieve(query, top_k=budget)
    return [r.node.node_id for r in results]


def run_rag_hybrid(query: str, budget: int, **kwargs) -> List[str]:
    """v3 RAG_HYBRID: Semantic + PPR + causal constraints."""
    hybrid = kwargs["hybrid_retriever"]
    results = hybrid.retrieve(query, budget=budget)
    return [r.node.node_id for r in results]


# --- Original Methods ---

def run_semantic_only(query: str, budget: int, **kwargs) -> List[str]:
    """Original SemanticOnlyRetriever."""
    from retrievers import SemanticOnlyRetriever
    cols = kwargs["cols"]
    retriever = SemanticOnlyRetriever(cols)
    return retriever.retrieve(query, k=budget)


def run_threshold_hybrid(query: str, budget: int, **kwargs) -> List[str]:
    """Original ThresholdHybridRetriever."""
    from retrievers import ThresholdHybridRetriever, ThresholdHybridConfig
    cols = kwargs["cols"]
    nx_graph = kwargs["nx_graph"]
    retriever = ThresholdHybridRetriever(cols, nx_graph)
    config = ThresholdHybridConfig(k=budget, semantic_threshold=0.4, max_depth=2)
    return retriever.retrieve(query, config)


def run_rrf_hybrid(query: str, budget: int, **kwargs) -> List[str]:
    """RRFHybridRetriever with Markov Blanket."""
    from retrievers import RRFHybridRetriever, RRFHybridConfig
    cols = kwargs["cols"]
    nx_digraph = kwargs["nx_digraph"]
    col_descriptions = kwargs.get("col_descriptions", {})

    retriever = RRFHybridRetriever(
        columns=cols,
        graph=nx_digraph,
        column_descriptions=col_descriptions,
    )
    config = RRFHybridConfig(k=budget, causal_mode='parents_only')
    return retriever.retrieve(query, config)


# --- LLM Methods ---

def run_llm_simple(query: str, budget: int, **kwargs) -> List[str]:
    """LLM_SIMPLE: Query + column names only."""
    selector = kwargs["llm_selector"]
    cols = kwargs["cols"]
    cache_key = f"bench::simple::{query[:50]}"
    return selector.select_simple(query, cols, budget, cache_key)


def run_llm_description(query: str, budget: int, **kwargs) -> List[str]:
    """LLM_DESCRIPTION: Query + names + descriptions."""
    selector = kwargs["llm_selector"]
    cols = kwargs["cols"]
    col_descriptions = kwargs["col_descriptions"]
    cache_key = f"bench::desc::{query[:50]}"
    return selector.select_description(query, col_descriptions, cols, budget, cache_key)


def run_llm_graph(query: str, budget: int, **kwargs) -> List[str]:
    """LLM_GRAPH: Query + names + descriptions + graph edges."""
    selector = kwargs["llm_selector"]
    cols = kwargs["cols"]
    col_descriptions = kwargs["col_descriptions"]
    edges = kwargs["edges"]
    cache_key = f"bench::graph::{query[:50]}"
    return selector.select_graph(query, col_descriptions, cols, edges, budget, cache_key)


# Register all methods
REGISTRY.register("RAG_SEMANTIC", run_rag_semantic)
REGISTRY.register("RAG_HYBRID", run_rag_hybrid)
REGISTRY.register("SemanticOnly", run_semantic_only)
REGISTRY.register("ThresholdHybrid", run_threshold_hybrid)
REGISTRY.register("RRFHybrid", run_rrf_hybrid)
REGISTRY.register("LLM_SIMPLE", run_llm_simple)
REGISTRY.register("LLM_DESCRIPTION", run_llm_description)
REGISTRY.register("LLM_GRAPH", run_llm_graph)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    tasks: List[BenchmarkTask],
    methods: List[str],
    budget: int,
    include_llm: bool = True,
    output_dir: Path = Path("."),
) -> List[TaskResult]:
    """Run benchmark and return results."""

    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(PATIENTS_PATH)
    cols = list(df.columns)
    arcs = pd.read_csv(ARCS_PATH)

    # Build graphs
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(cols)
    for _, row in arcs.iterrows():
        nx_graph.add_edge(row["from"], row["to"])

    nx_digraph = nx.DiGraph()
    nx_digraph.add_nodes_from(cols)
    for _, row in arcs.iterrows():
        nx_digraph.add_edge(row["from"], row["to"])

    edges = [(row["from"], row["to"]) for _, row in arcs.iterrows()]

    # Build schema nodes for v3 methods
    from schema_v3 import build_column_infos, build_schema_nodes, HEPAR_SYNONYMS
    infos = build_column_infos(df, synonyms=HEPAR_SYNONYMS)
    nodes, _ = build_schema_nodes(infos)
    col_descriptions = {col: infos[col].description for col in cols}

    # Build v3 causal graph
    from causal_graph_v3 import CausalGraph
    causal_graph = CausalGraph.from_arcs_csv(str(ARCS_PATH), nodes=cols)

    # Initialize v3 retrievers
    logger.info("Initializing v3 retrievers...")
    semantic_retriever = _get_v3_semantic_retriever(nodes, cols)
    hybrid_retriever = _get_v3_hybrid_retriever(nodes, semantic_retriever, causal_graph)

    # Initialize LLM selector if needed
    llm_selector = None
    llm_methods = {"LLM_SIMPLE", "LLM_DESCRIPTION", "LLM_GRAPH"}
    if include_llm and any(m in llm_methods for m in methods):
        logger.info("Initializing LLM...")
        from llm_baselines_v3 import HuggingFaceLLMWrapper, LLMColumnSelector, LLMConfig
        config = LLMConfig(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            temperature=0.1,
            max_new_tokens=256,
        )
        llm = HuggingFaceLLMWrapper(config)
        llm_selector = LLMColumnSelector(llm=llm, cache_path=str(output_dir / "llm_cache.json"))

    # Prepare kwargs for methods
    kwargs = {
        "cols": cols,
        "nx_graph": nx_graph,
        "nx_digraph": nx_digraph,
        "col_descriptions": col_descriptions,
        "edges": edges,
        "nodes": nodes,
        "semantic_retriever": semantic_retriever,
        "hybrid_retriever": hybrid_retriever,
        "llm_selector": llm_selector,
    }

    # Run benchmark
    results: List[TaskResult] = []
    total = len(tasks) * len(methods)
    completed = 0

    for task in tasks:
        for method in methods:
            # Skip LLM methods if not included
            if method in llm_methods and not include_llm:
                continue

            logger.info(f"[{completed+1}/{total}] {method} on {task.id}")

            try:
                start = time.time()
                selected = REGISTRY.run(method, task.query, budget, **kwargs)
                runtime_ms = (time.time() - start) * 1000
            except Exception as e:
                logger.error(f"Error running {method} on {task.id}: {e}")
                selected = cols[:budget]  # fallback
                runtime_ms = 0.0

            metrics = compute_metrics(selected, task.gold_columns, task.leakage_columns)

            results.append(TaskResult(
                task_id=task.id,
                method=method,
                query=task.query,
                selected_columns=selected,
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                leakage_rate=metrics["leakage_rate"],
                label=metrics["label"],
                runtime_ms=runtime_ms,
            ))

            completed += 1

    return results


def save_results(results: List[TaskResult], output_dir: Path):
    """Save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Results CSV
    results_path = output_dir / "results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "method", "query", "selected_columns",
            "precision", "recall", "f1", "leakage_rate", "label", "runtime_ms"
        ])
        for r in results:
            writer.writerow([
                r.task_id, r.method, r.query[:50] + "...",
                ";".join(r.selected_columns),
                f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}",
                f"{r.leakage_rate:.4f}", r.label, f"{r.runtime_ms:.1f}"
            ])
    logger.info(f"Results saved to {results_path}")

    # Summary CSV
    summary_path = output_dir / "summary.csv"
    method_stats: Dict[str, Dict] = {}
    for r in results:
        if r.method not in method_stats:
            method_stats[r.method] = {
                "precision": [], "recall": [], "f1": [], "leakage": [],
                "correct": 0, "partial": 0, "wrong": 0
            }
        stats = method_stats[r.method]
        stats["precision"].append(r.precision)
        stats["recall"].append(r.recall)
        stats["f1"].append(r.f1)
        stats["leakage"].append(r.leakage_rate)
        stats[r.label] += 1

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "mean_precision", "mean_recall", "mean_f1", "mean_leakage",
            "n_correct", "n_partial", "n_wrong", "n_tasks"
        ])
        for method, stats in sorted(method_stats.items()):
            n = len(stats["precision"])
            writer.writerow([
                method,
                f"{sum(stats['precision'])/n:.4f}",
                f"{sum(stats['recall'])/n:.4f}",
                f"{sum(stats['f1'])/n:.4f}",
                f"{sum(stats['leakage'])/n:.4f}",
                stats["correct"], stats["partial"], stats["wrong"], n
            ])
    logger.info(f"Summary saved to {summary_path}")


def print_summary(results: List[TaskResult]):
    """Print summary to console."""
    method_stats: Dict[str, Dict] = {}
    for r in results:
        if r.method not in method_stats:
            method_stats[r.method] = {
                "precision": [], "recall": [], "f1": [], "leakage": [],
                "correct": 0, "partial": 0, "wrong": 0
            }
        stats = method_stats[r.method]
        stats["precision"].append(r.precision)
        stats["recall"].append(r.recall)
        stats["f1"].append(r.f1)
        stats["leakage"].append(r.leakage_rate)
        stats[r.label] += 1

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Leak':>8} {'Corr':>6} {'Part':>6} {'Wrong':>6}")
    print("-" * 80)

    for method in sorted(method_stats.keys()):
        stats = method_stats[method]
        n = len(stats["precision"])
        print(f"{method:<20} "
              f"{sum(stats['precision'])/n:>8.3f} "
              f"{sum(stats['recall'])/n:>8.3f} "
              f"{sum(stats['f1'])/n:>8.3f} "
              f"{sum(stats['leakage'])/n:>8.3f} "
              f"{stats['correct']:>6} "
              f"{stats['partial']:>6} "
              f"{stats['wrong']:>6}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark for CausalAgentRag")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run (default: all)")
    parser.add_argument("--budget", type=int, default=10,
                        help="Column budget (default: 10)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM methods (faster)")
    parser.add_argument("--output", type=str, default="benchmark_results",
                        help="Output directory (default: benchmark_results)")
    parser.add_argument("--list-methods", action="store_true",
                        help="List available methods and exit")

    args = parser.parse_args()

    if args.list_methods:
        print("Available methods:")
        for m in REGISTRY.list_methods():
            print(f"  - {m}")
        return

    # Load tasks
    tasks = load_benchmark(BENCHMARK_PATH)
    logger.info(f"Loaded {len(tasks)} benchmark tasks")

    # Determine methods
    all_methods = REGISTRY.list_methods()
    if args.methods:
        methods = [m for m in args.methods if m in all_methods]
        if not methods:
            logger.error(f"No valid methods specified. Available: {all_methods}")
            return
    else:
        methods = all_methods

    # Filter LLM methods if requested
    if args.no_llm:
        methods = [m for m in methods if not m.startswith("LLM_")]

    logger.info(f"Running methods: {methods}")
    logger.info(f"Budget: {args.budget}")

    # Run benchmark
    output_dir = Path(args.output)
    results = run_benchmark(
        tasks=tasks,
        methods=methods,
        budget=args.budget,
        include_llm=not args.no_llm,
        output_dir=output_dir,
    )

    # Save and print results
    save_results(results, output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
