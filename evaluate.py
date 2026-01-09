#!/usr/bin/env python3
"""
Evaluation Module for CausalAgentRag Column Selection

This module evaluates the quality of column selection by comparing
system predictions against ground truth annotations.

Metrics computed:
- Precision: Fraction of predicted columns that are relevant
- Recall: Fraction of relevant columns that were predicted
- F1 Score: Harmonic mean of precision and recall
- Jaccard Similarity: Intersection over union
- Exact Match: Whether prediction exactly matches ground truth

Usage:
    # Evaluate predictions against benchmark
    uv run evaluate.py --predictions results.json --benchmark data/benchmark.json

    # Evaluate with detailed output
    uv run evaluate.py --predictions results.json --benchmark data/benchmark.json --verbose

    # Output results to JSON file
    uv run evaluate.py --predictions results.json --benchmark data/benchmark.json --output evaluation_results.json
"""

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""

    query_id: str
    query: str
    precision: float
    recall: float
    f1_score: float
    jaccard: float
    exact_match: bool
    predicted_columns: list[str]
    expected_columns: list[str]
    true_positives: list[str]
    false_positives: list[str]
    false_negatives: list[str]
    # New classification metrics
    classification: str  # "correct", "partial", "wrong"


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries."""

    num_queries: int
    mean_precision: float
    mean_recall: float
    mean_f1_score: float
    mean_jaccard: float
    exact_match_rate: float
    std_precision: float
    std_recall: float
    std_f1_score: float
    std_jaccard: float
    # New classification counts
    num_correct: int  # Exact match (predicted == gold_columns)
    num_partial: int  # All gold found but extra columns (recall=1.0, precision<1.0)
    num_wrong: int  # Missing at least 1 gold column (recall<1.0)


def compute_precision(predicted: set, expected: set) -> float:
    """
    Compute precision: TP / (TP + FP).

    Args:
        predicted: Set of predicted columns
        expected: Set of expected columns

    Returns:
        Precision score (0.0 to 1.0)
    """
    if len(predicted) == 0:
        return 0.0
    true_positives = len(predicted & expected)
    return true_positives / len(predicted)


def compute_recall(predicted: set, expected: set) -> float:
    """
    Compute recall: TP / (TP + FN).

    Args:
        predicted: Set of predicted columns
        expected: Set of expected columns

    Returns:
        Recall score (0.0 to 1.0)
    """
    if len(expected) == 0:
        return 1.0  # If no expected columns, recall is perfect
    true_positives = len(predicted & expected)
    return true_positives / len(expected)


def compute_f1_score(precision: float, recall: float) -> float:
    """
    Compute F1 score: 2 * (precision * recall) / (precision + recall).

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_jaccard(predicted: set, expected: set) -> float:
    """
    Compute Jaccard similarity: |intersection| / |union|.

    Args:
        predicted: Set of predicted columns
        expected: Set of expected columns

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    intersection = len(predicted & expected)
    union = len(predicted | expected)
    if union == 0:
        return 1.0  # Both sets empty
    return intersection / union


def compute_exact_match(predicted: set, expected: set) -> bool:
    """
    Check if prediction exactly matches expected.

    Args:
        predicted: Set of predicted columns
        expected: Set of expected columns

    Returns:
        True if sets are equal
    """
    return predicted == expected


def classify_prediction(predicted_set: set, expected_set: set) -> str:
    """
    Classify prediction into correct, partial, or wrong.

    Args:
        predicted_set: Set of predicted columns
        expected_set: Set of expected columns

    Returns:
        "correct": Exact match (predicted == expected)
        "partial": All expected found but extra columns (recall=1.0, precision<1.0)
        "wrong": Missing at least 1 expected column (recall<1.0)
    """
    if predicted_set == expected_set:
        return "correct"
    elif expected_set.issubset(predicted_set):
        # All gold columns found, but extra columns predicted
        return "partial"
    else:
        # Missing at least 1 gold column
        return "wrong"


def evaluate_query(
    query_id: str,
    query: str,
    predicted_columns: list[str],
    expected_columns: list[str],
) -> QueryMetrics:
    """
    Evaluate predictions for a single query.

    Args:
        query_id: Query identifier
        query: Query text
        predicted_columns: List of predicted column names
        expected_columns: List of expected column names

    Returns:
        QueryMetrics with all computed metrics
    """
    predicted_set = set(predicted_columns)
    expected_set = set(expected_columns)

    # Compute metrics
    precision = compute_precision(predicted_set, expected_set)
    recall = compute_recall(predicted_set, expected_set)
    f1 = compute_f1_score(precision, recall)
    jaccard = compute_jaccard(predicted_set, expected_set)
    exact_match = compute_exact_match(predicted_set, expected_set)

    # Compute TP, FP, FN for detailed analysis
    true_positives = list(predicted_set & expected_set)
    false_positives = list(predicted_set - expected_set)
    false_negatives = list(expected_set - predicted_set)

    # Classify prediction
    classification = classify_prediction(predicted_set, expected_set)

    return QueryMetrics(
        query_id=query_id,
        query=query,
        precision=precision,
        recall=recall,
        f1_score=f1,
        jaccard=jaccard,
        exact_match=exact_match,
        predicted_columns=predicted_columns,
        expected_columns=expected_columns,
        true_positives=sorted(true_positives),
        false_positives=sorted(false_positives),
        false_negatives=sorted(false_negatives),
        classification=classification,
    )


def compute_aggregate_metrics(query_metrics: list[QueryMetrics]) -> AggregateMetrics:
    """
    Compute aggregate metrics across all queries.

    Args:
        query_metrics: List of QueryMetrics for each query

    Returns:
        AggregateMetrics with mean and std for all metrics
    """
    import statistics

    n = len(query_metrics)
    if n == 0:
        raise ValueError("No queries to evaluate")

    precisions = [m.precision for m in query_metrics]
    recalls = [m.recall for m in query_metrics]
    f1_scores = [m.f1_score for m in query_metrics]
    jaccards = [m.jaccard for m in query_metrics]
    exact_matches = [1 if m.exact_match else 0 for m in query_metrics]

    # Compute standard deviations (handle single element case)
    std_precision = statistics.stdev(precisions) if n > 1 else 0.0
    std_recall = statistics.stdev(recalls) if n > 1 else 0.0
    std_f1 = statistics.stdev(f1_scores) if n > 1 else 0.0
    std_jaccard = statistics.stdev(jaccards) if n > 1 else 0.0

    # Count classifications
    num_correct = sum(1 for m in query_metrics if m.classification == "correct")
    num_partial = sum(1 for m in query_metrics if m.classification == "partial")
    num_wrong = sum(1 for m in query_metrics if m.classification == "wrong")

    return AggregateMetrics(
        num_queries=n,
        mean_precision=statistics.mean(precisions),
        mean_recall=statistics.mean(recalls),
        mean_f1_score=statistics.mean(f1_scores),
        mean_jaccard=statistics.mean(jaccards),
        exact_match_rate=statistics.mean(exact_matches),
        std_precision=std_precision,
        std_recall=std_recall,
        std_f1_score=std_f1,
        std_jaccard=std_jaccard,
        num_correct=num_correct,
        num_partial=num_partial,
        num_wrong=num_wrong,
    )


def load_benchmark(benchmark_path: Path) -> dict:
    """
    Load benchmark file with ground truth.

    Args:
        benchmark_path: Path to benchmark JSON file

    Returns:
        Dictionary with benchmark data

    Raises:
        FileNotFoundError: If benchmark file doesn't exist
    """
    logger.info(f"Loading benchmark from {benchmark_path}")

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with open(benchmark_path, encoding="utf-8") as f:
        benchmark = json.load(f)

    logger.info(f"Loaded {len(benchmark['queries'])} benchmark queries")
    return benchmark


def load_predictions(predictions_path: Path) -> dict:
    """
    Load predictions file.

    Expected format:
    {
        "model": "...",
        "mode": "...",
        "results": [
            {"query_id": "q1", "query": "...", "predicted_columns": [...]}
        ]
    }

    Args:
        predictions_path: Path to predictions JSON file

    Returns:
        Dictionary with predictions data

    Raises:
        FileNotFoundError: If predictions file doesn't exist
    """
    logger.info(f"Loading predictions from {predictions_path}")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    logger.info(f"Loaded {len(predictions['results'])} predictions")
    return predictions


def evaluate_predictions(
    predictions: dict,
    benchmark: dict,
) -> tuple[list[QueryMetrics], AggregateMetrics]:
    """
    Evaluate all predictions against benchmark.

    Args:
        predictions: Dictionary with prediction results
        benchmark: Dictionary with benchmark ground truth

    Returns:
        Tuple of (list of QueryMetrics, AggregateMetrics)
    """
    logger.info("Evaluating predictions...")

    # Build lookup for benchmark queries
    benchmark_lookup = {q["id"]: q for q in benchmark["queries"]}

    query_metrics = []
    for pred in predictions["results"]:
        query_id = pred["query_id"]

        if query_id not in benchmark_lookup:
            logger.warning(f"Query {query_id} not found in benchmark, skipping")
            continue

        bench_query = benchmark_lookup[query_id]

        # Support both formats: expected_columns (v1) and gold_columns (v3)
        expected_columns = bench_query.get("expected_columns") or bench_query.get("gold_columns", [])

        metrics = evaluate_query(
            query_id=query_id,
            query=pred["query"],
            predicted_columns=pred["predicted_columns"],
            expected_columns=expected_columns,
        )
        query_metrics.append(metrics)

    # Compute aggregate metrics
    aggregate = compute_aggregate_metrics(query_metrics)

    logger.info(f"Evaluated {len(query_metrics)} queries")
    return query_metrics, aggregate


def print_results(
    query_metrics: list[QueryMetrics],
    aggregate: AggregateMetrics,
    predictions_meta: dict,
    verbose: bool = False,
) -> None:
    """
    Print evaluation results to console.

    Args:
        query_metrics: List of per-query metrics
        aggregate: Aggregate metrics
        predictions_meta: Metadata from predictions file
        verbose: If True, print detailed per-query results
    """
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Print metadata
    print(f"\nModel: {predictions_meta.get('model', 'N/A')}")
    print(f"Mode: {predictions_meta.get('mode', 'N/A')}")
    print(f"Queries Evaluated: {aggregate.num_queries}")

    # Print classification summary
    print("\n" + "-" * 70)
    print("CLASSIFICATION SUMMARY")
    print("-" * 70)
    print(f"{'Correct (exact match):':<30} {aggregate.num_correct:>5}/{aggregate.num_queries} ({100*aggregate.num_correct/aggregate.num_queries:.1f}%)")
    print(f"{'Partial (all gold + extra):':<30} {aggregate.num_partial:>5}/{aggregate.num_queries} ({100*aggregate.num_partial/aggregate.num_queries:.1f}%)")
    print(f"{'Wrong (missing gold cols):':<30} {aggregate.num_wrong:>5}/{aggregate.num_queries} ({100*aggregate.num_wrong/aggregate.num_queries:.1f}%)")

    # Print aggregate metrics
    print("\n" + "-" * 70)
    print("AGGREGATE METRICS")
    print("-" * 70)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 40)
    print(f"{'Precision':<20} {aggregate.mean_precision:>10.4f} {aggregate.std_precision:>10.4f}")
    print(f"{'Recall':<20} {aggregate.mean_recall:>10.4f} {aggregate.std_recall:>10.4f}")
    print(f"{'F1 Score':<20} {aggregate.mean_f1_score:>10.4f} {aggregate.std_f1_score:>10.4f}")
    print(f"{'Jaccard':<20} {aggregate.mean_jaccard:>10.4f} {aggregate.std_jaccard:>10.4f}")
    print(f"{'Exact Match Rate':<20} {aggregate.exact_match_rate:>10.4f}")

    if verbose:
        print("\n" + "-" * 70)
        print("PER-QUERY RESULTS")
        print("-" * 70)

        for m in query_metrics:
            # Classification indicator
            if m.classification == "correct":
                indicator = "[CORRECT]"
            elif m.classification == "partial":
                indicator = "[PARTIAL]"
            else:
                indicator = "[WRONG]"

            print(f"\n{indicator} [{m.query_id}] {m.query}")
            print(f"  Precision: {m.precision:.4f}, Recall: {m.recall:.4f}, F1: {m.f1_score:.4f}, Jaccard: {m.jaccard:.4f}")
            print(f"  Predicted: {m.predicted_columns}")
            print(f"  Expected:  {m.expected_columns}")
            if m.true_positives:
                print(f"  TP (correct): {m.true_positives}")
            if m.false_positives:
                print(f"  FP (extra):   {m.false_positives}")
            if m.false_negatives:
                print(f"  FN (missed):  {m.false_negatives}")

    print("\n" + "=" * 70)


def save_results(
    query_metrics: list[QueryMetrics],
    aggregate: AggregateMetrics,
    predictions_meta: dict,
    output_path: Path,
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        query_metrics: List of per-query metrics
        aggregate: Aggregate metrics
        predictions_meta: Metadata from predictions file
        output_path: Path to output JSON file
    """
    logger.info(f"Saving results to {output_path}")

    results = {
        "metadata": {
            "model": predictions_meta.get("model", "N/A"),
            "mode": predictions_meta.get("mode", "N/A"),
            "num_queries": aggregate.num_queries,
        },
        "aggregate_metrics": asdict(aggregate),
        "per_query_metrics": [asdict(m) for m in query_metrics],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate column selection predictions against benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  uv run evaluate.py --predictions results.json --benchmark data/benchmark.json

  # Verbose output with per-query details
  uv run evaluate.py --predictions results.json --benchmark data/benchmark.json --verbose

  # Save results to JSON
  uv run evaluate.py --predictions results.json --benchmark data/benchmark.json --output eval_results.json
        """,
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSON file (output from llm_huggingface.py)",
    )

    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmark.json"),
        help="Path to benchmark JSON file with ground truth. Default: data/benchmark.json",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results as JSON. If not provided, results are only printed.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-query results",
    )

    return parser.parse_args()


def main() -> None:
    """Main function for evaluation."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Column Selection Evaluation")
    logger.info("=" * 60)

    # Load data
    benchmark = load_benchmark(args.benchmark)
    predictions = load_predictions(args.predictions)

    # Evaluate
    query_metrics, aggregate = evaluate_predictions(predictions, benchmark)

    # Print results
    print_results(
        query_metrics=query_metrics,
        aggregate=aggregate,
        predictions_meta=predictions,
        verbose=args.verbose,
    )

    # Save results if output path provided
    if args.output:
        save_results(
            query_metrics=query_metrics,
            aggregate=aggregate,
            predictions_meta=predictions,
            output_path=args.output,
        )

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
