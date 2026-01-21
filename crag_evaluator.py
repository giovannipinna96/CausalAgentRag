"""
CRAG (Corrective Retrieval-Augmented Generation) for self-correcting retrieval.

CRAG evaluates retrieval quality BEFORE sending results to the LLM.
If confidence is low, it refines the query and retries retrieval.

Key insight: Low-quality retrieval leads to poor LLM outputs.
Better to detect and fix retrieval issues early than to propagate errors.

Reference: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CRAGConfig:
    """Configuration for Corrective RAG."""
    high_threshold: float = 0.7  # Above this = high confidence
    low_threshold: float = 0.4   # Below this = low confidence, retry
    max_retries: int = 2  # Maximum query refinement attempts
    score_variance_weight: float = 0.3
    mention_coverage_weight: float = 0.4
    diversity_weight: float = 0.3
    min_top_score: float = 0.3  # Minimum score for top result


@dataclass
class ConfidenceResult:
    """Result from confidence evaluation."""
    level: str  # 'high' | 'medium' | 'low'
    score: float  # 0-1 confidence score
    metrics: Dict[str, float]  # Individual metric values
    reasons: List[str]  # Explanations for the confidence level


class RetrievalConfidenceEvaluator:
    """
    Evaluate confidence in retrieval results without ground truth.

    Uses multiple heuristic signals:
    1. Score distribution - uniform is suspicious, clear winners are good
    2. Query mention coverage - explicit mentions should be found
    3. Top score quality - very low top score is bad
    4. Score gap - big gap between top results is good
    """

    def __init__(self, config: Optional[CRAGConfig] = None):
        """Initialize evaluator."""
        self.config = config or CRAGConfig()

    def evaluate(
        self,
        query: str,
        retrieved_columns: List[str],
        retrieval_scores: Dict[str, float],
        column_descriptions: Optional[Dict[str, str]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate retrieval confidence.

        Args:
            query: Original query
            retrieved_columns: Retrieved column names
            retrieval_scores: Column -> score mapping
            column_descriptions: Optional column descriptions

        Returns:
            ConfidenceResult with level, score, and breakdown
        """
        metrics = {}
        reasons = []

        # Get scores for retrieved columns
        scores = [retrieval_scores.get(col, 0.0) for col in retrieved_columns]

        if not scores:
            return ConfidenceResult(
                level="low",
                score=0.0,
                metrics={"empty_results": 1.0},
                reasons=["No columns retrieved"]
            )

        # Metric 1: Top score quality
        top_score = max(scores)
        metrics["top_score"] = top_score
        if top_score < self.config.min_top_score:
            reasons.append(f"Top score too low ({top_score:.2f} < {self.config.min_top_score})")

        # Metric 2: Score variance (normalized)
        if len(scores) > 1:
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            # High variance relative to mean is good (clear differentiation)
            variance_ratio = score_std / (score_mean + 1e-6)
            metrics["score_variance"] = min(variance_ratio, 1.0)
        else:
            metrics["score_variance"] = 0.5

        # Metric 3: Query mention coverage
        mention_coverage = self._compute_mention_coverage(
            query, retrieved_columns, column_descriptions or {}
        )
        metrics["mention_coverage"] = mention_coverage
        if mention_coverage < 0.5:
            reasons.append("Query terms not well covered in results")

        # Metric 4: Score gap (between 1st and 2nd)
        if len(scores) > 1:
            sorted_scores = sorted(scores, reverse=True)
            gap = sorted_scores[0] - sorted_scores[1]
            metrics["score_gap"] = min(gap / (sorted_scores[0] + 1e-6), 1.0)
        else:
            metrics["score_gap"] = 1.0

        # Metric 5: Results diversity (not all from same category)
        diversity = self._compute_diversity(retrieved_columns, column_descriptions or {})
        metrics["diversity"] = diversity

        # Compute overall confidence score
        confidence_score = (
            self.config.score_variance_weight * metrics["score_variance"] +
            self.config.mention_coverage_weight * metrics["mention_coverage"] +
            self.config.diversity_weight * metrics["diversity"]
        )

        # Adjust by top score quality
        confidence_score *= min(1.0, top_score / self.config.min_top_score)

        # Determine level
        if confidence_score >= self.config.high_threshold:
            level = "high"
        elif confidence_score >= self.config.low_threshold:
            level = "medium"
        else:
            level = "low"
            if not reasons:
                reasons.append("Overall confidence score below threshold")

        return ConfidenceResult(
            level=level,
            score=confidence_score,
            metrics=metrics,
            reasons=reasons
        )

    def _compute_mention_coverage(
        self,
        query: str,
        columns: List[str],
        descriptions: Dict[str, str],
    ) -> float:
        """Check if query terms appear in retrieved columns."""
        # Extract key terms from query
        query_lower = query.lower()
        query_terms = set(re.findall(r'\b\w{3,}\b', query_lower))

        # Remove common words
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'will', 'are', 'can', 'which', 'want', 'patients', 'model', 'predict', 'build', 'identify'}
        query_terms -= stopwords

        if not query_terms:
            return 0.5  # Neutral if no specific terms

        # Check coverage in columns and descriptions
        found_terms = set()
        for col in columns:
            col_text = col.lower()
            desc_text = descriptions.get(col, "").lower()
            combined = f"{col_text} {desc_text}"

            for term in query_terms:
                if term in combined:
                    found_terms.add(term)

        return len(found_terms) / len(query_terms) if query_terms else 0.5

    def _compute_diversity(
        self,
        columns: List[str],
        descriptions: Dict[str, str],
    ) -> float:
        """Measure semantic diversity of results."""
        if len(columns) <= 1:
            return 1.0

        # Simple heuristic: count unique prefixes (column name patterns)
        prefixes = set()
        for col in columns:
            # Take first word or first 4 chars
            parts = col.split("_")
            if parts:
                prefixes.add(parts[0][:4])

        # More unique prefixes = more diverse
        diversity = len(prefixes) / len(columns)
        return min(diversity * 1.5, 1.0)  # Scale up slightly


class QueryRefiner:
    """
    Refine queries when retrieval confidence is low.

    Strategies:
    1. Add medical synonyms
    2. Expand abbreviations
    3. Make query more specific
    4. Use LLM for intelligent expansion (if available)
    """

    # Medical term expansions
    MEDICAL_EXPANSIONS = {
        "liver": ["hepatic", "hepato", "alt", "ast", "ggtp"],
        "hepatitis": ["viral", "hbv", "hcv", "hbsag", "chronic", "acute"],
        "cirrhosis": ["fibrosis", "scarring", "portal", "ascites"],
        "fatty": ["steatosis", "lipid", "fat", "obesity"],
        "enzyme": ["alt", "ast", "ggtp", "transaminase", "aminotransferase"],
        "risk": ["factor", "predisposing", "history", "exposure"],
        "blood": ["serum", "plasma", "hematology", "inr", "platelet"],
        "test": ["marker", "indicator", "measurement", "level"],
        "disease": ["condition", "disorder", "pathology"],
        "predict": ["risk", "develop", "progression", "outcome"],
    }

    def __init__(self, llm: Optional[Any] = None):
        """Initialize refiner with optional LLM."""
        self.llm = llm

    def refine(
        self,
        original_query: str,
        confidence_result: ConfidenceResult,
        attempt: int = 0,
    ) -> str:
        """
        Refine query based on confidence feedback.

        Args:
            original_query: Original query
            confidence_result: Confidence evaluation result
            attempt: Current attempt number (for varying strategies)

        Returns:
            Refined query
        """
        # Strategy varies by attempt
        if attempt == 0:
            # First attempt: add synonyms
            return self._expand_with_synonyms(original_query)
        elif attempt == 1:
            # Second attempt: use LLM if available, else more synonyms
            if self.llm is not None:
                return self._expand_with_llm(original_query, confidence_result)
            else:
                return self._expand_aggressive(original_query)
        else:
            # Later attempts: combine strategies
            return self._expand_aggressive(original_query)

    def _expand_with_synonyms(self, query: str) -> str:
        """Add medical synonyms to query."""
        query_lower = query.lower()
        additions = []

        for term, synonyms in self.MEDICAL_EXPANSIONS.items():
            if term in query_lower:
                # Add first 2 synonyms not already in query
                for syn in synonyms[:2]:
                    if syn not in query_lower:
                        additions.append(syn)

        if additions:
            return f"{query} (related: {', '.join(additions)})"
        return query

    def _expand_aggressive(self, query: str) -> str:
        """More aggressive expansion with multiple synonyms."""
        query_lower = query.lower()
        additions = []

        for term, synonyms in self.MEDICAL_EXPANSIONS.items():
            if term in query_lower:
                # Add all synonyms not already in query
                for syn in synonyms:
                    if syn not in query_lower and syn not in additions:
                        additions.append(syn)

        if additions:
            return f"{query} considering: {', '.join(additions[:5])}"
        return query

    def _expand_with_llm(
        self,
        query: str,
        confidence_result: ConfidenceResult,
    ) -> str:
        """Use LLM to intelligently expand query."""
        prompt = f"""The following medical database query returned low-confidence results:
Query: "{query}"

Issues identified:
{chr(10).join('- ' + r for r in confidence_result.reasons)}

Rewrite the query to be more specific about what medical features or measurements are needed.
Focus on medical terminology that would match database column names.

Return ONLY the refined query, nothing else."""

        try:
            refined = self.llm.generate(prompt).strip()
            # Sanity check - not too different from original
            if len(refined) > 0 and len(refined) < len(query) * 3:
                return refined
        except Exception as e:
            logger.warning(f"LLM query refinement failed: {e}")

        # Fallback to synonym expansion
        return self._expand_with_synonyms(query)


@dataclass
class CRAGResult:
    """Result from CRAG retrieval."""
    columns: List[str]
    scores: Dict[str, float]
    confidence: ConfidenceResult
    attempts: int
    query_history: List[str]  # Original + refined queries


class CRAGRetriever:
    """
    Corrective RAG - self-correcting retrieval wrapper.

    Wraps any base retriever and adds confidence evaluation + retry logic.

    Flow:
    1. Retrieve with base retriever
    2. Evaluate confidence
    3. If low: refine query and retry (up to max_retries)
    4. Return best results
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str, int], Tuple[List[str], Dict[str, float]]],
        config: Optional[CRAGConfig] = None,
        llm: Optional[Any] = None,
        column_descriptions: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize CRAG wrapper.

        Args:
            retrieve_fn: Function that takes (query, k) and returns (columns, scores)
            config: CRAG configuration
            llm: Optional LLM for query refinement
            column_descriptions: Column descriptions for confidence evaluation
        """
        self.retrieve_fn = retrieve_fn
        self.config = config or CRAGConfig()
        self.evaluator = RetrievalConfidenceEvaluator(self.config)
        self.refiner = QueryRefiner(llm)
        self.descriptions = column_descriptions or {}

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Retrieve with self-correction.

        Returns:
            Tuple of (columns, scores)
        """
        result = self.retrieve_detailed(query, k)
        return result.columns, result.scores

    def retrieve_detailed(
        self,
        query: str,
        k: int = 10,
    ) -> CRAGResult:
        """
        Retrieve with full CRAG details.

        Returns:
            CRAGResult with columns, scores, confidence, and retry info
        """
        current_query = query
        query_history = [query]
        best_result = None
        best_confidence = None

        for attempt in range(self.config.max_retries + 1):
            # Retrieve
            columns, scores = self.retrieve_fn(current_query, k)

            # Evaluate confidence
            confidence = self.evaluator.evaluate(
                current_query, columns, scores, self.descriptions
            )

            logger.info(
                f"CRAG attempt {attempt + 1}: confidence={confidence.level} "
                f"(score={confidence.score:.2f})"
            )

            # Track best result
            if best_result is None or confidence.score > best_confidence.score:
                best_result = (columns, scores)
                best_confidence = confidence

            # If confidence is acceptable, return
            if confidence.level in ["high", "medium"]:
                return CRAGResult(
                    columns=columns,
                    scores=scores,
                    confidence=confidence,
                    attempts=attempt + 1,
                    query_history=query_history,
                )

            # If this was the last attempt, return best
            if attempt >= self.config.max_retries:
                break

            # Refine query for next attempt
            refined_query = self.refiner.refine(current_query, confidence, attempt)

            if refined_query != current_query:
                logger.info(f"CRAG refined query: {refined_query}")
                current_query = refined_query
                query_history.append(refined_query)
            else:
                # No refinement possible, stop
                break

        # Return best result from all attempts
        return CRAGResult(
            columns=best_result[0],
            scores=best_result[1],
            confidence=best_confidence,
            attempts=len(query_history),
            query_history=query_history,
        )


def create_crag_retriever(
    retrieve_fn: Callable[[str, int], Tuple[List[str], Dict[str, float]]],
    column_descriptions: Optional[Dict[str, str]] = None,
    llm: Optional[Any] = None,
    max_retries: int = 2,
) -> CRAGRetriever:
    """
    Convenience function to create a CRAG retriever.

    Args:
        retrieve_fn: Base retrieval function
        column_descriptions: Column descriptions
        llm: Optional LLM for query refinement
        max_retries: Maximum retry attempts

    Returns:
        Configured CRAGRetriever
    """
    config = CRAGConfig(max_retries=max_retries)
    return CRAGRetriever(
        retrieve_fn=retrieve_fn,
        config=config,
        llm=llm,
        column_descriptions=column_descriptions,
    )
