"""
HyDE (Hypothetical Document Embeddings) for column retrieval.

HyDE generates hypothetical column descriptions using an LLM before retrieval,
then uses these hypothetical descriptions to find semantically similar actual columns.

This addresses the vocabulary mismatch problem: user queries use natural language
while column names are technical/abbreviated.

Reference: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class HyDEConfig:
    """Configuration for HyDE retrieval."""
    n_hypotheses: int = 3  # Number of hypothetical docs to generate
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    aggregation: str = "mean"  # 'mean' | 'max' | 'weighted'
    hypothesis_weight: float = 0.6  # Weight for hypothesis-based scores vs original query
    max_hypothesis_tokens: int = 150
    temperature: float = 0.7  # Higher temperature for diverse hypotheses


# Default prompt template for medical column hypothesis generation
HYDE_PROMPT_TEMPLATE = """You are a medical database expert. Given a query about patient data, generate {n} hypothetical database column descriptions that would be relevant to answering this query.

Query: "{query}"

Generate {n} column descriptions in JSON format. Each should include:
- A realistic column name (snake_case, like real database columns)
- A brief medical description

Focus on columns that would help answer the query. Include a mix of:
- Direct measurements (lab values, vitals)
- Risk factors and history
- Diagnostic indicators

Return ONLY a JSON array like:
["column_name: description", "column_name2: description2", ...]"""


@dataclass
class HyDEResult:
    """Result from HyDE scoring."""
    scores: Dict[str, float]  # column -> similarity score
    hypotheses: List[str]  # Generated hypothetical descriptions
    query_embedding: np.ndarray = field(default=None, repr=False)
    hypothesis_embeddings: np.ndarray = field(default=None, repr=False)


class HyDEScorer:
    """
    Hypothetical Document Embeddings scorer for column retrieval.

    Flow:
    1. Query -> LLM generates hypothetical column descriptions
    2. Embed hypothetical descriptions
    3. Compare hypothetical embeddings vs actual column embeddings
    4. Return similarity scores (optionally combined with direct query similarity)
    """

    def __init__(
        self,
        columns: List[str],
        column_descriptions: Optional[Dict[str, str]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm: Optional[Any] = None,  # HuggingFaceLLM instance
    ):
        """
        Initialize HyDE scorer.

        Args:
            columns: List of column names
            column_descriptions: Optional dict mapping column -> description
            model_name: Sentence transformer model for embeddings
            llm: HuggingFaceLLM instance for hypothesis generation
        """
        self.columns = columns
        self.descriptions = column_descriptions or {}
        self.llm = llm

        # Load embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embed_model = SentenceTransformer(model_name)

        # Pre-compute column embeddings (with descriptions if available)
        self._compute_column_embeddings()

    def _compute_column_embeddings(self) -> None:
        """Pre-compute embeddings for all columns."""
        # Create text representations
        texts = []
        for col in self.columns:
            if col in self.descriptions:
                texts.append(f"{col}: {self.descriptions[col]}")
            else:
                # Expand column name (snake_case -> words)
                expanded = col.replace("_", " ")
                texts.append(f"{col}: {expanded}")

        # Compute embeddings
        self.column_embeddings = self.embed_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        logger.info(f"Computed embeddings for {len(self.columns)} columns")

    def generate_hypotheses(
        self,
        query: str,
        n: int = 3,
        config: Optional[HyDEConfig] = None,
    ) -> List[str]:
        """
        Generate hypothetical column descriptions for a query.

        Args:
            query: User query
            n: Number of hypotheses to generate
            config: Optional configuration

        Returns:
            List of hypothetical column descriptions
        """
        if self.llm is None:
            logger.warning("No LLM provided, using fallback hypothesis generation")
            return self._fallback_hypotheses(query, n)

        config = config or HyDEConfig()

        # Build prompt
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query, n=n)

        # Generate with LLM
        try:
            response = self.llm.generate(prompt)
            hypotheses = self._parse_hypotheses(response, n)

            if len(hypotheses) < n:
                logger.warning(f"Only parsed {len(hypotheses)} hypotheses, expected {n}")
                # Add fallback hypotheses
                hypotheses.extend(self._fallback_hypotheses(query, n - len(hypotheses)))

            return hypotheses[:n]

        except Exception as e:
            logger.error(f"LLM hypothesis generation failed: {e}")
            return self._fallback_hypotheses(query, n)

    def _parse_hypotheses(self, response: str, n: int) -> List[str]:
        """Parse hypotheses from LLM response."""
        hypotheses = []

        # Try to extract JSON array
        try:
            # Find JSON array in response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                arr = json.loads(match.group())
                if isinstance(arr, list):
                    hypotheses = [str(item) for item in arr if item]
        except json.JSONDecodeError:
            pass

        # Fallback: split by newlines and clean
        if not hypotheses:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove list markers
                line = re.sub(r'^[\d\.\-\*\[\]]+\s*', '', line)
                line = line.strip('"\'')
                if line and ':' in line:
                    hypotheses.append(line)

        return hypotheses

    def _fallback_hypotheses(self, query: str, n: int) -> List[str]:
        """Generate simple hypotheses without LLM (keyword expansion)."""
        # Extract key terms from query
        words = query.lower().split()

        # Medical term mappings for common queries
        medical_expansions = {
            'liver': ['alt: alanine aminotransferase liver enzyme',
                     'ast: aspartate aminotransferase liver enzyme',
                     'hepatomegaly: liver enlargement'],
            'hepatitis': ['hbsag: hepatitis B surface antigen',
                         'hcv_anti: hepatitis C antibody',
                         'ChHepatitis: chronic hepatitis status'],
            'cirrhosis': ['fibrosis: liver fibrosis stage',
                         'ascites: abdominal fluid accumulation',
                         'Cirrhosis: cirrhosis diagnosis'],
            'predict': ['age: patient age category',
                       'sex: patient biological sex',
                       'alcoholism: alcohol consumption history'],
            'risk': ['diabetes: diabetes mellitus status',
                    'obesity: body mass index category',
                    'hepatotoxic: hepatotoxic drug exposure'],
        }

        hypotheses = []
        for word in words:
            if word in medical_expansions:
                hypotheses.extend(medical_expansions[word])
                if len(hypotheses) >= n:
                    break

        # Pad with generic if needed
        if len(hypotheses) < n:
            generic = [
                'patient_feature: clinical observation or measurement',
                'diagnostic_marker: laboratory test result',
                'risk_factor: condition predisposing to disease',
            ]
            hypotheses.extend(generic[:n - len(hypotheses)])

        return hypotheses[:n]

    def score(
        self,
        query: str,
        config: Optional[HyDEConfig] = None,
    ) -> HyDEResult:
        """
        Score all columns using HyDE approach.

        1. Generate hypothetical column descriptions
        2. Embed hypotheses
        3. Average hypothesis embeddings
        4. Compare to actual column embeddings
        5. Optionally combine with direct query similarity

        Args:
            query: User query
            config: Optional configuration

        Returns:
            HyDEResult with scores and metadata
        """
        config = config or HyDEConfig()

        # Generate hypotheses
        hypotheses = self.generate_hypotheses(query, config.n_hypotheses, config)
        logger.info(f"Generated {len(hypotheses)} hypotheses for query")

        # Embed query directly
        query_embedding = self.embed_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Embed hypotheses
        hypothesis_embeddings = self.embed_model.encode(
            hypotheses,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # Aggregate hypothesis embeddings
        if config.aggregation == "mean":
            combined_hyp_embedding = np.mean(hypothesis_embeddings, axis=0)
        elif config.aggregation == "max":
            combined_hyp_embedding = np.max(hypothesis_embeddings, axis=0)
        else:  # weighted - weight by position (first hypothesis most important)
            weights = np.array([1.0 / (i + 1) for i in range(len(hypotheses))])
            weights = weights / weights.sum()
            combined_hyp_embedding = np.average(hypothesis_embeddings, axis=0, weights=weights)

        # Normalize combined embedding
        combined_hyp_embedding = combined_hyp_embedding / (np.linalg.norm(combined_hyp_embedding) + 1e-9)

        # Compute similarities
        # 1. Hypothesis-based similarity
        hyp_similarities = np.dot(self.column_embeddings, combined_hyp_embedding)

        # 2. Direct query similarity
        query_similarities = np.dot(self.column_embeddings, query_embedding)

        # 3. Combine scores (weighted average)
        w = config.hypothesis_weight
        combined_scores = w * hyp_similarities + (1 - w) * query_similarities

        # Build result dict
        scores = {
            col: float(score)
            for col, score in zip(self.columns, combined_scores)
        }

        return HyDEResult(
            scores=scores,
            hypotheses=hypotheses,
            query_embedding=query_embedding,
            hypothesis_embeddings=hypothesis_embeddings,
        )

    def score_direct(self, query: str) -> Dict[str, float]:
        """Score using direct query similarity (no HyDE, for comparison)."""
        query_embedding = self.embed_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = np.dot(self.column_embeddings, query_embedding)

        return {
            col: float(score)
            for col, score in zip(self.columns, similarities)
        }


class HyDERetriever:
    """
    HyDE-based column retrieval.

    Wraps HyDEScorer to provide a simple retrieve interface.
    """

    def __init__(
        self,
        columns: List[str],
        column_descriptions: Optional[Dict[str, str]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm: Optional[Any] = None,
        config: Optional[HyDEConfig] = None,
    ):
        """
        Initialize HyDE retriever.

        Args:
            columns: List of column names
            column_descriptions: Optional column descriptions
            model_name: Embedding model
            llm: LLM for hypothesis generation
            config: HyDE configuration
        """
        self.config = config or HyDEConfig()
        self.scorer = HyDEScorer(
            columns=columns,
            column_descriptions=column_descriptions,
            model_name=model_name,
            llm=llm,
        )
        self.columns = columns

    def retrieve(
        self,
        query: str,
        k: int = 10,
        config: Optional[HyDEConfig] = None,
    ) -> List[str]:
        """
        Retrieve top-k columns using HyDE.

        Args:
            query: User query
            k: Number of columns to retrieve
            config: Optional configuration override

        Returns:
            List of column names
        """
        result = self.retrieve_detailed(query, k, config)
        return result[0]

    def retrieve_detailed(
        self,
        query: str,
        k: int = 10,
        config: Optional[HyDEConfig] = None,
    ) -> Tuple[List[str], Dict[str, float], HyDEResult]:
        """
        Retrieve with full details.

        Returns:
            Tuple of (columns, scores, hyde_result)
        """
        config = config or self.config

        # Score using HyDE
        hyde_result = self.scorer.score(query, config)

        # Sort by score
        sorted_cols = sorted(
            hyde_result.scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Take top k
        top_cols = [col for col, _ in sorted_cols[:k]]
        top_scores = {col: hyde_result.scores[col] for col in top_cols}

        return top_cols, top_scores, hyde_result


def create_hyde_retriever(
    columns: List[str],
    descriptions: Optional[Dict[str, str]] = None,
    llm: Optional[Any] = None,
    n_hypotheses: int = 3,
) -> HyDERetriever:
    """
    Convenience function to create a HyDE retriever.

    Args:
        columns: Column names
        descriptions: Optional column descriptions
        llm: HuggingFaceLLM instance (optional, falls back to keyword expansion)
        n_hypotheses: Number of hypotheses to generate

    Returns:
        Configured HyDERetriever
    """
    config = HyDEConfig(n_hypotheses=n_hypotheses)
    return HyDERetriever(
        columns=columns,
        column_descriptions=descriptions,
        llm=llm,
        config=config,
    )
