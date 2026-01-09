"""
Threshold-based hybrid retrieval.

Core idea: keep semantic columns only when similarity is high enough,
then fill the remaining slots with causally related columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Set, Optional

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer


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
