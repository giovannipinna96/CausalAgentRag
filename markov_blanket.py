"""
Markov Blanket and causal graph utilities for precision-focused retrieval.

Key insight: For prediction tasks, we want CAUSES (parents) not EFFECTS (children).
Including effects causes data leakage - the effect is correlated with the target
but using it as a feature would be cheating.

The Markov Blanket of a node is: parents + children + spouses (co-parents of children).
For prediction tasks, we often want only the 'parents_only' or 'ancestors' mode
to avoid leakage.
"""

from __future__ import annotations

from typing import Dict, Set, List, Tuple, Optional, Literal
from dataclasses import dataclass

import networkx as nx


CausalMode = Literal['parents_only', 'ancestors', 'full_mb', 'full_blanket']


def get_parents(G: nx.DiGraph, node: str) -> Set[str]:
    """Get direct parents (causes) of a node."""
    if node not in G:
        return set()
    return set(G.predecessors(node))


def get_children(G: nx.DiGraph, node: str) -> Set[str]:
    """Get direct children (effects) of a node."""
    if node not in G:
        return set()
    return set(G.successors(node))


def get_spouses(G: nx.DiGraph, node: str) -> Set[str]:
    """
    Get spouses of a node (other parents of the node's children).

    Spouses are important for conditional independence but can be
    tricky for prediction - they may introduce confounding.
    """
    if node not in G:
        return set()

    spouses = set()
    for child in G.successors(node):
        for parent in G.predecessors(child):
            if parent != node:
                spouses.add(parent)
    return spouses


def get_markov_blanket(G: nx.DiGraph, node: str) -> Set[str]:
    """
    Get the full Markov Blanket: parents + children + spouses.

    The Markov Blanket is the minimal set of nodes that makes the target
    conditionally independent of all other nodes in the graph.

    WARNING: For prediction tasks, using the full MB may include effects
    (children) which causes data leakage. Use get_causal_parents_only() instead.
    """
    if node not in G:
        return set()

    parents = get_parents(G, node)
    children = get_children(G, node)
    spouses = get_spouses(G, node)

    return parents | children | spouses


def get_causal_parents_only(G: nx.DiGraph, node: str) -> Set[str]:
    """
    Get only the direct parents (causes) of a node.

    PREDICTION MODE: This is the key to avoiding data leakage.
    Only direct causes can be used as valid predictive features.
    Effects (children) should be excluded as they would leak information.

    Example:
        For "Predict cirrhosis":
        - Parents (valid): Steatosis, fibrosis -> these CAUSE cirrhosis
        - Children (leakage): albumin, alt, ast -> these are EFFECTS of cirrhosis
    """
    return get_parents(G, node)


def get_causal_ancestors(
    G: nx.DiGraph,
    node: str,
    max_depth: int = 2
) -> Dict[str, float]:
    """
    Get ancestors (causes) of a node with distance-based decay scores.

    For prediction tasks, ancestors are valid features (no leakage).
    Score decays exponentially with distance: 1/(2^(depth-1))

    Args:
        G: Directed graph (edges go from cause to effect)
        node: Target node to find ancestors of
        max_depth: Maximum depth to traverse (1 = parents only)

    Returns:
        Dict mapping ancestor nodes to their scores (higher = closer/more relevant)
    """
    if node not in G:
        return {}

    scores: Dict[str, float] = {}
    visited = {node}
    current_level = {node}

    for depth in range(1, max_depth + 1):
        next_level = set()
        score = 1.0 / (2 ** (depth - 1))

        for n in current_level:
            for parent in G.predecessors(n):
                if parent not in visited:
                    visited.add(parent)
                    next_level.add(parent)
                    # Keep the highest score (closest path)
                    if parent not in scores or score > scores[parent]:
                        scores[parent] = score

        current_level = next_level
        if not current_level:
            break

    return scores


def get_causal_descendants(
    G: nx.DiGraph,
    node: str,
    max_depth: int = 2
) -> Dict[str, float]:
    """
    Get descendants (effects) of a node with distance-based decay scores.

    WARNING: These are EFFECTS and should NOT be used as predictive features
    for the target node (data leakage). This function is provided for analysis
    and for identifying what to EXCLUDE.

    Args:
        G: Directed graph (edges go from cause to effect)
        node: Target node to find descendants of
        max_depth: Maximum depth to traverse

    Returns:
        Dict mapping descendant nodes to their scores
    """
    if node not in G:
        return {}

    scores: Dict[str, float] = {}
    visited = {node}
    current_level = {node}

    for depth in range(1, max_depth + 1):
        next_level = set()
        score = 1.0 / (2 ** (depth - 1))

        for n in current_level:
            for child in G.successors(n):
                if child not in visited:
                    visited.add(child)
                    next_level.add(child)
                    if child not in scores or score > scores[child]:
                        scores[child] = score

        current_level = next_level
        if not current_level:
            break

    return scores


def get_weighted_markov_blanket(
    G: nx.DiGraph,
    node: str,
    mode: CausalMode = 'parents_only',
    max_depth: int = 2,
    parent_weight: float = 1.0,
    child_weight: float = 0.5,
    spouse_weight: float = 0.3
) -> Dict[str, float]:
    """
    Get Markov Blanket members with configurable mode and weights.

    Args:
        G: Directed graph (edges go from cause to effect)
        node: Target node
        mode:
            - 'parents_only': Only causes (highest precision for prediction)
            - 'ancestors': Parents + grandparents with decay
            - 'full_mb': Full Markov Blanket with configurable weights
            - 'full_blanket': Same as 'full_mb' (alias)
        max_depth: For 'ancestors' mode, how deep to traverse
        parent_weight: Base weight for parents
        child_weight: Base weight for children (only used in full_mb mode)
        spouse_weight: Base weight for spouses (only used in full_mb mode)

    Returns:
        Dict mapping nodes to their relevance scores
    """
    if node not in G:
        return {}

    scores: Dict[str, float] = {}

    if mode == 'parents_only':
        # Only direct causes - highest precision
        for parent in get_parents(G, node):
            scores[parent] = parent_weight

    elif mode == 'ancestors':
        # All ancestors with decay
        scores = get_causal_ancestors(G, node, max_depth=max_depth)
        # Scale by parent_weight
        scores = {k: v * parent_weight for k, v in scores.items()}

    elif mode in ('full_mb', 'full_blanket'):
        # Full Markov Blanket with different weights
        for parent in get_parents(G, node):
            scores[parent] = parent_weight

        for child in get_children(G, node):
            if child not in scores or child_weight > scores[child]:
                scores[child] = child_weight

        for spouse in get_spouses(G, node):
            if spouse not in scores or spouse_weight > scores[spouse]:
                scores[spouse] = spouse_weight
    else:
        raise ValueError(f"Unknown mode: {mode}. Expected one of: parents_only, ancestors, full_mb, full_blanket")

    return scores


def is_leakage_column(
    G: nx.DiGraph,
    target: str,
    candidate: str,
    max_depth: int = 3
) -> bool:
    """
    Check if candidate is an EFFECT of target (would cause data leakage).

    A column is a leakage risk if:
    1. It's a direct child (effect) of the target
    2. It's a descendant of the target (indirect effect)

    Args:
        G: Directed graph (edges go from cause to effect)
        target: The target variable being predicted
        candidate: The candidate feature column
        max_depth: How many hops to check for descendant relationship

    Returns:
        True if candidate is an effect of target (leakage risk), False otherwise
    """
    if target not in G or candidate not in G:
        return False

    # Check if candidate is a descendant of target
    descendants = get_causal_descendants(G, target, max_depth=max_depth)
    return candidate in descendants


def get_leakage_free_columns(
    G: nx.DiGraph,
    target: str,
    all_columns: List[str],
    max_depth: int = 3
) -> List[str]:
    """
    Filter columns to exclude those that would cause data leakage.

    Args:
        G: Directed graph (edges go from cause to effect)
        target: The target variable being predicted
        all_columns: List of all candidate columns
        max_depth: How many hops to check for descendant relationship

    Returns:
        List of columns that are safe to use as features (not effects of target)
    """
    if target not in G:
        return all_columns

    descendants = set(get_causal_descendants(G, target, max_depth=max_depth).keys())
    return [col for col in all_columns if col not in descendants and col != target]


def get_graph_relevance_scores(
    G: nx.DiGraph,
    target: str,
    all_columns: List[str],
    mode: CausalMode = 'parents_only',
    max_depth: int = 2,
    decay_factor: float = 0.5
) -> List[Tuple[str, float]]:
    """
    Score all columns by their causal relevance to target.
    Returns ranked list suitable for RRF fusion.

    Args:
        G: Directed graph
        target: Target node to score relevance to
        all_columns: All columns to score
        mode: Causal mode for scoring
        max_depth: Maximum depth for traversal
        decay_factor: Decay factor for non-MB nodes

    Returns:
        List of (column, score) tuples sorted by score descending
    """
    mb_scores = get_weighted_markov_blanket(G, target, mode=mode, max_depth=max_depth)

    scores = []
    for col in all_columns:
        if col == target:
            scores.append((col, 0.0))  # Don't include target itself
        elif col in mb_scores:
            scores.append((col, mb_scores[col]))
        elif col in G:
            # Give small score to graph-connected but not in MB
            try:
                path_len = nx.shortest_path_length(G.to_undirected(), target, col)
                scores.append((col, decay_factor ** path_len))
            except nx.NetworkXNoPath:
                scores.append((col, 0.0))
        else:
            scores.append((col, 0.0))

    return sorted(scores, key=lambda x: x[1], reverse=True)


@dataclass
class MarkovBlanketResult:
    """Result from Markov Blanket scoring."""
    parents: Set[str]
    children: Set[str]
    spouses: Set[str]
    scores: Dict[str, float]
    mode: CausalMode

    @property
    def all_members(self) -> Set[str]:
        return self.parents | self.children | self.spouses

    def get_valid_predictors(self) -> Set[str]:
        """Get only the valid predictors (parents/causes)."""
        return self.parents


class MarkovBlanketScorer:
    """
    Causal scorer using Markov Blanket instead of undirected BFS.

    This preserves causal direction, which is critical for prediction tasks
    where we need to distinguish causes (valid features) from effects (leakage).
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initialize with a directed graph.

        Args:
            graph: NetworkX DiGraph where edges go from cause to effect
        """
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("MarkovBlanketScorer requires a directed graph (nx.DiGraph)")
        self.graph = graph

    def score(
        self,
        targets: List[str],
        mode: CausalMode = 'parents_only',
        max_depth: int = 2
    ) -> Dict[str, float]:
        """
        Score all nodes based on their causal relationship to targets.

        Args:
            targets: List of target nodes to score relative to
            mode: Causal mode ('parents_only', 'ancestors', 'full_mb')
            max_depth: Maximum depth for ancestor traversal

        Returns:
            Dict mapping all graph nodes to their relevance scores
        """
        scores: Dict[str, float] = {node: 0.0 for node in self.graph.nodes}

        for target in targets:
            if target not in self.graph:
                continue

            target_scores = get_weighted_markov_blanket(
                self.graph, target, mode=mode, max_depth=max_depth
            )

            # Merge scores, keeping maximum
            for node, score in target_scores.items():
                if score > scores.get(node, 0.0):
                    scores[node] = score

        return scores

    def score_detailed(
        self,
        target: str,
        mode: CausalMode = 'parents_only',
        max_depth: int = 2
    ) -> MarkovBlanketResult:
        """
        Get detailed Markov Blanket information for a single target.

        Returns MarkovBlanketResult with full breakdown of parents, children, spouses.
        """
        if target not in self.graph:
            return MarkovBlanketResult(
                parents=set(),
                children=set(),
                spouses=set(),
                scores={},
                mode=mode
            )

        parents = get_parents(self.graph, target)
        children = get_children(self.graph, target)
        spouses = get_spouses(self.graph, target)
        scores = get_weighted_markov_blanket(
            self.graph, target, mode=mode, max_depth=max_depth
        )

        return MarkovBlanketResult(
            parents=parents,
            children=children,
            spouses=spouses,
            scores=scores,
            mode=mode
        )

    def get_leakage_columns(self, target: str, max_depth: int = 3) -> Set[str]:
        """Get all columns that would cause data leakage for this target."""
        if target not in self.graph:
            return set()
        return set(get_causal_descendants(self.graph, target, max_depth=max_depth).keys())
