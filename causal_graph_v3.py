"""
causal_graph_v3.py

Causal graph utilities from causal_rag_hepar_v3.

Core primitives:
- parents/children/ancestors/descendants
- d-separation via moralized ancestral graph
- enumerate minimal backdoor adjustment sets
- personalized PageRank for causal expansion (graph diffusion)
- forbidden sets for prediction and total effect tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import itertools

import networkx as nx
import pandas as pd


NodeLike = Union[str, Iterable[str]]


def _as_set(x: NodeLike) -> Set[str]:
    if isinstance(x, str):
        return {x}
    return set(x)


@dataclass
class CausalGraph:
    """
    Causal graph wrapper with causal reasoning utilities.

    Supports:
    - Edge weights (for discovered graphs with bootstrap confidence)
    - Personalized PageRank for causal expansion
    - d-separation testing
    - Backdoor adjustment set enumeration
    """
    G: nx.DiGraph

    @staticmethod
    def from_arcs_csv(
        arcs_csv: str,
        nodes: Optional[Sequence[str]] = None,
        weight_col: Optional[str] = None
    ) -> "CausalGraph":
        """Load graph from CSV with from/to or src/dst or source/target columns."""
        arcs = pd.read_csv(arcs_csv)

        # Detect column names
        if {"from", "to"} <= set(arcs.columns):
            src_col, dst_col = "from", "to"
        elif {"src", "dst"} <= set(arcs.columns):
            src_col, dst_col = "src", "dst"
        elif {"source", "target"} <= set(arcs.columns):
            src_col, dst_col = "source", "target"
        else:
            raise ValueError("arcs_csv must have columns: from/to (or src/dst, or source/target)")

        G = nx.DiGraph()
        if nodes is not None:
            G.add_nodes_from(nodes)

        if weight_col and weight_col in arcs.columns:
            for u, v, w in arcs[[src_col, dst_col, weight_col]].itertuples(index=False, name=None):
                G.add_edge(str(u), str(v), weight=float(w))
        else:
            for u, v in arcs[[src_col, dst_col]].itertuples(index=False, name=None):
                G.add_edge(str(u), str(v), weight=1.0)

        return CausalGraph(G)

    @staticmethod
    def from_networkx(G: nx.DiGraph) -> "CausalGraph":
        """Create from existing NetworkX DiGraph."""
        return CausalGraph(G.copy())

    def copy(self) -> "CausalGraph":
        return CausalGraph(self.G.copy())

    # -----------------------------
    # Local neighborhoods
    # -----------------------------
    def parents(self, node: str, min_weight: float = 0.0) -> List[str]:
        """Get parents of a node (direct causes)."""
        return [p for p in self.G.predecessors(node)
                if float(self.G[p][node].get("weight", 1.0)) >= min_weight]

    def children(self, node: str, min_weight: float = 0.0) -> List[str]:
        """Get children of a node (direct effects)."""
        return [c for c in self.G.successors(node)
                if float(self.G[node][c].get("weight", 1.0)) >= min_weight]

    def ancestors(self, node: str, min_weight: float = 0.0) -> Set[str]:
        """Get all ancestors (indirect causes)."""
        return _reachable(self.G, sources=[node], direction="up", min_weight=min_weight)

    def descendants(self, node: str, min_weight: float = 0.0) -> Set[str]:
        """Get all descendants (indirect effects)."""
        return _reachable(self.G, sources=[node], direction="down", min_weight=min_weight)

    def has_path(self, u: str, v: str, min_weight: float = 0.0) -> bool:
        """Check if there's a directed path from u to v."""
        if u == v:
            return True
        q = [u]
        seen = {u}
        while q:
            x = q.pop()
            for y in self.children(x, min_weight=min_weight):
                if y == v:
                    return True
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        return False

    def shortest_path_length(self, u: str, v: str, min_weight: float = 0.0) -> Optional[int]:
        """Get shortest path length from u to v."""
        if u == v:
            return 0
        q = [(u, 0)]
        seen = {u}
        while q:
            x, d = q.pop(0)
            for y in self.children(x, min_weight=min_weight):
                if y == v:
                    return d + 1
                if y not in seen:
                    seen.add(y)
                    q.append((y, d + 1))
        return None

    # -----------------------------
    # d-separation
    # -----------------------------
    def d_separated(self, X: NodeLike, Y: NodeLike, Z: Set[str], min_weight: float = 0.0) -> bool:
        """
        Test d-separation using the moralized ancestral graph criterion.
        X _||_ Y | Z in G
        """
        G = _thresholded_subgraph(self.G, min_weight=min_weight)
        Xs = _as_set(X)
        Ys = _as_set(Y)
        Zs = set(Z)

        # ancestral set
        anc = set()
        for n in Xs | Ys | Zs:
            anc |= nx.ancestors(G, n)
            anc.add(n)

        H = G.subgraph(anc).copy()

        # moralize: connect co-parents
        U = nx.Graph()
        U.add_nodes_from(H.nodes())
        for u, v in H.edges():
            U.add_edge(u, v)
        for n in H.nodes():
            ps = list(H.predecessors(n))
            for a, b in itertools.combinations(ps, 2):
                U.add_edge(a, b)

        # remove conditioned nodes
        U.remove_nodes_from(Zs)

        # check separation
        for x in Xs:
            if x not in U:
                continue
            for y in Ys:
                if y not in U:
                    continue
                if nx.has_path(U, x, y):
                    return False
        return True

    # -----------------------------
    # Backdoor sets
    # -----------------------------
    def enumerate_backdoor_sets(
        self,
        treatment: str,
        outcome: str,
        max_set_size: int = 4,
        restrict_to_ancestors: bool = True,
        min_weight: float = 0.0,
    ) -> List[List[str]]:
        """
        Enumerate minimal backdoor adjustment sets for P(outcome | do(treatment)).

        Uses the backdoor criterion:
          outcome _||_ treatment | Z  in G_{underline{treatment}}
          AND Z contains no descendants of treatment.

        Returns list of minimal valid adjustment sets up to max_set_size.
        """
        if treatment == outcome:
            return []

        G = _thresholded_subgraph(self.G, min_weight=min_weight)

        # G_{underline{treatment}} = remove outgoing edges from treatment
        Gb = G.copy()
        for c in list(Gb.successors(treatment)):
            if Gb.has_edge(treatment, c):
                Gb.remove_edge(treatment, c)

        # Candidate covariates (not treatment, outcome, or descendants of treatment)
        forbidden = set(nx.descendants(G, treatment))
        forbidden.add(treatment)
        forbidden.add(outcome)

        if restrict_to_ancestors:
            cand = (nx.ancestors(G, treatment) | nx.ancestors(G, outcome)) - forbidden
        else:
            cand = set(G.nodes()) - forbidden

        cand = sorted(cand)

        valid: List[Set[str]] = []
        for k in range(0, max_set_size + 1):
            for subset in itertools.combinations(cand, k):
                Z = set(subset)
                # must not contain descendants of treatment
                if Z & set(nx.descendants(G, treatment)):
                    continue
                # d-separation in backdoor graph
                if _d_separated_moral(Gb, {treatment}, {outcome}, Z):
                    valid.append(Z)

        # Keep only minimal sets
        minimal: List[Set[str]] = []
        for Z in sorted(valid, key=lambda s: (len(s), sorted(s))):
            if any(m.issubset(Z) for m in minimal):
                continue
            minimal.append(Z)

        return [sorted(list(s)) for s in minimal]

    # -----------------------------
    # Forbidden sets
    # -----------------------------
    def forbidden_for_total_effect(self, treatment: str, outcome: str, min_weight: float = 0.0) -> Set[str]:
        """
        Forbidden set for total effect estimation:
        - descendants of treatment (post-treatment), excluding outcome
        - descendants of outcome (post-outcome leakage)
        """
        G = _thresholded_subgraph(self.G, min_weight=min_weight)
        forb = set(nx.descendants(G, treatment)) - {outcome}
        forb |= set(nx.descendants(G, outcome))
        return forb

    def forbidden_for_prediction(self, target: str, min_weight: float = 0.0) -> Set[str]:
        """Forbidden columns for leak-free prediction: descendants of the target."""
        G = _thresholded_subgraph(self.G, min_weight=min_weight)
        return set(nx.descendants(G, target))

    # -----------------------------
    # Graph diffusion (PPR)
    # -----------------------------
    def personalized_pagerank(
        self,
        seeds: Dict[str, float],
        reverse: bool = True,
        alpha: float = 0.85,
        min_weight: float = 0.0,
    ) -> Dict[str, float]:
        """
        Personalized PageRank scores from seeds.

        reverse=True means we run on the *reversed* graph to diffuse
        "upstream" towards causes.
        """
        if not seeds:
            return {}

        G = _thresholded_subgraph(self.G, min_weight=min_weight)
        if reverse:
            G = G.reverse(copy=True)

        # Personalization vector (normalized)
        pers = {n: 0.0 for n in G.nodes()}
        total = float(sum(max(0.0, w) for w in seeds.values()))
        if total <= 0:
            total = 1.0
        for n, w in seeds.items():
            if n in pers:
                pers[n] = float(max(0.0, w)) / total

        pr = nx.pagerank(G, alpha=alpha, personalization=pers, weight="weight")
        return {k: float(v) for k, v in pr.items()}


def _thresholded_subgraph(G: nx.DiGraph, min_weight: float = 0.0) -> nx.DiGraph:
    """Filter edges by minimum weight."""
    if min_weight <= 0.0:
        return G
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        if w >= min_weight:
            H.add_edge(u, v, **data)
    return H


def _reachable(G: nx.DiGraph, sources: Sequence[str], direction: str, min_weight: float = 0.0) -> Set[str]:
    """Find all reachable nodes in given direction (up=predecessors, down=successors)."""
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")
    get_next = G.predecessors if direction == "up" else G.successors

    seen: Set[str] = set()
    stack = list(sources)
    while stack:
        n = stack.pop()
        for nxt in get_next(n):
            if direction == "up":
                w = float(G[nxt][n].get("weight", 1.0))
            else:
                w = float(G[n][nxt].get("weight", 1.0))
            if w < min_weight:
                continue
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _d_separated_moral(G: nx.DiGraph, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
    """Test d-separation using moralized ancestral graph."""
    anc = set()
    for n in X | Y | Z:
        anc |= nx.ancestors(G, n)
        anc.add(n)
    H = G.subgraph(anc).copy()

    U = nx.Graph()
    U.add_nodes_from(H.nodes())
    for u, v in H.edges():
        U.add_edge(u, v)
    for n in H.nodes():
        ps = list(H.predecessors(n))
        for a, b in itertools.combinations(ps, 2):
            U.add_edge(a, b)

    U.remove_nodes_from(Z)

    for x in X:
        if x not in U:
            continue
        for y in Y:
            if y not in U:
                continue
            if nx.has_path(U, x, y):
                return False
    return True


def demo_causal_graph():
    """Demo: test CausalGraph utilities on HEPAR data."""
    import pandas as pd

    print("=" * 60)
    print("CausalGraph (v3) Demo")
    print("=" * 60)

    # Load graph
    df = pd.read_csv("data/HEPAR_simulated_patients.csv")
    cols = list(df.columns)
    graph = CausalGraph.from_arcs_csv("data/HEPAR_arcs.csv", nodes=cols)

    print(f"\nLoaded graph with {graph.G.number_of_nodes()} nodes and {graph.G.number_of_edges()} edges")

    # Test basic operations
    print("\n--- Basic Operations ---")
    target = "Cirrhosis"
    print(f"\nTarget: {target}")
    print(f"Parents: {graph.parents(target)}")
    print(f"Children: {graph.children(target)}")
    print(f"Ancestors: {graph.ancestors(target)}")
    print(f"Descendants: {graph.descendants(target)}")

    # Test forbidden sets
    print("\n--- Forbidden Sets ---")
    print(f"\nForbidden for prediction of {target}:")
    forb_pred = graph.forbidden_for_prediction(target)
    print(f"  {sorted(forb_pred)}")

    treatment, outcome = "alcoholism", "Cirrhosis"
    print(f"\nForbidden for total effect {treatment} -> {outcome}:")
    forb_effect = graph.forbidden_for_total_effect(treatment, outcome)
    print(f"  {sorted(forb_effect)}")

    # Test backdoor sets
    print("\n--- Backdoor Adjustment Sets ---")
    print(f"\nMinimal backdoor sets for effect of {treatment} on {outcome}:")
    backdoor_sets = graph.enumerate_backdoor_sets(treatment, outcome, max_set_size=3)
    for i, adj_set in enumerate(backdoor_sets[:5], 1):
        print(f"  {i}. {adj_set}")
    if len(backdoor_sets) > 5:
        print(f"  ... and {len(backdoor_sets) - 5} more")

    # Test Personalized PageRank
    print("\n--- Personalized PageRank ---")
    seeds = {target: 1.0}
    ppr = graph.personalized_pagerank(seeds, reverse=True)
    top_ppr = sorted(ppr.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop-10 nodes by PPR (diffusing upstream from {target}):")
    for node, score in top_ppr:
        print(f"  {node:20s}: {score:.6f}")

    # Test d-separation
    print("\n--- d-Separation Tests ---")
    test_cases = [
        ("alcoholism", "jaundice", set()),
        ("alcoholism", "jaundice", {"Cirrhosis"}),
        ("alcoholism", "jaundice", {"Steatosis"}),
    ]
    for x, y, z in test_cases:
        sep = graph.d_separated(x, y, z)
        z_str = "{" + ", ".join(z) + "}" if z else "{}"
        print(f"  {x} _||_ {y} | {z_str}: {sep}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")


if __name__ == "__main__":
    demo_causal_graph()
