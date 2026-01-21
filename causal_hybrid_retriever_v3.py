"""
causal_hybrid_retriever_v3.py

A Causal RAG retriever for tabular schemas.
Adapted from causal_rag_hepar_v3.

Key design constraints:
- No oracle: the retriever only gets the user query + schema index + optional causal graph.
  It does NOT get (treatment, outcome) or (target) as explicit inputs.
- Budget-controlled: returns exactly `budget` columns.
- Hybrid: starts from semantic relevance, then expands using causal graph upstream
  and enforces causal constraints to avoid leakage / post-treatment adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from llama_index.core.schema import BaseNode, NodeWithScore

from causal_graph_v3 import CausalGraph
from semantic_retriever_v3 import SemanticSchemaRetriever


# Intent constants
_INTENT_EFFECT = "total_effect"
_INTENT_PRED = "leakfree_prediction"
_INTENT_SEM = "semantic_schema"


def detect_intent(query: str) -> str:
    """
    Lightweight intent detector (rule-based).

    Detects:
    - total_effect: queries about causal effects, interventions
    - leakfree_prediction: queries about prediction, screening, risk
    - semantic_schema: general queries
    """
    q = query.lower()

    effect_kw = [
        "causal effect", "effect of", "impact of", "treatment effect", "intervention",
        "estimate effect", "average treatment effect", "ate",
        "does ", "cause", "causes", "affect", "affects",
        "do(",
    ]
    pred_kw = [
        "predict", "prediction", "forecast", "risk", "screening",
        "early", "future", "onset", "prognosis", "diagnose", "diagnosis"
    ]

    if any(k in q for k in effect_kw):
        return _INTENT_EFFECT
    if any(k in q for k in pred_kw):
        return _INTENT_PRED
    return _INTENT_SEM


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (float(v) - lo) / (hi - lo) for k, v in scores.items()}


@dataclass
class InferredFocus:
    """Inferred focus variables from query analysis."""
    intent: str
    target: Optional[str] = None
    treatment: Optional[str] = None
    outcome: Optional[str] = None


@dataclass
class CausalHybridSchemaRetriever:
    """
    Hybrid causal+semantic retriever with ablation toggles.

    Ablations you can run without changing code:
    - Semantic-only: set graph=None
    - Constraints-only: enable_ppr=False, enable_backdoor=False
    - Graph-only: alpha_semantic=0.0, alpha_graph=1.0
    """

    nodes: Sequence[BaseNode]
    semantic: SemanticSchemaRetriever
    graph: Optional[CausalGraph] = None

    # Score fusion weights
    alpha_semantic: float = 0.55
    alpha_graph: float = 0.45

    # Inference & selection
    max_focus_candidates: int = 10
    budget_must_include: int = 2  # reserved for T,Y in effect tasks

    # Causal set heuristics
    max_adj_set_size: int = 4
    include_top_parents: int = 4

    # Causal constraint knobs
    forbid_min_weight: float = 0.0
    hard_forbid: bool = True  # if False, allow forbidden nodes as last resort

    # Ablation toggles
    enable_ppr: bool = True
    enable_backdoor: bool = True
    enable_parent_heuristic: bool = True

    def retrieve(self, query: str, budget: int) -> List[NodeWithScore]:
        """Main retrieval: infer focus then retrieve with focus."""
        focus = self.infer_focus(query)
        return self.retrieve_with_focus(query=query, budget=budget, focus=focus)

    def infer_focus(self, query: str) -> InferredFocus:
        """
        Infer (intent, target) or (treatment, outcome) from query using:
        - semantic top candidates
        - causal graph paths to disambiguate T->Y direction
        """
        intent = detect_intent(query)
        sem_scores = self.semantic.retrieve_scores(query)

        cand = sorted(sem_scores.items(), key=lambda kv: kv[1], reverse=True)
        cand_ids = [c for c, _ in cand[:self.max_focus_candidates]]

        if intent == _INTENT_PRED:
            target = cand_ids[0] if cand_ids else None
            return InferredFocus(intent=intent, target=target)

        if intent == _INTENT_EFFECT:
            if len(cand_ids) < 2:
                return InferredFocus(intent=intent, treatment=cand_ids[0] if cand_ids else None)

            # If no graph: fallback to top2 semantic
            if self.graph is None or self.graph.G.number_of_edges() == 0:
                return InferredFocus(intent=intent, treatment=cand_ids[0], outcome=cand_ids[1])

            best_pair = (cand_ids[0], cand_ids[1])
            best_score = -1e9

            # Evaluate ordered pairs connected by a directed path
            for i, a in enumerate(cand_ids):
                for j, b in enumerate(cand_ids):
                    if i == j:
                        continue
                    if not self.graph.has_path(a, b, min_weight=self.forbid_min_weight):
                        continue
                    sa = sem_scores.get(a, 0.0)
                    sb = sem_scores.get(b, 0.0)
                    dist = self.graph.shortest_path_length(a, b, min_weight=self.forbid_min_weight)
                    path_bonus = 0.0 if dist is None else 1.0 / (1.0 + dist)
                    score = sa + sb + 0.25 * path_bonus
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b)

            t, y = best_pair
            return InferredFocus(intent=intent, treatment=t, outcome=y)

        return InferredFocus(intent=intent)

    def retrieve_with_focus(self, query: str, budget: int, focus: InferredFocus) -> List[NodeWithScore]:
        """Retrieve with explicit focus (for ablation or oracle experiments)."""
        budget = int(budget)
        if budget <= 0:
            return []

        sem_scores = self.semantic.retrieve_scores(query)

        # No graph => semantic-only
        if self.graph is None or self.graph.G.number_of_edges() == 0:
            return self._topk_nodes_from_scores(sem_scores, top_k=budget)

        # Semantic intent => semantic-only
        if focus.intent == _INTENT_SEM:
            return self._topk_nodes_from_scores(sem_scores, top_k=budget)

        if focus.intent == _INTENT_PRED and focus.target:
            return self._retrieve_prediction(query, budget, focus.target, sem_scores)

        if focus.intent == _INTENT_EFFECT and focus.treatment and focus.outcome:
            return self._retrieve_total_effect(query, budget, focus.treatment, focus.outcome, sem_scores)

        # Fallback
        return self._topk_nodes_from_scores(sem_scores, top_k=budget)

    # -----------------------
    # Task-specific policies
    # -----------------------
    def _retrieve_prediction(
        self, query: str, budget: int, target: str, sem_scores: Dict[str, float]
    ) -> List[NodeWithScore]:
        """Retrieval policy for leak-free prediction tasks."""
        assert self.graph is not None

        forb = self.graph.forbidden_for_prediction(target, min_weight=self.forbid_min_weight)
        forb.discard(target)

        pr_n: Dict[str, float] = {}
        if self.enable_ppr:
            pr = self.graph.personalized_pagerank(
                seeds={target: 1.0}, reverse=True, min_weight=self.forbid_min_weight
            )
            pr_n = _normalize(pr)

        combined = self._combine_scores(sem_scores, pr_n)

        chosen: List[str] = [target]

        if self.enable_parent_heuristic:
            parents = self.graph.parents(target, min_weight=self.forbid_min_weight)
            parents = sorted(parents, key=lambda c: sem_scores.get(c, 0.0), reverse=True)
            for p in parents[:self.include_top_parents]:
                if p not in chosen and p not in forb:
                    chosen.append(p)

        chosen = self._fill_with_ranked(chosen, combined, budget=budget, forbidden=forb)
        return self._nodes_with_scores_from_order(chosen, combined)

    def _retrieve_total_effect(
        self, query: str, budget: int, t: str, y: str, sem_scores: Dict[str, float]
    ) -> List[NodeWithScore]:
        """Retrieval policy for total effect estimation tasks."""
        assert self.graph is not None

        forb = self.graph.forbidden_for_total_effect(t, y, min_weight=self.forbid_min_weight)
        forb.discard(t)
        forb.discard(y)

        pr_n: Dict[str, float] = {}
        if self.enable_ppr:
            seeds = {
                t: max(0.1, sem_scores.get(t, 0.0)),
                y: max(0.1, sem_scores.get(y, 0.0)),
            }
            pr = self.graph.personalized_pagerank(
                seeds=seeds, reverse=True, min_weight=self.forbid_min_weight
            )
            pr_n = _normalize(pr)

        combined = self._combine_scores(sem_scores, pr_n)

        chosen: List[str] = [t, y]

        if self.enable_backdoor:
            adj_sets = self.graph.enumerate_backdoor_sets(
                t, y, max_set_size=self.max_adj_set_size, min_weight=self.forbid_min_weight
            )
            if adj_sets:
                # Choose "best" adjustment set using available scores
                def node_score(n: str) -> float:
                    if self.alpha_semantic > 0:
                        return sem_scores.get(n, 0.0)
                    return pr_n.get(n, 0.0)

                def set_score(S: List[str]) -> float:
                    if not S:
                        return 0.0
                    return float(np.mean([node_score(s) for s in S]))

                best = sorted(adj_sets, key=lambda S: (-set_score(S), len(S), S))[0]
                for z in best:
                    if z not in chosen and z not in forb:
                        chosen.append(z)

        chosen = self._fill_with_ranked(chosen, combined, budget=budget, forbidden=forb)
        return self._nodes_with_scores_from_order(chosen, combined)

    # -----------------------
    # Helpers
    # -----------------------
    def _combine_scores(self, sem_scores: Dict[str, float], pr_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine semantic and graph scores."""
        combined: Dict[str, float] = {}
        keys = set(sem_scores) | set(pr_scores)
        for k in keys:
            combined[k] = (
                self.alpha_semantic * sem_scores.get(k, 0.0) +
                self.alpha_graph * pr_scores.get(k, 0.0)
            )
        return combined

    def _topk_nodes_from_scores(self, scores: Dict[str, float], top_k: int) -> List[NodeWithScore]:
        """Get top-k nodes by score."""
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        id2node = {n.node_id: n for n in self.nodes}
        out: List[NodeWithScore] = []
        for node_id, s in ranked:
            node = id2node.get(node_id)
            if node is None:
                continue
            out.append(NodeWithScore(node=node, score=float(s)))
        return out

    def _fill_with_ranked(
        self, chosen: List[str], ranked_scores: Dict[str, float],
        budget: int, forbidden: Optional[set] = None
    ) -> List[str]:
        """Fill remaining budget with ranked candidates, respecting forbidden set."""
        forbidden = forbidden or set()
        ranked = sorted(ranked_scores.items(), key=lambda kv: kv[1], reverse=True)

        # First pass: non-forbidden only
        for node_id, _ in ranked:
            if len(chosen) >= budget:
                break
            if node_id in chosen:
                continue
            if node_id in forbidden and self.hard_forbid:
                continue
            chosen.append(node_id)

        # Second pass: allow forbidden if hard_forbid=False
        if len(chosen) < budget and not self.hard_forbid and forbidden:
            for node_id, _ in ranked:
                if len(chosen) >= budget:
                    break
                if node_id in chosen:
                    continue
                if node_id not in forbidden:
                    continue
                chosen.append(node_id)

        # Pad if needed
        if len(chosen) < budget:
            all_ids = [n.node_id for n in self.nodes]
            for nid in all_ids:
                if len(chosen) >= budget:
                    break
                if nid not in chosen:
                    chosen.append(nid)

        return chosen[:budget]

    def _nodes_with_scores_from_order(
        self, ordered_ids: List[str], scores: Dict[str, float]
    ) -> List[NodeWithScore]:
        """Create NodeWithScore list from ordered column IDs."""
        id2node = {n.node_id: n for n in self.nodes}
        out: List[NodeWithScore] = []
        for nid in ordered_ids:
            node = id2node.get(nid)
            if node is None:
                continue
            out.append(NodeWithScore(node=node, score=float(scores.get(nid, 0.0))))
        return out


def demo_causal_hybrid_retriever():
    """Demo: test CausalHybridSchemaRetriever on HEPAR data."""
    import pandas as pd
    from schema_v3 import build_column_infos, build_schema_nodes, HEPAR_SYNONYMS

    print("=" * 60)
    print("CausalHybridSchemaRetriever (v3) Demo")
    print("=" * 60)

    # Load data and build schema
    df = pd.read_csv("data/HEPAR_simulated_patients.csv")
    cols = list(df.columns)
    print(f"\nLoaded {len(df)} records with {len(cols)} columns")

    infos = build_column_infos(df, synonyms=HEPAR_SYNONYMS)
    nodes, col2node = build_schema_nodes(infos)
    print(f"Built {len(nodes)} schema nodes")

    # Load causal graph
    graph = CausalGraph.from_arcs_csv("data/HEPAR_arcs.csv", nodes=cols)
    print(f"Loaded graph with {graph.G.number_of_edges()} edges")

    # Initialize semantic retriever
    print("\nInitializing SemanticSchemaRetriever...")
    semantic = SemanticSchemaRetriever(
        nodes=nodes,
        embed_model="BAAI/bge-small-en-v1.5",
        vector_top_k=24,
        bm25_top_k=24,
        alpha_dense=0.6,
    )

    # Initialize hybrid retriever
    print("Initializing CausalHybridSchemaRetriever...")
    hybrid = CausalHybridSchemaRetriever(
        nodes=nodes,
        semantic=semantic,
        graph=graph,
        alpha_semantic=0.55,
        alpha_graph=0.45,
        enable_ppr=True,
        enable_backdoor=True,
        enable_parent_heuristic=True,
    )

    # Test queries with different intents
    test_queries = [
        # Prediction intent
        ("Which columns should I use to predict cirrhosis?", "PREDICTION"),
        ("What features are useful for early screening of liver problems?", "PREDICTION"),

        # Effect intent
        ("What is the effect of alcoholism on liver disease?", "EFFECT"),
        ("Does obesity affect cirrhosis risk?", "EFFECT"),

        # Semantic intent
        ("What columns relate to hepatitis B?", "SEMANTIC"),
        ("Show me liver enzyme measurements", "SEMANTIC"),
    ]

    for query, expected_type in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Expected intent: {expected_type}")
        print("-" * 60)

        # Infer focus
        focus = hybrid.infer_focus(query)
        print(f"Detected intent: {focus.intent}")
        if focus.target:
            print(f"Inferred target: {focus.target}")
        if focus.treatment and focus.outcome:
            print(f"Inferred treatment: {focus.treatment} -> outcome: {focus.outcome}")

        # Retrieve
        results = hybrid.retrieve(query, budget=8)
        print(f"\nTop-8 columns (causal hybrid):")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.node.node_id:20s} (score: {r.score:.4f})")

        # Show forbidden columns if applicable
        if focus.intent == _INTENT_PRED and focus.target:
            forb = graph.forbidden_for_prediction(focus.target)
            retrieved_cols = [r.node.node_id for r in results]
            leaked = set(retrieved_cols) & forb
            print(f"\nForbidden (descendants of {focus.target}): {len(forb)} columns")
            print(f"Leakage in results: {leaked if leaked else 'None'}")

        if focus.intent == _INTENT_EFFECT and focus.treatment and focus.outcome:
            forb = graph.forbidden_for_total_effect(focus.treatment, focus.outcome)
            retrieved_cols = [r.node.node_id for r in results]
            leaked = set(retrieved_cols) & forb
            print(f"\nForbidden (post-treatment): {len(forb)} columns")
            print(f"Leakage in results: {leaked if leaked else 'None'}")

    # Compare with semantic-only
    print(f"\n{'=' * 60}")
    print("Comparison: Semantic-only vs Causal Hybrid")
    print("=" * 60)

    query = "Predict cirrhosis from patient data"
    print(f"\nQuery: {query}")

    sem_results = semantic.retrieve(query, top_k=8)
    hybrid_results = hybrid.retrieve(query, budget=8)

    print("\nSemantic-only:")
    for i, r in enumerate(sem_results, 1):
        print(f"  {i}. {r.node.node_id}")

    print("\nCausal Hybrid:")
    for i, r in enumerate(hybrid_results, 1):
        print(f"  {i}. {r.node.node_id}")

    # Check leakage
    focus = hybrid.infer_focus(query)
    if focus.target:
        forb = graph.forbidden_for_prediction(focus.target)
        sem_cols = [r.node.node_id for r in sem_results]
        hyb_cols = [r.node.node_id for r in hybrid_results]
        print(f"\nLeakage analysis (forbidden = descendants of {focus.target}):")
        print(f"  Semantic-only leakage: {set(sem_cols) & forb}")
        print(f"  Causal Hybrid leakage: {set(hyb_cols) & forb}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")


if __name__ == "__main__":
    demo_causal_hybrid_retriever()
