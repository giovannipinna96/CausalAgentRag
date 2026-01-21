"""
downstream_v3.py

Optional downstream evaluation to validate that schema-level column selection
translates into practical gains.
Adapted from causal_rag_hepar_v3.

Evaluations:
- Leak-free prediction: AUC of a simple logistic regression trained on the
  retrieved predictors, after removing leakage variables (descendants of target).
- Total effect estimation: ATE via g-formula with a simple predictive model,
  after removing forbidden post-treatment variables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


_BIN_ABSENT = {"absent", "no", "false", "0", 0}
_BIN_PRESENT = {"present", "yes", "true", "1", 1}


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Downstream evaluation requires scikit-learn. "
            "Install with: uv add scikit-learn"
        ) from e


_BIN_RE = re.compile(r"^a(?P<hi>\d+)_?(?P<lo>\d+)$")


def _parse_binned_numeric_value(v: object) -> Optional[float]:
    """Parse HEPAR-style binned numeric strings like 'a99_35' -> midpoint."""
    if not isinstance(v, str):
        return None
    m = _BIN_RE.match(v.strip())
    if not m:
        return None
    hi = float(m.group("hi"))
    lo = float(m.group("lo"))
    return 0.5 * (hi + lo)


def to_binary_non_absent(s: pd.Series) -> np.ndarray:
    """
    Convert a categorical series to binary:
    - 0 if value is 'absent' (or equivalent)
    - 1 otherwise

    This matches the benchmark interpretation for multi-level disease variables
    (e.g., Cirrhosis: absent/compensate/decompensate).
    """
    vals = s.astype(str).str.lower()
    return (vals != "absent").astype(int).to_numpy()


@dataclass
class OutcomeVector:
    y: np.ndarray
    kind: str  # "binary" | "continuous"


def parse_outcome_vector(s: pd.Series) -> Optional[OutcomeVector]:
    """
    Try to coerce an outcome series into either:
    - binary (0/1)
    - continuous numeric

    Returns None if unsupported.
    """
    # binned numeric?
    parsed = s.map(_parse_binned_numeric_value)
    if parsed.notna().all():
        return OutcomeVector(y=parsed.astype(float).to_numpy(), kind="continuous")

    # numeric strings?
    try:
        y_num = pd.to_numeric(s, errors="raise")
        return OutcomeVector(y=y_num.astype(float).to_numpy(), kind="continuous")
    except Exception:
        pass

    # binary?
    uniq = list(pd.Series(s.astype(str).str.lower().unique()))
    if len(uniq) == 2:
        return OutcomeVector(y=to_binary_non_absent(s), kind="binary")

    # contains 'absent' => binarize non-absent
    if "absent" in set(uniq):
        return OutcomeVector(y=to_binary_non_absent(s), kind="binary")

    return None


def leakfree_prediction_auc(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    leakage: Iterable[str],
    test_size: float = 0.3,
    random_state: int = 0,
) -> float:
    """
    Train a simple classifier for target and return ROC-AUC.

    Removes leakage features (descendants) from the feature set to mimic
    an "early screening" scenario where post-diagnosis variables are not available.

    Args:
        df: DataFrame with all columns
        target: Target column to predict
        features: List of feature columns to use
        leakage: Set of columns to exclude (descendants of target)
        test_size: Fraction for test split
        random_state: Random seed

    Returns:
        ROC-AUC score (0.5 if no usable predictors)
    """
    _require_sklearn()
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if target not in df.columns:
        return float("nan")

    leakage_set = set(leakage)
    X_cols = [c for c in features if c in df.columns and c != target and c not in leakage_set]
    if len(X_cols) == 0:
        return 0.5

    y_vec = parse_outcome_vector(df[target])
    if y_vec is None or y_vec.kind != "binary":
        y = to_binary_non_absent(df[target])
    else:
        y = y_vec.y.astype(int)

    X = df[X_cols].copy()

    # all categorical => one-hot
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), X_cols),
        ]
    )

    clf = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=500, n_jobs=1)),
        ]
    )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, p))
    except Exception:
        return float("nan")


def ate_gformula(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjust: Sequence[str],
    forbidden: Iterable[str] = (),
    random_state: int = 0,
) -> Optional[float]:
    """
    Estimate ATE via g-formula using a simple predictive model.

    Args:
        df: DataFrame with all columns
        treatment: Treatment column name
        outcome: Outcome column name
        adjust: Adjustment variables (confounders)
        forbidden: Columns to exclude (post-treatment)
        random_state: Random seed

    Returns:
        Estimated ATE or None if cannot compute
    """
    _require_sklearn()
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression, Ridge

    if treatment not in df.columns or outcome not in df.columns:
        return None

    forb = set(forbidden)
    adj_cols = [c for c in adjust if c in df.columns and c not in forb and c not in {treatment, outcome}]

    # treatment binary
    T = to_binary_non_absent(df[treatment]).astype(int)

    y_vec = parse_outcome_vector(df[outcome])
    if y_vec is None:
        return None
    Y = y_vec.y

    # build feature frame: treatment + adjustment cols
    X = df[adj_cols].copy()
    X[treatment] = T

    all_cols = adj_cols + [treatment]

    pre = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), all_cols),
        ]
    )

    if y_vec.kind == "binary":
        model = LogisticRegression(max_iter=800, n_jobs=1)
    else:
        model = Ridge(alpha=1.0, random_state=random_state)

    pipe = Pipeline([("pre", pre), ("m", model)])

    try:
        pipe.fit(X, Y)
        X1 = X.copy()
        X0 = X.copy()
        X1[treatment] = 1
        X0[treatment] = 0

        if y_vec.kind == "binary":
            p1 = pipe.predict_proba(X1)[:, 1]
            p0 = pipe.predict_proba(X0)[:, 1]
            return float(np.mean(p1 - p0))
        else:
            y1 = pipe.predict(X1)
            y0 = pipe.predict(X0)
            return float(np.mean(y1 - y0))
    except Exception:
        return None


def ate_reference_from_valid_sets(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    valid_sets: Sequence[Sequence[str]],
    forbidden: Iterable[str] = (),
    random_state: int = 0,
) -> Optional[float]:
    """
    Compute a reference ATE by averaging g-formula estimates across all valid adjustment sets.

    This provides a stable reference to quantify how much a retrieved adjustment set
    deviates from a plausible valid strategy.
    """
    ests: List[float] = []
    for Z in valid_sets:
        e = ate_gformula(df, treatment, outcome, adjust=list(Z), forbidden=forbidden, random_state=random_state)
        if e is not None and not np.isnan(e):
            ests.append(float(e))
    if not ests:
        return None
    return float(np.mean(ests))


def demo_downstream_validation():
    """Demo: test downstream validation on HEPAR data."""
    import pandas as pd
    from causal_graph_v3 import CausalGraph

    print("=" * 60)
    print("Downstream Validation (v3) Demo")
    print("=" * 60)

    # Load data
    df = pd.read_csv("data/HEPAR_simulated_patients.csv")
    cols = list(df.columns)
    graph = CausalGraph.from_arcs_csv("data/HEPAR_arcs.csv", nodes=cols)

    print(f"\nLoaded {len(df)} records with {len(cols)} columns")
    print(f"Graph: {graph.G.number_of_edges()} edges")

    # Test 1: Leak-free prediction AUC
    print("\n" + "=" * 60)
    print("Test 1: Leak-free Prediction AUC")
    print("=" * 60)

    target = "Cirrhosis"
    leakage = graph.forbidden_for_prediction(target)
    parents = graph.parents(target)

    print(f"\nTarget: {target}")
    print(f"Parents (causes): {parents}")
    print(f"Leakage (descendants): {len(leakage)} columns")

    # Good features (parents + ancestors)
    good_features = list(graph.ancestors(target)) + parents + [target]
    good_features = list(set(good_features))

    # Bad features (includes leakage)
    bad_features = list(leakage)[:5] + parents

    print(f"\n--- Good features (parents/ancestors only) ---")
    print(f"Features: {[f for f in good_features if f != target][:8]}")
    auc_good = leakfree_prediction_auc(df, target, good_features, leakage)
    print(f"AUC: {auc_good:.4f}")

    print(f"\n--- Bad features (includes leakage) ---")
    print(f"Features: {bad_features}")
    auc_bad = leakfree_prediction_auc(df, target, bad_features, set())  # No leakage removal
    print(f"AUC (with leakage): {auc_bad:.4f}")

    # Test 2: ATE via g-formula
    print("\n" + "=" * 60)
    print("Test 2: ATE via G-formula")
    print("=" * 60)

    treatment = "alcoholism"
    outcome = "Cirrhosis"
    forbidden = graph.forbidden_for_total_effect(treatment, outcome)

    print(f"\nTreatment: {treatment}")
    print(f"Outcome: {outcome}")
    print(f"Forbidden (post-treatment): {len(forbidden)} columns")

    # Get valid adjustment sets
    valid_sets = graph.enumerate_backdoor_sets(treatment, outcome, max_set_size=3)
    print(f"\nValid adjustment sets ({len(valid_sets)}):")
    for i, adj_set in enumerate(valid_sets[:3], 1):
        print(f"  {i}. {adj_set}")

    # Compute ATE with valid adjustment
    if valid_sets:
        best_adj = valid_sets[0]
        ate = ate_gformula(df, treatment, outcome, adjust=best_adj, forbidden=forbidden)
        print(f"\nATE with adjustment {best_adj}: {ate:.4f}" if ate else "\nATE: could not compute")

    # Compute reference ATE
    ate_ref = ate_reference_from_valid_sets(df, treatment, outcome, valid_sets, forbidden)
    print(f"Reference ATE (avg across valid sets): {ate_ref:.4f}" if ate_ref else "Reference ATE: could not compute")

    # Compare with naive (no adjustment)
    ate_naive = ate_gformula(df, treatment, outcome, adjust=[], forbidden=forbidden)
    print(f"Naive ATE (no adjustment): {ate_naive:.4f}" if ate_naive else "Naive ATE: could not compute")

    # Compare with bad adjustment (post-treatment)
    bad_adj = list(forbidden)[:3]
    ate_bad = ate_gformula(df, treatment, outcome, adjust=bad_adj, forbidden=set())  # No forbidden removal
    print(f"Bad ATE (post-treatment adjustment): {ate_bad:.4f}" if ate_bad else "Bad ATE: could not compute")

    print(f"\n{'=' * 60}")
    print("Demo complete!")


if __name__ == "__main__":
    demo_downstream_validation()
