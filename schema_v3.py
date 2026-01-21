"""
schema_v3.py

Schema-to-Nodes utilities for tabular RAG where the retrieval unit is a column.
Adapted from causal_rag_hepar_v3.

Each column is represented as a LlamaIndex TextNode containing:
- column name
- human-friendly description (auto-generated from synonyms + value bins)
- value set / bin ranges (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

import pandas as pd


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    description: str
    values: List[str]
    kind: str  # "binary", "ordinal_bin", "categorical"


def _humanize_col_name(col: str) -> str:
    return col.replace("_", " ").strip()


def _parse_bin_token(tok: str) -> Optional[Tuple[float, float]]:
    """
    Parse tokens like:
      - a34_0      -> (0, 34)
      - a199_100   -> (100, 199)
    Returns (low, high) or None if not parseable.
    """
    if not isinstance(tok, str):
        return None
    m = re.match(r"^[a-zA-Z]([0-9]+)_([0-9]+)$", tok)
    if not m:
        return None
    high = float(m.group(1))
    low = float(m.group(2))
    if low > high:
        low, high = high, low
    return (low, high)


def _describe_bins(values: List[str]) -> Optional[str]:
    parsed = []
    for v in values:
        r = _parse_bin_token(v)
        if r is not None:
            parsed.append((r[0], r[1], v))
    if len(parsed) < 2:
        return None
    parsed.sort(key=lambda x: (x[0], x[1]))
    ranges = [f"{int(lo)}-{int(hi)}" for lo, hi, _ in parsed]
    return "discretized into bins: " + ", ".join(ranges)


# HEPAR synonyms for medical terms
HEPAR_SYNONYMS: Dict[str, List[str]] = {
    # Diseases / conditions
    "Cirrhosis": ["liver cirrhosis", "advanced liver scarring", "end-stage liver disease"],
    "Steatosis": ["fatty liver disease", "hepatic steatosis", "fatty liver"],
    "ChHepatitis": ["chronic viral hepatitis", "chronic hepatitis", "long-term hepatitis infection"],
    "THepatitis": ["toxic hepatitis", "drug-induced hepatitis", "hepatitis due to toxins"],
    "RHepatitis": ["reactive hepatitis", "acute reactive hepatitis"],
    "PBC": ["primary biliary cholangitis", "primary biliary cirrhosis", "autoimmune bile duct disease"],
    "Hyperbilirubinemia": ["high bilirubin", "hyperbilirubinemia", "elevated bilirubin levels"],

    # Labs / biomarkers
    "alt": ["ALT", "alanine aminotransferase", "SGPT"],
    "ast": ["AST", "aspartate aminotransferase", "SGOT"],
    "ggtp": ["GGT", "gamma-glutamyl transferase", "gamma-GT"],
    "bilirubin": ["bilirubin", "total bilirubin", "bilirubin level"],
    "inr": ["INR", "blood clotting time", "coagulation index"],
    "platelet": ["platelet count", "thrombocytes"],
    "cholesterol": ["cholesterol level", "serum cholesterol"],
    "triglycerides": ["triglycerides", "blood triglycerides"],
    "phosphatase": ["alkaline phosphatase", "ALP"],

    # Symptoms / signs
    "pain_ruq": ["right upper quadrant pain", "RUQ pain", "pain in the upper right abdomen"],
    "upper_pain": ["upper abdominal pain", "epigastric pain"],
    "itching": ["pruritus", "itching"],
    "jaundice": ["jaundice", "yellow skin/eyes"],

    # Exposures / history
    "alcoholism": ["alcohol abuse", "heavy drinking", "alcohol dependence"],
    "hepatotoxic": ["hepatotoxic medication", "liver-toxic drugs", "drug-induced liver injury exposure"],
    "vh_amn": ["history of viral hepatitis", "viral hepatitis exposure", "hepatitis anamnesis"],
    "transfusion": ["blood transfusion", "transfusion history"],
    "injections": ["injection exposure", "parenteral injections", "needle exposure"],

    # Metabolic
    "obesity": ["obesity", "high BMI", "overweight"],
    "diabetes": ["diabetes", "diabetes mellitus"],
}


def build_column_infos(
    df: pd.DataFrame,
    synonyms: Optional[Dict[str, List[str]]] = None,
    include_values: bool = True,
    max_values_in_text: int = 12,
) -> Dict[str, ColumnInfo]:
    """
    Build a ColumnInfo for each dataframe column.

    The descriptions are auto-generated and act as a simple data dictionary.
    """
    synonyms = synonyms or HEPAR_SYNONYMS
    infos: Dict[str, ColumnInfo] = {}

    for col in df.columns:
        ser = df[col].astype(str)
        uniq = sorted(ser.dropna().unique().tolist())

        # Identify kind
        uniq_set = set([u.lower() for u in uniq])
        if uniq_set == {"absent", "present"}:
            kind = "binary"
        else:
            parse_count = sum(1 for u in uniq if _parse_bin_token(u) is not None)
            if parse_count >= max(2, int(0.8 * len(uniq))):
                kind = "ordinal_bin"
            else:
                kind = "categorical"

        # Concept name from synonyms
        aliases = synonyms.get(col, [])
        concept = aliases[0] if aliases else _humanize_col_name(col)

        # Build description text
        desc_parts = [f"{concept}."]

        if kind == "binary":
            desc_parts.append("Binary indicator (absent/present).")
        elif kind == "ordinal_bin":
            bdesc = _describe_bins(uniq)
            if bdesc:
                desc_parts.append(bdesc + ".")
            else:
                desc_parts.append("Ordinal/discretized measurement.")
        else:
            desc_parts.append("Categorical variable.")

        if aliases:
            uniq_aliases = list(dict.fromkeys(aliases))
            alias_str = ", ".join(uniq_aliases[:8])
            desc_parts.append(f"Aliases: {alias_str}.")

        if include_values:
            shown = uniq[:max_values_in_text]
            if len(uniq) > max_values_in_text:
                shown.append("...")
            desc_parts.append(f"Values: {', '.join(shown)}.")

        description = " ".join(desc_parts)
        infos[col] = ColumnInfo(name=col, description=description, values=uniq, kind=kind)

    return infos


def build_schema_nodes(infos: Dict[str, ColumnInfo]):
    """
    Convert ColumnInfo objects into LlamaIndex TextNodes.

    Returns:
        nodes: List[TextNode]
        col2node: Dict[column_name, node]
    """
    from llama_index.core.schema import TextNode

    nodes = []
    col2node = {}
    for col, info in infos.items():
        text = f"Column: {info.name}\nDescription: {info.description}"
        node = TextNode(
            text=text,
            id_=info.name,  # stable node id == column name
            metadata={
                "col_name": info.name,
                "kind": info.kind,
            },
        )
        nodes.append(node)
        col2node[col] = node
    return nodes, col2node


if __name__ == "__main__":
    # Demo: load HEPAR data and build schema nodes
    import json

    df = pd.read_csv("data/HEPAR_simulated_patients.csv")
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")

    infos = build_column_infos(df, synonyms=HEPAR_SYNONYMS)
    nodes, col2node = build_schema_nodes(infos)

    print(f"\nBuilt {len(nodes)} schema nodes")
    print("\nSample node (alt):")
    if "alt" in col2node:
        print(col2node["alt"].text)

    print("\nSample node (Cirrhosis):")
    if "Cirrhosis" in col2node:
        print(col2node["Cirrhosis"].text)
