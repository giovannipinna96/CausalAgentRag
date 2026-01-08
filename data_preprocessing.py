#!/usr/bin/env python3
"""
Data Preprocessing Module for CausalAgentRag

This module processes HEPAR dataset files and generates:
1. JSON file with column names
2. JSON file with causal graph structure (nodes and edges)
3. PNG visualization of the causal Bayesian network

The HEPAR II dataset is a medical Bayesian network for liver disorder diagnosis.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path(__file__).parent / "data"
PATIENTS_FILE = DATA_DIR / "HEPAR_simulated_patients.csv"
ARCS_FILE = DATA_DIR / "HEPAR_arcs.csv"
OUTPUT_COLUMNS_JSON = DATA_DIR / "hepar_columns.json"
OUTPUT_GRAPH_JSON = DATA_DIR / "hepar_graph.json"
OUTPUT_GRAPH_PNG = DATA_DIR / "hepar_graph.png"

# Node categories for visualization coloring
# Based on HEPAR II Bayesian network documentation
DISEASE_NODES = {
    "THepatitis",
    "ChHepatitis",
    "PBC",
    "Steatosis",
    "Cirrhosis",
    "Hyperbilirubinemia",
    "RHepatitis",
    "fibrosis",
    "carcinoma",
}

RISK_FACTOR_NODES = {
    "alcoholism",
    "vh_amn",
    "hepatotoxic",
    "hospital",
    "surgery",
    "gallstones",
    "choledocholithotomy",
    "injections",
    "transfusion",
    "sex",
    "age",
    "diabetes",
    "obesity",
}


def load_patient_data(filepath: Path) -> pd.DataFrame:
    """
    Load patient data from CSV file.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing patient data

    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.ParserError: If the file cannot be parsed
    """
    logger.info(f"Loading patient data from {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(f"Patient data file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} patient records with {len(df.columns)} columns")

    return df


def extract_columns(df: pd.DataFrame) -> list[str]:
    """
    Extract column names from DataFrame.

    Args:
        df: DataFrame containing patient data

    Returns:
        List of column names
    """
    columns = df.columns.tolist()
    logger.info(f"Extracted {len(columns)} column names")
    return columns


def build_causal_graph(arcs_filepath: Path) -> nx.DiGraph:
    """
    Build directed causal graph from arcs CSV file.

    The arcs file contains directed edges in format: from,to
    representing causal relationships in the Bayesian network.

    Args:
        arcs_filepath: Path to the arcs CSV file

    Returns:
        NetworkX DiGraph representing the causal structure

    Raises:
        FileNotFoundError: If the file does not exist
    """
    logger.info(f"Building causal graph from {arcs_filepath}")

    if not arcs_filepath.exists():
        raise FileNotFoundError(f"Arcs file not found: {arcs_filepath}")

    # Read arcs CSV
    arcs_df = pd.read_csv(arcs_filepath)
    logger.info(f"Loaded {len(arcs_df)} causal edges")

    # Create directed graph
    graph = nx.DiGraph()

    # Add edges from the arcs data
    for _, row in arcs_df.iterrows():
        graph.add_edge(row["from"], row["to"])

    logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    return graph


def export_columns_json(columns: list[str], output_path: Path) -> None:
    """
    Export column names to JSON file.

    Args:
        columns: List of column names
        output_path: Path for the output JSON file
    """
    logger.info(f"Exporting column names to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(columns, f, indent=2)

    logger.info(f"Saved {len(columns)} column names to JSON")


def export_graph_json(graph: nx.DiGraph, output_path: Path) -> None:
    """
    Export graph structure to JSON file.

    The output format is:
    {
        "nodes": ["node1", "node2", ...],
        "edges": [["from1", "to1"], ["from2", "to2"], ...]
    }

    Args:
        graph: NetworkX DiGraph to export
        output_path: Path for the output JSON file
    """
    logger.info(f"Exporting graph structure to {output_path}")

    graph_data = {
        "nodes": list(graph.nodes()),
        "edges": [[u, v] for u, v in graph.edges()],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    logger.info(f"Saved graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges to JSON")


def get_node_colors(graph: nx.DiGraph) -> list[str]:
    """
    Assign colors to nodes based on their category.

    - Disease nodes: orange
    - Risk factor nodes: blue
    - Observation/test nodes: green

    Args:
        graph: NetworkX DiGraph

    Returns:
        List of color strings for each node
    """
    colors = []
    for node in graph.nodes():
        if node in DISEASE_NODES:
            colors.append("#FF8C00")  # Orange for diseases
        elif node in RISK_FACTOR_NODES:
            colors.append("#4169E1")  # Royal blue for risk factors
        else:
            colors.append("#32CD32")  # Lime green for observations/tests

    return colors


def visualize_graph(graph: nx.DiGraph, output_path: Path) -> None:
    """
    Create and save PNG visualization of the causal graph.

    Uses spring layout for node positioning and colors nodes
    by category (diseases, risk factors, observations).

    Args:
        graph: NetworkX DiGraph to visualize
        output_path: Path for the output PNG file
    """
    logger.info(f"Generating graph visualization to {output_path}")

    # Create figure with large size for readability
    fig, ax = plt.subplots(figsize=(24, 20))

    # Use spring layout with increased k for better spacing
    logger.info("Computing graph layout...")
    pos = nx.spring_layout(graph, k=2.5, iterations=100, seed=42)

    # Get node colors by category
    node_colors = get_node_colors(graph)

    # Draw the graph
    logger.info("Drawing graph...")
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.9,
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="#666666",
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=7,
        font_weight="bold",
        ax=ax,
    )

    # Add legend
    legend_elements = [
        plt.scatter([], [], c="#FF8C00", s=100, label="Diseases"),
        plt.scatter([], [], c="#4169E1", s=100, label="Risk Factors"),
        plt.scatter([], [], c="#32CD32", s=100, label="Observations/Tests"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    # Set title
    ax.set_title(
        "HEPAR II Causal Bayesian Network\n(Liver Disorder Diagnosis)",
        fontsize=14,
        fontweight="bold",
    )

    # Remove axes
    ax.axis("off")

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved graph visualization to {output_path}")


def main() -> None:
    """
    Main function to run all preprocessing steps.

    Steps:
    1. Load patient data and extract column names
    2. Build causal graph from arcs file
    3. Export column names to JSON
    4. Export graph structure to JSON
    5. Generate PNG visualization of the graph
    """
    logger.info("=" * 60)
    logger.info("Starting HEPAR data preprocessing")
    logger.info("=" * 60)

    # Step 1: Load patient data and extract columns
    logger.info("\n--- Step 1: Loading patient data ---")
    df = load_patient_data(PATIENTS_FILE)
    columns = extract_columns(df)

    # Step 2: Build causal graph
    logger.info("\n--- Step 2: Building causal graph ---")
    graph = build_causal_graph(ARCS_FILE)

    # Step 3: Export columns to JSON
    logger.info("\n--- Step 3: Exporting columns to JSON ---")
    export_columns_json(columns, OUTPUT_COLUMNS_JSON)

    # Step 4: Export graph to JSON
    logger.info("\n--- Step 4: Exporting graph to JSON ---")
    export_graph_json(graph, OUTPUT_GRAPH_JSON)

    # Step 5: Generate graph visualization
    logger.info("\n--- Step 5: Generating graph visualization ---")
    visualize_graph(graph, OUTPUT_GRAPH_PNG)

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing completed successfully!")
    logger.info("=" * 60)
    logger.info(f"Output files:")
    logger.info(f"  - Columns JSON: {OUTPUT_COLUMNS_JSON}")
    logger.info(f"  - Graph JSON:   {OUTPUT_GRAPH_JSON}")
    logger.info(f"  - Graph PNG:    {OUTPUT_GRAPH_PNG}")


if __name__ == "__main__":
    main()
