"""
Attention-optimized prompting for LLM column selection.

Addresses the "Lost in the Middle" phenomenon where LLMs show 10-40%
performance degradation when relevant information is in the middle of context.

Key insight: Place most relevant columns at START and END of the list,
less relevant in the MIDDLE where attention is weakest.

Reference: "Lost in the Middle: How Language Models Use Long Contexts"
(Liu et al., 2023)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class PromptConfig:
    """Configuration for prompt optimization."""
    format_type: str = "structured"  # "structured" | "compact" | "create_table"
    max_sample_values: int = 3
    use_attention_ordering: bool = True
    include_descriptions: bool = True


def reorder_for_attention(
    columns: List[str],
    scores: List[float],
) -> List[str]:
    """
    Reorder columns for optimal LLM attention.

    Strategy:
    - Most relevant → START (best attention)
    - Second most relevant → END (good attention)
    - Others → MIDDLE (lower attention, acceptable for less relevant)

    This interleaving places high-relevance items at attention hotspots.

    Args:
        columns: List of column names (should already be sorted by relevance)
        scores: Corresponding relevance scores

    Returns:
        Reordered list with best items at start/end
    """
    if len(columns) <= 2:
        return columns

    # Sort by score to get relevance order
    sorted_pairs = sorted(zip(columns, scores), key=lambda x: x[1], reverse=True)
    sorted_cols = [col for col, _ in sorted_pairs]

    # Interleave: odd indices go to start, even to end
    reordered = [None] * len(sorted_cols)
    left, right = 0, len(sorted_cols) - 1

    for i, col in enumerate(sorted_cols):
        if i % 2 == 0:
            reordered[left] = col
            left += 1
        else:
            reordered[right] = col
            right -= 1

    return reordered


def reorder_with_indices(
    items: List[Any],
    scores: List[float],
) -> Tuple[List[Any], List[int]]:
    """
    Reorder items for attention and return original indices.

    Useful when you need to track which positions items came from.

    Args:
        items: List of items to reorder
        scores: Corresponding relevance scores

    Returns:
        Tuple of (reordered_items, original_indices)
    """
    if len(items) <= 2:
        return items, list(range(len(items)))

    # Get sorted indices
    indexed = [(i, item, score) for i, (item, score) in enumerate(zip(items, scores))]
    indexed.sort(key=lambda x: x[2], reverse=True)

    # Interleave
    reordered_items = [None] * len(items)
    reordered_indices = [None] * len(items)
    left, right = 0, len(items) - 1

    for i, (orig_idx, item, _) in enumerate(indexed):
        if i % 2 == 0:
            reordered_items[left] = item
            reordered_indices[left] = orig_idx
            left += 1
        else:
            reordered_items[right] = item
            reordered_indices[right] = orig_idx
            right -= 1

    return reordered_items, reordered_indices


def format_column_structured(
    column: str,
    description: Optional[str] = None,
    score: Optional[float] = None,
    rank: Optional[int] = None,
) -> str:
    """
    Format a single column in structured format.

    Args:
        column: Column name
        description: Optional description
        score: Optional relevance score
        rank: Optional rank number

    Returns:
        Formatted string
    """
    parts = []
    if rank is not None:
        parts.append(f"{rank}.")
    parts.append(column)
    if description:
        parts.append(f"- {description}")
    if score is not None:
        parts.append(f"(score: {score:.2f})")
    return " ".join(parts)


def format_column_compact(
    column: str,
    description: Optional[str] = None,
) -> str:
    """
    Format column in compact format (minimal tokens).

    Args:
        column: Column name
        description: Optional description

    Returns:
        Compact formatted string
    """
    if description:
        return f"{column}: {description}"
    return column


def format_columns_create_table(
    columns: List[str],
    descriptions: Optional[Dict[str, str]] = None,
    data_types: Optional[Dict[str, str]] = None,
    sample_values: Optional[Dict[str, List[Any]]] = None,
    table_name: str = "dataset",
    max_samples: int = 3,
) -> str:
    """
    Format columns as CREATE TABLE statement.

    Research shows CREATE TABLE format with sample rows achieves best accuracy
    for schema understanding tasks.

    Args:
        columns: List of column names
        descriptions: Optional column descriptions
        data_types: Optional column data types
        sample_values: Optional sample values per column
        table_name: Name for the table
        max_samples: Maximum sample values to include

    Returns:
        CREATE TABLE formatted string
    """
    descriptions = descriptions or {}
    data_types = data_types or {}
    sample_values = sample_values or {}

    lines = [f"CREATE TABLE {table_name} ("]

    for col in columns:
        dtype = data_types.get(col, "VARCHAR")
        desc = descriptions.get(col, "")
        comment = f"-- {desc}" if desc else ""
        lines.append(f"    {col} {dtype}, {comment}")

    lines.append(");")

    # Add sample values comment (max 3 - more decreases performance)
    if sample_values:
        sample_parts = []
        for col in columns[:max_samples]:
            if col in sample_values and sample_values[col]:
                val = sample_values[col][0]
                sample_parts.append(str(val))
        if sample_parts:
            lines.append(f"/* Sample row: ({', '.join(sample_parts)}) */")

    return "\n".join(lines)


def format_schema_prompt(
    columns: List[str],
    scores: List[float],
    query: str,
    descriptions: Optional[Dict[str, str]] = None,
    config: Optional[PromptConfig] = None,
) -> str:
    """
    Generate attention-optimized prompt for column selection.

    Args:
        columns: List of candidate columns
        scores: Relevance scores for each column
        query: User's query
        descriptions: Optional column descriptions
        config: Prompt configuration

    Returns:
        Optimized prompt string
    """
    config = config or PromptConfig()
    descriptions = descriptions or {}

    # Reorder for attention if enabled
    if config.use_attention_ordering:
        columns = reorder_for_attention(columns, scores)

    # Format based on type
    if config.format_type == "create_table":
        schema_text = format_columns_create_table(
            columns, descriptions, table_name="medical_data"
        )
    elif config.format_type == "compact":
        lines = []
        for col in columns:
            lines.append(format_column_compact(col, descriptions.get(col) if config.include_descriptions else None))
        schema_text = "\n".join(lines)
    else:  # structured
        lines = []
        for i, col in enumerate(columns, 1):
            lines.append(format_column_structured(
                col,
                descriptions.get(col) if config.include_descriptions else None,
                rank=i
            ))
        schema_text = "\n".join(lines)

    return f"""=== AVAILABLE COLUMNS ===
{schema_text}

=== USER QUESTION ===
{query}

Select the columns needed to answer this question. Return ONLY a JSON array of column names."""


def build_optimized_prompt(
    query: str,
    columns: List[str],
    scores: Optional[List[float]] = None,
    descriptions: Optional[Dict[str, str]] = None,
    graph_edges: Optional[List[Dict[str, str]]] = None,
    use_attention_ordering: bool = True,
) -> str:
    """
    Build an optimized prompt for column selection.

    This is the main function to use for benchmark integration.

    Args:
        query: User query
        columns: Candidate columns
        scores: Optional relevance scores (if None, assumes already ordered)
        descriptions: Optional column descriptions
        graph_edges: Optional causal graph edges
        use_attention_ordering: Whether to reorder for attention

    Returns:
        Optimized prompt string
    """
    # Apply attention ordering if scores provided
    if use_attention_ordering and scores is not None:
        columns = reorder_for_attention(columns, scores)

    descriptions = descriptions or {}

    # Build header
    header = (
        "You are selecting dataset columns needed to answer the user's question.\n"
        "Rules:\n"
        "- Return a JSON array of column names.\n"
        "- Use ONLY column names from the provided list.\n"
        "- Include all necessary columns; exclude unrelated columns.\n"
        "- For prediction tasks, avoid data leakage (do not include effects of the target).\n"
        "\n"
    )

    # Build column block with descriptions
    if descriptions:
        col_lines = []
        for col in columns:
            desc = descriptions.get(col, "")
            if desc:
                col_lines.append(f"- {col}: {desc}")
            else:
                col_lines.append(f"- {col}")
        col_block = "\n".join(col_lines)
    else:
        col_block = "\n".join([f"- {c}" for c in columns])

    # Build graph block if provided
    graph_block = ""
    if graph_edges is not None:
        import json
        graph_block = "\nCausal graph (JSON edges):\n" + json.dumps(graph_edges, indent=2)

    return (
        f"{header}"
        f"Question: {query}\n\n"
        f"Columns:\n{col_block}\n"
        f"{graph_block}\n\n"
        "Return ONLY the JSON array."
    )


class PromptOptimizer:
    """
    Stateful prompt optimizer with caching and configuration.

    Use this class when you need to optimize multiple prompts with
    consistent configuration.
    """

    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize prompt optimizer.

        Args:
            config: Prompt configuration
        """
        self.config = config or PromptConfig()

    def optimize(
        self,
        query: str,
        columns: List[str],
        scores: Optional[List[float]] = None,
        descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate optimized prompt.

        Args:
            query: User query
            columns: Candidate columns
            scores: Relevance scores
            descriptions: Column descriptions

        Returns:
            Optimized prompt string
        """
        return build_optimized_prompt(
            query=query,
            columns=columns,
            scores=scores,
            descriptions=descriptions,
            use_attention_ordering=self.config.use_attention_ordering,
        )

    def get_column_order(
        self,
        columns: List[str],
        scores: List[float],
    ) -> List[str]:
        """
        Get attention-optimized column order.

        Args:
            columns: Original columns
            scores: Relevance scores

        Returns:
            Reordered column list
        """
        if self.config.use_attention_ordering:
            return reorder_for_attention(columns, scores)
        return columns
