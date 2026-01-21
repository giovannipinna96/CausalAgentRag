"""
llm_baselines_v3.py

LLM baselines for column selection using HuggingFace.
Adapted from causal_rag_hepar_v3 (which used Ollama).

Three modes:
- SIMPLE: LLM sees only query + list of column names
- DESCRIPTION: LLM sees query + column names + descriptions
- GRAPH: LLM sees query + names + descriptions + causal graph edges

These measure what an LLM can do just from schema context (no retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for HuggingFace LLM."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    temperature: float = 0.1
    max_new_tokens: int = 512
    device: str = "auto"
    use_quantization: bool = False
    quantization_bits: int = 4


class HuggingFaceLLMWrapper:
    """
    Lightweight wrapper for HuggingFace LLM for column selection.

    This provides a simpler interface than the full HuggingFaceLLM class,
    focused on JSON output for column selection.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        logger.info(f"Initializing LLM with model: {self.config.model_name}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for HuggingFace LLM inference")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        # Prepare model kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.config.device,
        }

        if self.config.use_quantization:
            if self.config.quantization_bits == 4:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.config.quantization_bits == 8:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16

        # Load model
        logger.info(f"Loading model {self.config.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        logger.info("Model loaded successfully")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = generated_ids[0][input_length:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)


def _format_columns_only(col_names: Sequence[str]) -> str:
    return "\n".join([f"- {c}" for c in col_names])


def _format_columns_with_desc(col_desc: Dict[str, str], col_names: Sequence[str]) -> str:
    lines = []
    for c in col_names:
        d = col_desc.get(c, "")
        if d:
            lines.append(f"- {c}: {d}")
        else:
            lines.append(f"- {c}")
    return "\n".join(lines)


def _format_graph_edges(edges: Sequence[Tuple[str, str]]) -> str:
    obj = [{"from": u, "to": v} for u, v in edges]
    return json.dumps(obj, indent=2)


def _selector_system_prompt(k: int) -> str:
    return (
        "You are a careful data science assistant. "
        "Your job is to select dataset column names needed to answer the user's request.\n"
        f"Return ONLY valid JSON with the following schema:\n"
        f'{{"selected_columns": ["col1", "col2", ...]}}\n'
        f"Rules:\n"
        f"- Output exactly {k} unique column names.\n"
        f"- Every column name must be EXACTLY one of the provided column names.\n"
        f"- Do not output any explanation.\n"
    )


@dataclass
class LLMColumnSelector:
    """LLM-based column selector with caching."""
    llm: HuggingFaceLLMWrapper
    cache_path: Optional[str] = None

    def __post_init__(self):
        self._cache: Dict[str, Dict] = {}
        if self.cache_path and Path(self.cache_path).exists():
            self._cache = json.loads(Path(self.cache_path).read_text(encoding="utf-8"))

    def _save(self):
        if not self.cache_path:
            return
        Path(self.cache_path).write_text(json.dumps(self._cache, indent=2), encoding="utf-8")

    def select_simple(
        self, query: str, col_names: Sequence[str], k: int, cache_key: str
    ) -> List[str]:
        """Select columns using only column names (no descriptions)."""
        return self._select(
            query=query,
            context=_format_columns_only(col_names),
            col_names=col_names,
            k=k,
            cache_key=cache_key,
        )

    def select_description(
        self, query: str, col_desc: Dict[str, str], col_names: Sequence[str], k: int, cache_key: str
    ) -> List[str]:
        """Select columns using names + descriptions."""
        return self._select(
            query=query,
            context=_format_columns_with_desc(col_desc, col_names),
            col_names=col_names,
            k=k,
            cache_key=cache_key,
        )

    def select_graph(
        self, query: str, col_desc: Dict[str, str], col_names: Sequence[str],
        edges: Sequence[Tuple[str, str]], k: int, cache_key: str
    ) -> List[str]:
        """Select columns using names + descriptions + causal graph."""
        cols = _format_columns_with_desc(col_desc, col_names)
        graph = _format_graph_edges(edges)
        context = cols + "\n\nCausal graph edges (cause -> effect):\n" + graph
        return self._select(
            query=query,
            context=context,
            col_names=col_names,
            k=k,
            cache_key=cache_key,
        )

    def _select(
        self, query: str, context: str, col_names: Sequence[str], k: int, cache_key: str
    ) -> List[str]:
        """Internal selection method with caching."""
        if cache_key in self._cache:
            return list(self._cache[cache_key].get("selected_columns", []))

        sys = _selector_system_prompt(k)
        user = f"USER QUERY:\n{query}\n\nCOLUMNS:\n{context}\n\nReturn JSON now."

        text = self.llm.generate(prompt=user, system_prompt=sys)

        try:
            # Try to extract JSON from response
            # Handle cases where LLM might add extra text
            import re
            json_match = re.search(r'\{[^{}]*"selected_columns"[^{}]*\}', text, re.DOTALL)
            if json_match:
                text = json_match.group()

            obj = json.loads(text)
            cols = obj.get("selected_columns", [])

            # Strict filter to valid columns
            valid = [c for c in cols if c in set(col_names)]

            # Enforce exactly k unique (pad/truncate)
            dedup = []
            for c in valid:
                if c not in dedup:
                    dedup.append(c)
            if len(dedup) < k:
                for c in col_names:
                    if c not in dedup:
                        dedup.append(c)
                    if len(dedup) >= k:
                        break
            out = dedup[:k]
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.warning(f"Raw response: {text[:200]}...")
            # Fallback: alphabetical
            out = list(sorted(col_names))[:k]

        self._cache[cache_key] = {"selected_columns": out, "raw_response": text}
        self._save()
        return out


def demo_llm_baselines():
    """Demo: test LLM baselines on HEPAR data."""
    import pandas as pd
    from schema_v3 import build_column_infos, build_schema_nodes, HEPAR_SYNONYMS
    from causal_graph_v3 import CausalGraph

    print("=" * 60)
    print("LLM Baselines (v3) Demo - HuggingFace")
    print("=" * 60)

    # Load data
    df = pd.read_csv("data/HEPAR_simulated_patients.csv")
    cols = list(df.columns)
    print(f"\nLoaded {len(df)} records with {len(cols)} columns")

    # Build schema
    infos = build_column_infos(df, synonyms=HEPAR_SYNONYMS)
    col_desc = {col: infos[col].description for col in cols}

    # Load graph
    graph = CausalGraph.from_arcs_csv("data/HEPAR_arcs.csv", nodes=cols)
    edges = list(graph.G.edges())
    print(f"Graph: {len(edges)} edges")

    # Initialize LLM
    print("\nInitializing HuggingFace LLM...")
    config = LLMConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        temperature=0.1,
        max_new_tokens=256,
        use_quantization=False,  # Set True if low on VRAM
    )
    llm = HuggingFaceLLMWrapper(config)

    # Initialize selector
    selector = LLMColumnSelector(llm=llm, cache_path="llm_baselines_cache.json")

    # Test queries
    test_queries = [
        "Which columns should I use to predict cirrhosis?",
        "What is the effect of alcoholism on liver disease?",
        "Find columns related to hepatitis B markers",
    ]

    k = 8  # Budget

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Budget: {k} columns")
        print("-" * 60)

        # SIMPLE mode
        print("\n[LLM_SIMPLE] Column names only:")
        simple_cols = selector.select_simple(query, cols, k, cache_key=f"demo::simple::{query}")
        for i, col in enumerate(simple_cols, 1):
            print(f"  {i}. {col}")

        # DESCRIPTION mode
        print("\n[LLM_DESCRIPTION] Names + descriptions:")
        desc_cols = selector.select_description(query, col_desc, cols, k, cache_key=f"demo::desc::{query}")
        for i, col in enumerate(desc_cols, 1):
            print(f"  {i}. {col}")

        # GRAPH mode
        print("\n[LLM_GRAPH] Names + descriptions + causal graph:")
        graph_cols = selector.select_graph(query, col_desc, cols, edges, k, cache_key=f"demo::graph::{query}")
        for i, col in enumerate(graph_cols, 1):
            print(f"  {i}. {col}")

    print(f"\n{'=' * 60}")
    print("Demo complete!")


if __name__ == "__main__":
    demo_llm_baselines()
