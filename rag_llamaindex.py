#!/usr/bin/env python3
"""
RAG Module with LlamaIndex for CausalAgentRag

This module implements a Retrieval-Augmented Generation (RAG) system
that indexes medical column names with their meanings for semantic retrieval.

The system uses:
- LlamaIndex for document indexing and retrieval
- HuggingFace transformers for the LLM component
- all-MiniLM-L6-v2 for embeddings

Prerequisites:
    - A CUDA-capable GPU with sufficient VRAM (~6GB for Qwen2.5-3B-Instruct)
    - HuggingFace transformers and accelerate libraries

Usage:
    # Demo mode
    uv run rag_llamaindex.py

    # Benchmark mode with top-k retrieval
    uv run rag_llamaindex.py --mode topk --benchmark data/causal_rag_benchmark_v3.json --output results_rag_topk.json

    # Benchmark mode with threshold retrieval
    uv run rag_llamaindex.py --mode threshold --benchmark data/causal_rag_benchmark_v3.json --output results_rag_threshold.json
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.5
DEFAULT_MAX_NEW_TOKENS = 512

# File paths
DATA_DIR = Path(__file__).parent / "data"
COLUMNS_JSON = DATA_DIR / "hepar_columns.json"


# HEPAR II Column Meanings Dictionary
# Based on HEPAR II Bayesian Network documentation for liver disorder diagnosis
# Source: Onisko et al. "HEPAR II: A Probabilistic Model for Diagnosis of Liver Disorders"
HEPAR_COLUMN_MEANINGS: dict[str, str] = {
    # === DISEASES ===
    "THepatitis": "Toxic Hepatitis - liver inflammation caused by exposure to toxic substances, drugs, or chemicals",
    "ChHepatitis": "Chronic Hepatitis - persistent liver inflammation lasting more than six months, can be active or persistent",
    "RHepatitis": "Reactive Hepatitis - secondary liver inflammation as a reaction to other diseases or conditions",
    "PBC": "Primary Biliary Cirrhosis (now called Primary Biliary Cholangitis) - autoimmune disease causing bile duct destruction",
    "Steatosis": "Hepatic Steatosis (Fatty Liver Disease) - accumulation of fat in liver cells, can be alcoholic or non-alcoholic",
    "Cirrhosis": "Liver Cirrhosis - late-stage scarring (fibrosis) of the liver, can be compensated or decompensated",
    "Hyperbilirubinemia": "Elevated bilirubin levels in blood causing jaundice, indicates liver dysfunction or bile obstruction",
    "fibrosis": "Liver Fibrosis - formation of scar tissue in the liver, precursor to cirrhosis",
    "carcinoma": "Hepatocellular Carcinoma (HCC) - primary liver cancer, often associated with cirrhosis or chronic hepatitis",
    # === RISK FACTORS ===
    "alcoholism": "Alcohol Use Disorder - chronic alcohol abuse, major risk factor for alcoholic liver disease",
    "vh_amn": "Viral Hepatitis Anamnesis - patient history of viral hepatitis infection (past or present)",
    "hepatotoxic": "Hepatotoxic Drug/Substance Exposure - exposure to medications or substances toxic to liver",
    "hospital": "Hospital Admission History - previous hospitalizations which increase infection risk",
    "surgery": "Surgical History - previous surgical procedures with potential blood exposure",
    "gallstones": "Cholelithiasis (Gallstones) - presence of stones in gallbladder, can cause bile duct obstruction",
    "choledocholithotomy": "Choledocholithotomy - surgical removal of stones from bile duct, indicates history of biliary disease",
    "injections": "Injection History - history of injections which may transmit blood-borne infections",
    "transfusion": "Blood Transfusion History - previous blood transfusions, risk for viral hepatitis transmission",
    "diabetes": "Diabetes Mellitus - metabolic disorder associated with non-alcoholic fatty liver disease",
    "obesity": "Obesity - excessive body weight, major risk factor for non-alcoholic steatohepatitis (NASH)",
    "sex": "Patient Sex/Gender - biological sex (male/female), some liver diseases show gender predisposition",
    "age": "Patient Age - age group of the patient, risk factor for various liver conditions",
    # === LABORATORY TESTS - LIVER ENZYMES ===
    "alt": "ALT (Alanine Aminotransferase/SGPT) - liver enzyme, elevated in hepatocellular damage",
    "ast": "AST (Aspartate Aminotransferase/SGOT) - enzyme found in liver and muscle, elevated in liver damage",
    "ggtp": "GGTP/GGT (Gamma-Glutamyl Transpeptidase) - enzyme elevated in bile duct disease and alcohol use",
    "phosphatase": "Alkaline Phosphatase (ALP) - enzyme elevated in cholestatic liver disease and bile obstruction",
    "amylase": "Amylase - digestive enzyme, elevated in pancreatic and biliary diseases",
    # === LABORATORY TESTS - LIVER FUNCTION ===
    "bilirubin": "Total Bilirubin - breakdown product of hemoglobin, elevated in liver disease and jaundice",
    "albumin": "Serum Albumin - protein produced by liver, decreased in chronic liver disease",
    "proteins": "Total Serum Proteins - indicator of liver synthetic function",
    "cholesterol": "Total Cholesterol - lipid marker, can be altered in liver disease",
    "triglycerides": "Triglycerides - blood lipids, elevated in fatty liver disease",
    # === LABORATORY TESTS - COAGULATION ===
    "platelet": "Platelet Count - blood clotting cells, decreased in cirrhosis due to portal hypertension",
    "inr": "INR (International Normalized Ratio) - coagulation test, prolonged in liver failure",
    "bleeding": "Bleeding Tendency/Coagulopathy - abnormal bleeding due to impaired liver clotting factor synthesis",
    # === LABORATORY TESTS - INFLAMMATION ===
    "ESR": "ESR (Erythrocyte Sedimentation Rate) - inflammatory marker, elevated in hepatitis",
    # === LABORATORY TESTS - VIRAL MARKERS ===
    "hbsag": "HBsAg (Hepatitis B Surface Antigen) - marker of active Hepatitis B infection",
    "hbsag_anti": "Anti-HBs (Hepatitis B Surface Antibody) - marker of immunity to Hepatitis B",
    "hbc_anti": "Anti-HBc (Hepatitis B Core Antibody) - marker of past or current Hepatitis B infection",
    "hbeag": "HBeAg (Hepatitis B e Antigen) - marker of high Hepatitis B viral replication",
    "hcv_anti": "Anti-HCV (Hepatitis C Antibody) - marker of Hepatitis C exposure or infection",
    # === LABORATORY TESTS - AUTOIMMUNE ===
    "ama": "AMA (Antimitochondrial Antibodies) - autoantibody marker for Primary Biliary Cholangitis",
    "le_cells": "LE Cells (Lupus Erythematosus Cells) - marker for autoimmune conditions",
    # === LABORATORY TESTS - OTHER ===
    "urea": "Blood Urea Nitrogen (BUN) - kidney function marker, can be elevated in hepatorenal syndrome",
    "density": "Liver Density - imaging measurement indicating fat content or fibrosis",
    # === SYMPTOMS ===
    "fatigue": "Fatigue/Weakness - common symptom of chronic liver disease and hepatitis",
    "anorexia": "Anorexia/Loss of Appetite - reduced appetite common in liver diseases",
    "nausea": "Nausea - feeling of sickness, common in hepatitis and liver dysfunction",
    "itching": "Pruritus (Itching) - skin itching caused by bile salt accumulation in cholestatic disease",
    "jaundice": "Jaundice/Icterus - yellowing of skin and eyes due to elevated bilirubin",
    "pain": "Abdominal Pain - general abdominal discomfort",
    "upper_pain": "Upper Abdominal Pain - pain in upper abdomen, often related to liver or gallbladder",
    "pain_ruq": "Right Upper Quadrant Pain - pain in RUQ where liver is located",
    "pressure_ruq": "Right Upper Quadrant Pressure - pressure sensation in liver area",
    "hepatalgia": "Hepatalgia - liver pain, direct pain from liver capsule distension",
    "flatulence": "Flatulence/Bloating - abdominal gas, common in digestive disorders",
    "joints": "Joint Pain/Arthralgia - joint pain associated with autoimmune hepatitis or viral hepatitis",
    # === PHYSICAL EXAMINATION FINDINGS ===
    "hepatomegaly": "Hepatomegaly - enlarged liver detected on physical examination",
    "spleen": "Splenomegaly - enlarged spleen, common in portal hypertension",
    "ascites": "Ascites - fluid accumulation in abdomen, sign of decompensated cirrhosis",
    "edema": "Peripheral Edema - swelling in legs/feet due to low albumin or heart failure",
    "spiders": "Spider Angiomas/Nevi - small spider-like blood vessels on skin, sign of chronic liver disease",
    "palms": "Palmar Erythema - reddening of palms, associated with chronic liver disease",
    "skin": "Skin Changes - various dermatological manifestations of liver disease",
    "encephalopathy": "Hepatic Encephalopathy - brain dysfunction due to liver failure, confusion to coma",
    "consciousness": "Altered Consciousness - mental status changes in hepatic encephalopathy",
    "edge": "Liver Edge - palpable liver edge characteristics on examination",
    "irregular_liver": "Irregular Liver Surface - nodular or irregular liver surface suggesting cirrhosis",
    # === OTHER ===
    "alcohol": "Current Alcohol Consumption - present alcohol intake level",
    "fat": "Dietary Fat Intake - fat consumption in diet, relevant for fatty liver",
}


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""

    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    llm_temperature: float = 0.1
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    top_k: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    use_quantization: bool = False
    quantization_bits: int = 4  # 4 or 8 bit quantization when enabled


class ColumnRAG:
    """
    RAG system for medical column retrieval.

    This class indexes medical column names with their descriptions
    and retrieves relevant columns based on semantic similarity.

    Attributes:
        config: RAG configuration parameters
        index: LlamaIndex vector store index
    """

    def __init__(self, config: RAGConfig | None = None):
        """
        Initialize the Column RAG system.

        Args:
            config: RAG configuration. Uses defaults if not provided.
        """
        self.config = config or RAGConfig()
        self.index = None
        self._documents = []

        logger.info("Initializing ColumnRAG")
        logger.info(f"  LLM Model: {self.config.llm_model}")
        logger.info(f"  Embedding Model: {self.config.embed_model}")

        # Setup LlamaIndex settings
        self._setup_settings()

    def _setup_settings(self) -> None:
        """Configure LlamaIndex global settings."""
        logger.info("Setting up LlamaIndex components...")

        # Check GPU availability
        logger.info("Verifying GPU availability...")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. A GPU is required for HuggingFace LLM inference. "
                "Please ensure you have a CUDA-capable GPU and the correct PyTorch version installed."
            )
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")

        # Setup embedding model (local, no API needed)
        logger.info(f"Loading embedding model: {self.config.embed_model}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embed_model,
        )
        logger.info("Embedding model loaded successfully")

        # Setup quantization config if enabled
        model_kwargs = {"trust_remote_code": True}
        if self.config.use_quantization:
            logger.info(f"Setting up {self.config.quantization_bits}-bit quantization...")
            if self.config.quantization_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif self.config.quantization_bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(
                    f"Invalid quantization_bits: {self.config.quantization_bits}. "
                    "Must be 4 or 8."
                )
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.float16
            logger.info("Using float16 precision (no quantization)")

        # Setup LLM via HuggingFace
        logger.info(f"Loading HuggingFace LLM: {self.config.llm_model}")
        logger.info("This may take a few minutes on first run (downloading model weights)...")
        Settings.llm = HuggingFaceLLM(
            model_name=self.config.llm_model,
            tokenizer_name=self.config.llm_model,
            max_new_tokens=self.config.max_new_tokens,
            generate_kwargs={"temperature": self.config.llm_temperature, "do_sample": True},
            model_kwargs=model_kwargs,
            device_map="auto",
        )
        logger.info("LLM loaded successfully")

    def build_index(
        self,
        columns: list[str],
        column_meanings: dict[str, str] | None = None,
    ) -> None:
        """
        Build vector index from column names and their meanings.

        Each column is indexed as a document containing:
        - Column name
        - Medical description/meaning
        - Metadata for retrieval

        Args:
            columns: List of column names to index
            column_meanings: Dictionary mapping column names to descriptions.
                            Uses HEPAR_COLUMN_MEANINGS if not provided.
        """
        logger.info(f"Building index for {len(columns)} columns")

        meanings = column_meanings or HEPAR_COLUMN_MEANINGS

        # Create documents for each column
        self._documents = []
        for col in columns:
            meaning = meanings.get(col, f"Medical feature: {col}")

            # Create document with column name and meaning
            doc_text = f"Column: {col}\nDescription: {meaning}"

            doc = Document(
                text=doc_text,
                metadata={
                    "column_name": col,
                    "description": meaning,
                },
            )
            self._documents.append(doc)

        logger.info(f"Created {len(self._documents)} documents")

        # Build vector index
        logger.info("Building vector store index...")
        self.index = VectorStoreIndex.from_documents(
            self._documents,
            show_progress=True,
        )
        logger.info("Index built successfully")

    def retrieve_top_k(self, query: str, k: int | None = None) -> list[dict]:
        """
        Retrieve top-k most relevant columns for a query.

        Args:
            query: Natural language query
            k: Number of results to return. Uses config default if not provided.

        Returns:
            List of dictionaries with column_name, description, and score

        Raises:
            ValueError: If index hasn't been built
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        k = k or self.config.top_k
        logger.info(f"Retrieving top-{k} columns for query: '{query}'")

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=k,
        )

        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "column_name": node.metadata.get("column_name", ""),
                "description": node.metadata.get("description", ""),
                "score": node.score,
            })

        logger.info(f"Retrieved {len(results)} columns")
        for r in results:
            logger.info(f"  - {r['column_name']}: {r['score']:.4f}")

        return results

    def retrieve_by_threshold(
        self,
        query: str,
        threshold: float | None = None,
        max_results: int = 20,
    ) -> list[dict]:
        """
        Retrieve columns above a similarity threshold.

        Args:
            query: Natural language query
            threshold: Minimum similarity score. Uses config default if not provided.
            max_results: Maximum number of results to evaluate

        Returns:
            List of dictionaries with column_name, description, and score

        Raises:
            ValueError: If index hasn't been built
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        threshold = threshold or self.config.similarity_threshold
        logger.info(
            f"Retrieving columns with similarity >= {threshold} for query: '{query}'"
        )

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=max_results,
        )

        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            if node.score >= threshold:
                results.append({
                    "column_name": node.metadata.get("column_name", ""),
                    "description": node.metadata.get("description", ""),
                    "score": node.score,
                })

        logger.info(f"Retrieved {len(results)} columns above threshold {threshold}")
        for r in results:
            logger.info(f"  - {r['column_name']}: {r['score']:.4f}")

        return results

    def query_with_generation(self, query: str, top_k: int | None = None) -> str:
        """
        Query the RAG system with LLM-generated response.

        Retrieves relevant columns and uses the LLM to generate
        a natural language response explaining the column selection.

        Args:
            query: Natural language query
            top_k: Number of columns to consider

        Returns:
            LLM-generated response explaining the relevant columns

        Raises:
            ValueError: If index hasn't been built
            RuntimeError: If LLM generation fails
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        k = top_k or self.config.top_k
        logger.info(f"Querying RAG with LLM generation for: '{query}'")

        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=k,
        )

        # Query with LLM generation
        response = query_engine.query(query)

        logger.info("Generated response successfully")
        return str(response)


def load_hepar_columns() -> list[str]:
    """
    Load HEPAR column names from JSON file.

    Returns:
        List of column names

    Raises:
        FileNotFoundError: If columns JSON file doesn't exist
    """
    logger.info(f"Loading columns from {COLUMNS_JSON}")

    if not COLUMNS_JSON.exists():
        raise FileNotFoundError(
            f"Columns file not found: {COLUMNS_JSON}\n"
            "Please run data_preprocessing.py first."
        )

    with open(COLUMNS_JSON, encoding="utf-8") as f:
        columns = json.load(f)

    logger.info(f"Loaded {len(columns)} columns")
    return columns


def load_benchmark_queries(benchmark_path: str) -> list[dict]:
    """
    Load queries from benchmark file.

    Args:
        benchmark_path: Path to benchmark JSON file

    Returns:
        List of query dictionaries with 'id' and 'query' keys

    Raises:
        FileNotFoundError: If benchmark file doesn't exist
    """
    logger.info(f"Loading benchmark queries from {benchmark_path}")

    benchmark_file = Path(benchmark_path)
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    with open(benchmark_file, encoding="utf-8") as f:
        benchmark = json.load(f)

    queries = [{"id": q["id"], "query": q["query"]} for q in benchmark["queries"]]
    logger.info(f"Loaded {len(queries)} benchmark queries")
    return queries


def save_results_json(
    results: list[dict],
    output_path: str,
    mode: str,
    embed_model: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> None:
    """
    Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
        mode: Retrieval mode (topk or threshold)
        embed_model: Embedding model used
        top_k: Top-k value (for topk mode)
        threshold: Threshold value (for threshold mode)
    """
    logger.info(f"Saving results to {output_path}")

    output = {
        "model": f"RAG ({embed_model})",
        "mode": f"rag_{mode}",
        "config": {
            "embed_model": embed_model,
            "mode": mode,
        },
        "results": results,
    }

    if top_k is not None:
        output["config"]["top_k"] = top_k
    if threshold is not None:
        output["config"]["threshold"] = threshold

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG-based column retrieval for medical data analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo
  uv run rag_llamaindex.py

  # Run benchmark with top-k retrieval
  uv run rag_llamaindex.py --mode topk --benchmark data/causal_rag_benchmark_v3.json --output results_rag_topk.json --top-k 10

  # Run benchmark with threshold retrieval
  uv run rag_llamaindex.py --mode threshold --benchmark data/causal_rag_benchmark_v3.json --output results_rag_threshold.json --threshold 0.3
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["topk", "threshold"],
        default="topk",
        help="Retrieval mode: 'topk' for top-k retrieval, 'threshold' for similarity threshold. Default: topk",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark JSON file. If provided, runs on benchmark queries.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of columns to retrieve (for topk mode). Default: {DEFAULT_TOP_K}",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold (for threshold mode). Default: {DEFAULT_SIMILARITY_THRESHOLD}",
    )

    parser.add_argument(
        "--embed-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model to use. Default: {DEFAULT_EMBED_MODEL}",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM loading (retrieval only, faster startup)",
    )

    return parser.parse_args()


class ColumnRAGRetrievalOnly:
    """
    Lightweight RAG system for retrieval only (no LLM).

    This class is faster to initialize as it skips LLM loading.
    Use this for benchmarking retrieval performance.
    """

    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL):
        """
        Initialize the retrieval-only RAG system.

        Args:
            embed_model: Embedding model name
        """
        self.embed_model = embed_model
        self.index = None
        self._documents = []

        logger.info("Initializing ColumnRAG (retrieval only, no LLM)")
        logger.info(f"  Embedding Model: {embed_model}")

        # Setup embedding model only
        logger.info(f"Loading embedding model: {embed_model}")
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)
        logger.info("Embedding model loaded successfully")

    def build_index(
        self,
        columns: list[str],
        column_meanings: dict[str, str] | None = None,
    ) -> None:
        """Build vector index from column names and their meanings."""
        logger.info(f"Building index for {len(columns)} columns")

        meanings = column_meanings or HEPAR_COLUMN_MEANINGS

        self._documents = []
        for col in columns:
            meaning = meanings.get(col, f"Medical feature: {col}")
            doc_text = f"Column: {col}\nDescription: {meaning}"
            doc = Document(
                text=doc_text,
                metadata={"column_name": col, "description": meaning},
            )
            self._documents.append(doc)

        logger.info(f"Created {len(self._documents)} documents")
        logger.info("Building vector store index...")
        self.index = VectorStoreIndex.from_documents(self._documents, show_progress=True)
        logger.info("Index built successfully")

    def retrieve_top_k(self, query: str, k: int = 10) -> list[dict]:
        """Retrieve top-k most relevant columns for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=k)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            results.append({
                "column_name": node.metadata.get("column_name", ""),
                "description": node.metadata.get("description", ""),
                "score": node.score,
            })
        return results

    def retrieve_by_threshold(
        self, query: str, threshold: float = 0.5, max_results: int = 20
    ) -> list[dict]:
        """Retrieve columns above a similarity threshold."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=max_results)
        nodes = retriever.retrieve(query)

        results = []
        for node in nodes:
            if node.score >= threshold:
                results.append({
                    "column_name": node.metadata.get("column_name", ""),
                    "description": node.metadata.get("description", ""),
                    "score": node.score,
                })
        return results


def run_benchmark(
    rag: ColumnRAGRetrievalOnly,
    benchmark_path: str,
    output_path: str,
    mode: str,
    top_k: int,
    threshold: float,
    embed_model: str,
) -> None:
    """
    Run benchmark evaluation on RAG retrieval.

    Args:
        rag: Initialized RAG system
        benchmark_path: Path to benchmark JSON file
        output_path: Path to save results
        mode: Retrieval mode ('topk' or 'threshold')
        top_k: Number of columns for top-k mode
        threshold: Similarity threshold for threshold mode
        embed_model: Embedding model name
    """
    # Load benchmark queries
    benchmark_queries = load_benchmark_queries(benchmark_path)

    results = []
    for q in benchmark_queries:
        query_id = q["id"]
        query = q["query"]

        logger.info(f"\n--- Query [{query_id}] ---")
        logger.info(f"Query: {query}")

        # Retrieve columns based on mode
        if mode == "topk":
            retrieved = rag.retrieve_top_k(query, k=top_k)
        else:  # threshold mode
            retrieved = rag.retrieve_by_threshold(query, threshold=threshold)

        predicted_columns = [r["column_name"] for r in retrieved]

        logger.info(f"Selected columns ({len(predicted_columns)}): {predicted_columns}")

        results.append({
            "query_id": query_id,
            "query": query,
            "predicted_columns": predicted_columns,
        })

        print(f"\nResult: {predicted_columns}")

    # Save results
    if output_path:
        save_results_json(
            results=results,
            output_path=output_path,
            mode=mode,
            embed_model=embed_model,
            top_k=top_k if mode == "topk" else None,
            threshold=threshold if mode == "threshold" else None,
        )


def main() -> None:
    """
    Main function for RAG-based column retrieval.

    Supports two modes:
    1. Demo mode: Runs sample queries with full RAG (including LLM)
    2. Benchmark mode: Runs retrieval-only on benchmark queries
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RAG Column Retrieval with LlamaIndex")
    logger.info("=" * 60)

    # Load columns
    logger.info("\n--- Loading HEPAR columns ---")
    columns = load_hepar_columns()

    if args.benchmark:
        # Benchmark mode - use lightweight retrieval-only system
        logger.info(f"\n--- Benchmark Mode ---")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Benchmark: {args.benchmark}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Embed model: {args.embed_model}")
        if args.mode == "topk":
            logger.info(f"Top-k: {args.top_k}")
        else:
            logger.info(f"Threshold: {args.threshold}")

        # Initialize retrieval-only RAG (faster, no LLM)
        rag = ColumnRAGRetrievalOnly(embed_model=args.embed_model)
        rag.build_index(columns, HEPAR_COLUMN_MEANINGS)

        # Run benchmark
        run_benchmark(
            rag=rag,
            benchmark_path=args.benchmark,
            output_path=args.output,
            mode=args.mode,
            top_k=args.top_k,
            threshold=args.threshold,
            embed_model=args.embed_model,
        )

        logger.info("\n" + "=" * 60)
        logger.info("Benchmark completed!")
        logger.info("=" * 60)

    else:
        # Demo mode - use full RAG with LLM
        logger.info("\n--- Demo Mode ---")

        config = RAGConfig(
            top_k=8,
            similarity_threshold=0.4,
            embed_model=args.embed_model,
        )

        if args.no_llm:
            # Use lightweight version
            rag = ColumnRAGRetrievalOnly(embed_model=args.embed_model)
            rag.build_index(columns, HEPAR_COLUMN_MEANINGS)
        else:
            rag = ColumnRAG(config)
            rag.build_index(columns, HEPAR_COLUMN_MEANINGS)

        # Sample queries
        sample_queries = [
            "liver enzyme levels and hepatitis markers",
            "risk factors for cirrhosis including alcohol",
            "symptoms of bile duct obstruction",
            "viral hepatitis B infection markers",
            "fatty liver disease indicators",
        ]

        # Demonstrate top-k retrieval
        logger.info("\n" + "=" * 60)
        logger.info("TOP-K RETRIEVAL DEMO")
        logger.info("=" * 60)

        for query in sample_queries[:2]:
            logger.info(f"\n--- Query: '{query}' ---")
            results = rag.retrieve_top_k(query, k=5)
            logger.info("Top 5 relevant columns:")
            for r in results:
                logger.info(f"  [{r['score']:.3f}] {r['column_name']}: {r['description'][:60]}...")

        # Demonstrate threshold retrieval
        logger.info("\n" + "=" * 60)
        logger.info("THRESHOLD RETRIEVAL DEMO")
        logger.info("=" * 60)

        for query in sample_queries[2:4]:
            logger.info(f"\n--- Query: '{query}' ---")
            results = rag.retrieve_by_threshold(query, threshold=0.35)
            logger.info(f"Columns with similarity >= 0.35:")
            for r in results:
                logger.info(f"  [{r['score']:.3f}] {r['column_name']}")

        # Demonstrate LLM generation (only if full RAG)
        if not args.no_llm and isinstance(rag, ColumnRAG):
            logger.info("\n" + "=" * 60)
            logger.info("LLM GENERATION DEMO (HuggingFace)")
            logger.info("=" * 60)

            query = "What columns should I use to analyze hepatitis B infection status?"
            logger.info(f"\n--- Query: '{query}' ---")
            response = rag.query_with_generation(query)
            logger.info(f"LLM Response:\n{response}")

        logger.info("\n" + "=" * 60)
        logger.info("Demo completed!")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
