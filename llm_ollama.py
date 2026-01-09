#!/usr/bin/env python3
"""
LLM Module with Ollama Framework for CausalAgentRag

This module provides an interface to interact with LLMs via Ollama
for column selection tasks in medical/hepatological data analysis.

Supports two prompt modes:
1. Simple mode: Query + column names only
2. Enriched mode: Query + column names + descriptions + causal graph

Prerequisites:
    - Ollama must be installed and running (https://ollama.com)
    - The model must be pulled: `ollama pull qwen2.5:3b`

Usage:
    python llm_ollama.py
    # or with uv:
    uv run llm_ollama.py
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import ollama
import tiktoken
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 120.0

# File paths
DATA_DIR = Path(__file__).parent / "data"
COLUMNS_JSON = DATA_DIR / "hepar_columns.json"
GRAPH_JSON = DATA_DIR / "hepar_graph.json"

# Global tokenizer cache
_tiktoken_encoder = None
_qwen_tokenizer = None


# HEPAR II Column Meanings Dictionary
# Based on HEPAR II Bayesian Network documentation for liver disorder diagnosis
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
class LLMConfig:
    """Configuration for the Ollama LLM."""

    model_name: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    timeout: float = DEFAULT_TIMEOUT


@dataclass
class TokenCount:
    """Token count results from multiple tokenizers."""

    tiktoken_count: int
    qwen_count: int
    text_length: int

    def __str__(self) -> str:
        return (
            f"Token count (tiktoken cl100k): {self.tiktoken_count} tokens\n"
            f"Token count (Qwen2.5 tokenizer): {self.qwen_count} tokens\n"
            f"Text length: {self.text_length} characters"
        )


def get_tiktoken_encoder():
    """Get or create tiktoken encoder (cached)."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        logger.info("Loading tiktoken encoder (cl100k_base)...")
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


def get_qwen_tokenizer():
    """Get or create Qwen tokenizer (cached)."""
    global _qwen_tokenizer
    if _qwen_tokenizer is None:
        logger.info("Loading Qwen2.5 tokenizer from HuggingFace...")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            trust_remote_code=True,
        )
    return _qwen_tokenizer


def count_tokens(text: str) -> TokenCount:
    """
    Count tokens using multiple tokenizers.

    Args:
        text: Text to tokenize

    Returns:
        TokenCount with tiktoken and Qwen tokenizer counts
    """
    # Tiktoken (OpenAI tokenizer)
    tiktoken_enc = get_tiktoken_encoder()
    tiktoken_count = len(tiktoken_enc.encode(text))

    # Qwen tokenizer (model-specific)
    qwen_tok = get_qwen_tokenizer()
    qwen_count = len(qwen_tok.encode(text))

    return TokenCount(
        tiktoken_count=tiktoken_count,
        qwen_count=qwen_count,
        text_length=len(text),
    )


class OllamaLLM:
    """
    Wrapper class for Ollama LLM client.

    This class provides a clean interface to interact with LLMs
    through the Ollama framework for text generation tasks.

    Attributes:
        config: LLM configuration parameters
        client: Ollama client instance
    """

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize the Ollama LLM client.

        Args:
            config: LLM configuration. Uses defaults if not provided.

        Raises:
            ConnectionError: If Ollama server is not running
        """
        self.config = config or LLMConfig()
        logger.info(f"Initializing OllamaLLM with model: {self.config.model_name}")

        # Verify Ollama connection
        self._verify_connection()

        logger.info("OllamaLLM initialized successfully")

    def _verify_connection(self) -> None:
        """
        Verify connection to Ollama server.

        Raises:
            ConnectionError: If Ollama server is not accessible
        """
        logger.info("Verifying connection to Ollama server...")
        try:
            # List models to verify connection
            models_response = ollama.list()
            # Handle both old dict format and new object format
            if hasattr(models_response, 'models'):
                available_models = [m.model for m in models_response.models]
            else:
                available_models = [m.get("name", m.get("model", "")) for m in models_response.get("models", [])]
            logger.info(f"Connected to Ollama. Available models: {available_models}")

            # Check if required model is available
            model_base = self.config.model_name.split(":")[0]
            model_found = any(model_base in m for m in available_models)

            if not model_found:
                logger.warning(
                    f"Model '{self.config.model_name}' not found locally. "
                    f"Please pull it with: ollama pull {self.config.model_name}"
                )

        except Exception as e:
            error_msg = (
                f"Failed to connect to Ollama server: {e}\n"
                "Please ensure Ollama is installed and running:\n"
                "  1. Install: https://ollama.com/download\n"
                "  2. Start: ollama serve\n"
                f"  3. Pull model: ollama pull {self.config.model_name}"
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Generate text response from the LLM.

        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt for context

        Returns:
            Generated text response

        Raises:
            RuntimeError: If generation fails
        """
        logger.info(f"Generating response for prompt (length: {len(prompt)} chars)")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.config.model_name,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                },
            )

            result = response["message"]["content"]
            logger.info(f"Generated response (length: {len(result)} chars)")
            return result

        except Exception as e:
            error_msg = f"Failed to generate response: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def generate_stream(self, prompt: str, system_prompt: str | None = None):
        """
        Generate text response with streaming output.

        Args:
            prompt: User prompt/question
            system_prompt: Optional system prompt for context

        Yields:
            Text chunks as they are generated

        Raises:
            RuntimeError: If generation fails
        """
        logger.info(f"Streaming response for prompt (length: {len(prompt)} chars)")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = ollama.chat(
                model=self.config.model_name,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                },
                stream=True,
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            error_msg = f"Failed to stream response: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


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


def load_hepar_graph() -> dict:
    """
    Load HEPAR causal graph from JSON file.

    Returns:
        Dictionary with 'nodes' and 'edges' lists

    Raises:
        FileNotFoundError: If graph JSON file doesn't exist
    """
    logger.info(f"Loading causal graph from {GRAPH_JSON}")

    if not GRAPH_JSON.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_JSON}\n"
            "Please run data_preprocessing.py first."
        )

    with open(GRAPH_JSON, encoding="utf-8") as f:
        graph = json.load(f)

    logger.info(f"Loaded graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    return graph


def select_columns_with_llm(
    query: str,
    columns: list[str],
    llm: OllamaLLM,
    max_columns: int = 10,
) -> list[str]:
    """
    Use LLM to select relevant columns for a natural language query.
    (Simple mode: column names only)

    The LLM analyzes the query and determines which database columns
    would be needed to answer it (for use in SQL or Python code).

    Args:
        query: Natural language question about the data
        columns: List of available column names
        llm: OllamaLLM instance
        max_columns: Maximum number of columns to return

    Returns:
        List of selected column names

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    logger.info(f"[SIMPLE MODE] Selecting columns for query: '{query}'")

    system_prompt = """You are a medical data analyst assistant specializing in hepatological (liver) data.
Your task is to select the most relevant database columns needed to answer a query.

The database contains medical patient data with the following column categories:
- Diseases: THepatitis, ChHepatitis, PBC, Steatosis, Cirrhosis, etc.
- Risk factors: alcoholism, diabetes, obesity, hepatotoxic, etc.
- Lab tests: alt, ast, ggtp, bilirubin, albumin, etc.
- Symptoms: fatigue, jaundice, itching, pain, nausea, etc.
- Physical findings: hepatomegaly, spleen, ascites, edema, etc.

IMPORTANT: Respond ONLY with a JSON array of column names, nothing else.
Example response: ["column1", "column2", "column3"]
"""

    prompt = f"""Given the following query about hepatological patient data, select the columns needed to answer it.

Available columns:
{json.dumps(columns, indent=2)}

Query: {query}

Select up to {max_columns} most relevant columns. Return ONLY a JSON array of column names."""

    response = llm.generate(prompt, system_prompt)

    return _parse_column_response(response, columns)


def select_columns_with_enriched_prompt(
    query: str,
    columns: list[str],
    column_meanings: dict[str, str],
    graph_data: dict,
    llm: OllamaLLM,
    max_columns: int = 10,
) -> list[str]:
    """
    Use LLM with enriched context including descriptions and causal graph.
    (Enriched mode: column names + descriptions + causal graph)

    The prompt includes:
    - Query
    - Column names with medical descriptions
    - Causal graph structure (nodes and edges)

    Prints token count before sending to LLM.

    Args:
        query: Natural language question about the data
        columns: List of available column names
        column_meanings: Dictionary mapping column names to descriptions
        graph_data: Causal graph with 'nodes' and 'edges'
        llm: OllamaLLM instance
        max_columns: Maximum number of columns to return

    Returns:
        List of selected column names

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    logger.info(f"[ENRICHED MODE] Selecting columns for query: '{query}'")

    # Build column descriptions section
    columns_with_desc = []
    for col in columns:
        desc = column_meanings.get(col, f"Medical feature: {col}")
        columns_with_desc.append(f"- {col}: {desc}")
    columns_section = "\n".join(columns_with_desc)

    # Build causal graph section
    edges_formatted = [f"  {edge[0]} -> {edge[1]}" for edge in graph_data["edges"]]
    graph_section = f"""Causal Graph Structure:
Nodes: {len(graph_data['nodes'])} medical variables
Edges: {len(graph_data['edges'])} causal relationships

Causal relationships (cause -> effect):
{chr(10).join(edges_formatted)}"""

    system_prompt = """You are a medical data analyst assistant specializing in hepatological (liver) data.
Your task is to select the most relevant database columns needed to answer a query.

You have access to:
1. Column names with their medical descriptions
2. The causal graph showing relationships between medical variables

Use both semantic understanding of the column descriptions AND the causal relationships
to select the most relevant columns. Consider:
- Columns directly mentioned or related to the query
- Columns causally connected to the target variables
- Columns that are effects of relevant causes, or causes of relevant effects

IMPORTANT: Respond ONLY with a JSON array of column names, nothing else.
Example response: ["column1", "column2", "column3"]
"""

    prompt = f"""Given the following query about hepatological patient data, select the columns needed to answer it.

## Available Columns with Descriptions:
{columns_section}

## {graph_section}

## Query:
{query}

Select up to {max_columns} most relevant columns considering both semantic relevance and causal relationships.
Return ONLY a JSON array of column names."""

    # Count and display tokens BEFORE sending to LLM
    full_prompt = system_prompt + "\n\n" + prompt
    token_counts = count_tokens(full_prompt)

    logger.info("=" * 50)
    logger.info("PROMPT TOKEN COUNT:")
    logger.info(f"  {token_counts}")
    logger.info("=" * 50)

    # Also print to stdout for visibility
    print("\n" + "=" * 50)
    print("PROMPT TOKEN COUNT:")
    print(f"  Token count (tiktoken cl100k): {token_counts.tiktoken_count} tokens")
    print(f"  Token count (Qwen2.5 tokenizer): {token_counts.qwen_count} tokens")
    print(f"  Text length: {token_counts.text_length} characters")
    print("=" * 50 + "\n")

    response = llm.generate(prompt, system_prompt)

    return _parse_column_response(response, columns)


def _parse_column_response(response: str, valid_columns: list[str]) -> list[str]:
    """
    Parse LLM response and extract valid column names.

    Args:
        response: LLM response text
        valid_columns: List of valid column names

    Returns:
        List of valid selected column names

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    try:
        # Clean the response - extract JSON array
        response = response.strip()
        # Handle potential markdown code blocks
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(
                line for line in lines if not line.startswith("```")
            )
        response = response.strip()

        selected = json.loads(response)

        if not isinstance(selected, list):
            raise ValueError("Response is not a list")

        # Validate columns exist
        valid_selected = [c for c in selected if c in valid_columns]

        if len(valid_selected) != len(selected):
            invalid = set(selected) - set(valid_selected)
            logger.warning(f"Removed invalid columns: {invalid}")

        logger.info(f"Selected {len(valid_selected)} columns: {valid_selected}")
        return valid_selected

    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM response as JSON: {response}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e


def main() -> None:
    """
    Main function demonstrating LLM-based column selection.

    Demonstrates both modes:
    1. Simple mode: column names only
    2. Enriched mode: column names + descriptions + causal graph
    """
    logger.info("=" * 60)
    logger.info("LLM Column Selection Demo with Ollama")
    logger.info("=" * 60)

    # Load data
    logger.info("\n--- Loading HEPAR data ---")
    columns = load_hepar_columns()
    graph = load_hepar_graph()
    logger.info(f"Available columns: {len(columns)}")

    # Initialize LLM
    logger.info("\n--- Initializing Ollama LLM ---")
    config = LLMConfig(
        model_name=DEFAULT_MODEL,
        temperature=0.1,
    )
    llm = OllamaLLM(config)

    # Sample queries for demonstration
    sample_queries = [
        "What columns do I need to analyze liver enzyme levels?",
        "Find patients at risk of cirrhosis based on alcohol consumption",
    ]

    # ========== SIMPLE MODE DEMO ==========
    logger.info("\n" + "=" * 60)
    logger.info("SIMPLE MODE DEMO (column names only)")
    logger.info("=" * 60)

    for i, query in enumerate(sample_queries, 1):
        logger.info(f"\n--- Simple Mode Query {i} ---")
        logger.info(f"Query: {query}")

        try:
            selected = select_columns_with_llm(query, columns, llm)
            logger.info(f"Selected columns: {selected}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")

    # ========== ENRICHED MODE DEMO ==========
    logger.info("\n" + "=" * 60)
    logger.info("ENRICHED MODE DEMO (column names + descriptions + causal graph)")
    logger.info("=" * 60)

    for i, query in enumerate(sample_queries, 1):
        logger.info(f"\n--- Enriched Mode Query {i} ---")
        logger.info(f"Query: {query}")

        try:
            selected = select_columns_with_enriched_prompt(
                query=query,
                columns=columns,
                column_meanings=HEPAR_COLUMN_MEANINGS,
                graph_data=graph,
                llm=llm,
            )
            logger.info(f"Selected columns: {selected}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
