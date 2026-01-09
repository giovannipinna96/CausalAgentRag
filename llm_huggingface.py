#!/usr/bin/env python3
"""
LLM Module with HuggingFace Transformers for CausalAgentRag

This module provides an interface to interact with LLMs via HuggingFace transformers
for column selection tasks in medical/hepatological data analysis.

Supports two prompt modes:
1. Simple mode: Query + column names only
2. Enriched mode: Query + column names + descriptions + causal graph

Prerequisites:
    - A CUDA-capable GPU with sufficient VRAM (~6GB for Qwen2.5-3B-Instruct)
    - HuggingFace transformers and accelerate libraries

Usage:
    python llm_huggingface.py
    # or with uv:
    uv run llm_huggingface.py
"""

import argparse
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import tiktoken
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_NEW_TOKENS = 512

# File paths
DATA_DIR = Path(__file__).parent / "data"
COLUMNS_JSON = DATA_DIR / "hepar_columns.json"
GRAPH_JSON = DATA_DIR / "hepar_graph.json"

# Global tokenizer cache for token counting
_tiktoken_encoder = None
_hf_tokenizer = None


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
    """Configuration for the HuggingFace LLM."""

    model_name: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    device: str = "auto"
    use_quantization: bool = False
    quantization_bits: int = 4  # 4 or 8 bit quantization when enabled


@dataclass
class TokenCount:
    """Token count results from multiple tokenizers."""

    tiktoken_count: int
    hf_count: int
    text_length: int

    def __str__(self) -> str:
        return (
            f"Token count (tiktoken cl100k): {self.tiktoken_count} tokens\n"
            f"Token count (HuggingFace tokenizer): {self.hf_count} tokens\n"
            f"Text length: {self.text_length} characters"
        )


def get_tiktoken_encoder():
    """Get or create tiktoken encoder (cached)."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        logger.info("Loading tiktoken encoder (cl100k_base)...")
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_encoder


def get_hf_tokenizer(model_name: str = DEFAULT_MODEL):
    """Get or create HuggingFace tokenizer (cached)."""
    global _hf_tokenizer
    if _hf_tokenizer is None:
        logger.info(f"Loading HuggingFace tokenizer for {model_name}...")
        _hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    return _hf_tokenizer


def count_tokens(text: str, model_name: str = DEFAULT_MODEL) -> TokenCount:
    """
    Count tokens using multiple tokenizers.

    Args:
        text: Text to tokenize
        model_name: HuggingFace model name for tokenizer

    Returns:
        TokenCount with tiktoken and HuggingFace tokenizer counts
    """
    # Tiktoken (OpenAI tokenizer)
    tiktoken_enc = get_tiktoken_encoder()
    tiktoken_count = len(tiktoken_enc.encode(text))

    # HuggingFace tokenizer (model-specific)
    hf_tok = get_hf_tokenizer(model_name)
    hf_count = len(hf_tok.encode(text))

    return TokenCount(
        tiktoken_count=tiktoken_count,
        hf_count=hf_count,
        text_length=len(text),
    )


class HuggingFaceLLM:
    """
    Wrapper class for HuggingFace LLM.

    This class provides a clean interface to interact with LLMs
    through HuggingFace transformers for text generation tasks.

    Attributes:
        config: LLM configuration parameters
        model: HuggingFace model instance
        tokenizer: HuggingFace tokenizer instance
    """

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize the HuggingFace LLM.

        Args:
            config: LLM configuration. Uses defaults if not provided.

        Raises:
            RuntimeError: If GPU is not available or model loading fails
        """
        self.config = config or LLMConfig()
        logger.info(f"Initializing HuggingFaceLLM with model: {self.config.model_name}")

        # Check GPU availability
        self._verify_gpu()

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        logger.info("Tokenizer loaded successfully")

        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.config.device,
        }

        # Setup quantization if enabled
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
            logger.info(f"{self.config.quantization_bits}-bit quantization configured")
        else:
            # Use float16 for non-quantized models
            model_kwargs["torch_dtype"] = torch.float16
            logger.info("Using float16 precision (no quantization)")

        # Load model
        logger.info(f"Loading model {self.config.model_name}...")
        logger.info("This may take a few minutes on first run (downloading model weights)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        logger.info("Model loaded successfully")

        # Log device info
        if hasattr(self.model, "device"):
            logger.info(f"Model device: {self.model.device}")
        elif hasattr(self.model, "hf_device_map"):
            logger.info(f"Model device map: {self.model.hf_device_map}")

        logger.info("HuggingFaceLLM initialized successfully")

    def _verify_gpu(self) -> None:
        """
        Verify GPU availability.

        Raises:
            RuntimeError: If CUDA is not available
        """
        logger.info("Verifying GPU availability...")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. A GPU is required for HuggingFace LLM inference. "
                "Please ensure you have a CUDA-capable GPU and the correct PyTorch version installed."
            )
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU available: {gpu_count} device(s)")
        logger.info(f"Primary GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    def _build_messages(
        self, prompt: str, system_prompt: str | None = None
    ) -> list[dict[str, str]]:
        """
        Build message list for chat template.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            List of message dictionaries
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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

        messages = self._build_messages(prompt, system_prompt)

        # Apply chat template
        logger.info("Applying chat template...")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize input
        logger.info("Tokenizing input...")
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]
        logger.info(f"Input tokens: {input_length}")

        # Generate response
        logger.info("Generating response...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Extract only the new tokens (response)
        response_ids = generated_ids[0][input_length:]
        result = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        logger.info(f"Generated response (length: {len(result)} chars)")
        return result

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

        messages = self._build_messages(prompt, system_prompt)

        # Apply chat template
        logger.info("Applying chat template...")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]
        logger.info(f"Input tokens: {input_length}")

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Generate in a separate thread
        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": self.config.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = threading.Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
        )
        thread.start()

        # Yield tokens as they are generated
        logger.info("Starting stream generation...")
        for text_chunk in streamer:
            yield text_chunk

        thread.join()
        logger.info("Stream generation completed")


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
    llm: HuggingFaceLLM,
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
        llm: HuggingFaceLLM instance
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


def select_columns_with_descriptions(
    query: str,
    columns: list[str],
    column_meanings: dict[str, str],
    llm: HuggingFaceLLM,
    max_columns: int = 10,
) -> list[str]:
    """
    Use LLM to select relevant columns with descriptions context.
    (Descriptions mode: column names + descriptions, NO causal graph)

    The LLM analyzes the query with column descriptions to make
    more informed selections based on medical meanings.

    Args:
        query: Natural language question about the data
        columns: List of available column names
        column_meanings: Dictionary mapping column names to descriptions
        llm: HuggingFaceLLM instance
        max_columns: Maximum number of columns to return

    Returns:
        List of selected column names

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    logger.info(f"[DESCRIPTIONS MODE] Selecting columns for query: '{query}'")

    # Build column descriptions section
    columns_with_desc = []
    for col in columns:
        desc = column_meanings.get(col, f"Medical feature: {col}")
        columns_with_desc.append(f"- {col}: {desc}")
    columns_section = "\n".join(columns_with_desc)

    system_prompt = """You are a medical data analyst assistant specializing in hepatological (liver) data.
Your task is to select the most relevant database columns needed to answer a query.

You have access to column names with their medical descriptions.
Use your understanding of the column descriptions to select the most relevant columns.

Consider:
- Columns directly mentioned or semantically related to the query
- Columns that would provide useful diagnostic or analytical information
- Medical relationships between conditions, symptoms, and test results

IMPORTANT: Respond ONLY with a JSON array of column names, nothing else.
Example response: ["column1", "column2", "column3"]
"""

    prompt = f"""Given the following query about hepatological patient data, select the columns needed to answer it.

## Available Columns with Descriptions:
{columns_section}

## Query:
{query}

Select up to {max_columns} most relevant columns based on their medical descriptions.
Return ONLY a JSON array of column names."""

    # Count and display tokens BEFORE sending to LLM
    full_prompt = system_prompt + "\n\n" + prompt
    token_counts = count_tokens(full_prompt, llm.config.model_name)

    logger.info("=" * 50)
    logger.info("PROMPT TOKEN COUNT:")
    logger.info(f"  {token_counts}")
    logger.info("=" * 50)

    # Also print to stdout for visibility
    print("\n" + "=" * 50)
    print("PROMPT TOKEN COUNT:")
    print(f"  Token count (tiktoken cl100k): {token_counts.tiktoken_count} tokens")
    print(f"  Token count (HuggingFace tokenizer): {token_counts.hf_count} tokens")
    print(f"  Text length: {token_counts.text_length} characters")
    print("=" * 50 + "\n")

    response = llm.generate(prompt, system_prompt)

    return _parse_column_response(response, columns)


def select_columns_with_enriched_prompt(
    query: str,
    columns: list[str],
    column_meanings: dict[str, str],
    graph_data: dict,
    llm: HuggingFaceLLM,
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
        llm: HuggingFaceLLM instance
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
    token_counts = count_tokens(full_prompt, llm.config.model_name)

    logger.info("=" * 50)
    logger.info("PROMPT TOKEN COUNT:")
    logger.info(f"  {token_counts}")
    logger.info("=" * 50)

    # Also print to stdout for visibility
    print("\n" + "=" * 50)
    print("PROMPT TOKEN COUNT:")
    print(f"  Token count (tiktoken cl100k): {token_counts.tiktoken_count} tokens")
    print(f"  Token count (HuggingFace tokenizer): {token_counts.hf_count} tokens")
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
    logger.info(f"Parsing LLM response: {response[:200]}...")

    # Clean the response - extract JSON array
    response = response.strip()
    # Handle potential markdown code blocks
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join(
            line for line in lines if not line.startswith("```")
        )
    response = response.strip()

    # Try to find JSON array in response
    start_idx = response.find("[")
    end_idx = response.rfind("]")
    if start_idx != -1 and end_idx != -1:
        response = response[start_idx : end_idx + 1]

    try:
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-based column selection for medical data analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with simple mode (column names only)
  uv run llm_huggingface.py --mode simple

  # Run with descriptions mode (column names + descriptions)
  uv run llm_huggingface.py --mode descriptions

  # Run with graph mode (column names + descriptions + causal graph)
  uv run llm_huggingface.py --mode graph

  # Run all modes
  uv run llm_huggingface.py --mode all

  # Custom query with specific model
  uv run llm_huggingface.py --mode simple --query "Find liver enzyme markers" --model "Qwen/Qwen2.5-1.5B-Instruct"

  # Enable 4-bit quantization to reduce VRAM usage
  uv run llm_huggingface.py --mode simple --quantization --quantization-bits 4
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "descriptions", "graph", "all"],
        default="all",
        help="Selection mode: 'simple' (column names only), 'descriptions' (names + descriptions), 'graph' (names + descriptions + causal graph), 'all' (run all modes). Default: all",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query to run. If not provided, uses sample queries.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Generation temperature. Default: {DEFAULT_TEMPERATURE}",
    )

    parser.add_argument(
        "--max-columns",
        type=int,
        default=10,
        help="Maximum number of columns to return. Default: 10",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum new tokens to generate. Default: {DEFAULT_MAX_NEW_TOKENS}",
    )

    parser.add_argument(
        "--quantization",
        action="store_true",
        help="Enable quantization to reduce VRAM usage.",
    )

    parser.add_argument(
        "--quantization-bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits (4 or 8). Only used if --quantization is enabled. Default: 4",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON file (for evaluation with evaluate.py)",
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to benchmark JSON file. If provided, runs on benchmark queries instead of sample queries.",
    )

    return parser.parse_args()


def run_mode(
    mode: str,
    query: str,
    columns: list[str],
    llm: HuggingFaceLLM,
    max_columns: int,
    graph: dict | None = None,
) -> list[str]:
    """
    Run column selection in the specified mode.

    Args:
        mode: Selection mode ('simple', 'descriptions', or 'graph')
        query: Natural language query
        columns: List of available column names
        llm: HuggingFaceLLM instance
        max_columns: Maximum columns to return
        graph: Causal graph data (required for 'graph' mode)

    Returns:
        List of selected column names
    """
    if mode == "simple":
        return select_columns_with_llm(query, columns, llm, max_columns)
    elif mode == "descriptions":
        return select_columns_with_descriptions(
            query, columns, HEPAR_COLUMN_MEANINGS, llm, max_columns
        )
    elif mode == "graph":
        if graph is None:
            raise ValueError("Graph data is required for 'graph' mode")
        return select_columns_with_enriched_prompt(
            query, columns, HEPAR_COLUMN_MEANINGS, graph, llm, max_columns
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


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
    model: str,
    mode: str,
) -> None:
    """
    Save results to JSON file for evaluation.

    Args:
        results: List of result dictionaries
        output_path: Path to output JSON file
        model: Model name used
        mode: Mode used for selection
    """
    logger.info(f"Saving results to {output_path}")

    output = {
        "model": model,
        "mode": mode,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main() -> None:
    """
    Main function for LLM-based column selection.

    Supports three modes via command line:
    1. simple: column names only
    2. descriptions: column names + medical descriptions
    3. graph: column names + descriptions + causal graph

    Can run on custom queries, sample queries, or benchmark queries.
    Results can be saved to JSON for evaluation with evaluate.py.
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("LLM Column Selection with HuggingFace")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Quantization: {args.quantization} ({args.quantization_bits}-bit)")
    if args.output:
        logger.info(f"Output: {args.output}")
    if args.benchmark:
        logger.info(f"Benchmark: {args.benchmark}")

    # Load data
    logger.info("\n--- Loading HEPAR data ---")
    columns = load_hepar_columns()
    logger.info(f"Loaded {len(columns)} columns")

    # Load graph only if needed
    graph = None
    if args.mode in ["graph", "all"]:
        graph = load_hepar_graph()
        logger.info(f"Loaded graph with {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")

    # Initialize LLM
    logger.info("\n--- Initializing HuggingFace LLM ---")
    config = LLMConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_quantization=args.quantization,
        quantization_bits=args.quantization_bits,
    )
    llm = HuggingFaceLLM(config)

    # Determine queries to run
    if args.benchmark:
        # Load queries from benchmark file
        benchmark_queries = load_benchmark_queries(args.benchmark)
        queries = [(q["id"], q["query"]) for q in benchmark_queries]
    elif args.query:
        queries = [("custom", args.query)]
    else:
        queries = [
            ("sample1", "What columns do I need to analyze liver enzyme levels?"),
            ("sample2", "Find patients at risk of cirrhosis based on alcohol consumption"),
        ]

    # Only run single mode when saving to output (for evaluation)
    if args.output and args.mode == "all":
        logger.warning("Output mode requires a single mode. Using 'graph' mode.")
        modes = ["graph"]
    elif args.mode == "all":
        modes = ["simple", "descriptions", "graph"]
    else:
        modes = [args.mode]

    # Collect results for JSON output
    all_results = []

    # Run selected modes
    for mode in modes:
        mode_names = {
            "simple": "SIMPLE MODE (column names only)",
            "descriptions": "DESCRIPTIONS MODE (column names + descriptions)",
            "graph": "GRAPH MODE (column names + descriptions + causal graph)",
        }

        logger.info("\n" + "=" * 60)
        logger.info(mode_names[mode])
        logger.info("=" * 60)

        for query_id, query in queries:
            logger.info(f"\n--- Query [{query_id}] ---")
            logger.info(f"Query: {query}")

            selected = run_mode(
                mode=mode,
                query=query,
                columns=columns,
                llm=llm,
                max_columns=args.max_columns,
                graph=graph,
            )

            logger.info(f"Selected columns ({len(selected)}): {selected}")
            print(f"\nResult: {selected}\n")

            # Collect result for JSON output
            if args.output:
                all_results.append({
                    "query_id": query_id,
                    "query": query,
                    "predicted_columns": selected,
                })

    # Save results to JSON if output path provided
    if args.output:
        save_results_json(
            results=all_results,
            output_path=args.output,
            model=args.model,
            mode=modes[0] if len(modes) == 1 else "multiple",
        )

    logger.info("\n" + "=" * 60)
    logger.info("Completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
