# CausalAgentRag

A hybrid retrieval system combining **Retrieval-Augmented Generation (RAG)** with **causal graph traversal** for intelligent column/feature selection in medical data analysis.

## Overview

This research project explores the integration of semantic similarity-based retrieval with causal reasoning for selecting relevant features from medical datasets. The primary application domain is hepatological (liver disorder) diagnosis, using the HEPAR II Bayesian network as the causal knowledge graph.

### Research Motivation

Traditional RAG systems rely solely on semantic similarity, which may miss causally relevant features that don't share obvious textual similarity with the query. By combining:
1. **Semantic Retrieval**: Embedding-based similarity matching
2. **Causal Graph Traversal**: BFS-based exploration of causal relationships

We aim to provide more comprehensive and medically meaningful feature selection.

## Project Structure

```
CausalAgentRag/
├── data/
│   ├── HEPAR_arcs.csv              # Causal graph edges (from -> to)
│   ├── HEPAR_simulated_patients.csv # 10,000 simulated patient records
│   ├── hepar_columns.json          # Extracted column names (generated)
│   ├── hepar_graph.json            # Graph structure as JSON (generated)
│   └── hepar_graph.png             # Causal network visualization (generated)
├── data_preprocessing.py           # Data processing and graph visualization
├── llm_huggingface.py              # LLM inference via HuggingFace transformers
├── llm_ollama.py                   # LLM inference via Ollama (legacy)
├── rag_llamaindex.py               # RAG system with LlamaIndex + HuggingFace
├── retrievers.py                   # Hybrid semantic-causal retrievers
├── main.py                         # Main entry point
├── pyproject.toml                  # Project dependencies
├── CLAUDE.md                       # Development instructions
└── README.md                       # This file
```

## Installation

### Prerequisites

- **Python 3.10+**
- **uv** package manager (recommended)
- **CUDA-capable GPU** with ~6GB+ VRAM (required for LLM inference with Qwen2.5-3B-Instruct)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CausalAgentRag
```

2. Install dependencies with uv:
```bash
uv sync
```

3. The first run will automatically download the Qwen2.5-3B-Instruct model from HuggingFace (~6GB).

## Usage

### 1. Data Preprocessing

Generate JSON files and visualization from the raw HEPAR data:

```bash
uv run data_preprocessing.py
```

**What it does:**
- Extracts column names from patient CSV -> `hepar_columns.json`
- Builds NetworkX DiGraph from causal edges -> `hepar_graph.json`
- Creates visualization of the Bayesian network -> `hepar_graph.png`

**Output Example:**
```
Loaded 10000 patient records with 70 columns
Built graph with 70 nodes and 123 edges
Saved graph visualization to data/hepar_graph.png
```

### 2. LLM Column Selection (HuggingFace)

Use a local LLM to intelligently select columns for natural language queries:

```bash
uv run llm_huggingface.py
```

**What it does:**
- Loads Qwen2.5-3B-Instruct model via HuggingFace transformers
- Takes natural language queries about the data
- Returns JSON array of relevant column names

**Example Query:**
```
Query: "What columns do I need to analyze liver enzyme levels?"
Selected columns: ["alt", "ast", "ggtp", "bilirubin", "albumin"]
```

**Configuration Options:**
```python
LLMConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # HuggingFace model
    temperature=0.1,                          # Low for deterministic outputs
    max_new_tokens=512,                       # Maximum response length
    use_quantization=False,                   # Enable 4/8-bit quantization
    quantization_bits=4,                      # 4 or 8 bit (when enabled)
)
```

### 3. RAG-based Column Retrieval

Use semantic similarity to retrieve relevant columns:

```bash
uv run rag_llamaindex.py
```

**What it does:**
- Indexes column names with medical descriptions
- Uses `all-MiniLM-L6-v2` embeddings for semantic search
- Supports both **top-k** and **similarity threshold** retrieval modes

**Example:**
```
Query: "viral hepatitis B infection markers"
Results:
  [0.587] hbc_anti - Anti-HBc (Hepatitis B Core Antibody)
  [0.575] hbsag - HBsAg (Hepatitis B Surface Antigen)
  [0.553] hbsag_anti - Anti-HBs (Hepatitis B Surface Antibody)
  [0.551] hbeag - HBeAg (Hepatitis B e Antigen)
```

## Module Descriptions

### `data_preprocessing.py`

Processes raw HEPAR data files and generates structured outputs.

**Key Functions:**
- `load_patient_data(filepath)` - Load CSV data into DataFrame
- `build_causal_graph(arcs_filepath)` - Construct NetworkX DiGraph
- `export_columns_json(columns, path)` - Save column list to JSON
- `export_graph_json(graph, path)` - Export graph structure
- `visualize_graph(graph, path)` - Generate PNG visualization

### `llm_huggingface.py`

LLM interface using HuggingFace transformers for local model inference.

**Key Classes:**
- `HuggingFaceLLM` - Wrapper for HuggingFace transformers
  - `generate(prompt, system_prompt)` - Text generation with chat template
  - `generate_stream(...)` - Streaming generation with TextIteratorStreamer

**Key Functions:**
- `select_columns_with_llm(query, columns, llm)` - LLM-based column selection (simple mode)
- `select_columns_with_enriched_prompt(...)` - LLM-based column selection with descriptions and causal graph

**Configuration:**
```python
LLMConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",  # ~6GB VRAM
    temperature=0.1,                          # Low for deterministic outputs
    max_new_tokens=512,                       # Maximum response length
    use_quantization=False,                   # Enable 4/8-bit quantization
    quantization_bits=4,                      # 4 or 8 bit (when enabled)
)
```

### `llm_ollama.py` (Legacy)

LLM interface using the Ollama framework for local model inference. This module is kept for backwards compatibility but is no longer actively used.

**Key Classes:**
- `OllamaLLM` - Wrapper for Ollama Python client

### `rag_llamaindex.py`

RAG system built with LlamaIndex for semantic column retrieval.

**Key Classes:**
- `ColumnRAG` - Main RAG system class
  - `build_index(columns, meanings)` - Create vector index
  - `retrieve_top_k(query, k)` - Get top-k columns by similarity
  - `retrieve_by_threshold(query, threshold)` - Get columns above threshold
  - `query_with_generation(query)` - RAG with LLM response

**Configuration:**
```python
RAGConfig(
    llm_model="Qwen/Qwen2.5-3B-Instruct",
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=10,
    similarity_threshold=0.5,
    use_quantization=False,  # Enable 4/8-bit quantization
    quantization_bits=4,     # 4 or 8 bit (when enabled)
)
```

**HEPAR Column Meanings:**
The module includes a comprehensive dictionary (`HEPAR_COLUMN_MEANINGS`) mapping all 70 medical columns to their clinical descriptions, based on the HEPAR II documentation.

### `retrievers.py`

Hybrid retrieval system combining semantic and causal approaches.

**Classes:**
- `SemanticScorer` - Embedding-based similarity scoring
- `CausalScorer` - BFS-based causal relevance scoring
- `SemanticOnlyRetriever` - Pure semantic baseline
- `ThresholdHybridRetriever` - Main hybrid retriever

**Key Parameters:**
```python
ThresholdHybridConfig(
    k=8,                    # Total columns to return
    semantic_threshold=0.5, # Min similarity for semantic selection
    max_depth=2,            # BFS traversal depth
    n_targets=1             # Causal seed columns
)
```

## Data Description

### HEPAR II Bayesian Network

The HEPAR II dataset is a well-established medical Bayesian network for liver disorder diagnosis, developed by Onisko et al.

**Statistics:**
- **70 variables** representing diseases, risk factors, and medical observations
- **123 causal edges** representing probabilistic dependencies
- **10,000 simulated patient records**

**Variable Categories:**
| Category | Examples | Count |
|----------|----------|-------|
| Diseases | THepatitis, Cirrhosis, PBC, Steatosis | 9 |
| Risk Factors | alcoholism, diabetes, obesity | 13 |
| Lab Tests | alt, ast, ggtp, bilirubin, albumin | ~20 |
| Symptoms | fatigue, jaundice, itching, pain | ~12 |
| Physical Findings | hepatomegaly, ascites, edema | ~16 |

### Graph Visualization

The generated `hepar_graph.png` displays the causal network with color-coded nodes:
- **Orange**: Diseases (THepatitis, Cirrhosis, etc.)
- **Blue**: Risk Factors (alcoholism, obesity, etc.)
- **Green**: Observations and Test Results

## Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `networkx` | Graph operations and BFS traversal |
| `matplotlib` | Graph visualization |
| `torch` | Deep learning framework |
| `transformers` | HuggingFace model loading |
| `accelerate` | Efficient model loading |
| `bitsandbytes` | Optional quantization support |
| `llama-index` | RAG framework |
| `llama-index-llms-huggingface` | HuggingFace LLM integration |
| `llama-index-embeddings-huggingface` | HuggingFace embeddings |
| `tiktoken` | Token counting |

### Model Requirements

- **LLM**: Qwen2.5-3B-Instruct (~6GB VRAM) via HuggingFace transformers
- **Embeddings**: all-MiniLM-L6-v2 (~90MB, runs on CPU)

### Scoring Mechanisms

**Semantic Scoring:**
- Cosine similarity between query embedding and column embeddings
- Medical abbreviation expansion (e.g., ALT -> "alanine transaminase")

**Causal Scoring:**
- BFS traversal from target nodes
- Exponential decay: `score = 1 / 2^(depth-1)`
- 1-hop neighbors: 1.0, 2-hop: 0.5, 3-hop: 0.25

## API Usage Examples

### Using ColumnRAG Programmatically

```python
from rag_llamaindex import ColumnRAG, RAGConfig, HEPAR_COLUMN_MEANINGS
import json

# Initialize RAG
config = RAGConfig(top_k=5, similarity_threshold=0.4)
rag = ColumnRAG(config)

# Load columns
with open("data/hepar_columns.json") as f:
    columns = json.load(f)

# Build index
rag.build_index(columns, HEPAR_COLUMN_MEANINGS)

# Retrieve columns
results = rag.retrieve_top_k("liver enzyme abnormalities", k=5)
for r in results:
    print(f"{r['column_name']}: {r['score']:.3f}")
```

### Using HuggingFaceLLM Programmatically

```python
from llm_huggingface import HuggingFaceLLM, LLMConfig, select_columns_with_llm

# Initialize LLM (requires CUDA GPU)
config = LLMConfig(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    use_quantization=False,  # Set True for 4-bit quantization (saves VRAM)
)
llm = HuggingFaceLLM(config)

# Select columns
columns = ["alt", "ast", "bilirubin", "albumin", "fatigue", "jaundice"]
selected = select_columns_with_llm(
    query="Find patients with liver damage",
    columns=columns,
    llm=llm
)
print(selected)  # ["alt", "ast", "bilirubin"]
```

## Research Applications

This system can be applied to:
1. **Medical Feature Selection**: Identifying relevant features for diagnosis models
2. **SQL Query Generation**: Selecting columns for database queries from natural language
3. **Explainable AI**: Tracing causal paths to explain feature relevance
4. **Knowledge Graph Augmented Retrieval**: Combining embedding similarity with structured knowledge

## References

- Onisko, A., et al. "HEPAR II: A Probabilistic Model for Diagnosis of Liver Disorders"
- bnlearn: Bayesian Network Repository (https://www.bnlearn.com/bnrepository/)
- LlamaIndex Documentation (https://docs.llamaindex.ai/)
- HuggingFace Transformers (https://huggingface.co/docs/transformers/)
- Qwen2.5 Model (https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## License

This project is for research purposes. The HEPAR II dataset is contributed by Agnieszka Onisko.
