# CausalAgentRag - Documentazione Completa dei Metodi

Questo documento descrive in dettaglio tutte le tecniche di retrieval implementate nel progetto CausalAgentRag per la selezione automatica di colonne/features da un dataset medico (HEPAR) basandosi su query in linguaggio naturale.

## Indice

1. [Overview del Progetto](#overview-del-progetto)
2. [Dataset e Benchmark](#dataset-e-benchmark)
3. [Metodi di Retrieval](#metodi-di-retrieval)
   - [V3 Methods (Best Performing)](#v3-methods-best-performing)
   - [Original Methods](#original-methods)
   - [LLM Baseline Methods](#llm-baseline-methods)
   - [Advanced Techniques](#advanced-techniques)
4. [Risultati del Benchmark Unificato](#risultati-del-benchmark-unificato)
5. [Analisi dei Risultati](#analisi-dei-risultati)
6. [Riferimenti e Paper](#riferimenti-e-paper)

---

## Overview del Progetto

CausalAgentRag è un sistema di retrieval ibrido che combina:
- **Similarity semantica** tra query e nomi/descrizioni delle colonne
- **Grafo causale** (Bayesian Network HEPAR) per identificare relazioni causa-effetto
- **Tecniche avanzate di NLP** (BM25, PPR, Intent Detection)

L'obiettivo è selezionare automaticamente le colonne più rilevanti per una task di predizione, evitando il **data leakage** (uso di variabili che sono effetti del target invece che cause).

---

## Dataset e Benchmark

### Dataset HEPAR

Il dataset HEPAR è una rete bayesiana per la diagnosi di malattie epatiche con:
- **70 variabili/colonne**
- **10,000 pazienti simulati**
- **Grafo causale** con relazioni direzionate tra variabili

Le colonne sono categorizzate in:
- **Malattie** (9): THepatitis, ChHepatitis, PBC, Steatosis, Cirrhosis, Hyperbilirubinemia, RHepatitis, fibrosis, carcinoma
- **Fattori di rischio** (13): alcoholism, vh_amn, hepatotoxic, hospital, surgery, gallstones, etc.
- **Osservazioni/Test** (48): alt, ast, ggtp, fatigue, jaundice, hepatomegaly, etc.

### Benchmark v3 - 16 Query

Il benchmark include 12 query di **predizione** (PRED) e 4 query **semantiche** (SEM):

| ID | Query | Gold Columns |
|----|-------|--------------|
| PRED1 | I want to predict which patients will develop cirrhosis | Steatosis, fibrosis |
| PRED2 | Build a model to identify patients at risk for chronic hepatitis | injections, transfusion, vh_amn |
| PRED3 | Predict which patients will develop fatty liver disease (steatosis) | alcoholism, obesity |
| PRED4 | I need to forecast elevated bilirubin levels in patients | ChHepatitis, Cirrhosis, Hyperbilirubinemia, PBC, gallstones |
| PRED5 | Predict toxic hepatitis occurrence | alcoholism, hepatotoxic |
| PRED6 | Build a classifier for primary biliary cholangitis (PBC) | age, sex |
| PRED7 | Identify patients likely to have encephalopathy | Cirrhosis, PBC |
| PRED8 | Predict bleeding risk in liver patients | inr, platelet |
| PRED9 | Forecast which patients will develop ascites | proteins |
| PRED10 | Predict reactive hepatitis in patients | hepatotoxic |
| PRED11 | Build a model to predict fibrosis progression | ChHepatitis |
| PRED12 | Predict hyperbilirubinemia in patients | age, sex |
| SEM1 | Show me all hepatitis-related columns in the dataset | THepatitis, ChHepatitis, RHepatitis |
| SEM2 | What demographic variables are available? | age, sex |
| SEM3 | List all antibody test columns | hbsag_anti, hbc_anti, hcv_anti |
| SEM4 | What liver enzyme measurements are in the data? | alt, ast, ggtp, phosphatase |

---

## Metodi di Retrieval

### V3 Methods (Best Performing)

#### 1. RAG_HYBRID (Semantic + PPR + Causal Constraints)

**Descrizione**: Il metodo migliore che combina retrieval semantico con diffusione sul grafo causale e vincoli per evitare leakage.

**Funzionamento**:
1. **Intent Detection**: Rileva automaticamente il tipo di query:
   - `leakfree_prediction`: Query di predizione (evita discendenti del target)
   - `total_effect`: Query di effetto causale (evita variabili post-treatment)
   - `semantic_schema`: Query semantiche pure
2. **Semantic Scoring**: Dense embeddings (BGE-small-en-v1.5) + BM25 fusion
3. **Graph Diffusion**: Personalized PageRank dal focus node
4. **Causal Constraints**: Esclude automaticamente variabili proibite (leakage)
5. **Weighted Fusion**: Combina semantic e graph scores

**Configurazione**:
```python
CausalHybridConfig(
    budget=10,
    alpha_dense=0.6,      # Weight for dense vs BM25
    ppr_alpha=0.85,       # PPR damping factor
    ppr_max_iter=100,
    semantic_weight=0.6,  # Semantic vs graph weight
    graph_weight=0.4,
)
```

**Pro**:
- **Miglior F1** (0.284) tra tutti i metodi
- **Leakage minimo** (1.3%) grazie ai vincoli causali
- **75% correct** (12/16 task)

**Contro**: Richiede grafo causale

**File**: `causal_hybrid_retriever_v3.py`

---

#### 2. RAG_SEMANTIC (Dense + BM25 Fusion)

**Descrizione**: Retrieval semantico puro con fusione di dense embeddings e BM25.

**Funzionamento**:
1. **Schema Building**: Crea TextNodes con nome colonna + sinonimi + descrizione
2. **Dense Retrieval**: VectorIndexRetriever con BGE-small-en-v1.5
3. **BM25 Retrieval**: Lexical matching con rank_bm25
4. **Min-Max Normalization**: Normalizza entrambi gli score a [0,1]
5. **Weighted Fusion**: `final = alpha * dense + (1-alpha) * bm25`

**Configurazione**:
```python
SemanticRetrieverConfig(
    embed_model="BAAI/bge-small-en-v1.5",
    alpha_dense=0.6,
    top_k=10
)
```

**Pro**: Non richiede grafo causale
**Contro**: Leakage rate più alto (9.4%)

**File**: `semantic_retriever_v3.py`

---

### Original Methods

#### 3. RRFHybrid (Reciprocal Rank Fusion)

**Descrizione**: Combina semantic, BM25 e causal scoring usando RRF.

**Funzionamento**:
1. **Semantic scoring**: Sentence-transformers all-MiniLM-L6-v2
2. **BM25 scoring**: TF-IDF migliorato
3. **Causal scoring**: BFS dal target con exponential decay
4. **RRF Fusion**: `score = sum(1/(k + rank_i))` con k=60

**Formula RRF**:
```
RRF_score(d) = Σ 1/(k + rank_i(d))
```

**Pro**: Robusto, buon recall (83.3%)
**Contro**: Leakage più alto di RAG_HYBRID (2.5%)

**File**: `retrievers.py`

---

#### 4. ThresholdHybrid

**Descrizione**: Seleziona colonne semantiche sopra una soglia, poi riempie con colonne causali.

**Funzionamento**:
1. Calcola similarity semantica per tutte le colonne
2. Seleziona colonne con similarity > threshold
3. Riempie slot rimanenti con colonne dal grafo causale (BFS)

**Configurazione**:
```python
ThresholdHybridConfig(
    k=10,
    semantic_threshold=0.5,
    max_depth=2,
    n_targets=1
)
```

**Pro**: Semplice da capire
**Contro**: **Alto leakage** (23.1%) - non filtra discendenti

**File**: `retrievers.py`

---

#### 5. SemanticOnly

**Descrizione**: Baseline puramente semantica.

**Funzionamento**:
1. Calcola embedding della query
2. Calcola similarity con tutte le colonne
3. Restituisce top-k per similarity

**Pro**: Semplice, nessuna dipendenza da grafo
**Contro**: Leakage rate 9.4%

**File**: `retrievers.py`

---

### LLM Baseline Methods

#### 6. LLM_SIMPLE

**Descrizione**: LLM seleziona colonne basandosi solo sui nomi.

**Funzionamento**:
1. Prompt con query + lista di 70 nomi colonne
2. LLM restituisce JSON con colonne selezionate
3. Parsing e validazione

**Modello**: `Qwen/Qwen2.5-3B-Instruct` (HuggingFace)

**Pro**: Basso leakage (1.9%)
**Contro**: Bassa precision e recall

**File**: `llm_baselines_v3.py` - mode `SIMPLE`

---

#### 7. LLM_DESCRIPTION

**Descrizione**: Come LLM_SIMPLE ma include descrizioni mediche.

**Funzionamento**:
1. Prompt con query + nome + descrizione per ogni colonna
2. LLM seleziona basandosi su informazioni più ricche

**Pro**: Migliore comprensione semantica
**Contro**: Prompt lungo, leakage 10%

**File**: `llm_baselines_v3.py` - mode `DESCRIPTION`

---

#### 8. LLM_GRAPH

**Descrizione**: LLM riceve anche informazioni sul grafo causale.

**Funzionamento**:
1. Prompt include: query, colonne, descrizioni, edges del grafo
2. LLM può ragionare su relazioni causali

**Pro**: Può usare informazione causale
**Contro**: Leakage 3.7%, performance limitata

**File**: `llm_baselines_v3.py` - mode `GRAPH`

---

### Advanced Techniques

#### 9. HyDE (Hypothetical Document Embeddings)

**Descrizione**: Genera documenti ipotetici dalla query prima del retrieval.

**File**: `hyde_retriever.py`

**Riferimenti**: [HyDE Paper](https://arxiv.org/abs/2212.10496)

---

#### 10. ColBERT (Late Interaction)

**Descrizione**: Usa late interaction per matching token-level.

**File**: `colbert_retriever.py`

**Riferimenti**: [ColBERT Paper](https://arxiv.org/abs/2004.12832)

---

#### 11. CRAG (Corrective RAG)

**Descrizione**: Self-correction con retry su query ambigue.

**File**: `crag_evaluator.py`

**Riferimenti**: [CRAG Paper](https://arxiv.org/abs/2401.15884)

---

## Risultati del Benchmark Unificato

### Benchmark eseguito il 2026-01-17

Tutti i metodi valutati sulle stesse 16 query con budget=10 colonne.

### Metriche Aggregate

| Method | Precision | Recall | F1 | Leakage | Correct | Partial | Wrong |
|--------|-----------|--------|-----|---------|---------|---------|-------|
| **RAG_HYBRID** | 0.181 | 0.781 | **0.284** | **0.013** | **12** | 1 | 3 |
| RRFHybrid | 0.175 | **0.833** | 0.281 | 0.025 | 9 | 1 | 6 |
| ThresholdHybrid | 0.175 | 0.802 | 0.278 | 0.231 | 3 | 4 | 9 |
| SemanticOnly | 0.119 | 0.480 | 0.185 | 0.094 | 4 | 4 | 8 |
| RAG_SEMANTIC | 0.112 | 0.483 | 0.177 | 0.094 | 3 | 2 | 11 |
| LLM_SIMPLE | 0.094 | 0.411 | 0.150 | 0.019 | 3 | 4 | 9 |
| LLM_DESCRIPTION | 0.087 | 0.335 | 0.137 | 0.100 | 1 | 3 | 12 |
| LLM_GRAPH | 0.081 | 0.365 | 0.130 | 0.037 | 3 | 3 | 10 |

### Classificazione delle Risposte

| Metodo | Correct (%) | Partial (%) | Wrong (%) |
|--------|-------------|-------------|-----------|
| **RAG_HYBRID** | **75.0%** | 6.2% | 18.8% |
| RRFHybrid | 56.2% | 6.2% | 37.5% |
| SemanticOnly | 25.0% | 25.0% | 50.0% |
| ThresholdHybrid | 18.8% | 25.0% | 56.2% |
| RAG_SEMANTIC | 18.8% | 12.5% | 68.8% |
| LLM_SIMPLE | 18.8% | 25.0% | 56.2% |
| LLM_GRAPH | 18.8% | 18.8% | 62.5% |
| LLM_DESCRIPTION | 6.2% | 18.8% | 75.0% |

### Data Leakage Analysis

| Metodo | Leakage Rate | Risk Level |
|--------|--------------|------------|
| **RAG_HYBRID** | 1.3% | **LOW** |
| LLM_SIMPLE | 1.9% | LOW |
| RRFHybrid | 2.5% | LOW |
| LLM_GRAPH | 3.7% | LOW |
| RAG_SEMANTIC | 9.4% | MEDIUM |
| SemanticOnly | 9.4% | MEDIUM |
| LLM_DESCRIPTION | 10.0% | MEDIUM |
| ThresholdHybrid | 23.1% | **HIGH** |

---

## Analisi dei Risultati

### Perché RAG_HYBRID è il migliore?

1. **Intent Detection**: Capisce automaticamente il tipo di query
2. **Causal Constraints**: Esclude discendenti del target per evitare leakage
3. **PPR Diffusion**: Propaga scores attraverso il grafo causale
4. **Fusion bilanciata**: Combina semantic e graph scores ottimalmente

### Perché ThresholdHybrid ha alto leakage?

ThresholdHybrid non implementa vincoli causali - include qualsiasi colonna collegata nel grafo, inclusi gli **effetti** (discendenti) del target che causano data leakage.

### Confronto: Retriever vs LLM Methods

| Categoria | F1 Range | Leakage Range | Note |
|-----------|----------|---------------|------|
| Retriever Methods | 0.177-0.284 | 0.013-0.231 | Migliore performance complessiva |
| LLM Methods | 0.130-0.150 | 0.019-0.100 | Basso leakage ma bassa recall |

### Raccomandazioni

| Use Case | Metodo Raccomandato |
|----------|---------------------|
| Massimizzare accuracy + safety | **RAG_HYBRID** |
| Massimizzare recall | RRFHybrid |
| Senza grafo causale | RAG_SEMANTIC |
| Minimo leakage assoluto | RAG_HYBRID o LLM_SIMPLE |
| Query semantiche semplici | SemanticOnly |

---

## Riferimenti e Paper

### Retrieval-Augmented Generation (RAG)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
  - [Paper](https://arxiv.org/abs/2005.11401)

### Reciprocal Rank Fusion (RRF)
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods", SIGIR 2009
  - [Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

### Personalized PageRank
- Page et al., "The PageRank Citation Ranking: Bringing Order to the Web", Stanford 1999
  - [Paper](http://ilpubs.stanford.edu:8090/422/)

### BM25
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009
  - [Paper](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)

### Sentence Transformers
- Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019
  - [Paper](https://arxiv.org/abs/1908.10084)
  - [Website](https://www.sbert.net/)

### BGE Embeddings
- Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding", 2023
  - [Paper](https://arxiv.org/abs/2309.07597)
  - [HuggingFace](https://huggingface.co/BAAI/bge-small-en-v1.5)

### HyDE (Hypothetical Document Embeddings)
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels", 2022
  - [Paper](https://arxiv.org/abs/2212.10496)

### ColBERT
- Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", SIGIR 2020
  - [Paper](https://arxiv.org/abs/2004.12832)

### CRAG (Corrective RAG)
- Yan et al., "Corrective Retrieval Augmented Generation", 2024
  - [Paper](https://arxiv.org/abs/2401.15884)

### HEPAR Dataset
- Onisko et al., "Learning Bayesian network parameters from small data sets: application of Noisy-OR gates", 2001
  - [Paper](https://www.sciencedirect.com/science/article/pii/S138650560100073X)

### Markov Blanket & Causal Inference
- Pearl, "Probabilistic Reasoning in Intelligent Systems", 1988
  - [Wikipedia](https://en.wikipedia.org/wiki/Markov_blanket)

### LlamaIndex
- [Documentation](https://docs.llamaindex.ai/)
- [GitHub](https://github.com/run-llama/llama_index)

---

## File del Progetto

### Core V3 Modules (Integrated)

| File | Descrizione |
|------|-------------|
| `schema_v3.py` | Schema builder con sinonimi e TextNodes |
| `semantic_retriever_v3.py` | Dense + BM25 fusion retriever |
| `causal_graph_v3.py` | PPR, d-separation, backdoor sets |
| `causal_hybrid_retriever_v3.py` | Intent detection + hybrid retrieval |
| `downstream_v3.py` | AUC e ATE validation |
| `llm_baselines_v3.py` | LLM column selection (HuggingFace) |

### Original Modules

| File | Descrizione |
|------|-------------|
| `retrievers.py` | SemanticOnly, ThresholdHybrid, RRFHybrid |
| `rag_llamaindex.py` | RAG con LlamaIndex |
| `llm_huggingface.py` | Interface LLM (Qwen2.5-3B) |

### Benchmark

| File | Descrizione |
|------|-------------|
| `unified_benchmark.py` | Runner unificato per tutti i metodi |
| `benchmark_results/results.csv` | Risultati per-task |
| `benchmark_results/summary.csv` | Metriche aggregate |

### Data Files

| File | Descrizione |
|------|-------------|
| `data/HEPAR_arcs.csv` | Edges del grafo causale |
| `data/HEPAR_simulated_patients.csv` | 10,000 pazienti |
| `data/hepar_columns.json` | Lista 70 colonne |
| `data/hepar_graph.json` | Grafo in formato JSON |
| `data/causal_rag_benchmark_v3.json` | 16 query benchmark |

---

*Documento aggiornato il 2026-01-17*
