"""
BM25 lexical retrieval for exact column name matching.

Semantic embeddings struggle with exact matches like `patient_id` vs `patientId`.
BM25 provides complementary lexical matching that catches these cases.

This module uses rank_bm25 for simplicity. For production with large
column sets, consider bm25s which offers 500x faster indexing.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Optional

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def _tokenize_column_name(text: str) -> List[str]:
    """
    Tokenize text preserving underscores and splitting camelCase.

    Examples:
        "patient_id" -> ["patient", "id"]
        "patientId" -> ["patient", "id"]
        "ALT_level" -> ["alt", "level"]
        "hepatomegaly" -> ["hepatomegaly"]
    """
    # First, replace underscores with spaces
    text = text.replace("_", " ")

    # Split camelCase: insert space before uppercase letters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Split on spaces and lowercase
    tokens = text.lower().split()

    # Filter empty tokens
    return [t for t in tokens if t]


def _tokenize_query(text: str) -> List[str]:
    """
    Tokenize a query string for BM25 matching.

    Similar to column tokenization but preserves more context.
    """
    # Replace punctuation with spaces (except underscores which are kept)
    text = re.sub(r'[^\w\s_]', ' ', text)

    # Replace underscores with spaces
    text = text.replace("_", " ")

    # Split camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Lowercase and split
    tokens = text.lower().split()

    # Filter short tokens and common stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
                 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once', 'and',
                 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                 'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
                 'also', 'now', 'here', 'there', 'when', 'where', 'why', 'how',
                 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                 'no', 'any', 'if', 'because', 'until', 'while', 'this', 'that',
                 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
                 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
                 'their', 'what', 'which', 'who', 'whom'}

    return [t for t in tokens if len(t) > 1 and t not in stopwords]


class BM25ColumnRetriever:
    """
    BM25-based retrieval for column names.

    Complements semantic retrieval by catching exact lexical matches
    that embeddings might miss.
    """

    def __init__(self):
        if not HAS_BM25:
            raise ImportError(
                "rank_bm25 is required for BM25ColumnRetriever. "
                "Install with: pip install rank-bm25"
            )
        self.bm25: Optional[BM25Okapi] = None
        self.columns: List[str] = []
        self.descriptions: Dict[str, str] = {}
        self.tokenized_corpus: List[List[str]] = []

    def index_columns(
        self,
        columns: List[str],
        descriptions: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Index columns for BM25 retrieval.

        Args:
            columns: List of column names
            descriptions: Optional dict mapping column names to descriptions
        """
        self.columns = columns
        self.descriptions = descriptions or {}

        # Create corpus: column name + optional description
        corpus = []
        for col in columns:
            text = col
            if col in self.descriptions:
                text = f"{col} {self.descriptions[col]}"
            corpus.append(text)

        # Tokenize corpus
        self.tokenized_corpus = [_tokenize_column_name(text) for text in corpus]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for columns matching the query.

        Args:
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of (column_name, bm25_score) sorted by score descending
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index_columns() first.")

        # Tokenize query
        query_tokens = _tokenize_query(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create (column, score) pairs and sort
        results = [(col, float(score)) for col, score in zip(self.columns, scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        # Filter zero scores and return top-k
        results = [(col, score) for col, score in results if score > 0]
        return results[:top_k]

    def get_all_scores(self, query: str) -> Dict[str, float]:
        """
        Get BM25 scores for all columns.

        Args:
            query: Search query

        Returns:
            Dict mapping column names to BM25 scores
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index_columns() first.")

        query_tokens = _tokenize_query(query)

        if not query_tokens:
            return {col: 0.0 for col in self.columns}

        scores = self.bm25.get_scores(query_tokens)
        return {col: float(score) for col, score in zip(self.columns, scores)}


class SimpleBM25:
    """
    Fallback simple BM25 implementation without external dependencies.

    Less accurate than rank_bm25 but works without installation.
    Use BM25ColumnRetriever for production.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.columns: List[str] = []
        self.tokenized: List[List[str]] = []
        self.avg_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.n_docs: int = 0

    def index_columns(self, columns: List[str]) -> None:
        self.columns = columns
        self.tokenized = [_tokenize_column_name(col) for col in columns]
        self.n_docs = len(columns)

        if self.n_docs == 0:
            self.avg_len = 0.0
            return

        self.avg_len = sum(len(doc) for doc in self.tokenized) / self.n_docs

        # Calculate document frequencies
        self.doc_freqs = {}
        for doc in self.tokenized:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency."""
        import math
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def _score_doc(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Score a single document against query."""
        score = 0.0
        doc_len = len(doc_tokens)

        if doc_len == 0:
            return 0.0

        # Count term frequencies in document
        tf: Dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1

        for term in query_tokens:
            if term not in tf:
                continue

            idf = self._idf(term)
            term_tf = tf[term]

            # BM25 formula
            numerator = term_tf * (self.k1 + 1)
            denominator = term_tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_tokens = _tokenize_query(query)

        if not query_tokens or not self.columns:
            return []

        scores = [
            (col, self._score_doc(query_tokens, doc_tokens))
            for col, doc_tokens in zip(self.columns, self.tokenized)
        ]

        scores.sort(key=lambda x: x[1], reverse=True)
        scores = [(col, score) for col, score in scores if score > 0]
        return scores[:top_k]

    def get_all_scores(self, query: str) -> Dict[str, float]:
        query_tokens = _tokenize_query(query)

        if not query_tokens:
            return {col: 0.0 for col in self.columns}

        return {
            col: self._score_doc(query_tokens, doc_tokens)
            for col, doc_tokens in zip(self.columns, self.tokenized)
        }


def create_bm25_retriever(
    columns: List[str],
    descriptions: Optional[Dict[str, str]] = None,
    use_fallback: bool = False
) -> BM25ColumnRetriever:
    """
    Factory function to create a BM25 retriever.

    Args:
        columns: List of column names to index
        descriptions: Optional column descriptions
        use_fallback: If True, use SimpleBM25 even if rank_bm25 is available

    Returns:
        Configured BM25 retriever
    """
    if use_fallback or not HAS_BM25:
        if not HAS_BM25 and not use_fallback:
            import warnings
            warnings.warn(
                "rank_bm25 not installed, using fallback SimpleBM25. "
                "Install rank_bm25 for better performance: pip install rank-bm25"
            )
        retriever = SimpleBM25()
        retriever.index_columns(columns)
        return retriever

    retriever = BM25ColumnRetriever()
    retriever.index_columns(columns, descriptions)
    return retriever
