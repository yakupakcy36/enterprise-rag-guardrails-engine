"""Multi-stage hybrid retrieval engine: Query Expansion, BM25, Dense TF-IDF, RRF & Cross-Reranker."""

import math
import re
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.chunker import DocumentChunk
from src.config import DEFAULT_RAG_CONFIG, RAGConfig


@dataclass
class SearchResult:
    chunk: DocumentChunk
    bm25_score: float
    dense_score: float
    rrf_score: float
    rerank_score: float
    rank: int


class QueryExpander:
    """Expands technical enterprise acronyms and domain synonyms prior to retrieval."""

    ACRONYM_MAP = {
        "p1": "p1 incident high-severity outage",
        "sla": "sla service level agreement",
        "rca": "rca root-cause analysis",
        "kms": "kms key management encryption",
        "tls": "tls cryptographic handshake",
        "mtls": "mtls zero trust mutual tls",
        "canary": "canary deployment rollout rollback",
    }

    @classmethod
    def expand(cls, query: str) -> str:
        tokens = re.findall(r"\b\w+\b", query.lower())
        expansions = [cls.ACRONYM_MAP[t] for t in tokens if t in cls.ACRONYM_MAP]
        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query


class BM25Index:
    """In-memory BM25 Okapi lexical search index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.tokenized_corpus: List[List[str]] = []

    def fit(self, texts: List[str]):
        self.corpus_size = len(texts)
        self.tokenized_corpus = [self._tokenize(t) for t in texts]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = float(sum(self.doc_lens) / max(1, self.corpus_size))

        self.doc_freqs = {}
        for doc in self.tokenized_corpus:
            for term in set(doc):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.idf = {
            term: math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in self.doc_freqs.items()
        }

    def score(self, query: str) -> np.ndarray:
        query_terms = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=float)
        if not query_terms or self.corpus_size == 0:
            return scores

        for term in query_terms:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]
            for doc_idx, doc in enumerate(self.tokenized_corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[doc_idx]
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-5, self.avgdl)))
                scores[doc_idx] += term_idf * (num / den)

        return scores

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())


class PairwiseCrossReranker:
    """
    Second-stage cross-reranker evaluating exact bi-gram overlap, token coverage,
    and proximity density between query and candidate chunks.
    """

    @staticmethod
    def score_pair(query: str, doc_text: str) -> float:
        q_tokens = re.findall(r"\b\w{2,}\b", query.lower())
        d_tokens = re.findall(r"\b\w{2,}\b", doc_text.lower())
        if not q_tokens or not d_tokens:
            return 0.0

        q_set = set(q_tokens)
        d_set = set(d_tokens)
        unigram_coverage = len(q_set.intersection(d_set)) / len(q_set)

        # Bi-gram exact phrase boost
        q_bigrams = set(zip(q_tokens, q_tokens[1:]))
        d_bigrams = set(zip(d_tokens, d_tokens[1:]))
        bigram_overlap = (
            len(q_bigrams.intersection(d_bigrams)) / len(q_bigrams)
            if q_bigrams
            else 0.0
        )

        return float(0.65 * unigram_coverage + 0.35 * bigram_overlap)


class HybridRetriever:
    """Production hybrid retriever with Query Expansion, BM25, Dense TF-IDF, RRF, and Cross-Reranking."""

    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        self.config = config
        self.chunks: List[DocumentChunk] = []
        self.bm25_index = BM25Index()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=10000,
        )
        self.dense_matrix = None
        self.is_indexed = False

    def index_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            self.chunks = []
            self.is_indexed = False
            return

        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.bm25_index.fit(texts)
        self.dense_matrix = self.vectorizer.fit_transform(texts)
        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        if not self.is_indexed or len(self.chunks) == 0:
            return []

        expanded_query = QueryExpander.expand(query)

        # 1. Sparse BM25 Scores
        bm25_scores = self.bm25_index.score(expanded_query)
        bm25_ranked = np.argsort(-bm25_scores)

        # 2. Dense Semantic Scores
        query_vec = self.vectorizer.transform([expanded_query])
        dense_scores = cosine_similarity(query_vec, self.dense_matrix)[0]
        dense_ranked = np.argsort(-dense_scores)

        # 3. Reciprocal Rank Fusion (RRF)
        k_const = self.config.rrf_k
        rrf_scores: Dict[int, float] = {}

        for rank, doc_idx in enumerate(bm25_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (
                (1.0 - self.config.hybrid_dense_weight) / (k_const + rank + 1)
            )
        for rank, doc_idx in enumerate(dense_ranked):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (
                self.config.hybrid_dense_weight / (k_const + rank + 1)
            )

        # 4. Cross-Reranking on candidate pool
        candidate_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[
            : max(top_k * 2, 6)
        ]

        reranked_pool = []
        for idx in candidate_indices:
            cross_score = PairwiseCrossReranker.score_pair(query, self.chunks[idx].text)
            # Combine normalized RRF with pairwise cross-reranker score
            norm_rrf = rrf_scores[idx] * (k_const + 1)
            final_score = (1.0 - self.config.rerank_weight) * norm_rrf + (
                self.config.rerank_weight * cross_score
            )
            reranked_pool.append((idx, final_score))

        reranked_pool.sort(key=lambda x: x[1], reverse=True)
        top_candidates = reranked_pool[:top_k]

        results = []
        for rank, (idx, r_score) in enumerate(top_candidates):
            results.append(SearchResult(
                chunk=self.chunks[idx],
                bm25_score=round(float(bm25_scores[idx]), 4),
                dense_score=round(float(dense_scores[idx]), 4),
                rrf_score=round(float(rrf_scores[idx]), 6),
                rerank_score=round(float(r_score), 4),
                rank=rank + 1,
            ))

        return results
