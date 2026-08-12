"""Semantic query cache with TF-IDF cosine similarity matching and LRU eviction."""

from collections import OrderedDict
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticQueryCache:
    """
    Caches verified RAG responses and serves semantically equivalent queries
    in sub-millisecond latency when cosine similarity >= threshold.
    """

    def __init__(self, similarity_threshold: float = 0.90, max_size: int = 256):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def lookup(self, sanitized_query: str) -> Optional[Dict[str, Any]]:
        """Check exact or semantic similarity match in cache."""
        norm_q = sanitized_query.strip().lower()
        if not norm_q or len(self._store) == 0:
            self.misses += 1
            return None

        # 1. Fast exact match
        if norm_q in self._store:
            self._store.move_to_end(norm_q)
            self.hits += 1
            return dict(self._store[norm_q])

        # 2. Semantic cosine similarity match across cached queries
        cached_keys: List[str] = list(self._store.keys())
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2)).fit(cached_keys + [norm_q])
            key_mat = vec.transform(cached_keys)
            q_vec = vec.transform([norm_q])
            sims = cosine_similarity(q_vec, key_mat)[0]
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= self.similarity_threshold:
                matched_key = cached_keys[best_idx]
                self._store.move_to_end(matched_key)
                self.hits += 1
                return dict(self._store[matched_key])
        except ValueError:
            pass

        self.misses += 1
        return None

    def store(self, sanitized_query: str, response_payload: Dict[str, Any]) -> None:
        """Insert verified response into LRU semantic cache."""
        norm_q = sanitized_query.strip().lower()
        if not norm_q:
            return
        if norm_q in self._store:
            self._store.move_to_end(norm_q)
        self._store[norm_q] = dict(response_payload)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "entries_count": len(self._store),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(float(hit_rate), 4),
            "similarity_threshold": self.similarity_threshold,
        }
