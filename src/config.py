"""Configuration settings for Enterprise RAG, Cross-Reranker, Semantic Cache & Guardrails."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    """Master configuration for chunking, hybrid retrieval, reranking, caching, and guardrails."""
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 80
    min_chunk_len: int = 40

    # Hybrid Retrieval & Cross-Reranking
    top_k_bm25: int = 6
    top_k_dense: int = 6
    top_k_final: int = 3
    rrf_k: int = 60
    hybrid_dense_weight: float = 0.50
    rerank_weight: float = 0.35

    # Semantic Cache
    enable_semantic_cache: bool = True
    semantic_cache_threshold: float = 0.90
    max_cache_size: int = 256

    # Security Guardrails & Evaluation
    max_query_chars: int = 1000
    jailbreak_alert_threshold: float = 0.70
    min_faithfulness_score: float = 0.65

    # Storage
    data_dir: Path = Path("data")
    index_storage_path: Path = Path("artifacts/index_store.joblib")


DEFAULT_RAG_CONFIG = RAGConfig()
