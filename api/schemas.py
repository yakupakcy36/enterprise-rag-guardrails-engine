"""Pydantic v2 schemas for the Enterprise RAG, Semantic Cache & Benchmark API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="User search or question prompt")


class CitationItem(BaseModel):
    chunk_id: str
    doc_id: str
    rrf_score: float
    rerank_score: float
    bm25_score: float
    dense_score: float
    excerpt: str


class GuardrailAudit(BaseModel):
    is_safe: bool
    risk_score: float
    violations: List[str]
    pii_redacted_count: int


class EvaluationMetricsSchema(BaseModel):
    faithfulness_score: float
    claim_support_ratio: float
    relevance_score: float
    context_precision: float
    is_grounded: bool
    violations: List[str]


class QueryResponse(BaseModel):
    status: str
    cache_hit: bool
    query: str
    sanitized_query: str
    answer: str
    citations: List[CitationItem]
    guardrail_audit: GuardrailAudit
    evaluation: Optional[EvaluationMetricsSchema]
    latency_ms: float


class IngestDocumentItem(BaseModel):
    doc_id: str
    text: str


class IngestRequest(BaseModel):
    documents: List[IngestDocumentItem]


class IngestResponse(BaseModel):
    status: str = "success"
    chunks_indexed: int
    total_indexed_chunks: int


class BenchmarkSuiteResponse(BaseModel):
    status: str = "success"
    queries_evaluated: int
    mrr_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    mean_faithfulness: float


class CacheStatsResponse(BaseModel):
    entries_count: int
    max_size: int
    hits: int
    misses: int
    hit_rate: float
    similarity_threshold: float


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    indexed_chunks: int
    hybrid_retrieval_enabled: bool
    semantic_cache_enabled: bool
    guardrails_active: bool
