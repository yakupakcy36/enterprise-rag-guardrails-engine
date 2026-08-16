"""FastAPI application for Enterprise RAG, Cross-Reranker, Semantic Cache & IR Benchmark."""

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_RAG_CONFIG
from src.engine import EnterpriseRAGEngine
from src.knowledge import ENTERPRISE_KNOWLEDGE_BASE
from api.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    BenchmarkSuiteResponse,
    CacheStatsResponse,
    HealthResponse,
    CitationItem,
    GuardrailAudit,
    EvaluationMetricsSchema,
)

rag_engine: Optional[EnterpriseRAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    rag_engine = EnterpriseRAGEngine(config=DEFAULT_RAG_CONFIG)
    rag_engine.ingest_documents(ENTERPRISE_KNOWLEDGE_BASE)
    yield


app = FastAPI(
    title="Enterprise RAG, Cross-Reranker & Security Guardrails Engine",
    description=(
        "Production API featuring multi-stage retrieval (BM25 + Dense Semantic RRF + Pairwise Cross-Reranking), "
        "Base64 de-obfuscating prompt injection defense, PII redaction, semantic caching, "
        "and an automated MRR/NDCG/Faithfulness evaluation harness."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    if rag_engine is None:
        return HealthResponse(
            status="unhealthy",
            engine_ready=False,
            indexed_chunks=0,
            hybrid_retrieval_enabled=False,
            semantic_cache_enabled=False,
            guardrails_active=False,
        )

    return HealthResponse(
        status="healthy",
        engine_ready=True,
        indexed_chunks=len(rag_engine.retriever.chunks),
        hybrid_retrieval_enabled=rag_engine.retriever.is_indexed,
        semantic_cache_enabled=DEFAULT_RAG_CONFIG.enable_semantic_cache,
        guardrails_active=True,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
def execute_query(request: QueryRequest):
    if rag_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG Engine unavailable")

    result = rag_engine.query(request.query)
    citations = [CitationItem(**c) for c in result.get("citations", [])]
    guardrail_audit = GuardrailAudit(**result["guardrail_audit"])
    evaluation = EvaluationMetricsSchema(**result["evaluation"]) if result.get("evaluation") else None

    return QueryResponse(
        status=result["status"],
        cache_hit=result.get("cache_hit", False),
        query=result["query"],
        sanitized_query=result["sanitized_query"],
        answer=result["answer"],
        citations=citations,
        guardrail_audit=guardrail_audit,
        evaluation=evaluation,
        latency_ms=result["latency_ms"],
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Knowledge Base"])
def ingest_documents(request: IngestRequest):
    if rag_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG Engine unavailable")

    docs = [{"doc_id": d.doc_id, "text": d.text} for d in request.documents]
    new_chunks = rag_engine.ingest_documents(docs)
    return IngestResponse(
        status="success",
        chunks_indexed=new_chunks,
        total_indexed_chunks=len(rag_engine.retriever.chunks),
    )


@app.get("/evaluate/benchmark", response_model=BenchmarkSuiteResponse, tags=["Evaluation Harness"])
def run_golden_benchmark():
    """Run automated Golden Dataset IR & Faithfulness regression harness (MRR@k, NDCG@k, HitRate@k)."""
    if rag_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG Engine unavailable")

    metrics = rag_engine.run_benchmark()
    return BenchmarkSuiteResponse(status="success", **metrics)


@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["Semantic Cache"])
def get_cache_stats():
    if rag_engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RAG Engine unavailable")
    return CacheStatsResponse(**rag_engine.cache.stats())
