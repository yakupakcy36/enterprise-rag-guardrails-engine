"""End-to-end RAG orchestrator integrating guardrails, semantic cache, hybrid retrieval, and IR benchmark."""

import time
from typing import List, Dict, Any
from src.config import DEFAULT_RAG_CONFIG, RAGConfig
from src.chunker import TextChunker, DocumentChunk
from src.guardrails import SecurityGuardrails, GuardrailResult
from src.retriever import HybridRetriever, SearchResult
from src.evaluator import RAGEvaluator, EvaluationReport
from src.cache import SemanticQueryCache
from src.knowledge import GOLDEN_BENCHMARK_QUERIES


class EnterpriseRAGEngine:
    """Production RAG engine with guardrails, semantic caching, cross-reranking, and IR benchmarking."""

    def __init__(self, config: RAGConfig = DEFAULT_RAG_CONFIG):
        self.config = config
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_len=config.min_chunk_len,
        )
        self.guardrails = SecurityGuardrails(risk_threshold=config.jailbreak_alert_threshold)
        self.retriever = HybridRetriever(config=config)
        self.evaluator = RAGEvaluator(min_faithfulness=config.min_faithfulness_score)
        self.cache = SemanticQueryCache(
            similarity_threshold=config.semantic_cache_threshold,
            max_size=config.max_cache_size,
        )

    def ingest_documents(self, documents: List[Dict[str, str]]) -> int:
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            doc_id = doc.get("doc_id", "doc_default")
            text = doc.get("text", "")
            chunks = self.chunker.chunk_document(doc_id=doc_id, text=text)
            all_chunks.extend(chunks)

        self.retriever.index_chunks(all_chunks)
        self.cache.clear()
        return len(all_chunks)

    def query(self, raw_query: str) -> Dict[str, Any]:
        start_time = time.perf_counter()

        # Step 1: Security Guardrails
        guard_result: GuardrailResult = self.guardrails.inspect_query(raw_query)
        if not guard_result.is_safe:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "blocked",
                "cache_hit": False,
                "query": raw_query,
                "sanitized_query": guard_result.sanitized_text,
                "answer": "Query blocked by security guardrails: Adversarial or instruction override detected.",
                "citations": [],
                "guardrail_audit": {
                    "is_safe": False,
                    "risk_score": guard_result.risk_score,
                    "violations": guard_result.violation_reasons,
                    "pii_redacted_count": guard_result.pii_redacted_count,
                },
                "evaluation": None,
                "latency_ms": round(elapsed_ms, 2),
            }

        query_text = guard_result.sanitized_text

        # Step 2: Semantic Cache Lookup
        if self.config.enable_semantic_cache:
            cached = self.cache.lookup(query_text)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                cached_resp = dict(cached)
                cached_resp["cache_hit"] = True
                cached_resp["query"] = raw_query
                cached_resp["latency_ms"] = round(elapsed_ms, 2)
                return cached_resp

        # Step 3: Hybrid Retrieval + Cross-Reranking
        search_results: List[SearchResult] = self.retriever.retrieve(
            query=query_text,
            top_k=self.config.top_k_final,
        )

        # Step 4: Grounded Synthesis
        if not search_results:
            answer = "No relevant context found in indexed knowledge base to answer the query."
            citations = []
        else:
            top_chunk = search_results[0].chunk
            citations = [
                {
                    "chunk_id": res.chunk.chunk_id,
                    "doc_id": res.chunk.doc_id,
                    "rrf_score": res.rrf_score,
                    "rerank_score": res.rerank_score,
                    "bm25_score": res.bm25_score,
                    "dense_score": res.dense_score,
                    "excerpt": res.chunk.text[:180] + "...",
                }
                for res in search_results
            ]
            answer = f"Based on {top_chunk.doc_id}: {top_chunk.text}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 5: Claim-Level Faithfulness Evaluation
        eval_report: EvaluationReport = self.evaluator.evaluate_output(
            query=query_text,
            retrieved_results=search_results,
            response=answer,
            latency_ms=elapsed_ms,
        )

        payload = {
            "status": "success",
            "cache_hit": False,
            "query": raw_query,
            "sanitized_query": guard_result.sanitized_text,
            "answer": answer,
            "citations": citations,
            "guardrail_audit": {
                "is_safe": True,
                "risk_score": guard_result.risk_score,
                "violations": guard_result.violation_reasons,
                "pii_redacted_count": guard_result.pii_redacted_count,
            },
            "evaluation": {
                "faithfulness_score": eval_report.faithfulness_score,
                "claim_support_ratio": eval_report.claim_support_ratio,
                "relevance_score": eval_report.relevance_score,
                "context_precision": eval_report.context_precision,
                "is_grounded": eval_report.is_grounded,
                "violations": eval_report.violations,
            },
            "latency_ms": round(elapsed_ms, 2),
        }

        if self.config.enable_semantic_cache and eval_report.is_grounded:
            self.cache.store(query_text, payload)

        return payload

    def run_benchmark(self) -> Dict[str, Any]:
        """Execute automated IR & Faithfulness benchmark over Golden Dataset."""
        rr_scores = []
        ndcg_scores = []
        hits = []
        faithfulness_scores = []

        for item in GOLDEN_BENCHMARK_QUERIES:
            q = item["query"]
            expected_doc = item["expected_doc_id"]

            results = self.retriever.retrieve(q, top_k=self.config.top_k_final)
            ranked_docs = [r.chunk.doc_id for r in results]
            ir = self.evaluator.compute_ir_metrics(ranked_docs, expected_doc)

            rr_scores.append(ir["reciprocal_rank"])
            ndcg_scores.append(ir["ndcg"])
            hits.append(ir["hit"])

            resp = self.query(q)
            if resp.get("evaluation"):
                faithfulness_scores.append(resp["evaluation"]["faithfulness_score"])

        n = max(1, len(GOLDEN_BENCHMARK_QUERIES))
        return {
            "queries_evaluated": len(GOLDEN_BENCHMARK_QUERIES),
            "mrr_at_k": round(sum(rr_scores) / n, 4),
            "ndcg_at_k": round(sum(ndcg_scores) / n, 4),
            "hit_rate_at_k": round(sum(hits) / n, 4),
            "mean_faithfulness": round(sum(faithfulness_scores) / max(1, len(faithfulness_scores)), 4),
        }
