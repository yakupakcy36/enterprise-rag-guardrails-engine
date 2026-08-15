"""Automated RAG evaluation harness: Claim-level Faithfulness, MRR@k, NDCG@k, and Groundedness."""

import math
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Set
from src.retriever import SearchResult


@dataclass
class EvaluationReport:
    query: str
    response: str
    faithfulness_score: float
    claim_support_ratio: float
    relevance_score: float
    context_precision: float
    is_grounded: bool
    latency_ms: float
    violations: List[str]


class RAGEvaluator:
    """Evaluates RAG pipeline outputs for claim-level faithfulness, relevance, and IR metrics."""

    def __init__(self, min_faithfulness: float = 0.65):
        self.min_faithfulness = min_faithfulness

    def evaluate_output(
        self,
        query: str,
        retrieved_results: List[SearchResult],
        response: str,
        latency_ms: float = 0.0,
    ) -> EvaluationReport:
        violations = []

        if not retrieved_results:
            return EvaluationReport(
                query=query,
                response=response,
                faithfulness_score=0.0,
                claim_support_ratio=0.0,
                relevance_score=0.0,
                context_precision=0.0,
                is_grounded=False,
                latency_ms=latency_ms,
                violations=["No context retrieved for query"],
            )

        context_corpus = " ".join([res.chunk.text for res in retrieved_results]).lower()
        context_tokens: Set[str] = set(re.findall(r"\b\w{3,}\b", context_corpus))

        response_tokens = re.findall(r"\b\w{3,}\b", response.lower())
        query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))

        # 1. Token-level Faithfulness
        if response_tokens:
            supported_count = sum(1 for t in response_tokens if t in context_tokens)
            faithfulness = supported_count / len(response_tokens)
        else:
            faithfulness = 0.0

        # 2. Claim-level Decomposition & Verification
        raw_claims = [s.strip() for s in re.split(r"[.!?]+", response) if len(s.strip()) > 10]
        if raw_claims:
            supported_claims = 0
            for claim in raw_claims:
                c_tokens = re.findall(r"\b\w{3,}\b", claim.lower())
                if not c_tokens:
                    continue
                overlap = sum(1 for t in c_tokens if t in context_tokens) / len(c_tokens)
                if overlap >= 0.60:
                    supported_claims += 1
            claim_ratio = supported_claims / len(raw_claims)
        else:
            claim_ratio = faithfulness

        # 3. Answer Relevance
        if query_tokens:
            relevant_count = sum(1 for t in query_tokens if t in set(response_tokens))
            relevance = relevant_count / len(query_tokens)
        else:
            relevance = 1.0

        # 4. Context Precision
        hit_chunks = 0
        for res in retrieved_results:
            chunk_tokens = set(re.findall(r"\b\w{3,}\b", res.chunk.text.lower()))
            if query_tokens and len(query_tokens.intersection(chunk_tokens)) > 0:
                hit_chunks += 1
        context_prec = hit_chunks / max(1, len(retrieved_results))

        is_grounded = (faithfulness >= self.min_faithfulness) and (claim_ratio >= 0.50)
        if not is_grounded:
            violations.append(
                f"Potential Hallucination: Faithfulness {faithfulness:.2f} / Claim Support {claim_ratio:.2f} below threshold"
            )

        return EvaluationReport(
            query=query,
            response=response,
            faithfulness_score=round(float(faithfulness), 4),
            claim_support_ratio=round(float(claim_ratio), 4),
            relevance_score=round(float(relevance), 4),
            context_precision=round(float(context_prec), 4),
            is_grounded=is_grounded,
            latency_ms=round(float(latency_ms), 2),
            violations=violations,
        )

    @staticmethod
    def compute_ir_metrics(ranked_doc_ids: List[str], expected_doc_id: str) -> Dict[str, float]:
        """Compute Reciprocal Rank (RR), NDCG@k, and Hit@k for a single benchmark query."""
        rr = 0.0
        ndcg = 0.0
        hit = 0.0

        for idx, doc_id in enumerate(ranked_doc_ids):
            rank = idx + 1
            if doc_id == expected_doc_id:
                hit = 1.0
                rr = 1.0 / rank
                ndcg = 1.0 / math.log2(rank + 1.0)
                break

        return {
            "reciprocal_rank": round(rr, 4),
            "ndcg": round(ndcg, 4),
            "hit": hit,
        }
