"""Test suite for RAG evaluation harness: claim support, IR metrics, and hallucination detection."""

from src.chunker import DocumentChunk
from src.retriever import SearchResult
from src.evaluator import RAGEvaluator


def test_evaluator_grounded_response():
    evaluator = RAGEvaluator(min_faithfulness=0.60)
    chunk = DocumentChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="The database replication lag must remain under 500ms to avoid failover.",
        start_char=0,
        end_char=75,
        token_count_approx=12,
    )
    search_res = SearchResult(
        chunk=chunk, bm25_score=1.0, dense_score=1.0, rrf_score=0.03, rerank_score=0.85, rank=1
    )

    query = "What is the replication lag threshold?"
    response = "Database replication lag must remain under 500ms to avoid failover."

    report = evaluator.evaluate_output(query=query, retrieved_results=[search_res], response=response)
    assert report.is_grounded is True
    assert report.faithfulness_score >= 0.60
    assert report.claim_support_ratio >= 0.60
    assert len(report.violations) == 0


def test_evaluator_hallucinated_response():
    evaluator = RAGEvaluator(min_faithfulness=0.65)
    chunk = DocumentChunk(
        chunk_id="c1",
        doc_id="doc1",
        text="All customer payments are processed exclusively through Stripe checkout.",
        start_char=0,
        end_char=70,
        token_count_approx=10,
    )
    search_res = SearchResult(
        chunk=chunk, bm25_score=1.0, dense_score=1.0, rrf_score=0.03, rerank_score=0.85, rank=1
    )

    query = "How are payments handled?"
    response = "We accept Bitcoin, Ethereum crypto transactions, and PayPal direct wire transfers."

    report = evaluator.evaluate_output(query=query, retrieved_results=[search_res], response=response)
    assert report.is_grounded is False
    assert report.faithfulness_score < 0.65


def test_ir_metrics_computation():
    ir = RAGEvaluator.compute_ir_metrics(["doc_a", "doc_b", "doc_c"], "doc_b")
    assert ir["reciprocal_rank"] == 0.5
    assert ir["hit"] == 1.0
    assert ir["ndcg"] > 0.6
