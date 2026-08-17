"""Test suite for document chunking, BM25, and hybrid RRF retrieval."""

import pytest
from src.chunker import TextChunker
from src.retriever import HybridRetriever, BM25Index


def test_chunker_sentence_boundary_and_overlap():
    """Verify sentence boundaries and overlap."""
    chunker = TextChunker(chunk_size=120, chunk_overlap=30, min_chunk_len=20)
    text = (
        "First sentence discussing incident management. "
        "Second sentence detailing database replication lag. "
        "Third sentence outlining failover protocol."
    )
    chunks = chunker.chunk_document(doc_id="test_doc", text=text)
    assert len(chunks) >= 2
    assert chunks[0].doc_id == "test_doc"
    assert chunks[0].chunk_id == "test_doc_c0"
    assert len(chunks[0].text) > 20


def test_bm25_index():
    """Verify BM25 accurately scores exact keyword matches."""
    index = BM25Index()
    docs = [
        "Kubernetes container orchestration and deployment",
        "PostgreSQL relational database replication and failover",
        "Redis in-memory caching and token bucket rate limiting",
    ]
    index.fit(docs)
    scores = index.score("PostgreSQL replication")
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_hybrid_retriever_rrf():
    """Verify hybrid retriever returns ranked results with RRF scores."""
    retriever = HybridRetriever()
    chunker = TextChunker(chunk_size=300)
    raw_docs = [
        {"doc_id": "doc_db", "text": "PostgreSQL database failover triggers when replica lag exceeds 500ms."},
        {"doc_id": "doc_rate", "text": "Token bucket rate limiter throttles API traffic to 100 requests per minute."},
        {"doc_id": "doc_sec", "text": "AES-256 encryption secures data at rest with automatic key rotation."},
    ]
    all_chunks = []
    for d in raw_docs:
        all_chunks.extend(chunker.chunk_document(d["doc_id"], d["text"]))

    retriever.index_chunks(all_chunks)
    assert retriever.is_indexed is True

    results = retriever.retrieve(query="database failover replica lag", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.doc_id == "doc_db"
    assert results[0].rrf_score > results[1].rrf_score
    assert results[0].bm25_score > 0.0
