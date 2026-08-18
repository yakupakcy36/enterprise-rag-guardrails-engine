# Enterprise RAG, Cross-Reranker & Security Guardrails Engine

[![CI Pipeline](https://github.com/yakupakcy36/enterprise-rag-guardrails-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yakupakcy36/enterprise-rag-guardrails-engine/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade Retrieval-Augmented Generation (RAG) engine featuring **multi-stage hybrid search (Query Expansion + BM25 + Dense Semantic RRF + Pairwise Cross-Reranking)**, **Base64 de-obfuscating prompt injection defenses**, **PII & secret credential redaction**, **LRU semantic caching**, and an **automated Golden Dataset evaluation harness (MRR@k, NDCG@k, Claim-Level Faithfulness)**.

Built on 100% local, open-source algorithmic foundations with **zero external paid API dependencies**.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph USER_INPUT["1. User Request"]
        Q["Incoming Query / Prompt"]
    end

    subgraph GUARDRAILS["2. Security & De-Obfuscation Layer"]
        G1["Base64 Smuggling Decoder & Injection Scanner"]
        G2["Chat Template / Delimiter Tampering Filter"]
        G3["PII & Credential Sanitizer: Email, Phone, API Keys, IPv4, CC"]
        Q --> G1
        G1 -->|Threat Detected| BLOCKED["HTTP 200 Blocked Audit Response"]
        G1 -->|Clean Query| G2 --> G3
    end

    subgraph CACHE["3. Semantic Cache Layer"]
        C1["LRU + TF-IDF Cosine Similarity Cache (>= 0.90)"]
        G3 --> C1
        C1 -->|Cache Hit (< 0.2ms)| SERVE_CACHED["Return Verified Cached Response"]
    end

    subgraph RETRIEVAL["4. Multi-Stage Hybrid Retrieval & Cross-Reranking"]
        direction TB
        EXP["Domain Acronym & Synonym Query Expander"]
        S1["Sparse BM25+ Okapi Lexical Index"]
        S2["Dense TF-IDF Bi-gram Semantic Index"]
        RRF["Reciprocal Rank Fusion (RRF)"]
        CROSS["Stage 2 Pairwise Cross-Reranker (Unigram + Bi-gram Proximity)"]
        C1 -->|Cache Miss| EXP
        EXP --> S1 & S2
        S1 & S2 --> RRF --> CROSS
    end

    subgraph EVALUATION["5. Claim-Level Groundedness & IR Benchmark"]
        direction TB
        E1["Atomic Claim Decomposition & Verification"]
        E2["Token Faithfulness & Context Precision"]
        E3["Golden Benchmark Suite: MRR@k, NDCG@k, HitRate@k"]
        CROSS --> E1 & E2
    end

    subgraph SERVING["6. FastAPI Serving"]
        API1["POST /query & /ingest"]
        API2["GET /evaluate/benchmark"]
        API3["GET /cache/stats & /health"]
    end

    E1 & E2 --> SERVING
    E3 --> API2
```

---

## 🔬 Algorithmic Foundations & Formulations

### 1. Reciprocal Rank Fusion (RRF) & Pairwise Cross-Reranking
First-stage candidates from lexical **BM25** ($k_1 = 1.5, b = 0.75$) and dense **TF-IDF sublinear n-gram vectors** are merged via RRF:

$$\text{RRF}(d) = \sum_{m \in \{\text{BM25},\,\text{Dense}\}} \frac{w_m}{k + r_m(d)}$$

Top candidates are subsequently re-scored by a **Pairwise Cross-Reranker** combining unigram coverage and exact bi-gram phrase overlap:

$$S_{\text{rerank}}(Q, D) = (1 - \lambda)\cdot \widehat{\text{RRF}}(D) + \lambda \cdot \left(0.65\frac{|Q_1 \cap D_1|}{|Q_1|} + 0.35\frac{|Q_2 \cap D_2|}{|Q_2|}\right)$$

### 2. Automated IR & Claim-Level Evaluation Metrics
* **Mean Reciprocal Rank ($\text{MRR}@k$):**
  $$\text{MRR}@k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
* **Normalized Discounted Cumulative Gain ($\text{NDCG}@k$):**
  $$\text{NDCG}@k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\log_2(\text{rank}_i + 1)}$$
* **Atomic Claim Support Ratio:** Decomposes responses into individual assertions $C = \{c_1, \dots, c_m\}$ and verifies token grounding against retrieved evidence.

---

## 📁 Repository Structure

```text
enterprise-rag-guardrails-engine/
├── .github/workflows/ci.yml     # Automated GitHub Actions CI pipeline
├── src/
│   ├── config.py                # Retrieval, reranking, cache & security parameters
│   ├── chunker.py               # Sentence-boundary sliding window chunker
│   ├── guardrails.py            # Base64 de-obfuscation, injection & PII guardrails
│   ├── cache.py                 # Semantic cosine similarity LRU cache
│   ├── retriever.py             # Query expansion, BM25, TF-IDF, RRF & Cross-Reranker
│   ├── evaluator.py             # Claim faithfulness, MRR@k, NDCG@k evaluator
│   ├── knowledge.py             # Baseline corpus & Golden Benchmark dataset
│   └── engine.py                # End-to-end RAG & benchmark orchestrator
├── api/
│   ├── schemas.py               # Strict Pydantic v2 schemas
│   └── app.py                   # FastAPI production server
├── tests/                       # Comprehensive unit & integration test suite
├── Dockerfile                   # Container image specification
├── docker-compose.yml           # Local service orchestration
└── requirements.txt
```

---

## 🚀 Quickstart & Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute test suite
pytest -v

# 3. Launch production API
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Reference

* **`POST /query`**: Runs input de-obfuscation & PII masking $\to$ semantic cache lookup $\to$ hybrid RRF + cross-reranking $\to$ claim-level faithfulness verification.
* **`GET /evaluate/benchmark`**: Executes the automated Golden Dataset regression suite and returns `mrr_at_k`, `ndcg_at_k`, `hit_rate_at_k`, and `mean_faithfulness`.
* **`GET /cache/stats`**: Reports semantic cache hit rate and active entry count.
* **`POST /ingest`**: Dynamically chunks and indexes new documents and invalidates stale cache entries.

---

## 📄 License
MIT License.
