# Reusable Milvus RAG Template with 2-Stage Reranking & RAG Triad Evaluation

A production-ready, extensible template for building **Retrieval-Augmented Generation (RAG)** systems. Built with **Milvus Standalone**, **FastAPI**, **Cross-Encoder Reranking**, and an **End-to-End RAG Triad & IR Evaluation Framework**.

Fully containerized and tested for seamless deployment with both **Podman** (`podman compose` / `podman-compose`) and **Docker** (`docker compose`).

---

## Architecture

```
                  +----------------------------------------------+
                  |         Client App / Attu UI / REST          |
                  +----------------------------------------------+
                                         |
                                         v
+-------------------------------------------------------------------------------+
|                             FastAPI RAG Service                               |
|                                                                               |
|  +---------------------+   +-----------------------+   +-------------------+  |
|  | Document Processor  |   |   Embedding Service   |   | Cross-Encoder     |  |
|  | - PDF / TXT / MD    |   | - SentenceTransformer |   |   Reranker        |  |
|  | - Recursive/Overlap |   | - FastEmbed / API     |   | - BGE-Reranker    |  |
|  +---------------------+   +-----------------------+   +-------------------+  |
|                                        |                         |            |
|  +-------------------------------------------------------------+ |            |
|  | Extensible Abstract Interfaces (BaseVectorStore, BaseReranker)|            |
|  +-------------------------------------------------------------+ |            |
|                                        |                         |            |
|  +-------------------------------------+                         |            |
|  | Milvus Service Implementation                                 |            |
|  +---------------------------------------------------------------+            |
|                                        |                         |            |
|  +-------------------------------------+-------------------------+---------+  |
|  |                    End-to-End Evaluation Engine                         |  |
|  |   * IR Metrics: Hit Rate@K, MRR@K, NDCG@K, Precision@K, Recall@K       |  |
|  |   * RAG Triad: Context Relevance, Context Recall, Faithfulness          |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
                                         | (gRPC :19530)
                                         v
+-------------------------------------------------------------------------------+
|                       Milvus Standalone Container Stack                       |
|                                                                               |
|   +-----------------------+   +-------------------+   +--------------------+  |
|   |   milvus-standalone   |   |    milvus-etcd    |   |    milvus-minio    |  |
|   |   (Vector Search)     |   | (Meta Storage)    |   |  (Segment Storage) |  |
|   +-----------------------+   +-------------------+   +--------------------+  |
|               |                         |                        |            |
+---------------+-------------------------+------------------------+------------+
                |                         |                        |
                v                         v                        v
        ./volumes/milvus           ./volumes/etcd           ./volumes/minio
```

---

## Directory Structure

```text
milvus-data/
├── .env.example                # Config template (Milvus, Embedding, Reranker, Eval)
├── Dockerfile                  # API service container definition
├── docker-compose.yml          # Docker Compose stack
├── compose.yaml                # Podman Compose compatible stack
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata & packaging
│
├── app/
│   ├── main.py                 # FastAPI app entrypoint, CORS & Lifespan
│   ├── core/
│   │   ├── config.py           # Pydantic Settings configuration
│   │   ├── logging.py          # Unified logger setup
│   │   └── milvus.py           # Milvus connection pool & health checker
│   ├── api/v1/
│   │   ├── health.py           # /health endpoint (liveness/readiness)
│   │   ├── collections.py      # /collections CRUD endpoints
│   │   ├── documents.py        # /documents/upload & /documents/ingest-text
│   │   ├── search.py           # /search & /search/two-stage & /rerank
│   │   └── eval.py             # /eval/benchmark & /eval/samples
│   ├── models/
│   │   ├── collection.py       # Pydantic collection schemas
│   │   ├── document.py         # Document ingestion & chunk schemas
│   │   ├── search.py           # Vector search & two-stage schemas
│   │   ├── rerank.py           # Reranking schemas
│   │   └── eval.py             # IR metrics & RAG Triad benchmark schemas
│   └── services/
│       ├── interfaces.py       # Abstract base classes (Swappable vector store/reranker)
│       ├── milvus_service.py   # Concrete BaseVectorStore for PyMilvus
│       ├── document_processor.py # Multi-format loader & recursive chunker
│       ├── embedding_service.py # Sentence-Transformers / FastEmbed
│       ├── rerank_service.py   # Cross-Encoder (BAAI/bge-reranker-base)
│       ├── evaluation_service.py # HitRate, MRR, NDCG & RAG Triad Calculator
│       └── synthetic_generator.py # Option B: Document-to-Benchmark Synthetic Generator
│
├── streamlit_app.py            # Streamlit UI Studio (Sidebar Ingestion & 3-Tab Explorer)
├── docker-compose.standalone.yml # Single-container Milvus standalone profile
│
├── data/
│   ├── raw/                    # Uploaded & raw source documents (PDF, TXT, MD, DOCX)
│   ├── processed/              # Extracted, normalized chunks
│   ├── embeddings/             # Exported vector checkpoints
│   └── eval/
│       └── sample_eval_dataset.json # Golden reference query-context benchmark suite
│
├── scripts/
│   ├── run-podman.sh           # Bash launch script for Podman
│   ├── run-docker.sh           # Bash launch script for Docker
│   ├── run.ps1                 # Windows PowerShell launcher (Podman & Docker)
│   ├── evaluate_rag.py         # Standalone CLI IR & RAG Triad benchmark runner
│   └── test_rag_pipeline.py    # End-to-end integration test
│
└── volumes/                    # Persistent storage for Milvus, MinIO, etcd
```

---

## Quickstart Guide

### 1. Launch with Podman

```bash
# Using the helper script:
./scripts/run-podman.sh up

# Or directly with podman compose:
podman compose -f compose.yaml up -d
```

### 2. Launch with Docker

```bash
# Using the helper script:
./scripts/run-docker.sh up

# Or directly with docker compose:
docker compose -f docker-compose.yml up -d
```

### 3. Launch on Windows (PowerShell)

```powershell
# Run with Docker:
.\scripts\run.ps1 -Engine docker -Action up

# Run with Podman:
.\scripts\run.ps1 -Engine podman -Action up
```

---

## Service URLs

| Service | Address | Purpose |
| :--- | :--- | :--- |
| **Streamlit UI** | `http://localhost:8502` | Interactive Sidebar Ingestion, 3-Tab Querying & RAG Triad Evaluation UI |
| **FastAPI Swagger Docs** | `http://localhost:8000/docs` | Interactive OpenAPI REST test client |
| **FastAPI Health Status** | `http://localhost:8000/api/v1/health` | Milvus & model health probe |
| **Attu Web UI** | `http://localhost:3000` | Milvus graphical web dashboard |
| **Milvus Standalone (gRPC)** | `localhost:19530` | Vector Database endpoint |
| **MinIO Console** | `http://localhost:9001` | Object storage admin (minioadmin / minioadmin) |

---

## Core API Endpoints

### 1. Ingest Text / Files
* `POST /api/v1/documents/ingest-text`
  ```json
  {
    "collection_name": "rag_documents",
    "text": "Milvus separates compute and storage, utilizing MinIO and etcd.",
    "doc_id": "doc_001",
    "chunk_size": 300,
    "chunk_overlap": 30
  }
  ```
* `POST /api/v1/documents/upload` (Form-data: `file=@sample.pdf`)

### 2. Two-Stage Retrieval (Milvus + Cross-Encoder Rerank)
* `POST /api/v1/search/two-stage`
  ```json
  {
    "query": "How does Milvus achieve scalability?",
    "top_k_candidates": 20,
    "top_n_final": 5,
    "rerank_score_threshold": 0.35
  }
  ```

### 3. End-to-End RAG Triad & IR Benchmark Evaluation
* `POST /api/v1/eval/benchmark`
  ```json
  {
    "use_sample_dataset_file": true,
    "top_k": 10,
    "top_n": 5
  }
  ```
  **Returns:**
  ```json
  {
    "total_samples": 4,
    "vector_only_evaluation": {
      "ir_metrics": { "hit_rate_at_k": 0.75, "mrr_at_k": 0.58, "ndcg_at_k": 0.62 },
      "rag_triad_metrics": { "context_relevance": 0.78, "context_recall": 0.82, "faithfulness": 0.85 }
    },
    "reranked_evaluation": {
      "ir_metrics": { "hit_rate_at_k": 1.0, "mrr_at_k": 0.91, "ndcg_at_k": 0.93 },
      "rag_triad_metrics": { "context_relevance": 0.94, "context_recall": 0.98, "faithfulness": 0.96 }
    },
    "accuracy_gain_mrr_percent": 56.89,
    "accuracy_gain_ndcg_percent": 50.0
  }
  ```

---

## Portability: Swapping Vector Databases

All components inherit from abstract interfaces defined in `app/services/interfaces.py`:
* `BaseVectorStore`: Interface for collections, indexing, inserting, and searching vectors.
* `BaseEmbeddingService`: Interface for text and batch embedding generation.
* `BaseReranker`: Interface for cross-encoder rerankers.
* `BaseEvaluator`: Interface for running IR and RAG Triad benchmarks.

To use with **Qdrant**, **Chroma**, or **PgVector**:
1. Create `app/services/qdrant_service.py` implementing `BaseVectorStore`.
2. Swap the vector store instance in `app/api/v1/` dependencies or config.
3. The Document Processor, Embedding Service, Cross-Encoder Reranker, and Evaluation Framework will work out-of-the-box without modifying business logic.
