# Reusable Milvus RAG Template, Reranking & Evaluation Framework Plan

## Top-Level Overview
This project provides a production-ready, modular, and reusable template for RAG (Retrieval-Augmented Generation) applications powered by Milvus vector database, FastAPI, Cross-Encoder rerankers, and an **End-to-End RAG Triad & IR Evaluation Framework**.

The entire architecture is designed with **loose coupling and vector-store abstraction**, allowing the document processing, embedding, reranking, and evaluation engines to be decoupled and reused seamlessly across other vector databases (e.g., Qdrant, Chroma, PgVector, Weaviate).

Container orchestration supports both **Podman** and **Docker** via compose configurations.

---

## Architecture Overview

### 1. Instance Architecture (Container Orchestration)
- **Engine Support:** Compatible with `podman compose` / `podman-compose` and `docker compose`.
- **Milvus Standalone Stack:**
  - `milvus-etcd`: Metadata storage.
  - `milvus-minio`: Object storage for vector indexes and log persistence.
  - `milvus-standalone`: Milvus core engine with gRPC (`19530`) and REST/Metrics (`9091`).
  - `milvus-attu` (Optional / Included): Web UI manager for Milvus on port `8000` or `3000`.
  - `rag-api`: FastAPI application service exposing ingestion, collection management, vector similarity search, reranking, and evaluation endpoints.
- **Persistence & Volumes:** Configurable volume bindings for data persistence across restarts (`./volumes/milvus`, `./volumes/etcd`, `./volumes/minio`).

### 2. Source Document & Evaluation Data Architecture
- `data/`
  - `raw/`: Unprocessed incoming files (PDF, Markdown, TXT, JSON, DOCX).
  - `processed/`: Cleaned, parsed, and normalized text/chunks with metadata tags.
  - `embeddings/`: Cached or exportable vector dumps / checkpoints.
  - `eval/`: Golden evaluation benchmarks (questions, ground-truth context chunks, reference answers).

### 3. Decoupled Modular Service Architecture
All services are implemented behind abstract base classes (interfaces) so that components can be extracted or swapped with other vector databases without rewriting pipeline logic.

- `app/`
  - `api/v1/`: Modular endpoints:
    - `/health`: Liveness and readiness probes checking Milvus connectivity.
    - `/collections`: Manage collections (create, describe, list, drop, index management).
    - `/documents`: Upload, parse, chunk, and embed documents.
    - `/search`: Vector similarity search and hybrid filtering.
    - `/rerank`: Standalone and 2-stage retrieval + reranking pipeline.
    - `/eval`: Trigger benchmark evaluations comparing Vector-only vs Reranked vs Ground Truth (IR metrics + RAG Triad).
  - `core/`: Config settings (Pydantic settings), logging, lifespan management.
  - `services/`:
    - `interfaces/`: Abstract base classes (`BaseVectorStore`, `BaseEmbeddingService`, `BaseReranker`, `BaseEvaluator`).
    - `milvus_client.py` & `milvus_service.py`: Implementation of `BaseVectorStore` for PyMilvus.
    - `document_processor.py`: Text extraction and configurable chunking strategies.
    - `embedding_service.py`: Implementation of `BaseEmbeddingService` (SentenceTransformers / FastEmbed / API-based).
    - `rerank_service.py`: Implementation of `BaseReranker` (Cross-Encoder / FlashRank / API-based).
    - `evaluation_service.py`: Implementation of `BaseEvaluator` executing:
      1. **IR Metrics**: Hit Rate@K, MRR@K (Mean Reciprocal Rank), NDCG@K, Precision@K, Recall@K.
      2. **RAG Triad Metrics**: Context Relevance / Precision, Context Recall, Faithfulness / Groundedness.
  - `models/`: Pydantic request/response schemas (`collection.py`, `document.py`, `search.py`, `rerank.py`, `eval.py`).

---

## Sub-Tasks

### Sub-Task 1: Infrastructure & Container Orchestration Setup
- **Intent**: Provide Docker Compose and Podman-compatible container definitions, environment configs, and directory structures.
- **Expected Outcomes**:
  - `compose.yaml` (and `docker-compose.yml`) configured for Milvus standalone + MinIO + etcd + Attu UI + API service.
  - Podman / Docker launch and helper scripts (`scripts/run-podman.sh`, `scripts/run-docker.sh` / `.ps1`).
  - `.env.example` with default connection parameters, model configurations, and eval settings.
  - Directory structure initialized with `.gitkeep` for volumes, raw/processed data, and eval benchmarks.
- **Todo List**:
  - [ ] Create `.env.example` defining ports, hostnames, Milvus credentials, model settings, and evaluation keys.
  - [ ] Create `docker-compose.yml` (and `compose.yaml`) optimized for Podman and Docker.
  - [ ] Create root directory structure: `data/raw/`, `data/processed/`, `data/eval/`, `volumes/`.
  - [ ] Add startup/teardown utility scripts for Podman and Docker (`scripts/`).
- **Relevant Context**: `docker-compose.yml`, `compose.yaml`, `.env.example`, `scripts/`
- **Status**: [ ] pending

---

### Sub-Task 2: Core FastAPI Application & Extensible Base Interfaces
- **Intent**: Build the FastAPI foundational app, lifespan connection management, and abstract interfaces (`BaseVectorStore`, `BaseEmbeddingService`, `BaseReranker`, `BaseEvaluator`) for easy portability to other vector DBs.
- **Expected Outcomes**:
  - Extensible abstract base classes enabling plug-and-play swapping of vector stores and evaluation engines.
  - Application startup/shutdown hooks handling Milvus connection lifecycle.
  - Health check endpoint testing Milvus server ping, collection state, and storage reachability.
  - Pydantic configuration loading settings from environment variables.
- **Todo List**:
  - [ ] Create `app/services/interfaces/` defining `BaseVectorStore`, `BaseEmbeddingService`, `BaseReranker`, and `BaseEvaluator`.
  - [ ] Create `app/core/config.py` using `pydantic-settings`.
  - [ ] Create `app/core/milvus.py` for connection pooling, health checks, and lifecycle handlers.
  - [ ] Create `app/main.py` with FastAPI initialization, CORS middleware, and lifespan handlers.
  - [ ] Create `app/api/v1/health.py` for health and readiness endpoints.
- **Relevant Context**: `app/services/interfaces/`, `app/main.py`, `app/core/`, `app/api/v1/health.py`
- **Status**: [ ] pending

---

### Sub-Task 3: Document Processing, Embedding & Reranking Services
- **Intent**: Implement modular utilities for parsing source documents, chunking strategies, generating embeddings, and cross-encoder reranking.
- **Expected Outcomes**:
  - Flexible document parsing (TXT, Markdown, PDF, JSON).
  - Chunking strategies (fixed size with overlap, sentence/markdown header based).
  - Concrete embedding provider implementation (`SentenceTransformers` / `FastEmbed` / API-based).
  - Concrete reranker provider implementation (`CrossEncoder` / `FlashRank` for lightweight CPU-friendly inference).
- **Todo List**:
  - [ ] Implement `app/services/document_processor.py` for text extraction and chunking with configurable overlap.
  - [ ] Implement `app/services/embedding_service.py` implementing `BaseEmbeddingService`.
  - [ ] Implement `app/services/rerank_service.py` implementing `BaseReranker`.
  - [ ] Create schemas in `app/models/document.py` and `app/models/rerank.py` for chunk payloads, metadata, and reranking request/response models.
- **Relevant Context**: `app/services/`, `app/models/document.py`, `app/models/rerank.py`
- **Status**: [ ] pending

---

### Sub-Task 4: Milvus Vector Store Service & API Endpoints
- **Intent**: Implement the PyMilvus vector store provider and REST endpoints for collection management, ingestion, and 2-stage retrieval.
- **Expected Outcomes**:
  - `MilvusService` implementing `BaseVectorStore` (create, describe, list, drop collections, create indexes, insert, and vector similarity search).
  - Document ingestion endpoint accepting file uploads or direct text payloads, vectorizing, and inserting into Milvus.
  - 2-Stage vector search endpoint retrieving top-$K$ from Milvus and reranking to top-$N$ with confidence score thresholding.
- **Todo List**:
  - [ ] Create `app/services/milvus_service.py` implementing `BaseVectorStore`.
  - [ ] Create `app/api/v1/collections.py` for collection CRUD operations.
  - [ ] Create `app/api/v1/documents.py` for file upload, chunking, and batch vector ingestion.
  - [ ] Create `app/api/v1/search.py` for semantic search, metadata filtering, and 2-stage reranked retrieval.
  - [ ] Create `app/models/` schemas for collection parameters and search requests/responses.
- **Relevant Context**: `app/services/milvus_service.py`, `app/api/v1/`, `app/models/`
- **Status**: [ ] pending

---

### Sub-Task 5: End-to-End RAG Triad & IR Evaluation Framework
- **Intent**: Implement a self-contained, portable evaluation module to benchmark retrieval accuracy (Vector Search vs Reranked) and compute RAG Triad metrics (Context Relevance, Context Recall, Faithfulness).
- **Expected Outcomes**:
  - `EvaluationService` implementing `BaseEvaluator` computing IR metrics (Hit Rate@K, MRR@K, NDCG@K, Precision@K, Recall@K).
  - End-to-End RAG Triad evaluation metrics (Context Relevance / Precision, Context Recall, Faithfulness / Groundedness).
  - REST endpoint `POST /api/v1/eval/benchmark` returning comprehensive comparative performance reports.
  - Standalone evaluation script `scripts/evaluate_rag.py` that can be run against any vector database implementing `BaseVectorStore`.
  - Sample golden benchmark dataset in `data/eval/sample_eval_dataset.json`.
- **Todo List**:
  - [ ] Create `app/services/evaluation_service.py` with IR metric calculators and RAG Triad scorers.
  - [ ] Create `app/models/eval.py` defining benchmark requests, golden sample schemas, and evaluation reports.
  - [ ] Create `app/api/v1/eval.py` exposing evaluation endpoints.
  - [ ] Create sample benchmark dataset `data/eval/sample_eval_dataset.json`.
  - [ ] Create standalone CLI evaluation runner `scripts/evaluate_rag.py`.
- **Relevant Context**: `app/services/evaluation_service.py`, `app/api/v1/eval.py`, `scripts/evaluate_rag.py`, `data/eval/`
- **Status**: [ ] pending

---

### Sub-Task 6: Packaging, Documentation & Portability Verification
- **Intent**: Provide Dockerfile, dependencies, verification tests, and clear documentation detailing how to run on Podman/Docker and how to reuse the evaluation/reranking modules with other vector stores.
- **Expected Outcomes**:
  - `Dockerfile` and `requirements.txt` / `pyproject.toml` for the FastAPI service.
  - `README.md` with complete architecture guide, Podman/Docker setup, API documentation, RAG Triad evaluation instructions, and vector database swapping guide.
  - End-to-end integration test script `scripts/test_rag_pipeline.py`.
- **Todo List**:
  - [ ] Create `Dockerfile` and `requirements.txt` for the FastAPI service.
  - [ ] Write `README.md` including architecture diagrams, Podman/Docker commands, API reference, RAG Triad evaluation usage, and a guide on plugging in other Vector DBs (Qdrant, Chroma, etc.).
  - [ ] Create an end-to-end pipeline test script `scripts/test_rag_pipeline.py` testing ingestion -> vector search -> reranking -> evaluation run.
- **Relevant Context**: `Dockerfile`, `requirements.txt`, `README.md`, `scripts/test_rag_pipeline.py`
- **Status**: [ ] pending
