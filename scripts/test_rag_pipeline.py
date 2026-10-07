#!/usr/bin/env python3
"""
End-to-End Pipeline Verification Script.
Tests:
1. Document chunking & ingestion.
2. Embedding generation.
3. Coarse vector similarity search.
4. Cross-Encoder reranking precision.
5. IR Metrics & RAG Triad evaluation reporting.
"""

import asyncio
import os
import sys

# Ensure local app path is resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models.eval import GoldenSample
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import embedding_service
from app.services.evaluation_service import evaluation_service
from app.services.interfaces import BaseVectorStore, DocumentChunk
from app.services.rerank_service import rerank_service


class InMemoryVectorStore(BaseVectorStore):
    """
    Lightweight in-memory vector store mock to verify pipeline without external Milvus instance running.
    """

    def __init__(self):
        self.store = {}

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def is_healthy(self):
        return {"status": "healthy", "engine": "in-memory-test"}

    async def create_collection(self, collection_name: str, dimension: int = 384, **kwargs) -> bool:
        self.store[collection_name] = []
        return True

    async def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.store

    async def drop_collection(self, collection_name: str) -> bool:
        self.store.pop(collection_name, None)
        return True

    async def list_collections(self):
        return list(self.store.keys())

    async def get_collection_stats(self, collection_name: str):
        return {"num_entities": len(self.store.get(collection_name, []))}

    async def insert_chunks(self, collection_name: str, chunks):
        if collection_name not in self.store:
            self.store[collection_name] = []
        self.store[collection_name].extend(chunks)
        return len(chunks)

    async def search(self, collection_name: str, query_vector, top_k=10, **kwargs):
        chunks = self.store.get(collection_name, [])
        # Cosine similarity simulation
        scored = []
        for c in chunks:
            if c.embedding:
                # Dot product as cosine similarity (vectors are unit length)
                sim = sum(a * b for a, b in zip(query_vector, c.embedding))
            else:
                sim = 0.5
            chunk_copy = c.model_copy(deep=True)
            chunk_copy.similarity_score = round(float(sim), 4)
            scored.append(chunk_copy)
        scored.sort(key=lambda x: x.similarity_score or 0.0, reverse=True)
        return scored[:top_k]


async def run_integration_test():
    print("=" * 80)
    print(">>> RUNNING MILVUS RAG PIPELINE & EVALUATION INTEGRATION TEST")
    print("=" * 80)

    # 1. Test Ingestion Documents
    print("\n[Step 1] Ingesting & Chunking Knowledge Base Documents...")
    docs = [
        {
            "doc_id": "doc_milvus_architecture",
            "text": (
                "Milvus is an open-source cloud-native vector database. It separates compute and storage, "
                "using MinIO for persistent segment storage and etcd for metadata. It scales horizontally "
                "by dividing workloads across QueryNodes, DataNodes, and IndexNodes."
            ),
        },
        {
            "doc_id": "doc_indexing_strategies",
            "text": (
                "Indexing in vector databases: HNSW (Hierarchical Navigable Small World) provides high "
                "recall and fast query speed at the expense of higher memory footprint. IVF_FLAT partitions "
                "vector space into Voronoi cells with lower memory usage."
            ),
        },
        {
            "doc_id": "doc_reranking_concepts",
            "text": (
                "Reranking in RAG: Bi-encoder vector search is efficient for coarse candidate retrieval. "
                "Cross-Encoder rerankers perform full self-attention across query and document pairs, "
                "substantially improving MRR and NDCG ranking accuracy."
            ),
        },
        {
            "doc_id": "doc_rag_triad",
            "text": (
                "The RAG Triad framework evaluates three critical dimensions: Context Relevance (signal-to-noise "
                "ratio in retrieval), Context Recall (ground-truth fact coverage), and Faithfulness (groundedness "
                "without LLM hallucinations)."
            ),
        },
    ]

    all_chunks = []
    for d in docs:
        chunks = DocumentProcessor.chunk_text(
            text=d["text"],
            doc_id=d["doc_id"],
            chunk_size=300,
            chunk_overlap=20,
        )
        texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_batch(texts)
        for c, emb in zip(chunks, embeddings):
            c.embedding = emb
        all_chunks.extend(chunks)

    print(f"  [+] Processed {len(docs)} documents into {len(all_chunks)} embedded chunks.")

    # 2. Insert into Vector Store
    vector_store = InMemoryVectorStore()
    col_name = "test_rag_collection"
    await vector_store.create_collection(col_name)
    await vector_store.insert_chunks(col_name, all_chunks)
    print(f"  [+] Seeded chunks into vector store '{col_name}'.")

    # 3. Test 2-Stage Retrieval & Reranking
    query = "Why should we use Cross-Encoder rerankers after vector search?"
    print(f"\n[Step 2] Executing Query: '{query}'")

    q_vec = embedding_service.embed_text(query)
    stage1_candidates = await vector_store.search(col_name, q_vec, top_k=4)
    print(f"  [+] Stage 1 Vector Search returned {len(stage1_candidates)} candidates.")
    for i, c in enumerate(stage1_candidates, 1):
        print(f"    {i}. [Doc: {c.doc_id}] Sim Score: {c.similarity_score}")

    stage2_reranked = rerank_service.rerank(query, stage1_candidates, top_n=2)
    print(f"\n  [+] Stage 2 Cross-Encoder Reranked Top 2 Results:")
    for i, c in enumerate(stage2_reranked, 1):
        print(f"    {i}. [Doc: {c.doc_id}] Rerank Score: {c.rerank_score} | Text: {c.text[:65]}...")

    # Validate correct top document
    assert stage2_reranked[0].doc_id == "doc_reranking_concepts", "Top ranked document mismatch!"
    print("\n  [PASS] Validation Passed: Target document correctly ranked at #1 position!")

    # 4. Test RAG Triad & IR Evaluation
    print("\n[Step 3] Running Full Benchmark Evaluation & RAG Triad Assessment...")
    golden_samples = [
        GoldenSample(
            sample_id="test_01",
            query="Why should we use Cross-Encoder rerankers after vector search?",
            ground_truth_doc_ids=["doc_reranking_concepts"],
            ground_truth_answer="Cross-encoders perform full self-attention to improve MRR and ranking precision.",
            relevant_keywords=["cross-encoder", "self-attention", "mrr", "ndcg"],
        ),
        GoldenSample(
            sample_id="test_02",
            query="What are the components of RAG Triad?",
            ground_truth_doc_ids=["doc_rag_triad"],
            ground_truth_answer="Context Relevance, Context Recall, and Faithfulness.",
            relevant_keywords=["context relevance", "context recall", "faithfulness", "triad"],
        ),
    ]

    report = await evaluation_service.evaluate_retrieval(
        benchmark_samples=[s.model_dump() for s in golden_samples],
        vector_store=vector_store,
        embedding_service=embedding_service,
        reranker=rerank_service,
        top_k=4,
        top_n=2,
    )

    print("\n" + "=" * 80)
    print(">>> PIPELINE VERIFICATION SUCCESSFUL")
    print("=" * 80)
    print(f"MRR Gain: +{report['accuracy_gain_mrr_percent']}%")
    print(f"NDCG Gain: +{report['accuracy_gain_ndcg_percent']}%")
    print(f"Summary: {report['summary']}")


if __name__ == "__main__":
    asyncio.run(run_integration_test())
