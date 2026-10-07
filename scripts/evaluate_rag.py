#!/usr/bin/env python3
"""
Standalone CLI Benchmark Evaluation Runner.
Demonstrates portability: works directly with BaseVectorStore, BaseEmbeddingService,
BaseReranker, and BaseEvaluator.
"""

import asyncio
import json
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.evaluation_service import evaluation_service
from app.services.milvus_service import milvus_service
from app.services.rerank_service import rerank_service


async def main():
    print("=" * 80)
    print(">>> MILVUS RAG PIPELINE - STANDALONE EVALUATION RUNNER")
    print("=" * 80)

    dataset_path = os.path.join("data", "eval", "sample_eval_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    samples = dataset.get("samples", [])
    print(f"Loaded {len(samples)} golden benchmark test queries.\n")

    # Connect to vector store
    print("1. Connecting to Vector Store...")
    await milvus_service.connect()

    print("2. Running Comparative Evaluation (Vector Search vs Cross-Encoder Rerank)...")
    results = await evaluation_service.evaluate_retrieval(
        benchmark_samples=samples,
        vector_store=milvus_service,
        embedding_service=embedding_service,
        reranker=rerank_service,
        top_k=settings.RERANKER_TOP_K_CANDIDATES,
        top_n=settings.RERANKER_DEFAULT_TOP_N,
    )

    print("\n" + "=" * 80)
    print(">>> EVALUATION RESULTS & RAG TRIAD COMPARISON")
    print("=" * 80)

    vec_eval = results["vector_only_evaluation"]
    rr_eval = results["reranked_evaluation"]

    print(f"\n[STAGE 1: {vec_eval.stage_name}]")
    print(f"  * Hit Rate @ {results['top_n_reranked']}:    {vec_eval.ir_metrics.hit_rate_at_k:.4f}")
    print(f"  * MRR @ {results['top_n_reranked']}:         {vec_eval.ir_metrics.mrr_at_k:.4f}")
    print(f"  * NDCG @ {results['top_n_reranked']}:        {vec_eval.ir_metrics.ndcg_at_k:.4f}")
    print(f"  * Context Relevance: {vec_eval.rag_triad_metrics.context_relevance:.4f}")
    print(f"  * Context Recall:    {vec_eval.rag_triad_metrics.context_recall:.4f}")
    print(f"  * Faithfulness:      {vec_eval.rag_triad_metrics.faithfulness:.4f}")

    print(f"\n[STAGE 2: {rr_eval.stage_name}]")
    print(f"  * Hit Rate @ {results['top_n_reranked']}:    {rr_eval.ir_metrics.hit_rate_at_k:.4f}")
    print(f"  * MRR @ {results['top_n_reranked']}:         {rr_eval.ir_metrics.mrr_at_k:.4f}")
    print(f"  * NDCG @ {results['top_n_reranked']}:        {rr_eval.ir_metrics.ndcg_at_k:.4f}")
    print(f"  * Context Relevance: {rr_eval.rag_triad_metrics.context_relevance:.4f}")
    print(f"  * Context Recall:    {rr_eval.rag_triad_metrics.context_recall:.4f}")
    print(f"  * Faithfulness:      {rr_eval.rag_triad_metrics.faithfulness:.4f}")

    print("\n[PERFORMANCE GAIN]")
    print(f"  [+] MRR Improvement:  +{results['accuracy_gain_mrr_percent']}%")
    print(f"  [+] NDCG Improvement: +{results['accuracy_gain_ndcg_percent']}%")
    print(f"\nSummary: {results['summary']}\n")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
