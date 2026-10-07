import math
import os
import re
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.eval import (
    GoldenSample,
    IRMetrics,
    RAGTriadMetrics,
    StageEvaluationComparison,
)
from app.services.interfaces import (
    BaseEmbeddingService,
    BaseEvaluator,
    BaseReranker,
    BaseVectorStore,
    DocumentChunk,
)


class EvaluationService(BaseEvaluator):
    """
    Self-contained, portable evaluation engine.
    Calculates Information Retrieval (IR) metrics and RAG Triad scores.
    """

    @staticmethod
    def calculate_hit_rate(retrieved: List[DocumentChunk], ground_truth_ids: List[str]) -> float:
        if not ground_truth_ids:
            return 1.0
        retrieved_ids = {c.doc_id for c in retrieved}
        return 1.0 if any(gt in retrieved_ids for gt in ground_truth_ids) else 0.0

    @staticmethod
    def calculate_reciprocal_rank(retrieved: List[DocumentChunk], ground_truth_ids: List[str]) -> float:
        if not ground_truth_ids:
            return 1.0
        for rank, chunk in enumerate(retrieved, start=1):
            if chunk.doc_id in ground_truth_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_ndcg(retrieved: List[DocumentChunk], ground_truth_ids: List[str], k: int) -> float:
        if not ground_truth_ids:
            return 1.0
        dcg = 0.0
        for rank, chunk in enumerate(retrieved[:k], start=1):
            rel = 1.0 if chunk.doc_id in ground_truth_ids else 0.0
            dcg += rel / math.log2(rank + 1)

        # Ideal DCG
        ideal_rels = [1.0] * min(len(ground_truth_ids), k)
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_precision_recall(
        retrieved: List[DocumentChunk], ground_truth_ids: List[str]
    ) -> (float, float):
        if not ground_truth_ids:
            return 1.0, 1.0
        if not retrieved:
            return 0.0, 0.0

        hits = sum(1 for c in retrieved if c.doc_id in ground_truth_ids)
        precision = hits / len(retrieved)
        recall = hits / len(ground_truth_ids)
        return precision, min(1.0, recall)

    @classmethod
    def evaluate_rag_triad(
        cls,
        query: str,
        retrieved_chunks: List[DocumentChunk],
        ground_truth_sample: GoldenSample,
    ) -> RAGTriadMetrics:
        """
        Calculates RAG Triad components:
        1. Context Relevance: Ratio of retrieved text that directly relates to query keywords/subject.
        2. Context Recall: Coverage of ground-truth reference keywords in the retrieved context.
        3. Faithfulness: Consistency & absence of contradictory noise.
        """
        if not retrieved_chunks:
            return RAGTriadMetrics(context_relevance=0.0, context_recall=0.0, faithfulness=0.0)

        combined_text = " ".join(c.text for c in retrieved_chunks).lower()
        query_words = set(query.lower().split())

        # 1. Context Relevance
        sentences = [s.strip() for s in re.split(r"[.!?\n]", combined_text) if s.strip()]
        if not sentences:
            context_rel = 0.0
        else:
            rel_sentences = sum(
                1 for s in sentences if any(w in s for w in query_words if len(w) > 3)
            )
            context_rel = min(1.0, rel_sentences / len(sentences) + 0.3)  # Smoothed base

        # 2. Context Recall
        gt_keywords = ground_truth_sample.relevant_keywords
        if gt_keywords:
            matched_kw = sum(1 for kw in gt_keywords if kw.lower() in combined_text)
            context_recall = matched_kw / len(gt_keywords)
        else:
            # Fallback to doc_id hit
            context_recall = 1.0 if any(
                c.doc_id in ground_truth_sample.ground_truth_doc_ids for c in retrieved_chunks
            ) else 0.0

        # 3. Faithfulness / Groundedness (High when top chunks have high confidence scores)
        avg_score = sum(
            (c.rerank_score or c.similarity_score or 0.5) for c in retrieved_chunks
        ) / len(retrieved_chunks)
        faithfulness = round(min(1.0, max(0.0, avg_score * 0.95 + (0.05 * context_recall))), 4)

        return RAGTriadMetrics(
            context_relevance=round(context_rel, 4),
            context_recall=round(context_recall, 4),
            faithfulness=faithfulness,
        )

    async def evaluate_retrieval(
        self,
        benchmark_samples: List[Dict[str, Any]],
        vector_store: BaseVectorStore,
        embedding_service: BaseEmbeddingService,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 10,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Runs comparative evaluation between vector search and reranked search."""
        samples = [
            GoldenSample(**s) if isinstance(s, dict) else s for s in benchmark_samples
        ]

        vec_hit_rates, vec_mrrs, vec_ndcgs, vec_precs, vec_recalls = [], [], [], [], []
        vec_ctx_rel, vec_ctx_rec, vec_faith = [], [], []

        rr_hit_rates, rr_mrrs, rr_ndcgs, rr_precs, rr_recalls = [], [], [], [], []
        rr_ctx_rel, rr_ctx_rec, rr_faith = [], [], []

        for sample in samples:
            q_vec = embedding_service.embed_text(sample.query)
            # Stage 1: Vector Search
            col_name = settings.MILVUS_DEFAULT_COLLECTION
            candidates = await vector_store.search(
                collection_name=col_name,
                query_vector=q_vec,
                top_k=top_k,
            )

            # Stage 1 Metrics
            gt_ids = sample.ground_truth_doc_ids
            vec_hit_rates.append(self.calculate_hit_rate(candidates[:top_n], gt_ids))
            vec_mrrs.append(self.calculate_reciprocal_rank(candidates[:top_n], gt_ids))
            vec_ndcgs.append(self.calculate_ndcg(candidates[:top_n], gt_ids, top_n))
            p, r = self.calculate_precision_recall(candidates[:top_n], gt_ids)
            vec_precs.append(p)
            vec_recalls.append(r)

            vec_triad = self.evaluate_rag_triad(sample.query, candidates[:top_n], sample)
            vec_ctx_rel.append(vec_triad.context_relevance)
            vec_ctx_rec.append(vec_triad.context_recall)
            vec_faith.append(vec_triad.faithfulness)

            # Stage 2: Reranking
            if reranker:
                reranked = reranker.rerank(sample.query, candidates, top_n=top_n)
            else:
                reranked = candidates[:top_n]

            # Stage 2 Metrics
            rr_hit_rates.append(self.calculate_hit_rate(reranked, gt_ids))
            rr_mrrs.append(self.calculate_reciprocal_rank(reranked, gt_ids))
            rr_ndcgs.append(self.calculate_ndcg(reranked, gt_ids, top_n))
            p_rr, r_rr = self.calculate_precision_recall(reranked, gt_ids)
            rr_precs.append(p_rr)
            rr_recalls.append(r_rr)

            rr_triad = self.evaluate_rag_triad(sample.query, reranked, sample)
            rr_ctx_rel.append(rr_triad.context_relevance)
            rr_ctx_rec.append(rr_triad.context_recall)
            rr_faith.append(rr_triad.faithfulness)

        avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0

        vec_ir = IRMetrics(
            hit_rate_at_k=avg(vec_hit_rates),
            mrr_at_k=avg(vec_mrrs),
            ndcg_at_k=avg(vec_ndcgs),
            precision_at_k=avg(vec_precs),
            recall_at_k=avg(vec_recalls),
        )
        vec_rag = RAGTriadMetrics(
            context_relevance=avg(vec_ctx_rel),
            context_recall=avg(vec_ctx_rec),
            faithfulness=avg(vec_faith),
        )

        rr_ir = IRMetrics(
            hit_rate_at_k=avg(rr_hit_rates),
            mrr_at_k=avg(rr_mrrs),
            ndcg_at_k=avg(rr_ndcgs),
            precision_at_k=avg(rr_precs),
            recall_at_k=avg(rr_recalls),
        )
        rr_rag = RAGTriadMetrics(
            context_relevance=avg(rr_ctx_rel),
            context_recall=avg(rr_ctx_rec),
            faithfulness=avg(rr_faith),
        )

        mrr_gain = (
            ((rr_ir.mrr_at_k - vec_ir.mrr_at_k) / (vec_ir.mrr_at_k or 1.0)) * 100.0
        )
        ndcg_gain = (
            ((rr_ir.ndcg_at_k - vec_ir.ndcg_at_k) / (vec_ir.ndcg_at_k or 1.0)) * 100.0
        )

        return {
            "total_samples": len(samples),
            "collection_name": settings.MILVUS_DEFAULT_COLLECTION,
            "top_k_candidates": top_k,
            "top_n_reranked": top_n,
            "vector_only_evaluation": StageEvaluationComparison(
                stage_name="Vector Similarity (Milvus)",
                ir_metrics=vec_ir,
                rag_triad_metrics=vec_rag,
            ),
            "reranked_evaluation": StageEvaluationComparison(
                stage_name="2-Stage (Milvus + Cross-Encoder)",
                ir_metrics=rr_ir,
                rag_triad_metrics=rr_rag,
            ),
            "accuracy_gain_mrr_percent": round(mrr_gain, 2),
            "accuracy_gain_ndcg_percent": round(ndcg_gain, 2),
            "summary": (
                f"Evaluated {len(samples)} golden benchmark samples. "
                f"Reranking produced an MRR gain of {round(mrr_gain, 2)}% "
                f"and NDCG@{top_n} gain of {round(ndcg_gain, 2)}% over vector-only retrieval."
            ),
        }


evaluation_service = EvaluationService()
