import math
from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.services.interfaces import BaseReranker, DocumentChunk


class RerankService(BaseReranker):
    """
    Reranker supporting Cross-Encoders (BAAI/bge-reranker-base, ms-marco, etc.),
    FlashRank, or lightweight term-overlap & vector-calibrated scoring.
    """

    def __init__(self):
        self._provider = settings.RERANKER_PROVIDER.lower()
        self._model_name = settings.RERANKER_MODEL_NAME
        self._model = None
        self._init_model()

    def _init_model(self):
        if self._provider == "cross-encoder":
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder model: {self._model_name}...")
                self._model = CrossEncoder(
                    self._model_name,
                    device=settings.RERANKER_DEVICE,
                    max_length=512,
                )
                logger.info(f"CrossEncoder {self._model_name} loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"CrossEncoder load failed ({e}). Falling back to algorithmic reranker."
                )
                self._model = None
        elif self._provider == "flashrank":
            try:
                from flashrank import Ranker
                logger.info(f"Loading FlashRank model: {self._model_name}...")
                self._model = Ranker(model_name=self._model_name)
            except Exception as e:
                logger.warning(f"FlashRank load failed ({e}). Falling back.")
                self._model = None
        else:
            logger.info("Using lightweight algorithmic reranking backend.")
            self._model = None

    def _fallback_rerank_score(self, query: str, text: str, similarity_score: Optional[float]) -> float:
        """
        Lightweight lexical + similarity fusion scoring when cross-encoder weights are not present.
        Calculates exact word overlap, position bonus, and blends with vector similarity.
        """
        query_terms = set(query.lower().split())
        doc_terms = text.lower().split()

        if not query_terms or not doc_terms:
            return similarity_score or 0.0

        # Term overlap ratio
        overlap = sum(1 for term in query_terms if term in doc_terms) / len(query_terms)

        # Exact substring boost
        substring_boost = 0.2 if query.lower() in text.lower() else 0.0

        # Blend with vector similarity
        sim = similarity_score if similarity_score is not None else 0.5

        # Normalize score into [0, 1]
        raw_score = (0.5 * sim) + (0.35 * overlap) + substring_boost
        return min(1.0, max(0.0, raw_score))

    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[DocumentChunk]:
        """
        Reranks a list of candidate document chunks against the user's query.
        """
        if not chunks:
            return []

        limit = top_n or settings.RERANKER_DEFAULT_TOP_N
        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.RERANKER_SCORE_THRESHOLD
        )

        scored_chunks: List[DocumentChunk] = []

        if self._model is not None and self._provider == "cross-encoder":
            try:
                pairs = [[query, chunk.text] for chunk in chunks]
                scores = self._model.predict(
                    pairs,
                    batch_size=settings.RERANKER_BATCH_SIZE,
                    show_progress_bar=False,
                )
                
                # Apply sigmoid normalization if scores are raw logits
                for chunk, score in zip(chunks, scores):
                    val = float(score)
                    # Sigmoid if unconstrained logit
                    norm_score = 1.0 / (1.0 + math.exp(-val)) if (val > 1.0 or val < 0.0) else val
                    chunk_copy = chunk.model_copy(deep=True)
                    chunk_copy.rerank_score = round(norm_score, 4)
                    scored_chunks.append(chunk_copy)
            except Exception as e:
                logger.error(f"CrossEncoder inference failed ({e}), using fallback.")
                scored_chunks = []

        if not scored_chunks:
            # Fallback algorithmic reranker
            for chunk in chunks:
                score = self._fallback_rerank_score(query, chunk.text, chunk.similarity_score)
                chunk_copy = chunk.model_copy(deep=True)
                chunk_copy.rerank_score = round(score, 4)
                scored_chunks.append(chunk_copy)

        # Sort descending by rerank score
        scored_chunks.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

        # Apply threshold filter
        filtered_chunks = [
            c for c in scored_chunks if (c.rerank_score or 0.0) >= threshold
        ]

        # If strict threshold removed all, retain top 1 to prevent empty context if candidates existed
        if not filtered_chunks and scored_chunks:
            filtered_chunks = scored_chunks[:1]

        return filtered_chunks[:limit]


rerank_service = RerankService()
