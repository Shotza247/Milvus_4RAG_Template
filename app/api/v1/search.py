from fastapi import APIRouter, HTTPException, status
from app.core.config import settings
from app.core.logging import logger
from app.models.rerank import RerankRequest, RerankResponse
from app.models.search import SearchResponse, TwoStageSearchRequest, VectorSearchRequest
from app.services.embedding_service import embedding_service
from app.services.milvus_service import milvus_service
from app.services.rerank_service import rerank_service

router = APIRouter(tags=["Search & Reranking"])


@router.post("/search", response_model=SearchResponse, summary="Vector Similarity Search (with optional reranking)")
async def search_vectors(payload: VectorSearchRequest):
    col_name = payload.collection_name or settings.MILVUS_DEFAULT_COLLECTION

    try:
        # Embed query text
        query_vector = embedding_service.embed_text(payload.query)

        # Stage 1: Coarse retrieval from Milvus
        candidates = await milvus_service.search(
            collection_name=col_name,
            query_vector=query_vector,
            top_k=payload.top_k or 10,
            filter_expr=payload.filter_expr,
        )

        stage = "vector_only"
        final_results = candidates

        # Stage 2 (Optional): Re-ranking
        if payload.rerank and candidates:
            final_results = rerank_service.rerank(
                query=payload.query,
                chunks=candidates,
                top_n=payload.top_n or 5,
                score_threshold=payload.score_threshold,
            )
            stage = "two_stage_reranked"

        return SearchResponse(
            query=payload.query,
            collection_name=col_name,
            total_retrieved=len(final_results),
            retrieval_stage=stage,
            results=final_results,
        )
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.post("/search/two-stage", response_model=SearchResponse, summary="Explicit 2-Stage Retrieval (Milvus Top-K -> Rerank Top-N)")
async def two_stage_retrieval(payload: TwoStageSearchRequest):
    """
    Standardized production RAG pattern:
    1. Retrieve broad candidate pool from Milvus (e.g. top 20-50).
    2. Deep Cross-Encoder scoring to extract top N (e.g. top 5).
    """
    col_name = payload.collection_name or settings.MILVUS_DEFAULT_COLLECTION

    try:
        query_vec = embedding_service.embed_text(payload.query)

        candidates = await milvus_service.search(
            collection_name=col_name,
            query_vector=query_vec,
            top_k=payload.top_k_candidates or settings.RERANKER_TOP_K_CANDIDATES,
            filter_expr=payload.filter_expr,
        )

        reranked = rerank_service.rerank(
            query=payload.query,
            chunks=candidates,
            top_n=payload.top_n_final or settings.RERANKER_DEFAULT_TOP_N,
            score_threshold=payload.rerank_score_threshold,
        )

        return SearchResponse(
            query=payload.query,
            collection_name=col_name,
            total_retrieved=len(reranked),
            retrieval_stage="two_stage_reranked",
            results=reranked,
        )
    except Exception as e:
        logger.error(f"Two-stage retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Two-stage search failed: {str(e)}",
        )


@router.post("/rerank", response_model=RerankResponse, summary="Standalone Reranker for any candidate chunks")
async def standalone_rerank(payload: RerankRequest):
    """
    Standalone reranking utility. Can be called on candidates returned by any vector DB or external source.
    """
    try:
        ranked = rerank_service.rerank(
            query=payload.query,
            chunks=payload.chunks,
            top_n=payload.top_n,
            score_threshold=payload.score_threshold,
        )

        return RerankResponse(
            query=payload.query,
            ranked_chunks=ranked,
            total_candidates=len(payload.chunks),
            top_n=len(ranked),
            reranker_model=settings.RERANKER_MODEL_NAME,
        )
    except Exception as e:
        logger.error(f"Rerank error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reranking failed: {str(e)}",
        )
