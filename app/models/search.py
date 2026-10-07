import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.services.interfaces import DocumentChunk


class VectorSearchRequest(BaseModel):
    collection_name: Optional[str] = None
    query: str
    top_k: Optional[int] = 10
    filter_expr: Optional[str] = None
    metric_type: Optional[str] = "COSINE"
    rerank: Optional[bool] = False
    top_n: Optional[int] = 5
    score_threshold: Optional[float] = None


class TwoStageSearchRequest(BaseModel):
    collection_name: Optional[str] = None
    query: str
    top_k_candidates: Optional[int] = 20
    top_n_final: Optional[int] = 5
    filter_expr: Optional[str] = None
    rerank_score_threshold: Optional[float] = 0.35


class SearchResponse(BaseModel):
    query: str
    collection_name: str
    total_retrieved: int
    retrieval_stage: str  # "vector_only" or "two_stage_reranked"
    results: List[DocumentChunk]
