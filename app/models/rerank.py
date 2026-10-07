from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.services.interfaces import DocumentChunk


class RerankRequest(BaseModel):
    query: str
    chunks: List[DocumentChunk]
    top_n: Optional[int] = 5
    score_threshold: Optional[float] = None


class RerankResponse(BaseModel):
    query: str
    ranked_chunks: List[DocumentChunk]
    total_candidates: int
    top_n: int
    reranker_model: str
