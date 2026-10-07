from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    collection_name: str
    dimension: Optional[int] = 384
    metric_type: Optional[str] = "COSINE"  # "COSINE", "L2", "IP"
    index_type: Optional[str] = "HNSW"     # "HNSW", "IVF_FLAT", "AUTOINDEX"
    description: Optional[str] = "RAG document vector collection"


class CollectionInfo(BaseModel):
    collection_name: str
    entities_count: int
    dimension: Optional[int] = None
    metric_type: Optional[str] = None
    index_type: Optional[str] = None
    loaded: bool = False
    description: Optional[str] = None


class CollectionListResponse(BaseModel):
    collections: List[str]
    total: int
