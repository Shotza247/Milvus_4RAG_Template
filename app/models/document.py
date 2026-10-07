from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source: str
    filename: Optional[str] = None
    file_type: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    chunk_index: int
    text: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    collection_name: Optional[str] = None
    doc_id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class IngestBatchRequest(BaseModel):
    collection_name: Optional[str] = None
    documents: List[IngestTextRequest]


class IngestResponse(BaseModel):
    success: bool
    collection_name: str
    doc_id: str
    chunks_created: int
    message: str
