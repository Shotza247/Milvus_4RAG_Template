from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Normalized chunk model representing a piece of document text and metadata."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResult(BaseModel):
    """Standardized search result returning matches from any vector database."""
    chunks: List[DocumentChunk]
    total_found: int
    query: str
    collection_name: str
    retrieval_stage: str = "vector"  # "vector" or "reranked"


class BaseVectorStore(ABC):
    """
    Abstract Vector Database interface.
    Enables swapping Milvus with Qdrant, Chroma, PgVector, Weaviate, etc.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close vector database connection."""
        pass

    @abstractmethod
    async def is_healthy(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        pass

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "COSINE",
        auto_id: bool = True,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a collection/index."""
        pass

    @abstractmethod
    async def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    async def drop_collection(self, collection_name: str) -> bool:
        """Drop/delete collection."""
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collection names."""
        pass

    @abstractmethod
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection entity count and schema metadata."""
        pass

    @abstractmethod
    async def insert_chunks(
        self, collection_name: str, chunks: List[DocumentChunk]
    ) -> int:
        """Insert embedded document chunks into the collection."""
        pass

    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Execute vector similarity search."""
        pass


class BaseEmbeddingService(ABC):
    """
    Abstract embedding generation interface.
    Enables swapping SentenceTransformers, FastEmbed, OpenAI, Watsonx, etc.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return vector dimension."""
        pass


class BaseReranker(ABC):
    """
    Abstract reranker interface.
    Enables swapping Cross-Encoders, FlashRank, Cohere, BGE, etc.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[DocumentChunk]:
        """Rerank a list of candidate chunks against the query."""
        pass


class BaseEvaluator(ABC):
    """
    Abstract evaluation interface for IR and RAG Triad benchmarking.
    """

    @abstractmethod
    async def evaluate_retrieval(
        self,
        benchmark_samples: List[Dict[str, Any]],
        vector_store: BaseVectorStore,
        embedding_service: BaseEmbeddingService,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 10,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Run evaluation benchmark across test queries."""
        pass
