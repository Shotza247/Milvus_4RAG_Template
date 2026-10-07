import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "milvus-rag-api"
    APP_ENV: str = "development"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    APP_DEBUG: bool = Field(default=True, validation_alias="APP_DEBUG")
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["*"]

    # Milvus Standalone Settings
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_SECURE: bool = False
    MILVUS_DEFAULT_COLLECTION: str = "rag_documents"

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "sentence-transformers"  # "sentence-transformers", "fastembed", "dummy"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    # Reranker Settings
    RERANKER_PROVIDER: str = "cross-encoder"  # "cross-encoder", "flashrank", "dummy"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_TOP_K_CANDIDATES: int = 20
    RERANKER_DEFAULT_TOP_N: int = 5
    RERANKER_SCORE_THRESHOLD: float = 0.35

    # Document Processing Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    CHUNK_STRATEGY: str = "recursive"

    # Evaluation Settings
    EVAL_DEFAULT_K: int = 5
    EVAL_SIMILARITY_METRIC: str = "cosine"
    OPENAI_API_KEY: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
