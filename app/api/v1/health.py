from fastapi import APIRouter
from app.core.milvus import MilvusConnectionManager
from app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health & Status"])


@router.get("", summary="Liveness & Readiness probe")
async def health_check():
    """
    Checks the status of the FastAPI service and the Milvus Vector DB connection.
    """
    milvus_health = MilvusConnectionManager.health_check()
    
    overall_status = "healthy" if milvus_health.get("connected") else "degraded"

    return {
        "status": overall_status,
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "reranker_provider": settings.RERANKER_PROVIDER,
        "reranker_model": settings.RERANKER_MODEL_NAME,
        "milvus": milvus_health,
    }
