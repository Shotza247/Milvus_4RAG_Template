from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.collections import router as collections_router
from app.api.v1.documents import router as documents_router
from app.api.v1.eval import router as eval_router
from app.api.v1.health import router as health_router
from app.api.v1.search import router as search_router
from app.core.config import settings
from app.core.logging import logger
from app.core.milvus import MilvusConnectionManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} in {settings.APP_ENV} mode...")
    MilvusConnectionManager.connect()
    yield
    logger.info(f"Shutting down {settings.APP_NAME}...")
    MilvusConnectionManager.disconnect()


app = FastAPI(
    title="Milvus RAG & Evaluation Engine API",
    description="Reusable RAG Service with Milvus, Cross-Encoder Reranking, and RAG Triad Evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(collections_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(search_router, prefix="/api/v1")
app.include_router(eval_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.APP_NAME,
        "status": "online",
        "documentation": "/docs",
        "health": "/api/v1/health",
        "version": "1.0.0",
    }
