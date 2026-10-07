from fastapi import APIRouter, HTTPException, status
from app.models.collection import (
    CollectionInfo,
    CollectionListResponse,
    CreateCollectionRequest,
)
from app.services.milvus_service import milvus_service

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.get("", response_model=CollectionListResponse, summary="List all collections")
async def list_collections():
    cols = await milvus_service.list_collections()
    return CollectionListResponse(collections=cols, total=len(cols))


@router.post("", summary="Create a new vector collection")
async def create_collection(payload: CreateCollectionRequest):
    try:
        success = await milvus_service.create_collection(
            collection_name=payload.collection_name,
            dimension=payload.dimension or 384,
            metric_type=payload.metric_type or "COSINE",
        )
        return {
            "success": success,
            "collection_name": payload.collection_name,
            "dimension": payload.dimension,
            "metric_type": payload.metric_type,
            "message": f"Collection '{payload.collection_name}' ready",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not create collection: {str(e)}",
        )


@router.get("/{collection_name}", summary="Get collection statistics & schema")
async def get_collection_info(collection_name: str):
    exists = await milvus_service.has_collection(collection_name)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found",
        )
    return await milvus_service.get_collection_stats(collection_name)


@router.delete("/{collection_name}", summary="Drop a collection")
async def drop_collection(collection_name: str):
    exists = await milvus_service.has_collection(collection_name)
    if not exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found",
        )
    success = await milvus_service.drop_collection(collection_name)
    return {
        "success": success,
        "collection_name": collection_name,
        "message": f"Collection '{collection_name}' dropped successfully",
    }
