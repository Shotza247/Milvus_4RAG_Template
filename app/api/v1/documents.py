import os
import shutil
import uuid
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from app.core.config import settings
from app.core.logging import logger
from app.models.document import IngestBatchRequest, IngestResponse, IngestTextRequest
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import embedding_service
from app.services.milvus_service import milvus_service

router = APIRouter(prefix="/documents", tags=["Documents & Ingestion"])

UPLOAD_DIR = "data/raw"


@router.post("/ingest-text", response_model=IngestResponse, summary="Ingest raw text string")
async def ingest_text(payload: IngestTextRequest):
    col_name = payload.collection_name or settings.MILVUS_DEFAULT_COLLECTION
    doc_id = payload.doc_id or str(uuid.uuid4())

    try:
        chunks = DocumentProcessor.chunk_text(
            text=payload.text,
            doc_id=doc_id,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            metadata=payload.metadata,
        )

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text produced 0 chunks",
            )

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        # Insert to Milvus
        count = await milvus_service.insert_chunks(col_name, chunks)

        return IngestResponse(
            success=True,
            collection_name=col_name,
            doc_id=doc_id,
            chunks_created=count,
            message=f"Successfully ingested {count} chunks into '{col_name}'",
        )
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post("/upload", response_model=IngestResponse, summary="Upload & ingest file (PDF, TXT, MD, DOCX)")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    doc_id: Optional[str] = Form(None),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
):
    col_name = collection_name or settings.MILVUS_DEFAULT_COLLECTION
    d_id = doc_id or str(uuid.uuid4())
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_path = os.path.join(UPLOAD_DIR, f"{d_id}_{file.filename}")
    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = DocumentProcessor.extract_text_from_file(saved_path)
        if not extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not extract text from file {file.filename}",
            )

        meta = {
            "source_file": file.filename,
            "content_type": file.content_type,
            "file_size": os.path.getsize(saved_path),
        }

        chunks = DocumentProcessor.chunk_text(
            text=extracted_text,
            doc_id=d_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=meta,
        )

        texts = [c.text for c in chunks]
        embeddings = embedding_service.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        count = await milvus_service.insert_chunks(col_name, chunks)

        return IngestResponse(
            success=True,
            collection_name=col_name,
            doc_id=d_id,
            chunks_created=count,
            message=f"File '{file.filename}' processed: {count} chunks added to '{col_name}'",
        )
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )
