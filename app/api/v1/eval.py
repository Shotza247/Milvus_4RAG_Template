import json
import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from app.core.config import settings
from app.core.logging import logger
from app.models.eval import (
    BenchmarkEvaluationReport,
    BenchmarkEvaluationRequest,
    GenerateSyntheticDatasetRequest,
    GenerateSyntheticDatasetResponse,
    GoldenSample,
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import embedding_service
from app.services.evaluation_service import evaluation_service
from app.services.milvus_service import milvus_service
from app.services.rerank_service import rerank_service
from app.services.synthetic_generator import synthetic_generator

router = APIRouter(prefix="/eval", tags=["Evaluation & Benchmarking"])

SAMPLE_EVAL_FILE = "data/eval/sample_eval_dataset.json"


@router.get("/samples", summary="Get bundled benchmark evaluation samples")
async def get_eval_samples():
    if os.path.exists(SAMPLE_EVAL_FILE):
        with open(SAMPLE_EVAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"samples": []}


@router.post(
    "/benchmark",
    response_model=BenchmarkEvaluationReport,
    summary="Run End-to-End IR & RAG Triad benchmark evaluation",
)
async def run_benchmark(payload: BenchmarkEvaluationRequest):
    """
    Executes an automated benchmark against the Milvus vector collection:
    1. Runs Vector-Only search (Stage 1).
    2. Runs 2-Stage Retrieval + Reranking (Stage 2).
    3. Computes IR Metrics (Hit@K, MRR@K, NDCG@K, Precision, Recall).
    4. Computes RAG Triad (Context Relevance, Context Recall, Faithfulness).
    5. Returns comparative accuracy report.
    """
    samples: List[GoldenSample] = []

    if payload.samples:
        samples = payload.samples
    elif payload.use_sample_dataset_file or not payload.samples:
        if os.path.exists(SAMPLE_EVAL_FILE):
            with open(SAMPLE_EVAL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = [GoldenSample(**s) for s in data.get("samples", [])]
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark dataset file '{SAMPLE_EVAL_FILE}' not found. Provide samples in payload.",
            )

    if not samples:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No benchmark samples provided for evaluation.",
        )

    try:
        results = await evaluation_service.evaluate_retrieval(
            benchmark_samples=[s.model_dump() for s in samples],
            vector_store=milvus_service,
            embedding_service=embedding_service,
            reranker=rerank_service,
            top_k=payload.top_k,
            top_n=payload.top_n,
        )
        return BenchmarkEvaluationReport(**results)
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.post(
    "/generate-synthetic",
    response_model=GenerateSyntheticDatasetResponse,
    summary="Generate realistic benchmark test set directly from document text",
)
async def generate_synthetic_benchmark(payload: GenerateSyntheticDatasetRequest):
    """
    Ragas / TruLens style synthetic benchmark generator.
    Extracts chunks from provided text or document, generating (Question, Ground-Truth Answer, Keywords) pairs.
    """
    col_name = payload.collection_name or settings.MILVUS_DEFAULT_COLLECTION
    doc_id = payload.doc_id or "uploaded_doc"

    if not payload.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text content is required to generate synthetic benchmark samples.",
        )

    try:
        chunks = DocumentProcessor.chunk_text(
            text=payload.text,
            doc_id=doc_id,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        samples = await synthetic_generator.generate_from_chunks(
            chunks=chunks,
            num_samples=payload.num_samples,
            use_llm=payload.use_llm,
        )

        if payload.save_as_benchmark and samples:
            os.makedirs("data/eval", exist_ok=True)
            with open(SAMPLE_EVAL_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_name": f"Synthetic Benchmarks for {doc_id}",
                        "version": "1.0",
                        "samples": [s.model_dump() for s in samples],
                    },
                    f,
                    indent=2,
                )

        return GenerateSyntheticDatasetResponse(
            success=True,
            num_generated=len(samples),
            collection_name=col_name,
            samples=samples,
            message=f"Generated {len(samples)} synthetic evaluation queries for '{doc_id}'",
        )
    except Exception as e:
        logger.error(f"Synthetic generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate synthetic dataset: {str(e)}",
        )
