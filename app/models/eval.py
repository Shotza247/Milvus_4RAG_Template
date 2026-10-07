from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GoldenSample(BaseModel):
    sample_id: str
    query: str
    ground_truth_doc_ids: List[str] = Field(
        default_factory=list,
        description="Expected document IDs that contain the true answer",
    )
    ground_truth_answer: Optional[str] = Field(
        None, description="Golden expected answer string"
    )
    relevant_keywords: List[str] = Field(
        default_factory=list,
        description="Key terms that must be present in relevant retrieved context",
    )


class IRMetrics(BaseModel):
    hit_rate_at_k: float = Field(..., description="Hit Rate @ K")
    mrr_at_k: float = Field(..., description="Mean Reciprocal Rank @ K")
    ndcg_at_k: float = Field(..., description="Normalized Discounted Cumulative Gain @ K")
    precision_at_k: float = Field(..., description="Precision @ K")
    recall_at_k: float = Field(..., description="Recall @ K")


class RAGTriadMetrics(BaseModel):
    context_relevance: float = Field(
        ...,
        description="Precision of retrieved context (ratio of relevant sentences vs noise)",
    )
    context_recall: float = Field(
        ...,
        description="Recall of ground-truth facts present in retrieved chunks",
    )
    faithfulness: float = Field(
        ...,
        description="Groundedness / Hallucination resistance metric",
    )


class BenchmarkEvaluationRequest(BaseModel):
    collection_name: Optional[str] = None
    samples: Optional[List[GoldenSample]] = None
    use_sample_dataset_file: bool = False
    top_k: int = 10
    top_n: int = 5
    score_threshold: Optional[float] = 0.35


class GenerateSyntheticDatasetRequest(BaseModel):
    collection_name: Optional[str] = None
    text: Optional[str] = None
    doc_id: Optional[str] = None
    num_samples: int = 5
    use_llm: bool = False
    save_as_benchmark: bool = True


class GenerateSyntheticDatasetResponse(BaseModel):
    success: bool
    num_generated: int
    collection_name: str
    samples: List[GoldenSample]
    message: str


class StageEvaluationComparison(BaseModel):
    stage_name: str
    ir_metrics: IRMetrics
    rag_triad_metrics: RAGTriadMetrics


class BenchmarkEvaluationReport(BaseModel):
    total_samples: int
    collection_name: str
    top_k_candidates: int
    top_n_reranked: int
    vector_only_evaluation: StageEvaluationComparison
    reranked_evaluation: StageEvaluationComparison
    accuracy_gain_mrr_percent: float
    accuracy_gain_ndcg_percent: float
    summary: str
