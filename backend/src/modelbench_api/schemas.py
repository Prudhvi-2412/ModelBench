from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RetrievalStrategy = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerationParameters(BaseModel):
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    top_p: float = Field(default=0.95, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


class GenerateRequest(GenerationParameters):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=100_000)
    model: str | None = None
    stream: bool = False


class ChatRequest(GenerationParameters):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    stream: bool = False


class GenerateResponse(BaseModel):
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    tokens_per_second: float
    request_id: str


class ModelInfo(BaseModel):
    name: str
    model_id: str
    provider: str
    loaded: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    provider: str
    model: str


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str | list[str]
    model: str | None = None


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int
    batch_size: int
    latency_ms: float
    embeddings_per_second: float
    request_id: str


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    embedding_model: str
    vector_store: str
    request_id: str


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None
    strategy: RetrievalStrategy = "dense"
    embedding_model: str | None = None
    reranker_model: str | None = None


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    page_number: int | None = None
    text: str
    score: float


class RetrievalResponse(BaseModel):
    query: str
    strategy: RetrievalStrategy
    embedding_model: str
    results: list[RetrievedChunk]
    latency_ms: float
    request_id: str


class RetrievalStrategyResult(BaseModel):
    strategy: RetrievalStrategy
    results: list[RetrievedChunk]
    latency_ms: float
    error: str | None = None


class RetrievalComparisonResponse(BaseModel):
    query: str
    document_id: str | None = None
    embedding_model: str
    comparisons: list[RetrievalStrategyResult]
    request_id: str


class RagQueryRequest(GenerationParameters):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None
    model: str | None = None
    retrieval_strategy: RetrievalStrategy = "dense"
    embedding_model: str | None = None
    reranker_model: str | None = None


class RagQueryResponse(BaseModel):
    generation: GenerateResponse
    sources: list[RetrievedChunk]
    retrieval_strategy: RetrievalStrategy
    retrieval_latency_ms: float


class DatasetInfo(BaseModel):
    name: str
    format: str
    examples: int


class DatasetUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    examples: list[dict] = Field(min_length=1, max_length=500)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1, max_length=80)
    mode: Literal["model", "rag"] = "model"
    model: str | None = None
    retrieval_strategy: RetrievalStrategy = "dense"
    top_k: int = Field(default=5, ge=1, le=50)
    limit: int | None = Field(default=None, ge=1, le=500)
    document_id: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    embedding_model: str | None = None
    semantic_metrics: bool = False
    semantic_embedding_model: str | None = None
    prompt_template: str | None = Field(default=None, max_length=20000)
    reranker_model: str | None = None


class HumanReview(BaseModel):
    label: Literal["correct", "partially_correct", "incorrect"]
    notes: str = Field(default="", max_length=2000)
    reviewed_at: str


class EvaluationReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["correct", "partially_correct", "incorrect"]
    notes: str = Field(default="", max_length=2000)


class EvaluationExampleResult(BaseModel):
    id: str
    question: str
    ground_truth: str
    answer: str
    metrics: dict[str, float | None]
    sources: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float
    repetition: int | None = None
    prompt: str = ""
    failure_category: str | None = None
    human_review: HumanReview | None = None


class EvaluationResponse(BaseModel):
    run_id: str
    created_at: str
    dataset: str
    mode: Literal["model", "rag"]
    config: dict
    dataset_sha256: str | None = None
    model: str
    retrieval_strategy: RetrievalStrategy | None = None
    example_count: int
    metrics: dict[str, float | None]
    results_path: str
    results: list[EvaluationExampleResult]


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1, max_length=80)
    models: list[str] = Field(min_length=2, max_length=8)
    mode: Literal["model", "rag"] = "model"
    retrieval_strategy: RetrievalStrategy = "dense"
    top_k: int = Field(default=5, ge=1, le=50)
    limit: int | None = Field(default=None, ge=1, le=100)
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    repetitions: int = Field(default=1, ge=1, le=10)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)
    embedding_model: str | None = None
    reranker_model: str | None = None


class BenchmarkModelResult(BaseModel):
    model: str
    provider: str | None = None
    model_id: str | None = None
    device: str | None = None
    loaded_before_run: bool | None = None
    cold_load_ms: float | None = None
    repetitions: int = 1
    latency_stddev_ms: float | None = None
    example_count: int
    exact_match: float
    token_f1: float
    p50_latency_ms: float
    p95_latency_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    requests_per_second: float
    tokens_per_second: float
    peak_rss_mb: float | None = None
    peak_cpu_percent: float | None = None
    peak_gpu_memory_mb: float | None = None
    estimated_cost: float | None = None
    configured_cost_per_1m_tokens: float | None = None
    examples: list[EvaluationExampleResult]


class BenchmarkResponse(BaseModel):
    run_id: str
    created_at: str
    dataset: str
    mode: Literal["model", "rag"]
    config: dict
    dataset_sha256: str | None = None
    retrieval_strategy: RetrievalStrategy | None = None
    results_path: str
    models: list[BenchmarkModelResult]


class ExperimentRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str = Field(min_length=1, max_length=80)


class ExperimentRunResponse(BaseModel):
    experiment_id: str
    config_name: str
    created_at: str
    config: dict
    evaluation: EvaluationResponse
    results_path: str
