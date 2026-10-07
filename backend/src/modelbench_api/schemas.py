from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class GenerationParameters(BaseModel):
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    top_p: float = Field(default=0.95, gt=0, le=1)


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


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    page_number: int | None = None
    text: str
    score: float


class RetrievalResponse(BaseModel):
    query: str
    strategy: Literal["dense"]
    embedding_model: str
    results: list[RetrievedChunk]
    latency_ms: float
    request_id: str


class RagQueryRequest(GenerationParameters):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: str | None = None
    model: str | None = None


class RagQueryResponse(BaseModel):
    generation: GenerateResponse
    sources: list[RetrievedChunk]
    retrieval_strategy: Literal["dense"]
    retrieval_latency_ms: float
