from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from ..settings import ROOT


class RagSettings(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=9999)
    max_upload_mb: int = Field(default=10, ge=1, le=100)
    candidate_k: int = Field(default=20, ge=1, le=200)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "auto"
    qdrant_path: str = ".modelbench/qdrant"

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def load_rag_settings() -> RagSettings:
    import os

    config_path = Path(os.environ.get("MODELBENCH_RAG_CONFIG", ROOT / "configs/rag/default.yaml"))
    if not config_path.exists():
        return RagSettings()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return RagSettings.model_validate(values.get("rag", values))
