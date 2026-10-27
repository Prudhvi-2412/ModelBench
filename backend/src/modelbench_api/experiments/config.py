import re
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import RetrievalStrategy
from ..settings import ROOT

EXPERIMENT_DIR = ROOT / "configs" / "experiments"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    dataset: str = Field(min_length=1, max_length=80)
    mode: Literal["model", "rag"] = "model"
    model: str | None = None
    retrieval_strategy: RetrievalStrategy = "dense"
    top_k: int = Field(default=5, ge=1, le=50)
    limit: int | None = Field(default=None, ge=1, le=500)
    document_id: str | None = None
    embedding_model: str | None = None
    prompt_template: str | None = Field(default=None, max_length=20_000)
    chunk_size: int | None = Field(default=None, ge=100, le=10_000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=9_999)
    reranker_model: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    top_p: float = Field(default=1, gt=0, le=1)
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def validate_chunk_settings(self):
        if self.chunk_size is not None and self.chunk_overlap is not None and self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def load_experiment(name: str) -> ExperimentConfig:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError("Invalid experiment config name")
    path = EXPERIMENT_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {name}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw.get("experiment", raw))
