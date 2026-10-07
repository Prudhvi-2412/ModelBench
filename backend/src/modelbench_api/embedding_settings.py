from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .settings import ROOT


DEFAULT_EMBEDDING_CONFIG = ROOT / "configs" / "models" / "embedding.yaml"


class EmbeddingSettings(BaseModel):
    name: str = "local-hashing"
    model_id: str = "hashing-384"
    provider: str = "hashing"
    device: str = "auto"
    dimension: int = Field(default=384, ge=32, le=8192)


def load_embedding_settings(config_path: Path | None = None) -> EmbeddingSettings:
    import os

    path = config_path or Path(os.environ.get("MODELBENCH_EMBEDDING_CONFIG", DEFAULT_EMBEDDING_CONFIG))
    if not path.exists():
        # The model configuration and embedding configuration are separate by design.
        return EmbeddingSettings()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    values = raw.get("embedding", raw)
    return EmbeddingSettings.model_validate(values)
