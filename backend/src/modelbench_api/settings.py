from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "models" / "default.yaml"


class ModelSettings(BaseModel):
    name: str = "local-development"
    model_id: str = "development-stub"
    provider: str = "mock"
    device: str = "auto"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    top_p: float = Field(default=0.95, gt=0, le=1)
    cost_per_1k_tokens: float | None = Field(default=None, ge=0)
    api_base_url: str = "http://localhost:8001/v1"
    api_key: str | None = None


def load_settings(config_path: Path | None = None) -> ModelSettings:
    path = config_path or Path(__import__("os").environ.get("MODELBENCH_MODEL_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        return ModelSettings()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ModelSettings.model_validate(raw.get("model", raw))
