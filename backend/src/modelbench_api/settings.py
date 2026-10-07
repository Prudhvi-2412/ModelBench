from pathlib import Path

import yaml
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "models" / "default.yaml"


class ModelSettings(BaseModel):
    name: str = "local-development"
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    provider: str = "mock"
    device: str = "auto"
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    top_p: float = Field(default=0.95, gt=0, le=1)


def load_settings(config_path: Path | None = None) -> ModelSettings:
    path = config_path or Path(__import__("os").environ.get("MODELBENCH_MODEL_CONFIG", DEFAULT_CONFIG))
    if not path.exists():
        return ModelSettings()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ModelSettings.model_validate(raw.get("model", raw))
