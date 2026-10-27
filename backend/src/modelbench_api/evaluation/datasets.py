import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..settings import ROOT

DATASET_DIR = ROOT / "datasets"
CUSTOM_DATASET_DIR = ROOT / ".modelbench" / "datasets"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


class DatasetExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=1, max_length=20_000)
    ground_truth: str = Field(min_length=1, max_length=20_000)
    contexts: list[str] = Field(default_factory=list)


def dataset_path(name: str) -> Path:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError("Dataset name may contain letters, digits, underscores, and hyphens")
    path = DATASET_DIR / f"{name}.jsonl"
    if not path.is_file():
        path = CUSTOM_DATASET_DIR / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {name}")
    return path


def dataset_sha256(name: str) -> str:
    """Fingerprint the exact JSONL bytes used by a run for reproducibility."""
    return hashlib.sha256(dataset_path(name).read_bytes()).hexdigest()


def load_dataset(name: str) -> list[DatasetExample]:
    path = dataset_path(name)
    examples: list[DatasetExample] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            example = DatasetExample.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"Invalid dataset row {line_number}: {exc}") from exc
        if example.id in seen_ids:
            raise ValueError(f"Duplicate example id {example.id!r} on row {line_number}")
        seen_ids.add(example.id)
        examples.append(example)
    if not examples:
        raise ValueError(f"Dataset is empty: {name}")
    return examples


def list_datasets() -> list[dict[str, str | int]]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    paths = {path.stem: path for path in DATASET_DIR.glob("*.jsonl")}
    paths.update({path.stem: path for path in CUSTOM_DATASET_DIR.glob("*.jsonl")})
    for name, path in sorted(paths.items()):
        try:
            examples = load_dataset(name)
            results.append({"name": name, "format": "jsonl", "examples": len(examples)})
        except ValueError:
            results.append({"name": name, "format": "jsonl", "examples": 0})
    return results


def save_custom_dataset(name: str, rows: list[dict]) -> int:
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError("Dataset name may contain letters, digits, underscores, and hyphens")
    if (DATASET_DIR / f"{name}.jsonl").exists():
        raise ValueError("A built-in dataset already uses this name")
    path = CUSTOM_DATASET_DIR / f"{name}.jsonl"
    if path.exists():
        raise ValueError("A custom dataset already uses this name; choose a different filename")
    if len(json.dumps(rows, ensure_ascii=False).encode("utf-8")) > 5 * 1024 * 1024:
        raise ValueError("Dataset upload must be 5 MB or smaller")
    examples = [DatasetExample.model_validate(row) for row in rows]
    ids = [example.id for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset example IDs must be unique")
    CUSTOM_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(example.model_dump_json() + "\n" for example in examples), encoding="utf-8")
    temporary.replace(path)
    return len(examples)
