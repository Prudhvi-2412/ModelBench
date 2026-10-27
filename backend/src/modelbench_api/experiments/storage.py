import json
from typing import Any

from ..run_store import save_run
from ..settings import ROOT

EXPERIMENT_RESULTS_DIR = ROOT / ".modelbench" / "experiments"


def _write(experiment_id: str, result: dict[str, Any]) -> str:
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT_RESULTS_DIR / f"{experiment_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return str(path.relative_to(ROOT))


async def save_experiment(experiment_id: str, result: dict[str, Any]) -> str:
    return await save_run(experiment_id, "experiment", result)
