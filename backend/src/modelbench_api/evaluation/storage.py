import json
from typing import Any

from ..run_store import save_run
from ..settings import ROOT

RESULTS_DIR = ROOT / ".modelbench" / "evaluations"


def _write_result(run_id: str, result: dict[str, Any]) -> str:
    path = ROOT / ".modelbench" / "evaluations" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path.relative_to(ROOT))


async def save_result(run_id: str, result: dict[str, Any]) -> str:
    return await save_run(run_id, "evaluation", result)
