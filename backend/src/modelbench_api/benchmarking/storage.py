import json
from typing import Any

from ..evaluation.storage import RESULTS_DIR
from ..run_store import save_run

BENCHMARK_DIR = RESULTS_DIR.parent / "benchmarks"


def _write_result(run_id: str, result: dict[str, Any]) -> str:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCHMARK_DIR / f"{run_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return str(path.relative_to(RESULTS_DIR.parents[1]))


async def save_benchmark(run_id: str, result: dict[str, Any]) -> str:
    return await save_run(run_id, "benchmark", result)
