"""Run persistence: PostgreSQL when configured, atomic JSON files for local-only use."""
import json
import os
from typing import Any

from .settings import ROOT

RUN_DIR = ROOT / ".modelbench" / "runs"
_CACHE_PREFIX = "modelbench:runs:v1:"


def _redis():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis
    except ImportError:
        return None
    try:
        client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=0.2,
                                     socket_timeout=0.2)
        client.ping()
        return client
    except redis.RedisError:
        return None


def _connect():
    url = os.getenv("DATABASE_URL")
    if not url:
        return None
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("DATABASE_URL is set but psycopg is missing; reinstall the backend dependencies") from exc
    return psycopg.connect(url, autocommit=True)


def _ensure_table(connection) -> None:
    connection.execute("""CREATE TABLE IF NOT EXISTS modelbench_runs (
        run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
        payload JSONB NOT NULL)""")


def _save(run_id: str, kind: str, result: dict[str, Any]) -> str:
    stored_result = {**result, "kind": kind}
    cache = _redis()
    if cache:
        import redis

        # Cache invalidation is best-effort; a Redis timeout must not turn a
        # successfully computed benchmark or evaluation into a failed request.
        try:
            keys = list(cache.scan_iter(f"{_CACHE_PREFIX}*"))
            if keys:
                cache.delete(*keys)
        except redis.RedisError:
            pass
    connection = _connect()
    if connection:
        from psycopg.types.json import Jsonb

        with connection:
            _ensure_table(connection)
            connection.execute(
                "INSERT INTO modelbench_runs (run_id,kind,created_at,payload) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (run_id) DO UPDATE SET payload=EXCLUDED.payload",
                (run_id, kind, result["created_at"], Jsonb(stored_result)),
            )
        connection.close()
        return f"postgres://modelbench_runs/{run_id}"
    directory = RUN_DIR / kind
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stored_result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return str(path.relative_to(ROOT))


def _list(kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    cache = _redis() if kind is None else None
    cache_key = f"{_CACHE_PREFIX}{limit}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return json.loads(cached)[:limit]
    connection = _connect()
    if connection:
        with connection:
            _ensure_table(connection)
            rows = connection.execute(
                "SELECT payload FROM modelbench_runs WHERE (%s::text IS NULL OR kind=%s) "
                "ORDER BY created_at DESC LIMIT %s", (kind, kind, limit)
            ).fetchall()
        connection.close()
        result = [row[0] for row in rows]
        if cache:
            cache.setex(cache_key, 5, json.dumps(result))
        return result
    if not RUN_DIR.exists():
        # Read records written by the earliest local implementation.
        legacy = ROOT / ".modelbench"
        result = []
        for folder in ("evaluations", "benchmarks", "experiments"):
            for path in (legacy / folder).glob("*.json"):
                try:
                    item = json.loads(path.read_text(encoding="utf-8"))
                    item.setdefault("kind", folder.rstrip("s"))
                    result.append(item)
                except (OSError, json.JSONDecodeError):
                    continue
        result = sorted(result, key=lambda item: item.get("created_at", ""), reverse=True)[:limit]
        if cache:
            cache.setex(cache_key, 5, json.dumps(result))
        return result
    result = []
    folders = [RUN_DIR / kind] if kind else RUN_DIR.iterdir()
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item.setdefault("kind", folder.name)
                result.append(item)
            except (OSError, json.JSONDecodeError):
                continue
    result = sorted(result, key=lambda item: item.get("created_at", ""), reverse=True)[:limit]
    if cache:
        cache.setex(cache_key, 5, json.dumps(result))
    return result


def _get(run_id: str, kind: str) -> dict[str, Any] | None:
    connection = _connect()
    if connection:
        with connection:
            _ensure_table(connection)
            row = connection.execute(
                "SELECT payload FROM modelbench_runs WHERE run_id=%s AND kind=%s",
                (run_id, kind),
            ).fetchone()
        connection.close()
        return row[0] if row else None

    path = RUN_DIR / kind / f"{run_id}.json"
    if not path.is_file() and kind == "evaluation":
        path = ROOT / ".modelbench" / "evaluations" / f"{run_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


async def save_run(run_id: str, kind: str, result: dict[str, Any]) -> str:
    import asyncio
    return await asyncio.to_thread(_save, run_id, kind, result)


async def list_runs(kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    import asyncio
    return await asyncio.to_thread(_list, kind, limit)


async def get_run(run_id: str, kind: str) -> dict[str, Any] | None:
    import asyncio
    return await asyncio.to_thread(_get, run_id, kind)
