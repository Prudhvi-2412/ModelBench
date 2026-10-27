import asyncio

import modelbench_api.main as api_module
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from modelbench_api.main import app
from modelbench_api.model_registry import ModelRuntime
from modelbench_api.providers.mock import DevelopmentProvider
from modelbench_api.schemas import EvaluationReviewRequest
from modelbench_api.settings import ModelSettings


def test_health_models_validation_and_metrics(monkeypatch):
    monkeypatch.setenv("MODELBENCH_API_KEY", "local-test-key")
    with TestClient(app) as client:
        auth = {"X-API-Key": "local-test-key"}
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.headers.get("x-request-id")
        embedding_profiles = client.get("/api/v1/embedding-models", headers=auth).json()
        assert {profile["model_id"] for profile in embedding_profiles} >= {"hashing-384", "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2"}
        vector = client.post("/api/v1/embeddings", json={"input": ["one", "two"]}, headers=auth)
        assert vector.status_code == 200
        assert vector.json()["batch_size"] == 2
        assert client.get("/api/v1/models").status_code == 401
        assert client.get("/api/v1/models", headers={"X-API-Key": "local-test-key"}).json()
        bad = client.post("/api/v1/generate", json={"prompt": "hi", "max_tokens": 0}, headers=auth)
        assert bad.status_code == 422
        missing = client.post("/api/v1/generate", json={"prompt": "hi", "model": "not-configured"}, headers=auth)
        assert missing.status_code == 404
        assert "modelbench_http_requests_total" in client.get("/metrics", headers={"X-API-Key": "local-test-key"}).text


def test_inference_stream_and_evaluation_saved():
    with TestClient(app) as client:
        response = client.post("/api/v1/generate", json={"prompt": "Say hello", "stream": True, "seed": 99})
        assert response.status_code == 200
        assert "event: done" in response.text
        evaluation = client.post("/api/v1/evaluations/run", json={"dataset": "general_qa", "limit": 1, "seed": 43})
        assert evaluation.status_code == 200
        body = evaluation.json()
        assert body["example_count"] == 1
        assert body["results"][0]["prompt"]
        assert body["results"][0]["human_review"] is None
        assert body["config"]["seed"] == 43
        assert len(body["dataset_sha256"]) == 64
        assert body["results_path"]
        assert client.get("/api/v1/runs").status_code == 200
        experiment = client.post("/api/v1/experiments/run", json={"config": "experiment_001"})
        assert experiment.status_code == 200, experiment.text
        assert experiment.json()["config"]["dataset"] == "general_qa"
        assert experiment.json()["evaluation"]["config"]["seed"] == 42
        history = client.get("/api/v1/runs?limit=1").json()
        assert history[0]["kind"] == "experiment"
        assert "results" not in history[0]


def test_review_api_persists_human_feedback_and_cors_configuration(monkeypatch):
    run_id = "review-run"
    record = {
        "run_id": run_id,
        "created_at": "2026-10-08T00:00:00+00:00",
        "dataset": "general_qa",
        "mode": "model",
        "config": {},
        "model": "local-development",
        "retrieval_strategy": None,
        "example_count": 1,
        "metrics": {"token_f1": 0.5},
        "results": [{
            "id": "gqa-001", "question": "A question", "ground_truth": "Answer",
            "answer": "Partial", "metrics": {"token_f1": 0.5}, "sources": [],
            "latency_ms": 2.0, "prompt": "A question", "failure_category": "reasoning failure",
        }],
    }
    saved = []

    async def fake_get_run(*_args):
        return record

    async def fake_save_run(*args):
        saved.append(args)
        return ".modelbench/runs/evaluation/review-run.json"

    monkeypatch.setattr(api_module, "get_run", fake_get_run)
    monkeypatch.setattr(api_module, "save_run", fake_save_run)

    reviewed = asyncio.run(api_module.review_evaluation_example(
        run_id, "gqa-001", EvaluationReviewRequest(label="partially_correct", notes="Needs a detail.")
    ))
    assert reviewed.results[0].human_review.label == "partially_correct"
    assert reviewed.results[0].human_review.notes == "Needs a detail."
    assert reviewed.results[0].human_review.reviewed_at
    assert saved and saved[0][0] == run_id and saved[0][1] == "evaluation"
    async def missing_example():
        await api_module.review_evaluation_example(
            run_id, "missing", EvaluationReviewRequest(label="incorrect", notes="")
        )
    with pytest.raises(HTTPException) as error:
        asyncio.run(missing_example())
    assert error.value.status_code == 404
    cors = next(item for item in app.user_middleware if item.cls.__name__ == "CORSMiddleware")
    assert "http://localhost:3003" in cors.kwargs["allow_origins"]
    assert "PATCH" in cors.kwargs["allow_methods"]


def test_document_rag_retrieval_and_failure_analysis():
    with TestClient(app) as client:
        uploaded = client.post("/api/v1/documents", files={"file": ("guide.txt", b"The Model Bench local platform uses Qdrant to store embedded document chunks.", "text/plain")})
        assert uploaded.status_code == 200, uploaded.text
        document_id = uploaded.json()["document_id"]
        retrieved = client.post("/api/v1/retrieve", json={"query": "What stores the chunks?", "document_id": document_id, "strategy": "bm25"})
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["results"]
        compared = client.post("/api/v1/retrieval/compare", json={"query": "What stores the chunks?", "document_id": document_id, "top_k": 2})
        assert compared.status_code == 200, compared.text
        assert [item["strategy"] for item in compared.json()["comparisons"]] == ["dense", "bm25", "hybrid", "hybrid_rerank"]
        answer = client.post("/api/v1/rag/query", json={"question": "What does Qdrant store?", "document_id": document_id})
        assert answer.status_code == 200, answer.text
        evaluation = client.post("/api/v1/evaluations/run", json={"dataset": "general_qa", "mode": "rag", "limit": 1})
        assert evaluation.status_code == 200
        assert client.get("/api/v1/failures").status_code == 200


def test_benchmark_runs_identical_workload_and_collects_resources():
    with TestClient(app) as client:
        observed_seeds = []

        class SeedRecordingProvider(DevelopmentProvider):
            async def stream(self, prompt, *, temperature, max_tokens, top_p, seed=None):
                observed_seeds.append(seed)
                async for fragment in super().stream(
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    seed=seed,
                ):
                    yield fragment

        app.state.model_registry._register(
            ModelRuntime(
                ModelSettings(name="second-development", model_id="development-stub-2", provider="mock"),
                SeedRecordingProvider(),
            )
        )
        response = client.post(
            "/api/v1/benchmarks/run",
            json={"dataset": "general_qa", "models": ["local-development", "second-development"], "limit": 1, "repetitions": 2, "seed": 23},
        )
        assert response.status_code == 200, response.text
        first = response.json()["models"][0]
        assert first["example_count"] == 1
        assert first["p95_latency_ms"] >= first["p50_latency_ms"]
        assert "peak_cpu_percent" in first
        assert "configured_cost_per_1m_tokens" in first
        assert response.json()["config"]["seed"] == 23
        assert len(response.json()["dataset_sha256"]) == 64
        assert observed_seeds == [23, 24]
