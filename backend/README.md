# FastAPI platform API

Install and run from the repository root:

```powershell
python -m pip install -e "backend[dev]"
python -m uvicorn modelbench_api.main:app --app-dir backend/src --reload --host 127.0.0.1 --port 8000
```

The no-download `mock` provider and hashing embeddings make CPU development available immediately. Mock completions are not real LLM outputs. Open `/docs` for the generated OpenAPI reference.

## Optional model runtimes

- Transformers: `python -m pip install -e "backend[transformers]"`, then point `MODELBENCH_MODEL_CONFIG` to a small profile YAML.
- Semantic embeddings: `python -m pip install -e "backend[embeddings]"`, then set `MODELBENCH_EMBEDDING_CONFIG` to `configs/models/embeddings/bge-small-en-v1.5.yaml` or `e5-small-v2.yaml`.
- Reranker: `python -m pip install -e "backend[reranking]"`; first use downloads the configured cross-encoder.
- GPU telemetry: `python -m pip install -e "backend[gpu-metrics]"` for NVML measurements.
- vLLM: run an OpenAI-compatible vLLM server separately and configure `provider: vllm` plus `api_base_url` in a model profile. Compose provides an optional GPU profile.

## Infrastructure

Standalone mode persists Qdrant and result JSON under the repository's `.modelbench/` folder. When `DATABASE_URL` is set, saved evaluations, experiments, and benchmarks go to PostgreSQL; `REDIS_URL` enables a five-second run-history cache. In Docker Compose, these are wired automatically, along with the remote Qdrant service, Prometheus, and Grafana.

To enable the optional API key check, set `MODELBENCH_API_KEY` in the process environment and provide it as a bearer token or `X-API-Key`; only `/api/v1/health` remains public. Do not place secrets in tracked files.

## Principal routes

- `GET /api/v1/health`, `/models`, `/embedding-models`, `/datasets`, `/runs`, `/failures`, `/system`
- `POST /api/v1/generate`, `/chat`, `/embeddings`
- `POST /api/v1/documents`, `/retrieve`, `/retrieval/compare`, `/rag/query`
- `POST /api/v1/evaluations/run`, `/benchmarks/run`, `/experiments/run`
- `GET /metrics` for Prometheus

See the root README and the architecture, inference, RAG, evaluation, benchmarking, experiments, and failure-analysis docs for the rationale and known limits.
