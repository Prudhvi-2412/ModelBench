# Model Bench

Model Bench is a local-first engineering workbench for serving open-weight models, building RAG applications, running repeatable evaluations and benchmarks, and investigating failures. It turns AI engineering responsibilities into inspectable software workflows instead of a single chatbot demo.

## What it demonstrates

- Open-weight generation via Hugging Face Transformers or an OpenAI-compatible vLLM server, plus a clearly labelled no-download development stub.
- Configurable sentence-transformer or deterministic hashing embeddings; local or remote Qdrant; PDF/TXT/Markdown ingestion; dense, BM25, hybrid RRF, and optional cross-encoder reranking.
- FastAPI inference, chat, embeddings, document, retrieval, RAG, evaluation, benchmark, experiment, history, failure, health, and system-metric APIs. Generation supports server-sent events and request IDs.
- Versionable JSONL datasets, YAML experiment profiles, repeatable model comparisons, local JSON result files, and optional PostgreSQL persistence with Redis run-history caching.
- A Next.js dashboard for all nine project areas, with model/retrieval comparisons and persisted failure examples. Prometheus and provisioned Grafana dashboards expose operational metrics.

## Architecture

```text
Next.js dashboard ── FastAPI platform API ── PostgreSQL (durable runs)
                               │               Redis (short-lived history cache)
              ┌────────────────┼─────────────────────┐
         Transformers /      Qdrant + BM25      evaluation, benchmarks,
         vLLM providers       retrieval          experiment records
              │                                      │
              └──────────── Prometheus ───────── Grafana
```

The default API uses a mock development provider and deterministic hashing embeddings. This starts without model downloads or a GPU. Mock output is intentionally not a real model completion and should not be interpreted as a quality score. Transformers, sentence-transformers, reranking, vLLM, and NVML GPU metrics are optional paths. Local single-process mode writes results and Qdrant data under `.modelbench/`; Compose mode uses PostgreSQL, Redis, and the Qdrant service. Remote Qdrant requests use a 30-second default timeout, configurable with `MODELBENCH_QDRANT_TIMEOUT_SECONDS` from 1 to 300 seconds, to tolerate cold collection creation and recovery.

## Technology stack

| Area | Technology |
|---|---|
| UI | Next.js, TypeScript, React, Tailwind CSS, Recharts |
| API | Python, FastAPI, Pydantic, Uvicorn |
| Model serving | Hugging Face Transformers or vLLM OpenAI-compatible server |
| Embeddings | sentence-transformers or built-in hashing fallback |
| Retrieval | Qdrant, BM25, reciprocal-rank fusion, optional CrossEncoder |
| Results | PostgreSQL in Compose; atomic local JSON files for standalone development; Redis cache |
| Operations | Docker Compose, Prometheus, Grafana, structured request logs |

## Local setup (CPU development)

Requirements: Python 3.11+, Node.js 22+, and npm. Docker is optional for the standalone mode.

From the repository root in PowerShell:

```powershell
python -m pip install -e "backend[dev]"
python -m uvicorn modelbench_api.main:app --app-dir backend/src --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000. API docs are at http://127.0.0.1:8000/docs, and health is `/api/v1/health`. The dashboard reads `NEXT_PUBLIC_API_BASE_URL`, defaulting to `http://127.0.0.1:8000`. Set `MODELBENCH_CORS_ORIGINS` if the UI runs on another origin.

Standalone mode uses local Qdrant storage and JSON files under `.modelbench/`. To use PostgreSQL and Redis outside Compose, set `DATABASE_URL` and `REDIS_URL`; PostgreSQL is durable storage and Redis caches run-history reads for five seconds. If PostgreSQL is configured, the database must be available before saving an evaluation or benchmark. Without those variables, run history remains local and persistent in JSON.

## Docker Compose

```powershell
docker compose up --build
```

This starts the dashboard (`:3000`), API (`:8000`), PostgreSQL (`:5432`), Redis (`:6379`), Qdrant (`:6333`), Prometheus (`:9090`), and Grafana (`:3001`). Grafana is provisioned with the local Prometheus datasource and Model Bench dashboard. The sample database credentials and Grafana password in Compose are for local development only; replace them before exposing services beyond localhost. Stop with `docker compose down`. Named volumes retain database, vector, cache, model, and dashboard data; `docker compose down -v` deletes that persisted data.

## Open-weight model setup

### CPU / Transformers

Install the optional provider dependencies:

```powershell
python -m pip install -e "backend[transformers]"
$env:MODELBENCH_MODEL_CONFIG = "configs/models/profiles/smollm2-360m.yaml"
python -m uvicorn modelbench_api.main:app --app-dir backend/src --reload
```

The first request downloads model weights from Hugging Face. The Qwen 0.5B and SmolLM2 360M profile examples are in `configs/models/profiles/`. Set `device: cpu` to force CPU execution. Model licenses and memory requirements vary; confirm both before downloading or redistributing weights.

### Optional GPU vLLM

The GPU profile requires Linux containers with a working NVIDIA Container Toolkit / Docker GPU runtime. The included Qwen 0.5B configuration is intentionally small, but available VRAM, CUDA compatibility, and vLLM version still determine whether it will load. It is not included in the default Compose startup.

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu up --build
```

The override points the API at vLLM using `configs/models/vllm.yaml`. For machines without a compatible GPU, use the CPU or development paths above. You can also run any compatible OpenAI Chat Completions server and set `provider: vllm` plus `api_base_url` in a model YAML file.

## Configure models and embeddings

- Default language-model config: `configs/models/default.yaml`.
- Discovered language-model profiles: `configs/models/profiles/*.yaml`.
- Non-loaded Llama and Mistral example profiles: `configs/models/examples/` (copy into `profiles/` to expose them to the registry; Llama may require an accepted Hugging Face license, and Mistral 7B exceeds the local GPU memory budget).
- Embedding config: `configs/models/embedding.yaml`.
- RAG chunk and upload settings: `configs/rag/default.yaml`.
- Do not put access tokens in YAML or commit them. Set `api_key` through a private local configuration or extend deployment secret injection; `MODELBENCH_API_KEY` optionally protects all non-health routes with a bearer/API key.

Providers are separated from HTTP routes in `backend/src/modelbench_api/providers/` and `embeddings/`; model IDs and sampling defaults are per-profile. The vLLM adapter speaks the OpenAI-compatible API and does not import vLLM inside the FastAPI process.

## RAG workflow

Use the dashboard or call `POST /api/v1/documents` with a PDF, TXT, or Markdown file. The API parses pages, chunks text with overlap, embeds each chunk, and stores metadata and vectors in Qdrant. Then use `POST /api/v1/retrieve` or `POST /api/v1/rag/query`. Supported retrieval strategies are `dense`, `bm25`, `hybrid`, and `hybrid_rerank`. Reranking downloads its configured CrossEncoder only on first use and requires `backend[reranking]`.

See [RAG design](docs/rag.md) for parsing, chunking, vector isolation, search strategy and trade-offs.

## Datasets, evaluations, benchmarks, and experiments

Dataset rows are JSONL records with `id`, `question`, `ground_truth`, and optional `contexts`. Validate a dataset with:

```powershell
$env:PYTHONPATH = "backend/src"
python scripts/validate_dataset.py general_qa
```

Run an evaluation through the dashboard or `POST /api/v1/evaluations/run`. Benchmarks run an identical workload sequentially against selected profiles and record cold model preparation separately from timed inference, repeated-sample latency variation, answer overlap (including ROUGE-L), TTFT, token/request throughput, process RAM/CPU, optional GPU allocation, estimated run cost, and configured cost per 1M output tokens. Evaluation, benchmark, and inference requests accept a seed; evaluation uses a derived per-example seed, and benchmarks derive the same per-example/per-repetition seeds for every compared model. Saved evaluations and benchmarks include a SHA-256 fingerprint of the exact JSONL dataset bytes alongside the request configuration and example-level prompts and outputs. A seed improves replayability, but GPU kernels, provider versions, and runtime hardware can still prevent bit-for-bit identical output. Cost is an estimate; mock providers and local hardware comparisons do not represent production pricing or multi-user throughput. Upload a JSONL question/answer set from the Evaluations view to use your own data in evaluations and benchmarks.

The `general_qa` starter dataset contains 23 examples. In the Evaluations view, label each generated answer as correct, partially correct, or incorrect and optionally add review notes; labels are persisted with the run. These seed examples still need independent human verification before the set can be called a human-reviewed benchmark. The System Metrics view can also switch the current browser tab to a different API base URL, such as the real local Transformers API; the API must allow the dashboard origin through CORS.

Run the saved YAML experiment:

```powershell
$env:PYTHONPATH = "backend/src"
python run_experiment.py --config configs/experiments/experiment_001.yaml
```

The CLI expects the API to be running. Runs preserve the resolved input configuration and example-level prompts, answers, sources, scores, and triage categories. Standalone records are atomic JSON files; Compose stores normalized run envelopes in PostgreSQL and uses Redis as a short-lived run-list cache.

Quality measures are explicitly baselines. Exact match and token F1 are lexical. Context precision/recall, answer relevance, and faithfulness/hallucination-risk are lexical proxies, not semantic judges; no RAGAS or LLM judge is enabled by default. Review examples and use a labeled evaluation set before making model-selection claims.

## Failure analysis

The dashboard’s Failure Analysis view calls `/api/v1/failures` and includes the question, model, prompt, expected and actual answers, retrieved text and scores, run ID, and heuristic category when available. Categories help triage retrieval misses, irrelevant context, possible hallucination, context-abstention, token-limit, and answer-quality problems; they are not proof of root cause. Inference errors and timeouts are logged and returned to the caller; failed-request persistence is a future extension. See [failure analysis](docs/failure-analysis.md).

## Observability

`GET /metrics` exports Prometheus metrics for HTTP request count, latency histogram, active requests, generation/retrieval/embedding/reranking time, TTFT, token rate, output tokens, and process RAM/CPU. GPU utilization and memory are sampled when NVML is available (`pip install -e "backend[gpu-metrics]"`); otherwise they are reported as unavailable. Prometheus derives P50/P95 from histograms, and the Grafana dashboard is provisioned at port `3001` (`admin` / `modelbench` in local Compose). Request-completion logs are JSON and carry correlation IDs in `X-Request-ID`.

## API reference

FastAPI OpenAPI docs are generated at `/docs`; the schemas are the source of truth. Main routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/v1/health`, `/api/v1/models`, `/api/v1/embedding-models`, `/api/v1/datasets`, `/api/v1/runs`, `/api/v1/benchmarks/{run_id}`, `/api/v1/failures`, `/api/v1/system` | Inspect configured services and load complete saved comparisons |
| POST | `/api/v1/generate`, `/api/v1/chat`, `/api/v1/embeddings` | Inference and vectorization |
| POST | `/api/v1/documents`, `/api/v1/retrieve`, `/api/v1/rag/query` | Ingest and query documents |
| POST | `/api/v1/evaluations/run`, `/api/v1/benchmarks/run`, `/api/v1/experiments/run` | Run and persist engineering workflows |
| PATCH | `/api/v1/evaluations/{run_id}/examples/{example_id}/review` | Save a human label and notes for an evaluation example |
| GET | `/metrics` | Prometheus scrape endpoint |

If `MODELBENCH_API_KEY` is configured, send `Authorization: Bearer <key>` or `X-API-Key`; health remains available for readiness checks. The dashboard accepts this key under System Metrics and keeps it in the current browser tab's session storage. This optional local guard is an auth integration point, not a replacement for production identity, TLS, network policy, key rotation, or user-level authorization.

## Tests and checks

```powershell
python -m pytest -q
python -m ruff check backend/src tests scripts run_experiment.py
cd frontend
npm run lint
npm run build
```

The production frontend build uses Next.js's documented Webpack option for a consistent build across supported environments. GitHub Actions runs backend tests and lint, frontend lint and production build, dataset validation, and Docker Compose configuration checks on pushes and pull requests.

Tests use the local development provider and a temporary Qdrant collection; they do not download language model weights. Docker integration checks require Docker Desktop / Docker Engine to be running.

## Documentation map

- [Architecture](docs/architecture.md)
- [Inference and serving](docs/inference.md)
- [RAG](docs/rag.md)
- [Evaluation](docs/evaluation.md)
- [Benchmarking](docs/benchmarking.md)
- [Failure analysis](docs/failure-analysis.md)
- [Experiments](docs/experiments.md)
- [AWS deployment plan](docs/aws-deployment.md) (design only; no cloud resources are created)

## AWS: future, optional phase

AWS is intentionally not implemented before local validation. A cost-conscious later design could put the dashboard in S3/CloudFront, the API and CPU workers on ECS, persistent results in RDS PostgreSQL, and Qdrant on a managed/self-hosted vector tier selected after workload testing. GPU serving can use an on-demand EC2 instance with vLLM, scale-to-zero/stop policies, ECR images, and CloudWatch/Prometheus integration. Estimate storage, network egress, and GPU time first; shut GPU capacity down when idle. No AWS resources or credentials are created by this repository.
