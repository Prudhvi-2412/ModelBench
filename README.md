# Model Bench

**A local-first platform for production-style LLM inference, RAG, benchmarking, and evaluation.**

Model Bench is an AI engineering portfolio project built to demonstrate the work described in the AI Engineering Intern role: serving open-weight language and embedding models, exposing inference APIs, building RAG systems, comparing models and retrieval strategies, investigating failures, and documenting reproducible experiments.

The goal is a working engineering platform, not a chatbot demo. Development starts locally and proceeds in small, verifiable milestones. Cloud deployment is intentionally deferred until the local system is useful and stable.

## Initial scope

The first usable milestone will provide a local API, model/provider interfaces, a basic document ingestion and retrieval path, and a simple experiment record. Later milestones add model comparison, evaluation and failure analysis, a dashboard, and observability. GPU serving is optional; the development path must remain usable on CPU.

## Planned architecture

```text
Next.js dashboard ── FastAPI platform API ── PostgreSQL / Redis
                              │
             ┌────────────────┼─────────────────┐
       Inference service   RAG pipeline    Evaluation & benchmarks
             │                 │                    │
      vLLM / Transformers   Qdrant + BM25     datasets + metrics
```

Services and model providers will be configurable. The dashboard will consume stable APIs; benchmark and evaluation logic will remain independent of the UI.

## Repository layout

```text
frontend/       Next.js dashboard
backend/        FastAPI application and persistence
inference/      model and embedding provider interfaces
rag/            parsing, chunking, retrieval, and reranking
evaluation/     datasets and evaluation metrics
benchmarking/   workload execution and result aggregation
configs/        model, RAG, and experiment YAML
datasets/       versionable sample evaluation data
scripts/        local development and experiment commands
docker/         service Dockerfiles and monitoring configuration
docs/           architecture and implementation guides
tests/          unit and integration tests
```

## Development sequence

1. Establish interfaces, configuration conventions, and a runnable local skeleton.
2. Implement inference and embedding APIs with a CPU-friendly development provider and optional GPU serving.
3. Add document ingestion, retrieval strategies, and RAG answers.
4. Add reproducible datasets, evaluations, experiments, and model benchmarks.
5. Build dashboard views for results and failure analysis.
6. Add operational metrics, container workflows, and complete documentation.
7. Consider an optional, cost-conscious AWS deployment only after local milestones work.

Each milestone should leave the project runnable and document its assumptions. Avoid integrating infrastructure that does not support a working feature yet.

## Status

The first local inference and dense-RAG path is in place: FastAPI endpoints for health, model discovery, generate, chat, SSE output, embeddings, PDF/text document ingestion, dense retrieval, and RAG queries. Uploaded documents are chunked and persisted in local Qdrant. Development providers run without model downloads; optional Hugging Face providers load configured language and embedding models. See [backend setup](backend/README.md).
