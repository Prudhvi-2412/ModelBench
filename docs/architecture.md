# Architecture and project intent

## Why Model Bench exists

The project translates AI engineering responsibilities into a demonstrable system. It should show that a model can be served behind an API, used in a retrieval-augmented application, evaluated against a dataset, compared with alternatives, and debugged when it fails. Results and configuration need to be recorded so an experiment can be repeated.

## Design boundaries

- **Local first:** the core workflow must run locally before any cloud deployment work.
- **Provider interfaces:** model and embedding providers can be changed through configuration rather than application rewrites.
- **Independent components:** serving, retrieval, evaluation, benchmarking, and presentation have clear boundaries.
- **Progressive infrastructure:** add PostgreSQL, Redis, Qdrant, monitoring, and containers as their corresponding workflows are implemented.
- **CPU development path:** hardware acceleration improves performance but is not required to explore core workflows.
- **Evidence over breadth:** prioritize end-to-end working paths over placeholder integrations.

## Core workflow

1. Configure a language model, embedding model, and inference parameters.
2. Call a validated inference API, optionally receiving streamed output.
3. Ingest documents, split them into chunks, index them, and retrieve evidence for a question.
4. Run a dataset through a model or RAG configuration and collect quality and performance measurements.
5. Compare runs and inspect failed examples, including their prompt, retrieved context, model response, and expected answer.

## Relationship to the role description

The source role description lists possible responsibilities, rather than mandatory technologies or a prescribed architecture. Model Bench uses those responsibilities as learning and demonstration goals: open-weight serving, embeddings, inference APIs, performance comparisons, RAG, evaluation, prompt/configuration experiments, GPU/container work, failure investigation, reproducibility, and production-style API design.

## Deferred decisions

Exact model IDs, GPU serving defaults, authentication mechanism, and cloud resources should be selected when the relevant milestone is implemented, based on available hardware and practical cost. AWS remains a later optional phase.

## Implemented local topology

The FastAPI process owns request validation and orchestration. Language model runtimes implement a provider interface: a no-download mock supports development, Transformers loads an open-weight causal model in-process, and the vLLM adapter calls a separately hosted OpenAI-compatible endpoint. Embedding providers are similarly interchangeable. This keeps model-runtime dependencies out of the API layer and lets a GPU server be stopped independently.

Qdrant stores vectors and chunk metadata. A deterministic hashing embedding provider supports smoke workflows without downloading weights; sentence-transformers is an optional semantic embedding provider. The local embedded Qdrant mode keeps setup short. Compose selects a Qdrant service through `QDRANT_URL`. Runs are JSON files without infrastructure and PostgreSQL rows when `DATABASE_URL` is set; Redis is only a short-TTL cache for run-history listing, never the source of truth.

Prometheus scrapes HTTP, generation, retrieval, embedding, token-rate and resource metrics. Grafana provisioning loads a ready-to-use dashboard. Open-weight model loading and reranking can download code/weights at first use, so configure trusted model IDs and verify their licenses. Secrets are read from environment/configuration and are not committed.
