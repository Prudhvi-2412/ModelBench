# Inference and serving

## Provider boundary

`InferenceProvider` defines asynchronous generation and streaming. The FastAPI routes do not import Transformers or vLLM directly. The `mock` provider is a no-download contract stub, not an LLM; the `transformers` provider lazily loads a Hugging Face causal model; `vllm` sends OpenAI-compatible chat-completions requests to a separately managed server.

Profiles in `configs/models/` choose a display name, model ID, provider, device, sampling defaults, and optional per-1K output token estimate. Every request can select a configured profile by name or model ID and override temperature, maximum output tokens, and top-p. The default configuration remains safe for a CPU-only machine.

## API behavior

`POST /api/v1/generate` and `/chat` return completion text plus model, whitespace-token estimates, latency, throughput, and a request ID. Set `stream: true` to receive SSE `delta` events and a final `done` event containing aggregate metadata; errors are emitted as `error` events. Embeddings return vectors, model, dimension, batch size, latency, throughput, and request ID.

The API validates bounds with Pydantic, rejects unknown model aliases, logs request completion as JSON, and echoes `X-Request-ID`. `MODELBENCH_API_KEY` optionally enables bearer or `X-API-Key` protection on non-health routes. Put the key in an environment secret, never in tracked YAML. The optional key is a local integration seam; public deployment still needs identity, authorization policy, TLS, quotas, and secret rotation.

## Runtime constraints

Transformers downloads weights on first request and consumes API-process RAM/VRAM. The vLLM deployment runs separately and is optional in Compose; it needs a supported NVIDIA runtime and enough VRAM for model weights plus KV cache. Small model examples are learning profiles, not performance recommendations. For CPU development use the mock provider; for actual CPU inference select a small Transformers profile and expect slower generation.

## Reproducibility

An evaluation/benchmark result records the resolved request configuration and model profile name. If a repository revision, model weights, provider library, hardware, or dataset changes, results can differ. Keep those versions in experiment notes when comparing runs; this project does not yet hash every external model artifact.
