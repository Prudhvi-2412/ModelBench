# Model benchmarking

## Run a comparison

1. Start the API and inspect `GET /api/v1/models` for configured profile names.
2. Use `POST /api/v1/benchmarks/run` with one evaluation dataset, two or more model names, and an optional RAG retrieval strategy. Set `repetitions` (1–10) to repeat each QA item; the dashboard defaults to two repetitions.
3. Inspect the normalized per-model output and saved JSON under `.modelbench/runs/benchmark/`, or the PostgreSQL `modelbench_runs` table in Compose mode.

Set the optional `seed` to make stochastic sampling replayable. The same per-example and per-repetition seed sequence is used for every selected model, so each model sees equivalent sampling settings. The saved run includes the SHA-256 fingerprint of the exact dataset JSONL bytes and the complete request config. Seeded runs can still differ across provider versions, runtimes, and GPU kernels.

The initial profiles are small Hugging Face Transformers models, loaded lazily. They download on first use and need the optional `transformers` dependencies. The default development provider is useful for API development, but its output and latency are not representative of real model serving.

## Measurements

- Cold load: `cold_load_ms` measures provider preparation before timed requests. Transformers includes local weight loading; remote vLLM load time is not observable and stays null. A provider already resident reports zero for Transformers.
- Quality: normalized exact match and token F1 against each ground truth; example metrics also include ROUGE-L.
- Latency: P50/P95 and population standard deviation across repeated samples. Timed latency excludes model preparation; RAG latency includes retrieval.
- TTFT: time from the benchmark request starting until the first streamed fragment; this includes retrieval for RAG runs.
- Throughput: requests/sec over this sequential workload and generated whitespace tokens/sec over generation time.
- Memory: process RSS sampled before and after each example; CUDA peak allocated memory from PyTorch where available.
- CPU: peak non-blocking process CPU percentage sampled after examples; short requests can be under-sampled.
- Cost: calculated only when a model profile configures `cost_per_1k_tokens`; output-token run estimate and normalized output cost per 1M tokens exclude input tokens and local electricity.
- GPU: peak CUDA allocation from PyTorch is recorded when inference runs in this process; NVML utilization/memory is exposed as a live metric when available. For remote vLLM, process allocation and GPU-wide live metrics have different measurement scopes.

Runs are sequential, so these measurements compare models under a simple single-request workload. They do not estimate concurrent capacity. Token counts use whitespace splitting until provider-native accounting is added. Process RSS/CPU are sampled and can miss short peaks. Treat results as engineering measurements for the same machine and workload, not a universal model ranking.
