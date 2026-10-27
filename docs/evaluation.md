# Evaluation datasets and metrics

## Dataset format

Evaluation files live in `datasets/` as JSON Lines, one object per row:

```json
{"id":"q001","question":"...","ground_truth":"...","contexts":["optional reference context"]}
```

IDs must be unique within a dataset. The API validates rows when listing and loading a dataset. `GET /api/v1/datasets` lists available files.

The dashboard accepts `.jsonl` files up to 5 MB and 500 rows. Each row must include `id`, `question`, and `ground_truth`; `contexts` is optional. The API validates every row and stores uploaded datasets under `.modelbench/datasets/`, so they survive restarts in local and Compose configurations. The dataset immediately becomes selectable for evaluations and benchmarks.

## Running an evaluation

Use `POST /api/v1/evaluations/run` with a dataset name and `mode` set to `model` or `rag`. Model runs include any per-example reference contexts in the prompt. RAG runs retrieve from the indexed document collection and accept a retrieval strategy and optional document ID. Runs are sequential, bounded to 500 examples per request, and persisted as JSON or in PostgreSQL, depending on configuration.

Set `seed` to record and use a sampling seed. Evaluation derives a stable seed per example. The result includes the exact source dataset's SHA-256 fingerprint and the resolved request configuration. GPU kernels and runtime versions can still cause differences between seeded reruns.

The dashboard's System Metrics page can point the current browser tab at another API base URL. The setting stays in that tab's session storage. For a local dashboard on port `3003` calling a different local API, the API must allow that origin in `MODELBENCH_CORS_ORIGINS`.

After a run, review each example against its question and reference in the dashboard. Labels are `correct`, `partially_correct`, or `incorrect`; notes and review time are saved with the evaluation run through `PATCH /api/v1/evaluations/{run_id}/examples/{example_id}/review`. A label is a human annotation, not an additional model-generated metric.

## Current deterministic metrics

- **Exact match:** normalized answer text equality.
- **Token F1:** token overlap precision and recall against the reference answer.
- **ROUGE-L:** longest-common-subsequence token F1, rewarding matching answer order as well as overlap.
- **Context recall:** fraction of reference answer terms present in the supplied/retrieved context.
- **Retrieval hit:** whether any one context contains at least half of the normalized reference terms.
- **Context precision:** share of returned passages with at least a small lexical overlap with the reference.
- **Answer relevance:** question-term coverage in the answer.
- **Faithfulness / hallucination-risk proxies:** lexical support for generated content in supplied context.

Context metrics are `null` when there is no context. These lexical metrics are inexpensive and reproducible, but can miss valid paraphrases and do not establish faithfulness or answer quality. Optionally enable embedding-based semantic metrics and select a sentence-transformers profile such as BGE or E5. The semantic profile is configured separately from the retrieval embedding profile so enabling evaluation does not change which document collection is searched. This adds answer-to-reference cosine similarity and, when contexts exist, answer-to-best-context similarity. They are semantic similarity signals, not entailment judgments; use human review or a calibrated judge for final correctness decisions. Semantic model loading and embedding time are included in evaluation latency.

`general_qa` now has 23 simple, manually authored seed examples for larger local review sessions. The set still needs independent human verification before it should be described as a human-reviewed benchmark or used for model-selection claims. RAG QA and context-abstention sets remain small smoke/demo datasets.
