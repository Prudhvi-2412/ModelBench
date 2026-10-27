# Failure analysis

Every evaluation example records its prompt, model answer, ground truth, retrieved chunks and scores, latency, and a heuristic failure category when the answer misses the reference. Benchmark examples use the same fields. The `/api/v1/failures` API and dashboard aggregate these saved examples across recent runs.

Current categories are triage hints: retrieval failure (no source retrieved), irrelevant context (weak lexical answer coverage by retrieved chunks), token limit (long answer near the configured limit), context ignored (abstention despite a reference mismatch), possible hallucination (unsupported lexical content), reasoning failure (some answer overlap but still wrong), and instruction-following failure (otherwise unclassified mismatch). These rules are deterministic and can be wrong; validate them against human-labeled failures before treating category counts as model quality metrics.

Inspect the prompt and every retrieved source when diagnosing a case. Ask whether the dataset reference is sufficient, whether the expected evidence exists in the indexed document, whether the retrieval strategy selected it, and whether the prompt/model used it. For model-level QA there may be no retrieved source. Failed inference requests are logged and returned to callers but are not currently stored as evaluation rows; timeouts and service faults should be inspected from request logs.

The answer and context quality signals in this repo are lexical proxies. A production evaluation should add a labeled rubric or a calibrated judge model and report judge version, prompt, and agreement with human labels.
