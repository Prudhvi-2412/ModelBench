# Reproducible experiments

Experiment configurations are YAML files under `configs/experiments/`. Each can record a dataset, language and embedding model profiles, model or RAG mode, retrieval strategy, reranker, prompt template, chunk size/overlap, generation parameters, and an optional sampling seed. Run one with:

```powershell
python run_experiment.py --config configs/experiments/experiment_001.yaml
```

The API reads the named config, executes an evaluation, and stores both the resolved config snapshot and evaluation output under `.modelbench/runs/experiment/` (and the evaluation row under `.modelbench/runs/evaluation/`). In Compose mode, both records are stored in PostgreSQL. Evaluation output includes the dataset SHA-256 fingerprint. RAG experiments use the selected embedding/reranker profiles during retrieval. Chunk size and overlap describe how the source documents should be ingested; re-ingest with those values before the run when they differ from the existing index.

The sample config uses the development provider and general QA dataset, so it needs no model download. For a real model, set its profile name and install the Transformers optional dependencies. Record changes as a new config file so previous experiment settings remain available for comparison.
