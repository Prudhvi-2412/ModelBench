# Backend API

The first milestone exposes a configurable FastAPI inference API. The default `mock` provider runs without downloading a model and clearly labels its output as a development response. It is useful for building the surrounding API and UI, but does not generate real model answers.

## Run locally

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .\backend
python -m uvicorn modelbench_api.main:app --app-dir backend\src --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API reference. Set `MODELBENCH_MODEL_CONFIG` to use a different YAML model configuration.

## Optional Transformers provider

Install `python -m pip install -e ".[transformers]"` from `backend`, then set `provider: transformers` in `configs/models/default.yaml`. The model downloads from Hugging Face on first inference. Use a small model for local experiments; the development GPU has 6 GB of VRAM, so larger model serving is not a reasonable default.

## Endpoints

- `GET /api/v1/health`
- `GET /api/v1/models`
- `POST /api/v1/generate`
- `POST /api/v1/chat`
- `POST /api/v1/embeddings`
- `POST /api/v1/documents` (multipart upload: `.txt`, `.md`, or `.pdf`)
- `POST /api/v1/retrieve` (dense vector search)
- `POST /api/v1/rag/query` (retrieve context, then generate an answer)

Requests accept temperature, max_tokens, top_p, optional model selection, and a stream flag. Streaming uses SSE and emits delta events followed by a done event containing the final metadata. Token counts are approximate whitespace counts in this initial milestone; provider-native token accounting is a later improvement.

The embedding endpoint uses offline feature hashing by default so its API can be developed without a model download. This is a lexical baseline, not a semantic embedding model. To use a real Sentence Transformers model, install `python -m pip install -e .[embeddings]` from `backend` and set `provider: sentence-transformers` and `model_id` in `configs/models/embedding.yaml` (for example, a BGE or E5 model).

Document uploads are parsed, split using `configs/rag/default.yaml`, embedded, and stored in a persistent local Qdrant collection under `.modelbench/qdrant`. PDF page numbers and document/chunk metadata are retained in point payloads. The default local store is intended for a single development API process; a Qdrant server can be configured as a later deployment option.

The retrieval endpoint embeds a query with the active embedding model and returns dense cosine-similarity matches. The RAG endpoint adds the retrieved chunks to a grounded prompt and returns the model response with source metadata and retrieval timing. BM25, hybrid search, and reranking are planned additions.
