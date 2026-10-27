# Retrieval augmented generation

## Current local workflow

1. Upload `.txt`, `.md`, or `.pdf` to `POST /api/v1/documents`.
2. Extract text, retain PDF page numbers, and split each page into overlapping chunks.
3. Embed each chunk and store its text, source metadata, embedding model, and vector in local Qdrant (embedded local mode or Qdrant service in Compose).
4. Call `POST /api/v1/retrieve` to inspect matching passages or `POST /api/v1/rag/query` to pass them to the configured language model.

The chunk size, overlap, upload limit, candidate count, and reranker model are in `configs/rag/default.yaml`. The active embedding model chooses a separate Qdrant collection so vectors from different embedding spaces are not mixed.

## Retrieval strategies

| Strategy | Method | Use |
| --- | --- | --- |
| `dense` | Qdrant cosine similarity over the configured embedding model | Semantic similarity |
| `bm25` | Okapi BM25 over locally stored chunk text | Exact terms and lexical matching |
| `hybrid` | Reciprocal rank fusion of dense and BM25 rankings | Combine semantic and lexical signals |
| `hybrid_rerank` | Hybrid candidate generation followed by a CrossEncoder | Reorder a smaller candidate set for relevance |

Set `strategy` on `/api/v1/retrieve`, or `retrieval_strategy` on `/api/v1/rag/query`. Scores from the strategies are method-specific and should not be compared as if they shared one scale.

## Prototype limits

The default hashing embedding is an offline lexical baseline, not a semantic model. For meaningful semantic retrieval, configure a Sentence Transformers BGE or E5 model. BM25 currently reads the local Qdrant corpus per request; that is suitable for a small local collection but should be replaced by an indexed lexical backend for larger datasets. The local Qdrant path is designed for one development API process.

Example BGE and E5 settings are provided under `configs/models/embeddings/`; set `MODELBENCH_EMBEDDING_CONFIG` to the selected YAML file and install `backend[embeddings]`. Every embedding profile uses an isolated Qdrant collection because vectors from different models should not be searched together. Re-embed documents after changing the configured embedding profile.

RAG answers include source chunks and their metadata. The prompt asks the model to use sources and say when they do not support an answer, but this instruction alone does not guarantee faithfulness; evaluation is needed to measure that behavior.
