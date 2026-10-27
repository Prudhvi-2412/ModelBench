# Model Bench dashboard

Next.js and TypeScript engineering dashboard for local inference, model profile inspection, document RAG, model benchmarks, evaluations, YAML experiments, persisted failure inspection, and process metrics. It compares all four retrieval strategies on the same question and displays retrieved snippets and per-method scores/latency.

## Local development

```powershell
Copy-Item .env.local.example .env.local
npm ci
npm run dev
```

Start the FastAPI API as described in `../backend/README.md`. The browser API base defaults to `http://127.0.0.1:8000`; configure `NEXT_PUBLIC_API_BASE_URL` for another local host. The API CORS allowlist defaults to the two usual local UI origins.

The dashboard keeps the last benchmark, evaluation, and experiment in browser storage for quick overview access; the API remains the persistent source for history and failure records. Scores are heuristic lexical baselines, and the UI labels that boundary.
