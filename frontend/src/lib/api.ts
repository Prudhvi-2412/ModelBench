export type ModelInfo = {
  name: string;
  model_id: string;
  provider: string;
  loaded: boolean;
};

export type DatasetInfo = { name: string; format: string; examples: number };
export type EmbeddingModelInfo = { name: string; model_id: string; provider: string; dimension: number; loaded: boolean };
export type SystemMetrics = { process_id: number; process_rss_mb: number; process_cpu_percent: number; gpu_utilization_percent: number | null; gpu_memory_used_mb: number | null; gpu_memory_total_mb: number | null };
export type RetrievalComparison = { query: string; comparisons: { strategy: string; latency_ms: number; error: string | null; results: RetrievedChunk[] }[] };
export type MetricSet = Record<string, number | null>;
export type RetrievedChunk = {
  document_id: string;
  chunk_id: string;
  filename: string;
  page_number: number | null;
  text: string;
  score: number;
};
export type EvaluationExample = {
  id: string;
  question: string;
  ground_truth: string;
  answer: string;
  metrics: MetricSet;
  sources: RetrievedChunk[];
  latency_ms: number;
  repetition?: number | null;
  prompt?: string;
  failure_category?: string | null;
  human_review?: { label: "correct" | "partially_correct" | "incorrect"; notes: string; reviewed_at: string } | null;
};
export type FailureExample = EvaluationExample & {
  run_id: string;
  dataset?: string;
  model?: string;
  created_at?: string;
};
export type RunRecord = Record<string, unknown> & {
  kind?: string;
  run_id?: string;
  experiment_id?: string;
  created_at: string;
  dataset?: string;
  model?: string;
  config_name?: string;
  metrics?: MetricSet;
  models?: Pick<BenchmarkModel, "model" | "example_count" | "exact_match" | "token_f1" | "p50_latency_ms" | "p95_latency_ms" | "tokens_per_second">[];
};
export type EvaluationResult = {
  run_id: string;
  created_at: string;
  dataset: string;
  mode: "model" | "rag";
  config: Record<string, unknown>;
  dataset_sha256?: string | null;
  model: string;
  retrieval_strategy: string | null;
  example_count: number;
  metrics: MetricSet;
  results_path: string;
  results: EvaluationExample[];
};
export type GenerationResult = {
  response: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  tokens_per_second: number;
  request_id: string;
};
export type RagQueryResult = {
  generation: GenerationResult;
  sources: RetrievedChunk[];
  retrieval_strategy: string;
  retrieval_latency_ms: number;
};
export type BenchmarkModel = {
  model: string;
  provider?: string | null;
  model_id?: string | null;
  device?: string | null;
  loaded_before_run?: boolean | null;
  cold_load_ms?: number | null;
  repetitions?: number;
  latency_stddev_ms?: number | null;
  example_count: number;
  exact_match: number;
  token_f1: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p50_ttft_ms: number;
  p95_ttft_ms: number;
  requests_per_second: number;
  tokens_per_second: number;
  peak_rss_mb: number | null;
  peak_cpu_percent: number | null;
  peak_gpu_memory_mb: number | null;
  estimated_cost: number | null;
  configured_cost_per_1m_tokens: number | null;
  examples: EvaluationExample[];
};
export type BenchmarkResult = {
  run_id: string;
  created_at: string;
  dataset: string;
  mode: "model" | "rag";
  config: Record<string, unknown>;
  dataset_sha256?: string | null;
  retrieval_strategy: string | null;
  results_path: string;
  models: BenchmarkModel[];
};
export type ExperimentResult = {
  experiment_id: string;
  config_name: string;
  created_at: string;
  results_path: string;
  evaluation: EvaluationResult;
};

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const SESSION_API_KEY = "modelbench.apiKey";
const SESSION_API_BASE = "modelbench.apiBaseUrl";

function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const configured = window.sessionStorage.getItem(SESSION_API_BASE);
    if (configured) return configured;
  }
  return API_BASE;
}

function authHeaders(headers?: HeadersInit): HeadersInit {
  const result = new Headers(headers);
  if (typeof window !== "undefined") {
    const key = window.sessionStorage.getItem(SESSION_API_KEY);
    if (key) result.set("Authorization", `Bearer ${key}`);
  }
  return result;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, { ...init, headers: authHeaders(init?.headers) });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      message = payload.detail || message;
    } catch {
      // Keep the status-based message for non-JSON error responses.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  getApiBaseUrl,
  setSessionApiBaseUrl: (value: string) => {
    const parsed = new URL(value.trim());
    if (!(parsed.protocol === "http:" || parsed.protocol === "https:") || parsed.username || parsed.password || parsed.search || parsed.hash) {
      throw new Error("Enter an HTTP or HTTPS API URL without credentials, query, or hash.");
    }
    const normalized = `${parsed.origin}${parsed.pathname.replace(/\/$/, "")}`;
    if (typeof window !== "undefined") window.sessionStorage.setItem(SESSION_API_BASE, normalized);
    return normalized;
  },
  setSessionApiKey: (key: string) => {
    if (typeof window !== "undefined") {
      if (key.trim()) window.sessionStorage.setItem(SESSION_API_KEY, key.trim());
      else window.sessionStorage.removeItem(SESSION_API_KEY);
    }
  },
  health: () => request<{ status: string; provider: string; model: string }>("/api/v1/health"),
  models: () => request<ModelInfo[]>("/api/v1/models"),
  prepareModel: (model: string) => request<{ model: string; provider: string; ready: boolean; cold_load_ms: number | null }>(`/api/v1/models/${encodeURIComponent(model)}/prepare`, { method: "POST" }),
  datasets: () => request<DatasetInfo[]>("/api/v1/datasets"),
  uploadDataset: (name: string, examples: unknown[]) => request<DatasetInfo>("/api/v1/datasets", json({ name, examples })),
  embeddingModels: () => request<EmbeddingModelInfo[]>("/api/v1/embedding-models"),
  failures: () => request<FailureExample[]>("/api/v1/failures?limit=100"),
  runs: () => request<RunRecord[]>("/api/v1/runs?limit=25"),
  benchmarkRun: (runId: string) => request<BenchmarkResult>(`/api/v1/benchmarks/${encodeURIComponent(runId)}`),
  system: () => request<SystemMetrics>("/api/v1/system"),
  compareRetrieval: (body: unknown) => request<RetrievalComparison>("/api/v1/retrieval/compare", json(body)),
  generate: (body: unknown) => request<GenerationResult>("/api/v1/generate", json(body)),
  ragQuery: (body: unknown) => request<RagQueryResult>("/api/v1/rag/query", json(body)),
  evaluate: (body: unknown) => request<EvaluationResult>("/api/v1/evaluations/run", json(body)),
  reviewEvaluationExample: (runId: string, exampleId: string, body: { label: "correct" | "partially_correct" | "incorrect"; notes: string }) =>
    request<EvaluationResult>(`/api/v1/evaluations/${encodeURIComponent(runId)}/examples/${encodeURIComponent(exampleId)}/review`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }),
  benchmark: (body: unknown) => request<BenchmarkResult>("/api/v1/benchmarks/run", json(body)),
  experiment: (body: unknown) => request<ExperimentResult>("/api/v1/experiments/run", json(body)),
  ingest: async (file: File, embeddingModel?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (embeddingModel) form.append("embedding_model", embeddingModel);
    return request<{ document_id: string; filename: string; page_count: number; chunk_count: number }>(
      "/api/v1/documents",
      { method: "POST", body: form },
    );
  },
  streamGenerate: async (
    body: unknown,
    onDelta: (delta: string) => void,
  ): Promise<GenerationResult> => {
    const response = await fetch(`${getApiBaseUrl()}/api/v1/generate`, {
      ...json({ ...(body as object), stream: true }),
      headers: authHeaders({ "Content-Type": "application/json" }),
    });
    if (!response.ok || !response.body) {
      let message = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch {
        message = `Request failed (${response.status})`;
      }
      throw new Error(message);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let final: GenerationResult | null = null;
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const event of events) {
        const eventName = event.match(/^event:\s*(.+)$/m)?.[1]?.trim();
        const data = event.split("\n").find((line) => line.startsWith("data: "))?.slice(6);
        if (!data) continue;
        const payload: Record<string, unknown> = JSON.parse(data);
        if (eventName === "done") final = payload as unknown as GenerationResult;
        else if (eventName === "error") throw new Error(String(payload.detail || "Streaming failed"));
        else if (payload.delta) onDelta(String(payload.delta));
      }
      if (done) break;
    }
    if (!final) throw new Error("The stream ended before completion metadata arrived");
    return final;
  },
};
