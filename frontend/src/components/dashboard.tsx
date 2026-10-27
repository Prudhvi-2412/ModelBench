"use client";

import { type FormEvent, type ReactNode, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type BenchmarkResult, type DatasetInfo, type EmbeddingModelInfo, type EvaluationResult, type ExperimentResult, type FailureExample, type ModelInfo, type RagQueryResult, type RetrievalComparison, type RetrievedChunk, type RunRecord, type SystemMetrics } from "@/lib/api";

type Section = "Overview" | "Models" | "Inference" | "Benchmarks" | "RAG" | "Experiments" | "Evaluations" | "Failure Analysis" | "System Metrics";
type Health = { status: string; provider: string; model: string } | null;

const sections: { name: Section; short: string; glyph: string }[] = [
  { name: "Overview", short: "01", glyph: "◫" },
  { name: "Models", short: "02", glyph: "◎" },
  { name: "Inference", short: "03", glyph: "↗" },
  { name: "Benchmarks", short: "04", glyph: "▥" },
  { name: "RAG", short: "05", glyph: "⌘" },
  { name: "Experiments", short: "06", glyph: "◌" },
  { name: "Evaluations", short: "07", glyph: "✓" },
  { name: "Failure Analysis", short: "08", glyph: "⌁" },
  { name: "System Metrics", short: "09", glyph: "⌗" },
];

const storageKeys = {
  benchmark: "modelbench.latestBenchmark",
  evaluation: "modelbench.latestEvaluation",
  experiment: "modelbench.latestExperiment",
};

function persist(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Dashboard state remains useful for the current session if storage is unavailable.
  }
}

function readStored<T>(key: string): T | null {
  try {
    const value = localStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : null;
  } catch {
    return null;
  }
}

function format(value: number | null | undefined, suffix = "") {
  return value == null || !Number.isFinite(value) ? "—" : `${Number(value.toFixed(2))}${suffix}`;
}

function formatPercent(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(1)}%`;
}

function shortModel(name: string) {
  return name.replace(/-instruct$/i, "").replace(/-instruct/i, "");
}

export default function Dashboard() {
  const [active, setActive] = useState<Section>("Overview");
  const [health, setHealth] = useState<Health>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelInfo[]>([]);
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [semanticEmbeddingModel, setSemanticEmbeddingModel] = useState("");
  const [failures, setFailures] = useState<FailureExample[]>([]);
  const [runHistory, setRunHistory] = useState<RunRecord[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [apiBaseUrl, setApiBaseUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedBenchModels, setSelectedBenchModels] = useState<string[]>([]);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);
  const [loadingBenchmarkRun, setLoadingBenchmarkRun] = useState(false);
  const [warmingModel, setWarmingModel] = useState<string | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [experiment, setExperiment] = useState<ExperimentResult | null>(null);
  const [dataset, setDataset] = useState("general_qa");
  const [benchmarkLimit, setBenchmarkLimit] = useState(3);
  const [benchmarkMaxTokens, setBenchmarkMaxTokens] = useState(64);
  const [benchmarkRepetitions, setBenchmarkRepetitions] = useState(2);
  const [generationSeed, setGenerationSeed] = useState(42);
  const [mode, setMode] = useState<"model" | "rag">("model");
  const [semanticMetrics, setSemanticMetrics] = useState(false);
  const [prompt, setPrompt] = useState("Explain how dense retrieval differs from BM25 in two sentences.");
  const [answer, setAnswer] = useState("");
  const [inferenceMeta, setInferenceMeta] = useState<{ latency_ms?: number; tokens_per_second?: number } | null>(null);
  const [documentId, setDocumentId] = useState("");
  const [documentLabel, setDocumentLabel] = useState("");
  const [ragQuestion, setRagQuestion] = useState("");
  const [ragStrategy, setRagStrategy] = useState("hybrid");
  const [ragTopK, setRagTopK] = useState(5);
  const [ragAnswer, setRagAnswer] = useState<RagQueryResult | null>(null);
  const [retrievalComparison, setRetrievalComparison] = useState<RetrievalComparison | null>(null);
  const [experimentConfig, setExperimentConfig] = useState("experiment_001");
  const [reviewingId, setReviewingId] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setError("");
    const [healthResult, modelsResult, datasetsResult, failuresResult, systemResult, embeddingsResult] = await Promise.allSettled([
      api.health(),
      api.models(),
      api.datasets(),
      api.failures(),
      api.system(),
      api.embeddingModels(),
    ]);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else setHealth(null);
    if (modelsResult.status === "fulfilled") {
      setModels(modelsResult.value);
      setSelectedModel((current) => current || modelsResult.value[0]?.name || "");
      setSelectedBenchModels((current) => {
        if (current.length) return current;
        return modelsResult.value.filter((model) => model.provider === "transformers").map((model) => model.name).slice(0, 3);
      });
    } else {
      setError("Could not reach the Model Bench API. Start the backend at http://127.0.0.1:8000.");
    }
    if (datasetsResult.status === "fulfilled") {
      setDatasets(datasetsResult.value);
      setDataset((current) => current || datasetsResult.value[0]?.name || "general_qa");
    }
    if (failuresResult.status === "fulfilled") setFailures(failuresResult.value);
    if (systemResult.status === "fulfilled") setSystemMetrics(systemResult.value);
    if (embeddingsResult.status === "fulfilled") {
      setEmbeddingModels(embeddingsResult.value);
      setEmbeddingModel((current) => current || embeddingsResult.value[0]?.model_id || "");
      setSemanticEmbeddingModel((current) => current || embeddingsResult.value.find((item) => item.provider === "sentence-transformers")?.model_id || "");
    }
    setLoading(false);
  }

  async function refreshHistory(loadLatestBenchmark = false) {
    try {
      const history = await api.runs();
      setRunHistory(history);
      const latestBenchmark = history.find((item) => item.kind === "benchmark" && item.run_id);
      const storedBenchmark = readStored<BenchmarkResult>(storageKeys.benchmark);
      if (loadLatestBenchmark && latestBenchmark?.run_id && latestBenchmark.run_id !== storedBenchmark?.run_id) {
        const saved = await api.benchmarkRun(latestBenchmark.run_id);
        setBenchmark(saved);
        persist(storageKeys.benchmark, saved);
      }
    } catch { /* API status already appears in the workspace header. */ }
  }

  async function loadBenchmarkRun(runId: string) {
    if (!runId) return;
    setLoadingBenchmarkRun(true); setError("");
    try {
      const saved = await api.benchmarkRun(runId);
      setBenchmark(saved); persist(storageKeys.benchmark, saved);
      setNotice(`Loaded benchmark ${runId.slice(0, 8)}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load saved benchmark");
    } finally { setLoadingBenchmarkRun(false); }
  }

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      setApiBaseUrl(api.getApiBaseUrl());
      setBenchmark(readStored<BenchmarkResult>(storageKeys.benchmark));
      setEvaluation(readStored<EvaluationResult>(storageKeys.evaluation));
      setExperiment(readStored<ExperimentResult>(storageKeys.experiment));
      void refresh();
      void refreshHistory(true);
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  async function runInference(event: FormEvent) {
    event.preventDefault();
    const form = event.currentTarget as HTMLFormElement;
    const values = new FormData(form);
    setBusy("inference"); setError(""); setAnswer(""); setInferenceMeta(null);
    try {
      const meta = await api.streamGenerate(
        {
          prompt,
          model: selectedModel || undefined,
          temperature: Number(values.get("temperature")),
          max_tokens: Number(values.get("max_tokens")),
          top_p: Number(values.get("top_p")),
          seed: values.get("seed") ? Number(values.get("seed")) : undefined,
        },
        (delta) => setAnswer((current) => current + delta),
      );
      setInferenceMeta(meta);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Inference request failed");
    } finally { setBusy(null); }
  }

  async function uploadDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const file = (form.elements.namedItem("document") as HTMLInputElement).files?.[0];
    if (!file) return;
    setBusy("upload"); setError(""); setNotice("");
    try {
      const result = await api.ingest(file, embeddingModel || undefined);
      setDocumentId(result.document_id);
      setDocumentLabel(`${result.filename} · ${result.chunk_count} chunks`);
      setNotice(`Indexed ${result.filename} into local Qdrant.`);
      form.reset();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Document upload failed");
    } finally { setBusy(null); }
  }

  async function queryRag(event: FormEvent) {
    event.preventDefault();
    setBusy("rag"); setError(""); setRagAnswer(null);
    try {
      const result = await api.ragQuery({
        question: ragQuestion,
        model: selectedModel || undefined,
        retrieval_strategy: ragStrategy,
        top_k: ragTopK,
        document_id: documentId || undefined,
        embedding_model: embeddingModel || undefined,
        temperature: 0.2,
        max_tokens: 512,
        seed: generationSeed,
      });
      setRagAnswer(result);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "RAG query failed");
    } finally { setBusy(null); }
  }

  async function compareStrategies() {
    if (!ragQuestion.trim()) { setError("Enter a question before comparing retrieval strategies."); return; }
    setBusy("compare"); setError(""); setRetrievalComparison(null);
    try {
      setRetrievalComparison(await api.compareRetrieval({ query: ragQuestion, document_id: documentId || undefined, top_k: ragTopK, embedding_model: embeddingModel || undefined }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Retrieval comparison failed");
    } finally { setBusy(null); }
  }

  async function runEvaluation(event: FormEvent) {
    event.preventDefault();
    setBusy("evaluation"); setError("");
    try {
      const result = await api.evaluate({
        dataset, mode, model: selectedModel || undefined, retrieval_strategy: ragStrategy,
        top_k: ragTopK, embedding_model: embeddingModel || undefined, semantic_metrics: semanticMetrics,
        semantic_embedding_model: semanticMetrics ? semanticEmbeddingModel || undefined : undefined,
        temperature: 0, max_tokens: 256, seed: generationSeed,
      });
      setEvaluation(result); persist(storageKeys.evaluation, result);
      void refreshHistory();
      setNotice(`Evaluation saved: ${result.run_id.slice(0, 8)}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Evaluation failed");
    } finally { setBusy(null); }
  }

  async function reviewEvaluationExample(exampleId: string, label: "correct" | "partially_correct" | "incorrect", notes: string) {
    if (!evaluation) return;
    setReviewingId(exampleId); setError("");
    try {
      const updated = await api.reviewEvaluationExample(evaluation.run_id, exampleId, { label, notes });
      setEvaluation(updated); persist(storageKeys.evaluation, updated);
      setNotice(`Saved human review for ${exampleId}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save the human review");
    } finally { setReviewingId(null); }
  }

  async function runBenchmark(event: FormEvent) {
    event.preventDefault();
    if (selectedBenchModels.length < 2) {
      setError("Select at least two models to compare."); return;
    }
    setBusy("benchmark"); setError("");
    try {
      const result = await api.benchmark({
        dataset, models: selectedBenchModels, mode, retrieval_strategy: ragStrategy,
        top_k: ragTopK, limit: benchmarkLimit, repetitions: benchmarkRepetitions, embedding_model: embeddingModel || undefined, temperature: 0, max_tokens: benchmarkMaxTokens, seed: generationSeed,
      });
      setBenchmark(result); persist(storageKeys.benchmark, result);
      void refreshHistory();
      setNotice(`Benchmark saved: ${result.run_id.slice(0, 8)}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Benchmark failed");
    } finally { setBusy(null); }
  }

  async function uploadDataset(file: File) {
    setBusy("dataset"); setError(""); setNotice("");
    try {
      if (file.size > 5 * 1024 * 1024) throw new Error("Dataset file must be 5 MB or smaller.");
      const rows = (await file.text()).split(/\r?\n/).filter((line) => line.trim()).map((line, index) => {
        try { return JSON.parse(line) as unknown; }
        catch { throw new Error(`Invalid JSON on line ${index + 1}.`); }
      });
      const name = file.name.replace(/\.jsonl$/i, "").toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^[^a-z0-9]+/, "").slice(0, 80);
      if (!name || !rows.length) throw new Error("Choose a non-empty .jsonl dataset with a valid filename.");
      const created = await api.uploadDataset(name, rows);
      setDatasets(await api.datasets()); setDataset(created.name);
      setNotice(`Loaded ${created.examples} evaluation examples from ${created.name}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Dataset upload failed");
    } finally { setBusy(null); }
  }

  async function warmupModel(modelName: string) {
    setWarmingModel(modelName); setError(""); setNotice("");
    try {
      const selected = models.find((model) => model.name === modelName);
      if (selected?.provider === "vllm") {
        const status = await api.prepareModel(modelName);
        setNotice(`${modelName} ${status.ready ? "is ready" : "is not ready"}; external model-load duration is not measurable here.`);
      } else {
        const result = await api.generate({ prompt: "Reply with OK.", model: modelName, temperature: 0, max_tokens: 1, top_p: 1 });
        setNotice(`${modelName} is ready. One-token warm-up took ${format(result.latency_ms, " ms")}.`);
      }
      setModels(await api.models());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not warm up ${modelName}`);
    } finally { setWarmingModel(null); }
  }

  async function runExperiment(event: FormEvent) {
    event.preventDefault();
    setBusy("experiment"); setError("");
    try {
      const result = await api.experiment({ config: experimentConfig.trim() });
      setExperiment(result); persist(storageKeys.experiment, result);
      void refreshHistory();
      setEvaluation(result.evaluation); persist(storageKeys.evaluation, result.evaluation);
      setNotice(`Experiment completed: ${result.experiment_id.slice(0, 8)}.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Experiment failed");
    } finally { setBusy(null); }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark"><span>m</span><i /></div>
          <div><strong>model<span>bench</span></strong><small>AI ENGINEERING LAB</small></div>
        </div>
        <div className="workspace-chip"><span className="workspace-dot" /> Local workspace <span className="chevron">⌄</span></div>
        <div className="side-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Dashboard sections">
          {sections.map((item) => (
            <button key={item.name} className={`nav-item ${active === item.name ? "active" : ""}`} onClick={() => { setActive(item.name); setError(""); }}>
              <span className="nav-glyph">{item.glyph}</span><span>{item.name}</span><span className="nav-short">{item.short}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-card"><span className="pulse-dot" /><div><b>LOCAL FIRST</b><small>Cloud deployment paused</small></div></div>
          <div className="user-row"><div className="avatar">MB</div><div><b>Model Bench</b><small>Engineering workspace</small></div><span className="more">···</span></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="crumb"><span>Workspace</span><b>/</b><strong>{active}</strong></div>
          <div className="top-actions">
            <div className={`api-status ${health ? "online" : "offline"}`}><span />{health ? "API connected" : "API offline"}</div>
            <button className="icon-button" title="Refresh workspace" onClick={() => void refresh()}>↻</button>
            <div className="top-avatar">MB</div>
          </div>
        </header>
        <div className="content-wrap">
          <div className="page-intro">
            <div><div className="eyebrow">MODEL BENCH <span>·</span> LOCAL LAB</div><h1>{active === "Overview" ? "Your AI engineering workspace" : active}</h1><p>{pageDescription(active)}</p></div>
            <div className="intro-meta"><span className="env-pill"><i /> DEVELOPMENT ENVIRONMENT</span><span className="date-stamp">LOCAL · CPU / GPU</span></div>
          </div>
          {(error || notice) && <div className={`toast ${error ? "toast-error" : "toast-success"}`}><span>{error ? "!" : "✓"}</span><div>{error || notice}</div><button onClick={() => { setError(""); setNotice(""); }}>×</button></div>}
          {loading && <div className="loading-strip"><span className="spinner" /> Connecting to local services…</div>}

          {active === "Overview" && <Overview health={health} models={models} benchmark={benchmark} evaluation={evaluation} experiment={experiment} onNavigate={setActive} />}
          {active === "Models" && <ModelsPage models={models} onSelect={(name) => { setSelectedModel(name); setActive("Inference"); }} />}
          {active === "Inference" && <InferencePage models={models} selectedModel={selectedModel} setSelectedModel={setSelectedModel} prompt={prompt} setPrompt={setPrompt} answer={answer} meta={inferenceMeta} busy={busy === "inference"} onSubmit={runInference} />}
          {active === "Benchmarks" && <BenchmarkPage models={models} datasets={datasets} dataset={dataset} setDataset={setDataset} seed={generationSeed} setSeed={setGenerationSeed} limit={benchmarkLimit} setLimit={setBenchmarkLimit} repetitions={benchmarkRepetitions} setRepetitions={setBenchmarkRepetitions} maxTokens={benchmarkMaxTokens} setMaxTokens={setBenchmarkMaxTokens} selectedModels={selectedBenchModels} setSelectedModels={setSelectedBenchModels} mode={mode} setMode={setMode} strategy={ragStrategy} setStrategy={setRagStrategy} topK={ragTopK} setTopK={setRagTopK} result={benchmark} history={runHistory.filter((item) => item.kind === "benchmark")} loadingRun={loadingBenchmarkRun} onSelectRun={loadBenchmarkRun} busy={busy === "benchmark"} warmingModel={warmingModel} onWarmup={warmupModel} onSubmit={runBenchmark} />}
          {active === "RAG" && <RagPage models={models} embeddingModels={embeddingModels} embeddingModel={embeddingModel} setEmbeddingModel={setEmbeddingModel} selectedModel={selectedModel} setSelectedModel={setSelectedModel} documentId={documentId} documentLabel={documentLabel} question={ragQuestion} setQuestion={setRagQuestion} strategy={ragStrategy} setStrategy={setRagStrategy} topK={ragTopK} setTopK={setRagTopK} answer={ragAnswer} comparison={retrievalComparison} busy={busy} onUpload={uploadDocument} onQuery={queryRag} onCompare={compareStrategies} />}
          {active === "Evaluations" && <EvaluationPage datasets={datasets} embeddingModels={embeddingModels} semanticEmbeddingModel={semanticEmbeddingModel} setSemanticEmbeddingModel={setSemanticEmbeddingModel} semanticMetrics={semanticMetrics} setSemanticMetrics={setSemanticMetrics} dataset={dataset} setDataset={setDataset} seed={generationSeed} setSeed={setGenerationSeed} mode={mode} setMode={setMode} strategy={ragStrategy} setStrategy={setRagStrategy} result={evaluation} busy={busy === "evaluation"} reviewingId={reviewingId} onReview={reviewEvaluationExample} onUploadDataset={uploadDataset} onSubmit={runEvaluation} />}
          {active === "Experiments" && <ExperimentsPage config={experimentConfig} setConfig={setExperimentConfig} result={experiment} history={runHistory} busy={busy === "experiment"} onSubmit={runExperiment} />}
          {active === "Failure Analysis" && <FailuresPage failures={failures} />}
          {active === "System Metrics" && <SystemPage health={health} models={models} benchmark={benchmark} metrics={systemMetrics} apiKey={apiKeyInput} setApiKey={setApiKeyInput} onSaveApiKey={(key) => { api.setSessionApiKey(key); void refresh(); }} apiBaseUrl={apiBaseUrl} setApiBaseUrl={setApiBaseUrl} onSaveApiBaseUrl={(value) => { const normalized = api.setSessionApiBaseUrl(value); setApiBaseUrl(normalized); window.location.reload(); }} />}
        </div>
      </main>
    </div>
  );
}

function pageDescription(section: Section) {
  const descriptions: Record<Section, string> = {
    Overview: "A clear view of the models, experiments, and signals that matter.",
    Models: "Inspect local model profiles and choose a provider for your next run.",
    Inference: "Send a prompt to a configured model and inspect streamed output and timing.",
    Benchmarks: "Run an identical dataset across model profiles and compare quality and speed.",
    RAG: "Ingest a source document, choose a retrieval strategy, and inspect grounded answers.",
    Experiments: "Replay a saved YAML configuration and retain its exact settings with the result.",
    Evaluations: "Measure answer overlap and retrieval coverage against small, versionable datasets.",
    "Failure Analysis": "Review examples where the answer missed the reference or retrieved context.",
    "System Metrics": "Check local service status and the latest process-level benchmark signals.",
  };
  return descriptions[section];
}

function Overview({ health, models, benchmark, evaluation, experiment, onNavigate }: {
  health: Health; models: ModelInfo[]; benchmark: BenchmarkResult | null; evaluation: EvaluationResult | null; experiment: ExperimentResult | null; onNavigate: (section: Section) => void;
}) {
  const best = benchmark?.models.reduce<BenchmarkResult["models"][number] | null>((current, candidate) => !current || candidate.token_f1 > current.token_f1 ? candidate : current, null);
  const fastest = benchmark?.models.reduce<BenchmarkResult["models"][number] | null>((current, candidate) => !current || candidate.p50_latency_ms < current.p50_latency_ms ? candidate : current, null);
  const bestP95 = benchmark?.models.reduce<BenchmarkResult["models"][number] | null>((current, candidate) => !current || candidate.p95_latency_ms < current.p95_latency_ms ? candidate : current, null);
  const fastestThroughput = benchmark?.models.reduce<BenchmarkResult["models"][number] | null>((current, candidate) => !current || candidate.tokens_per_second > current.tokens_per_second ? candidate : current, null);
  const qualityData = benchmark?.models.map((model) => ({ name: shortModel(model.model), exact: model.exact_match, f1: model.token_f1 })) || [];
  const latencyData = benchmark?.models.map((model) => ({ name: shortModel(model.model), p50: model.p50_latency_ms, p95: model.p95_latency_ms })) || [];
  return <>
    <div className="stat-grid">
      <StatCard label="MODEL PROFILES" value={models.length} detail={`${models.filter((model) => model.loaded).length} loaded in memory`} tone="mint" glyph="◎" />
      <StatCard label="BEST TOKEN F1" value={best ? formatPercent(best.token_f1) : "—"} detail={best?.model || "Run a benchmark to compare"} tone="blue" glyph="↗" />
      <StatCard label="FASTEST P50" value={fastest ? format(fastest.p50_latency_ms, " ms") : "—"} detail={fastest?.model || "Awaiting first benchmark"} tone="amber" glyph="◷" />
      <StatCard label="BEST P95" value={bestP95 ? format(bestP95.p95_latency_ms, " ms") : "—"} detail={bestP95?.model || "Awaiting first benchmark"} tone="amber" glyph="◷" />
      <StatCard label="PEAK TOKENS / SEC" value={fastestThroughput ? format(fastestThroughput.tokens_per_second) : "—"} detail={fastestThroughput?.model || "Awaiting first benchmark"} tone="mint" glyph="↗" />
      <StatCard label="LATEST EVALUATION" value={evaluation ? formatPercent(evaluation.metrics.token_f1) : "—"} detail={evaluation ? evaluation.dataset : "No evaluation saved"} tone="violet" glyph="✓" />
    </div>
    <div className="dashboard-grid overview-main-grid">
      <Panel title="Quality comparison" subtitle={benchmark ? `Latest run · ${benchmark.dataset}` : "Run a benchmark to populate this chart"} action={<button className="text-button" onClick={() => onNavigate("Benchmarks")}>Open benchmarks ↗</button>}>
        {qualityData.length ? <ChartFrame><ResponsiveContainer width="100%" height="100%"><BarChart data={qualityData} barGap={8}>
          <CartesianGrid strokeDasharray="3 3" stroke="#202b37" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#83909e", fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 1]} tick={{ fill: "#83909e", fontSize: 11 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} /><Bar dataKey="exact" name="Exact match" fill="#b5ef70" radius={[5, 5, 0, 0]} /><Bar dataKey="f1" name="Token F1" fill="#6ea8fe" radius={[5, 5, 0, 0]} />
        </BarChart></ResponsiveContainer></ChartFrame> : <EmptyChart icon="▥" title="No benchmark data yet" detail="Compare two configured model profiles to see quality signals here." />}
      </Panel>
      <Panel title="Latency profile" subtitle="P50 and P95 · milliseconds" action={<span className="mini-badge">{benchmark ? `${benchmark.models.length} models` : "NO RUNS"}</span>}>
        {latencyData.length ? <ChartFrame><ResponsiveContainer width="100%" height="100%"><AreaChart data={latencyData}>
          <defs><linearGradient id="p95fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#7e84ff" stopOpacity={0.22} /><stop offset="100%" stopColor="#7e84ff" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="#202b37" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#83909e", fontSize: 11 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#83909e", fontSize: 11 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} /><Area type="monotone" dataKey="p95" name="P95 latency" stroke="#8f8aff" fill="url(#p95fill)" strokeWidth={2} /><Area type="monotone" dataKey="p50" name="P50 latency" stroke="#50d4bd" fill="transparent" strokeWidth={2} />
        </AreaChart></ResponsiveContainer></ChartFrame> : <EmptyChart icon="◷" title="Latency appears after a run" detail="Metrics are measured on this machine and workload." />}
      </Panel>
    </div>
    <div className="dashboard-grid overview-bottom-grid">
      <Panel title="Configured models" subtitle="Profiles available to the API" action={<button className="text-button" onClick={() => onNavigate("Models")}>View all ↗</button>}>
        <div className="model-list compact-model-list">{models.slice(0, 4).map((model, index) => <ModelRow key={model.name} model={model} index={index} />)}{!models.length && <InlineEmpty>Connect to the API to load profiles.</InlineEmpty>}</div>
      </Panel>
      <Panel title="Workspace pulse" subtitle="Most recent local activity">
        <div className="activity-list">
          <ActivityItem glyph="▥" title={benchmark ? `Benchmark · ${benchmark.dataset}` : "No benchmark run yet"} detail={benchmark ? `${benchmark.models.length} model profiles compared` : "Run a comparison to see results"} time={benchmark ? benchmark.created_at : "—"} tone="mint" />
          <ActivityItem glyph="✓" title={evaluation ? `Evaluation · ${evaluation.dataset}` : "No evaluation run yet"} detail={evaluation ? `${evaluation.example_count} examples scored` : "Choose a dataset to begin"} time={evaluation ? evaluation.created_at : "—"} tone="blue" />
          <ActivityItem glyph="◌" title={experiment ? experiment.config_name : "Experiment configs ready"} detail={experiment ? experiment.results_path : "Try experiment_001"} time={experiment ? experiment.created_at : "configs/experiments"} tone="violet" />
        </div>
        <div className={`service-strip ${health ? "service-ok" : "service-down"}`}><span className="pulse-dot" />{health ? `Inference API online · default ${health.model} (${health.provider})` : "Waiting for backend API"}</div>
      </Panel>
    </div>
    <div className="callout-strip"><div className="callout-symbol">↗</div><div><b>Local-first by design</b><p>Model weights and Qdrant data stay on this machine. Cloud deployment remains a later phase.</p></div><span className="callout-tag">PHASE 01 — LOCAL</span></div>
  </>;
}

function ModelsPage({ models, onSelect }: { models: ModelInfo[]; onSelect: (name: string) => void }) {
  return <>
    <div className="model-summary-banner"><div><span className="eyebrow">MODEL REGISTRY</span><h2>{models.length} configured profiles</h2><p>Transformers profiles load lazily on first inference. The development stub never downloads weights.</p></div><div className="summary-emblem">◎</div></div>
    <div className="model-cards-grid">{models.map((model, index) => <article className="model-card" key={model.name}>
      <div className="model-card-head"><div className={`model-avatar avatar-${index % 4}`}>{model.name.slice(0, 1).toUpperCase()}</div><span className={`state-pill ${model.loaded ? "state-loaded" : "state-idle"}`}><i />{model.loaded ? "LOADED" : "ON DEMAND"}</span></div>
      <span className="model-provider">{model.provider === "mock" ? "DEVELOPMENT PROVIDER" : model.provider === "vllm" ? "OPENAI-COMPATIBLE · vLLM" : "HUGGING FACE · TRANSFORMERS"}</span><h3>{model.name}</h3><div className="model-id">{model.model_id}</div>
      <div className="model-specs"><span><small>PROVIDER</small><b>{model.provider}</b></span><span><small>STATUS</small><b>{model.loaded ? "Ready" : "Not loaded"}</b></span></div>
      <button className="button button-secondary button-full" onClick={() => onSelect(model.name)}>Use this model <span>↗</span></button>
    </article>)}</div>
    {models.some((model) => model.provider === "mock") && <div className="info-banner"><span>i</span><p><b>Development stub detected.</b> Its generated text is only an API placeholder and must not be treated as an LLM result in quality comparisons.</p></div>}
    {!models.length && <EmptyPanel title="Model registry unavailable" detail="Start the backend to read configured profiles." />}
  </>;
}

function InferencePage({ models, selectedModel, setSelectedModel, prompt, setPrompt, answer, meta, busy, onSubmit }: {
  models: ModelInfo[]; selectedModel: string; setSelectedModel: (value: string) => void; prompt: string; setPrompt: (value: string) => void; answer: string; meta: { latency_ms?: number; tokens_per_second?: number } | null; busy: boolean; onSubmit: (event: FormEvent) => void;
}) {
  return <div className="inference-layout">
    <Panel title="Inference request" subtitle="POST /api/v1/generate · SSE streaming" className="request-panel">
      <form onSubmit={onSubmit} className="form-stack">
        <Field label="MODEL PROFILE"><select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>{models.map((model) => <option key={model.name} value={model.name}>{model.name} · {model.provider}</option>)}</select></Field>
        <div className="parameter-row"><Field label="TEMPERATURE"><input name="temperature" type="number" min="0" max="2" step="0.1" defaultValue="0.2" /></Field><Field label="MAX TOKENS"><input name="max_tokens" type="number" min="1" max="8192" defaultValue="512" /></Field><Field label="TOP P"><input name="top_p" type="number" min="0.01" max="1" step="0.01" defaultValue="0.95" /></Field></div>
        <Field label="RANDOM SEED · OPTIONAL"><input name="seed" type="number" min="0" max="2147483647" placeholder="Leave blank for provider default" /></Field>
        <Field label="PROMPT"><textarea rows={9} value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Write a prompt…" /></Field>
        <div className="form-footer"><span className="subtle-note">Streaming enabled · request ID included</span><button className="button button-primary" disabled={busy || !models.length}>{busy ? <><span className="spinner" /> Generating…</> : <>Run inference <span>↗</span></>}</button></div>
      </form>
    </Panel>
    <Panel title="Model response" subtitle={meta ? `Completed in ${format(meta.latency_ms, " ms")}` : "Streamed output will appear here"} className="response-panel" action={<span className="mini-badge">{meta ? `${format(meta.tokens_per_second, " tok/s")}` : "WAITING"}</span>}>
      <div className={`response-body ${answer ? "has-answer" : ""}`}>{answer || <div className="empty-response"><div>⌁</div><b>Ready for a prompt</b><span>Output streams into this panel as tokens arrive.</span></div>}</div>
      {meta && <div className="response-metadata"><span><small>LATENCY</small><b>{format(meta.latency_ms, " ms")}</b></span><span><small>THROUGHPUT</small><b>{format(meta.tokens_per_second, " tokens/s")}</b></span><span><small>TRANSPORT</small><b>SSE stream</b></span></div>}
    </Panel>
  </div>;
}

function BenchmarkPage({ models, datasets, dataset, setDataset, seed, setSeed, limit, setLimit, repetitions, setRepetitions, maxTokens, setMaxTokens, selectedModels, setSelectedModels, mode, setMode, strategy, setStrategy, topK, setTopK, result, history, loadingRun, onSelectRun, busy, warmingModel, onWarmup, onSubmit }: {
  models: ModelInfo[]; datasets: DatasetInfo[]; dataset: string; setDataset: (v: string) => void; seed: number; setSeed: (v: number) => void; limit: number; setLimit: (v: number) => void; repetitions: number; setRepetitions: (v: number) => void; maxTokens: number; setMaxTokens: (v: number) => void; selectedModels: string[]; setSelectedModels: (v: string[]) => void; mode: "model" | "rag"; setMode: (v: "model" | "rag") => void; strategy: string; setStrategy: (v: string) => void; topK: number; setTopK: (v: number) => void; result: BenchmarkResult | null; history: RunRecord[]; loadingRun: boolean; onSelectRun: (runId: string) => void; busy: boolean; warmingModel: string | null; onWarmup: (name: string) => void; onSubmit: (event: FormEvent) => void;
}) {
  const chartData = result?.models.map((model) => ({ name: shortModel(model.model), exact: model.exact_match, f1: model.token_f1, p50: model.p50_latency_ms, p95: model.p95_latency_ms, tok: model.tokens_per_second, memory: model.peak_gpu_memory_mb })) || [];
  const benchmarkExamples = result?.models[0]?.examples.filter((example, index, rows) => rows.findIndex((row) => row.id === example.id) === index) || [];
  const toggle = (name: string) => setSelectedModels(selectedModels.includes(name) ? selectedModels.filter((item) => item !== name) : [...selectedModels, name]);
  return <>
    {history.length > 0 && <Panel title="Saved comparisons" subtitle="Load a complete result, including each model’s answer to every QA item"><div className="saved-run-picker"><Field label="BENCHMARK RUN"><select value={result?.run_id || ""} onChange={(event) => onSelectRun(event.target.value)}><option value="" disabled>Select a saved comparison</option>{history.map((item) => <option key={item.run_id} value={item.run_id}>{item.dataset || "dataset"} · {new Date(item.created_at).toLocaleString()} · {item.run_id?.slice(0, 8)}</option>)}</select></Field>{loadingRun && <span className="subtle-note"><span className="spinner" /> Loading saved comparison…</span>}</div></Panel>}
    <div className="dashboard-grid benchmark-config-grid">
      <Panel title="Benchmark workload" subtitle="Same dataset and settings across every selected model">
        <form onSubmit={onSubmit} className="form-stack">
          <Field label="EVALUATION DATASET"><select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.examples} examples</option>)}</select></Field>
          <div className="parameter-row"><Field label="EXAMPLES"><input type="number" min="1" max="100" value={limit} onChange={(event) => setLimit(Number(event.target.value))} /></Field><Field label="MAX NEW TOKENS"><input type="number" min="1" max="8192" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} /></Field><Field label="REPETITIONS"><input type="number" min="1" max="10" value={repetitions} onChange={(event) => setRepetitions(Number(event.target.value))} /></Field></div>
          <Field label="REPRODUCIBILITY SEED"><input type="number" min="0" max="2147483647" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field>
          <Field label="WORKLOAD MODE"><div className="segmented-control"><button type="button" className={mode === "model" ? "selected" : ""} onClick={() => setMode("model")}>Model QA</button><button type="button" className={mode === "rag" ? "selected" : ""} onClick={() => setMode("rag")}>RAG app</button></div></Field>
          {mode === "rag" && <div className="parameter-row"><Field label="RETRIEVAL"><select value={strategy} onChange={(event) => setStrategy(event.target.value)}><option value="dense">Dense</option><option value="bm25">BM25</option><option value="hybrid">Hybrid</option><option value="hybrid_rerank">Hybrid + reranker</option></select></Field><Field label="TOP K"><input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></Field></div>}
          <div className="field-label">MODEL PROFILES <span>Select at least two</span></div>
          <div className="profile-select-list">{models.map((model) => <div className="profile-option-row" key={model.name}><label className={`profile-option ${selectedModels.includes(model.name) ? "checked" : ""}`}><input type="checkbox" checked={selectedModels.includes(model.name)} onChange={() => toggle(model.name)} /><span className="check-box">✓</span><span className="profile-copy"><b>{model.name}</b><small>{model.provider === "mock" ? "development stub · not a real benchmark" : `${model.provider} · ${model.model_id}`}</small><small className={model.loaded ? "model-ready" : "model-idle"}>{model.loaded ? "READY · loaded in memory" : model.provider === "vllm" ? "CHECK · external server readiness" : "COLD · first inference loads weights"}</small></span></label>{(model.provider === "transformers" || model.provider === "vllm") && <button type="button" className="button button-secondary warmup-button" disabled={busy || warmingModel !== null || model.loaded} onClick={() => onWarmup(model.name)}>{warmingModel === model.name ? <><span className="spinner" /> Checking…</> : model.loaded ? "Ready" : model.provider === "vllm" ? "Check readiness" : "Warm up"}</button>}</div>)}</div>
          <div className="notice-box">Cold model loading is measured separately; inference latency is sampled after model preparation. Repetitions show variability (population standard deviation). Resource figures are process-level samples and GPU counters are optional.</div>
          <button className="button button-primary button-full" disabled={busy || selectedModels.length < 2}>{busy ? <><span className="spinner" /> Running workload…</> : <>Run comparison <span>↗</span></>}</button>
        </form>
      </Panel>
      <div className="benchmark-result-column">
        <Panel title="Quality score" subtitle={result ? `${result.dataset} · ${new Date(result.created_at).toLocaleString()}` : "Exact match and token F1"}>
          {chartData.length ? <ChartFrame><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke="#202b37" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#83909e", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 1]} tick={{ fill: "#83909e", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} /><Bar dataKey="exact" name="Exact match" fill="#b5ef70" radius={[5, 5, 0, 0]} /><Bar dataKey="f1" name="Token F1" fill="#6ea8fe" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></ChartFrame> : <EmptyChart icon="▥" title="No comparison loaded" detail="Run a workload to compare model quality." />}
        </Panel>
        <Panel title="Performance profile" subtitle="P50 / P95 latency · ms">
          {chartData.length ? <ChartFrame><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke="#202b37" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#83909e", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#83909e", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={tooltipStyle} /><Bar dataKey="p50" name="P50 latency" fill="#50d4bd" radius={[5, 5, 0, 0]} /><Bar dataKey="p95" name="P95 latency" fill="#8f8aff" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></ChartFrame> : <EmptyChart icon="◷" title="Performance data appears here" detail="Latency is measured on this machine." />}
        </Panel>
      </div>
    </div>
    {result && <Panel title="Normalized results" subtitle={`Run ${result.run_id.slice(0, 12)} · saved to ${result.results_path}`}><div className="table-scroll"><table className="data-table"><thead><tr><th>MODEL / RUNTIME</th><th>EXACT</th><th>TOKEN F1</th><th>ROUGE-L</th><th>COLD LOAD</th><th>REPEATS</th><th>LATENCY STDDEV</th><th>P50</th><th>P95</th><th>TTFT P50</th><th>TOKENS / SEC</th><th>REQUESTS / SEC</th><th>RAM MB</th><th>CPU %</th><th>GPU MB</th><th>EST. RUN COST</th><th>COST / 1M TOKENS</th></tr></thead><tbody>{result.models.map((item) => <tr key={item.model}><td className="model-cell">{item.model}<small className="model-runtime-note">{item.provider || "provider not recorded"} · {item.model_id || "model ID unavailable"} · {item.loaded_before_run == null ? "readiness unknown" : item.loaded_before_run ? "already loaded" : "cold load"} · device {item.device || "unknown"}</small></td><td>{format(item.exact_match)}</td><td>{format(item.token_f1)}</td><td>{format(item.examples.reduce((sum, example) => sum + (example.metrics.rouge_l || 0), 0) / (item.examples.length || 1))}</td><td>{format(item.cold_load_ms, " ms")}</td><td>{item.repetitions ?? 1}</td><td>{format(item.latency_stddev_ms, " ms")}</td><td>{format(item.p50_latency_ms, " ms")}</td><td>{format(item.p95_latency_ms, " ms")}</td><td>{format(item.p50_ttft_ms, " ms")}</td><td>{format(item.tokens_per_second)}</td><td>{format(item.requests_per_second)}</td><td>{format(item.peak_rss_mb)}</td><td>{format(item.peak_cpu_percent)}</td><td>{format(item.peak_gpu_memory_mb)}</td><td>{format(item.estimated_cost)}</td><td>{format(item.configured_cost_per_1m_tokens)}</td></tr>)}</tbody></table></div><div className="subtle-note table-footnote">{result.mode === "rag" ? `RAG workload · ${result.retrieval_strategy}` : "Model-level workload"} · seed {String(result.config.seed ?? "provider default")} · dataset SHA-256 {result.dataset_sha256 || "not recorded"} · process memory: sampled RSS · cost: configured output-token estimate only</div></Panel>}
    {result && <Panel title="Answer-by-answer comparison" subtitle={`${result.models[0]?.example_count || 0} shared QA items · reference and each model response`}>
      <div className="benchmark-answer-list">{benchmarkExamples.map((example) => <article className="benchmark-answer-card" key={example.id}>
        <div className="benchmark-question"><span>{example.id}</span><b>{example.question}</b><p><strong>Reference:</strong> {example.ground_truth}</p></div>
        <div className="benchmark-answer-grid" style={{ gridTemplateColumns: `repeat(${Math.max(result.models.length, 1)}, minmax(0, 1fr))` }}>
          {result.models.map((model) => {
            const answer = model.examples.find((item) => item.id === example.id && (item.repetition || 1) === (example.repetition || 1));
            return <div className="benchmark-model-answer" key={model.model}><div className="benchmark-model-heading"><b>{model.model}</b><span>F1 {format(answer?.metrics.token_f1)} · {format(answer?.latency_ms, " ms")}</span></div><p>{answer?.answer || "No answer recorded for this item."}</p>{answer?.failure_category && <span className="failure-tag">{answer.failure_category}</span>}</div>;
          })}
        </div>
      </article>)}</div>
    </Panel>}
  </>;
}

function RagPage({ models, embeddingModels, embeddingModel, setEmbeddingModel, selectedModel, setSelectedModel, documentId, documentLabel, question, setQuestion, strategy, setStrategy, topK, setTopK, answer, comparison, busy, onUpload, onQuery, onCompare }: {
  models: ModelInfo[]; embeddingModels: EmbeddingModelInfo[]; embeddingModel: string; setEmbeddingModel: (v: string) => void; selectedModel: string; setSelectedModel: (v: string) => void; documentId: string; documentLabel: string; question: string; setQuestion: (v: string) => void; strategy: string; setStrategy: (v: string) => void; topK: number; setTopK: (v: number) => void; answer: RagQueryResult | null; comparison: RetrievalComparison | null; busy: string | null; onUpload: (event: FormEvent<HTMLFormElement>) => void; onQuery: (event: FormEvent) => void; onCompare: () => void;
}) {
  return <div className="rag-workspace">
    <div className="rag-steps"><div className="rag-step current"><span>01</span><b>Ingest</b><small>PDF or text</small></div><i /><div className={`rag-step ${documentId ? "complete" : ""}`}><span>02</span><b>Retrieve</b><small>Dense / BM25 / hybrid</small></div><i /><div className={`rag-step ${answer ? "complete" : ""}`}><span>03</span><b>Generate</b><small>Grounded response</small></div></div>
    <div className="dashboard-grid rag-grid">
      <div className="rag-left-column">
        <Panel title="Knowledge source" subtitle="Files are chunked and stored in local Qdrant">
          <form onSubmit={onUpload} className="upload-form"><label className="upload-zone"><input name="document" type="file" accept=".pdf,.txt,.md" required /><span className="upload-icon">↑</span><b>Choose a document</b><small>PDF, TXT, or Markdown · up to 10 MB</small></label><button className="button button-secondary button-full" disabled={busy === "upload"}>{busy === "upload" ? "Indexing…" : "Upload and index"}</button></form>
          {documentId ? <div className="document-row"><div className="file-icon">▤</div><div><b>{documentLabel}</b><small>ID {documentId.slice(0, 16)}…</small></div><span className="state-pill state-loaded">INDEXED</span></div> : <div className="empty-inline">No document selected. Upload a source file to scope queries.</div>}
        </Panel>
        <Panel title="Retrieval configuration" subtitle="Compare search strategies on the same question">
          <div className="form-stack"><Field label="RETRIEVAL STRATEGY"><select value={strategy} onChange={(event) => setStrategy(event.target.value)}><option value="dense">Dense vector search</option><option value="bm25">BM25 keyword search</option><option value="hybrid">Hybrid · reciprocal rank fusion</option><option value="hybrid_rerank">Hybrid + CrossEncoder rerank</option></select></Field><Field label="EMBEDDING PROFILE"><select value={embeddingModel} onChange={(event) => setEmbeddingModel(event.target.value)}>{embeddingModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.name} · {model.dimension}d</option>)}</select></Field><div className="parameter-row"><Field label="TOP K"><input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></Field><Field label="GENERATION MODEL"><select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>{models.map((model) => <option key={model.name}>{model.name}</option>)}</select></Field></div><div className="subtle-note">Choose the same embedding profile used during document ingestion; each profile has a separate Qdrant collection.</div></div>
        </Panel>
      </div>
      <div className="rag-right-column">
        <Panel title="Ask your documents" subtitle="Answers include the retrieved source chunks">
          <form onSubmit={onQuery} className="form-stack"><Field label="QUESTION"><textarea rows={4} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask something grounded in your document…" required /></Field><div className="form-footer"><span className="subtle-note">{documentId ? "Scoped to selected document" : "Searching the full local collection"}</span><button type="button" className="button button-secondary" onClick={onCompare} disabled={busy === "compare" || !question}>{busy === "compare" ? <><span className="spinner" /> Comparing…</> : <>Compare all 4 strategies</>}</button><button className="button button-primary" disabled={busy === "rag" || !question}>{busy === "rag" ? <><span className="spinner" /> Retrieving…</> : <>Run RAG query <span>↗</span></>}</button></div></form>
        </Panel>
        <Panel title="Answer and evidence" subtitle={answer ? `Retrieval · ${answer.retrieval_strategy} · ${format(answer.retrieval_latency_ms, " ms")}` : "Grounded answer will appear here"}>
          {answer ? <><div className="answer-box">{answer.generation.response}</div><div className="sources-heading"><b>Retrieved sources</b><span>{answer.sources.length} passages</span></div><div className="source-list">{answer.sources.map((source: RetrievedChunk, index: number) => <div className="source-card" key={source.chunk_id}><div className="source-head"><span className="source-index">0{index + 1}</span><b>{source.filename}{source.page_number ? ` · page ${source.page_number}` : ""}</b><span className="score-pill">{format(source.score, " score")}</span></div><p>{source.text}</p></div>)}{!answer.sources.length && <div className="empty-inline">No chunks matched this question. Try another strategy or ingest a source.</div>}</div></> : <EmptyChart icon="⌘" title="Evidence comes first" detail="The answer will be shown with its retrieved passages and scores." />}
        </Panel>
        {comparison && <Panel title="Retrieval strategy comparison" subtitle={`Same query · ${comparison.query} · top K ${topK}`}>
          <div className="source-list">{comparison.comparisons.map((item) => { const score = item.results[0]?.score; return <article className="source-card" key={item.strategy}><div className="source-head"><span className="source-index">{item.strategy === "dense" ? "01" : item.strategy === "bm25" ? "02" : item.strategy === "hybrid" ? "03" : "04"}</span><b>{item.strategy.replaceAll("_", " ").toUpperCase()}</b><span className="score-pill">{format(item.latency_ms, " ms")}</span>{score != null && <span className="mini-badge">top score {format(score)}</span>}</div>{item.error ? <p>{item.error} · install the reranking extra and retry.</p> : item.results[0] ? <p>{item.results[0].filename}{item.results[0].page_number ? ` · page ${item.results[0].page_number}` : ""}: {item.results[0].text}</p> : <p>No passages returned for this strategy.</p>}</article>; })}</div>
        </Panel>}
      </div>
    </div>
  </div>;
}

function EvaluationPage({ datasets, embeddingModels, semanticEmbeddingModel, setSemanticEmbeddingModel, semanticMetrics, setSemanticMetrics, dataset, setDataset, seed, setSeed, mode, setMode, strategy, setStrategy, result, busy, reviewingId, onReview, onUploadDataset, onSubmit }: {
  datasets: DatasetInfo[]; embeddingModels: EmbeddingModelInfo[]; semanticEmbeddingModel: string; setSemanticEmbeddingModel: (v: string) => void; semanticMetrics: boolean; setSemanticMetrics: (v: boolean) => void; dataset: string; setDataset: (v: string) => void; seed: number; setSeed: (v: number) => void; mode: "model" | "rag"; setMode: (v: "model" | "rag") => void; strategy: string; setStrategy: (v: string) => void; result: EvaluationResult | null; busy: boolean; reviewingId: string | null; onReview: (id: string, label: "correct" | "partially_correct" | "incorrect", notes: string) => void; onUploadDataset: (file: File) => void; onSubmit: (event: FormEvent) => void;
}) {
  const data = result ? Object.entries(result.metrics).map(([name, value]) => ({ name: name.replaceAll("_", " "), value: value ?? 0 })) : [];
  const reviewed = result?.results.filter((item) => item.human_review).length || 0;
  const reviewedCounts = result?.results.reduce((counts, item) => {
    const label = item.human_review?.label;
    if (label) counts[label] += 1;
    return counts;
  }, { correct: 0, partially_correct: 0, incorrect: 0 }) || { correct: 0, partially_correct: 0, incorrect: 0 };
  return <div className="dashboard-grid benchmark-config-grid">
    <Panel title="Evaluation run" subtitle="Lexical baselines plus your human review labels">
      <form onSubmit={onSubmit} className="form-stack"><Field label="DATASET"><select value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.map((item) => <option key={item.name} value={item.name}>{item.name} · {item.examples} examples</option>)}</select></Field><Field label="ADD YOUR OWN QA DATASET"><input type="file" accept=".jsonl,application/jsonl" disabled={busy} onChange={(event) => { const file = event.target.files?.[0]; if (file) onUploadDataset(file); event.currentTarget.value = ""; }} /></Field><div className="subtle-note">JSONL rows: id, question, ground_truth; optional contexts array. Max 5 MB / 500 questions.</div><Field label="REPRODUCIBILITY SEED"><input type="number" min="0" max="2147483647" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field><Field label="EVALUATION MODE"><div className="segmented-control"><button type="button" className={mode === "model" ? "selected" : ""} onClick={() => setMode("model")}>Model + reference context</button><button type="button" className={mode === "rag" ? "selected" : ""} onClick={() => setMode("rag")}>Full RAG</button></div></Field>{mode === "rag" && <Field label="RETRIEVAL STRATEGY"><select value={strategy} onChange={(event) => setStrategy(event.target.value)}><option value="dense">Dense</option><option value="bm25">BM25</option><option value="hybrid">Hybrid</option><option value="hybrid_rerank">Hybrid + reranker</option></select></Field>}<label className="subtle-note" style={{display:"flex",gap:8,alignItems:"center"}}><input type="checkbox" checked={semanticMetrics} onChange={(event) => setSemanticMetrics(event.target.checked)} disabled={!embeddingModels.some((model) => model.provider === "sentence-transformers")} /> Include embedding-based semantic scores</label>{semanticMetrics && <><Field label="SEMANTIC EMBEDDING PROFILE"><select value={semanticEmbeddingModel} onChange={(event) => setSemanticEmbeddingModel(event.target.value)}>{embeddingModels.filter((model) => model.provider === "sentence-transformers").map((model) => <option key={model.model_id} value={model.model_id}>{model.name} · {model.dimension}d</option>)}</select></Field><div className="subtle-note">Loads the selected local embedding model. Similarity scores are semantic proxies; they do not prove entailment.</div></>}<div className="notice-box">Exact match, token F1, ROUGE-L, and context metrics are overlap-based baselines, not semantic quality judgments. Review answers manually below.</div><button className="button button-primary button-full" disabled={busy}>{busy ? <><span className="spinner" /> Evaluating…</> : <>Run evaluation <span>↗</span></>}</button></form>
    </Panel>
    <div className="evaluation-summary"><Panel title="Aggregate scores" subtitle={result ? `${result.example_count} examples · ${result.model}` : "Run an evaluation to see scores"}>{data.length ? <div className="metric-bar-list">{data.map((item) => <div className="metric-bar" key={item.name}><div><span>{item.name}</span><b>{format(item.value)}</b></div><div className="bar-track"><i style={{ width: `${Math.max(0, Math.min(100, item.value * 100))}%` }} /></div></div>)}<div className="subtle-note">Seed {String(result?.config?.seed ?? "provider default")} · dataset SHA-256 {result?.dataset_sha256 || "not recorded"} · saved to {result?.results_path}</div></div> : <EmptyChart icon="✓" title="No scores yet" detail="Exact match, token F1, context recall, and retrieval hit will appear here." />}</Panel></div>
    {result && <Panel title="Human review" subtitle={`${reviewed} of ${result.example_count} reviewed · ${reviewedCounts.correct} correct · ${reviewedCounts.partially_correct} partial · ${reviewedCounts.incorrect} incorrect`}>
      <div className="notice-box">Review the model answer against the question and reference. Labels are saved with this run; they are not generated by the model.</div>
      <div className="review-list">{result.results.map((item) => <EvaluationReviewCard key={item.id} item={item} busy={reviewingId === item.id} onSave={(label, notes) => onReview(item.id, label, notes)} />)}</div>
    </Panel>}
  </div>;
}

function EvaluationReviewCard({ item, busy, onSave }: { item: EvaluationResult["results"][number]; busy: boolean; onSave: (label: "correct" | "partially_correct" | "incorrect", notes: string) => void }) {
  const [label, setLabel] = useState<"" | "correct" | "partially_correct" | "incorrect">(item.human_review?.label || "");
  const [notes, setNotes] = useState(item.human_review?.notes || "");
  return <article className="review-card">
    <div className="review-copy"><small>{item.id} · {item.latency_ms.toFixed(0)} ms · token F1 {format(item.metrics.token_f1)}</small><b>{item.question}</b><p><strong>Reference:</strong> {item.ground_truth}</p><p><strong>Model answer:</strong> {item.answer || "(empty answer)"}</p></div>
    <div className="review-controls"><Field label="HUMAN LABEL"><select value={label} onChange={(event) => setLabel(event.target.value as typeof label)}><option value="" disabled>Select a label</option><option value="correct">Correct</option><option value="partially_correct">Partially correct</option><option value="incorrect">Incorrect</option></select></Field><Field label="REVIEW NOTES"><input value={notes} maxLength={2000} onChange={(event) => setNotes(event.target.value)} placeholder="Optional reason or correction" /></Field><button className="button button-secondary" disabled={busy || !label} onClick={() => { if (label) onSave(label, notes); }}>{busy ? "Saving…" : item.human_review ? "Update review" : "Save review"}</button>{item.human_review && <span className="subtle-note">Reviewed {new Date(item.human_review.reviewed_at).toLocaleString()}</span>}</div>
  </article>;
}

function ExperimentsPage({ config, setConfig, result, history, busy, onSubmit }: { config: string; setConfig: (v: string) => void; result: ExperimentResult | null; history: RunRecord[]; busy: boolean; onSubmit: (event: FormEvent) => void }) {
  return <div className="dashboard-grid experiment-grid">
    <Panel title="Replay a saved configuration" subtitle="YAML source · configs/experiments">
      <form onSubmit={onSubmit} className="form-stack"><Field label="EXPERIMENT CONFIG NAME"><input value={config} onChange={(event) => setConfig(event.target.value)} placeholder="experiment_001" required /></Field><div className="config-preview"><div className="code-top"><span><i /> {config || "experiment"}.yaml</span><span>YAML</span></div><pre>{`experiment:\n  dataset: general_qa\n  mode: model\n  model: local-development\n  temperature: 0\n  max_tokens: 128`}</pre></div><div className="notice-box">The API saves a resolved config snapshot with every result so the run can be reproduced.</div><button className="button button-primary" disabled={busy}>{busy ? <><span className="spinner" /> Running…</> : <>Run experiment <span>↗</span></>}</button></form>
    </Panel>
    <Panel title="Latest experiment" subtitle={result ? `${result.config_name} · ${new Date(result.created_at).toLocaleString()}` : "No experiment executed in this browser yet"}>
      {result ? <><div className="experiment-result-banner"><span className="experiment-check">✓</span><div><b>Experiment completed</b><small>{result.experiment_id}</small></div></div><div className="result-detail-grid"><Detail label="DATASET" value={result.evaluation.dataset} /><Detail label="MODEL" value={result.evaluation.model} /><Detail label="EXAMPLES" value={String(result.evaluation.example_count)} /><Detail label="TOKEN F1" value={format(result.evaluation.metrics.token_f1)} /></div><div className="path-card"><small>RESULT FILE</small><code>{result.results_path}</code></div></> : <EmptyChart icon="◌" title="Configuration is ready" detail="Start the API, then replay experiment_001." />}
    </Panel>
    <Panel title="Saved run history" subtitle="Latest records persisted by the API across browser sessions"><div className="activity-list">{history.map((item, index) => <ActivityItem key={`${item.kind}-${item.run_id || item.experiment_id || index}`} glyph={item.kind === "benchmark" ? "▥" : item.kind === "experiment" ? "◌" : "✓"} title={`${item.kind || "run"} · ${item.config_name || item.dataset || item.run_id || "record"}`} detail={`${item.dataset || "dataset unavailable"}${item.model ? ` · ${item.model}` : ""}`} time={item.created_at} tone={item.kind === "benchmark" ? "mint" : item.kind === "experiment" ? "violet" : "blue"} />)}{!history.length && <InlineEmpty>No persisted runs are available yet.</InlineEmpty>}</div></Panel>
  </div>;
}

function FailuresPage({ failures }: { failures: FailureExample[] }) {
  return <Panel title="Examples to inspect" subtitle="Persisted failure triage from evaluation and benchmark runs · heuristic labels require human review">
    {failures.length ? <div className="failure-list">{failures.map((item, index) => <article className="failure-card" key={`${item.run_id}-${item.id}-${index}`}><div className="failure-top"><span className="failure-tag">{(item.failure_category || "answer mismatch").toUpperCase()}</span><span className="mini-badge">{item.model || item.id}</span></div><div className="failure-columns"><div><small>QUESTION</small><p>{item.question}</p></div><div><small>EXPECTED</small><p>{item.ground_truth}</p></div><div><small>ACTUAL</small><p>{item.answer}</p></div></div>{item.prompt && <details className="failure-context"><summary>INSPECT PROMPT · RUN {item.run_id.slice(0, 10)}</summary><p>{item.prompt}</p></details>}{item.sources.length > 0 && <div className="failure-context"><small>RETRIEVED CONTEXT · SCORES</small><p>{item.sources.map((source) => `${source.filename} (${source.score.toFixed(3)}): ${source.text}`).join(" · ")}</p></div>}</article>)}</div> : <EmptyChart icon="✓" title="No saved failures yet" detail="Run an evaluation or benchmark to collect examples for inspection." />}
  </Panel>;
}

function SystemPage({ health, models, benchmark, metrics, apiKey, setApiKey, onSaveApiKey, apiBaseUrl, setApiBaseUrl, onSaveApiBaseUrl }: { health: Health; models: ModelInfo[]; benchmark: BenchmarkResult | null; metrics: SystemMetrics | null; apiKey: string; setApiKey: (value: string) => void; onSaveApiKey: (value: string) => void; apiBaseUrl: string; setApiBaseUrl: (value: string) => void; onSaveApiBaseUrl: (value: string) => void }) {
  const [apiUrlError, setApiUrlError] = useState("");
  function saveApiUrl(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setApiUrlError("");
    try { onSaveApiBaseUrl(apiBaseUrl); } catch (cause) { setApiUrlError(cause instanceof Error ? cause.message : "Invalid API URL"); }
  }
  return <><div className="stat-grid"><StatCard label="BACKEND API" value={health ? "ONLINE" : "OFFLINE"} detail={health ? "FastAPI local service" : "Start the configured Model Bench API"} tone={health ? "mint" : "amber"} glyph="⌁" /><StatCard label="MODEL PROFILES" value={models.length} detail="Configured in local YAML" tone="blue" glyph="◎" /><StatCard label="GPU UTILIZATION" value={metrics?.gpu_utilization_percent == null ? "N/A" : format(metrics.gpu_utilization_percent, "%")} detail={metrics?.gpu_memory_used_mb == null ? "NVML unavailable / CPU mode" : `${format(metrics.gpu_memory_used_mb)} / ${format(metrics.gpu_memory_total_mb)} MB`} tone="violet" glyph="▥" /><StatCard label="OBSERVABILITY" value="ACTIVE" detail="Prometheus scrape endpoint · /metrics" tone="mint" glyph="⌗" /></div><div className="dashboard-grid system-grid"><Panel title="Live process snapshot" subtitle="Measured on this API process"><div className="service-detail-list"><Detail label="API STATUS" value={health?.status || "Unavailable"} /><Detail label="ACTIVE PROVIDER" value={health?.provider || "—"} /><Detail label="DEFAULT PROFILE" value={health?.model || "—"} /><Detail label="PROCESS RAM" value={metrics ? `${format(metrics.process_rss_mb)} MB` : "—"} /><Detail label="CPU UTILIZATION" value={metrics ? `${format(metrics.process_cpu_percent)}%` : "—"} /><Detail label="LATEST RUN GPU MEMORY" value={benchmark ? format(Math.max(...benchmark.models.map((item) => item.peak_gpu_memory_mb || 0)), " MB") : "—"} /></div></Panel><div className="rag-left-column"><Panel title="Connect an inference API" subtitle="This setting is kept in this browser tab"><form className="form-stack" onSubmit={saveApiUrl}><Field label="API BASE URL"><input type="url" required value={apiBaseUrl} onChange={(event) => setApiBaseUrl(event.target.value)} placeholder="http://127.0.0.1:8001" /></Field><div className="subtle-note">Use the URL where the API serves /api/v1/health. For example, the local Transformers API can run at http://127.0.0.1:8001.</div>{apiUrlError && <div className="notice-box">{apiUrlError}</div>}<div className="form-footer"><span className="subtle-note">Changing the endpoint reloads this tab.</span><button className="button button-secondary">Connect API</button></div></form></Panel><Panel title="Metrics & dashboards" subtitle="Prometheus and Grafana are included in Docker Compose"><div className="activity-list"><ActivityItem glyph="⌗" title="Prometheus scrape" detail="http://localhost:9090 · API endpoint /metrics" time="AVAILABLE" tone="mint" /><ActivityItem glyph="▥" title="Grafana dashboard" detail="http://localhost:3001 · provisioned Model Bench panels" time="AVAILABLE" tone="blue" /><ActivityItem glyph="◷" title="Tracked metrics" detail="HTTP count/latency, active requests, inference, retrieval, embeddings, process RAM/CPU, optional GPU" time="LIVE" tone="violet" /></div></Panel><Panel title="Optional API authentication" subtitle="Key stays in this browser tab's session storage"><form className="form-stack" onSubmit={(event) => { event.preventDefault(); onSaveApiKey(apiKey); }}><Field label="BEARER KEY"><input type="password" autoComplete="off" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Set MODELBENCH_API_KEY from the backend environment" /></Field><div className="form-footer"><span className="subtle-note">Leave blank to clear the saved key.</span><button className="button button-secondary">Save key</button></div></form></Panel></div></div></>;
}

function StatCard({ label, value, detail, tone, glyph }: { label: string; value: string | number; detail: string; tone: string; glyph: string }) {
  return <article className="stat-card"><div className="stat-card-top"><span>{label}</span><i className={`stat-glyph ${tone}`}>{glyph}</i></div><strong>{value}</strong><small>{detail}</small><div className={`stat-accent ${tone}`} /></article>;
}

function Panel({ title, subtitle, action, className = "", children }: { title: string; subtitle?: string; action?: ReactNode; className?: string; children: ReactNode }) {
  return <section className={`panel ${className}`}><div className="panel-head"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return <label className="field"><span className="field-label">{label}</span>{children}</label>;
}

function ModelRow({ model, index }: { model: ModelInfo; index: number }) {
  return <div className="model-row"><div className={`model-avatar small-avatar avatar-${index % 4}`}>{model.name.slice(0, 1).toUpperCase()}</div><div className="model-row-copy"><b>{model.name}</b><small>{model.provider} · {model.model_id}</small></div><span className={`tiny-state ${model.loaded ? "is-loaded" : ""}`}><i />{model.loaded ? "loaded" : "idle"}</span></div>;
}

function ActivityItem({ glyph, title, detail, time, tone }: { glyph: string; title: string; detail: string; time: string; tone: string }) {
  return <div className="activity-item"><span className={`activity-glyph ${tone}`}>{glyph}</span><div className="activity-copy"><b>{title}</b><small>{detail}</small></div><time>{time === "—" || time.includes("/") || time === "AVAILABLE" || time === "NEXT PHASE" ? time : new Date(time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time></div>;
}

function ChartFrame({ children }: { children: ReactNode }) { return <div className="chart-frame">{children}</div>; }
function EmptyChart({ icon, title, detail }: { icon: string; title: string; detail: string }) { return <div className="empty-chart"><div className="empty-chart-icon">{icon}</div><b>{title}</b><span>{detail}</span></div>; }
function EmptyPanel({ title, detail }: { title: string; detail: string }) { return <div className="empty-panel"><b>{title}</b><span>{detail}</span></div>; }
function InlineEmpty({ children }: { children: ReactNode }) { return <div className="empty-inline">{children}</div>; }
function Detail({ label, value }: { label: string; value: string }) { return <div className="detail-item"><small>{label}</small><b>{value}</b></div>; }
const tooltipStyle = { background: "#101923", border: "1px solid #2b3745", borderRadius: 10, color: "#ecf1f5", fontSize: 12 };
