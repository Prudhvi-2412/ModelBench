
import pytest
import yaml
from modelbench_api import model_registry
from modelbench_api.benchmarking.metrics import percentile
from modelbench_api.evaluation.datasets import DatasetExample, load_dataset
from modelbench_api.evaluation.metrics import (
    aggregate_metrics,
    classify_failure,
    cosine_similarity,
    evaluate_answer,
    rouge_l,
    semantic_metrics,
)
from modelbench_api.providers.mock import DevelopmentProvider
from modelbench_api.rag.chunking import chunk_text
from modelbench_api.rag.retrieval import bm25_search, reciprocal_rank_fusion
from modelbench_api.settings import ModelSettings
from pydantic import ValidationError


def test_chunking_respects_overlap_and_covers_text():
    text = " ".join(f"word{i}" for i in range(40))
    chunks = chunk_text(text, chunk_size=70, chunk_overlap=15)
    assert len(chunks) > 1
    assert all(chunk.text.strip() for chunk in chunks)
    assert len(set(chunks[0].text.split()) & set(chunks[1].text.split())) >= 2


def test_metrics_and_aggregation():
    exact = evaluate_answer("The answer is 42.", "42", ["The answer is 42."])
    assert exact["exact_match"] == 0  # strict normalized full-answer metric
    assert exact["token_f1"] > 0
    assert exact["context_recall"] == 1
    assert aggregate_metrics([exact, {"exact_match": 1.0}])["exact_match"] == 0.5
    assert rouge_l("alpha beta gamma", "alpha gamma") > 0
    assert rouge_l("alpha beta", "gamma delta") == 0


def test_semantic_metrics_use_cosine_similarity_without_claiming_entailment():
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0
    metrics = semantic_metrics("answer", "reference", ["context"], [[1, 0], [0.8, 0.6], [0, 1]])
    assert metrics["semantic_similarity"] == pytest.approx(0.8)
    assert metrics["semantic_groundedness"] == 0
    assert metrics["semantic_hallucination_risk_proxy"] == 1


def test_cpu_sampler_reuses_process_handle(monkeypatch):
    from modelbench_api import main

    class CpuSampler:
        def cpu_percent(self, interval=None):
            assert interval is None
            return 37.5

    sampler = CpuSampler()
    monkeypatch.setattr(main, "_CPU_PROCESS", sampler)

    assert main._process_cpu_percent() == 37.5
    assert main._CPU_PROCESS is sampler


def test_custom_dataset_upload_is_persisted_and_loadable(tmp_path, monkeypatch):
    from modelbench_api.evaluation import datasets

    monkeypatch.setattr(datasets, "DATASET_DIR", tmp_path / "builtins")
    monkeypatch.setattr(datasets, "CUSTOM_DATASET_DIR", tmp_path / "custom")
    count = datasets.save_custom_dataset(
        "candidate-set",
        [{"id": "q1", "question": "What?", "ground_truth": "Answer", "contexts": ["context"]}],
    )
    first_fingerprint = datasets.dataset_sha256("candidate-set")
    assert count == 1
    assert datasets.load_dataset("candidate-set")[0].ground_truth == "Answer"
    assert datasets.list_datasets() == [{"name": "candidate-set", "format": "jsonl", "examples": 1}]
    assert len(first_fingerprint) == 64
    (tmp_path / "custom" / "candidate-set.jsonl").write_text("{}\n", encoding="utf-8")
    assert datasets.dataset_sha256("candidate-set") != first_fingerprint


def test_seed_is_validated_and_retained_in_experiment_config():
    from modelbench_api.experiments.config import ExperimentConfig
    from modelbench_api.schemas import EvaluationRequest, GenerateRequest

    assert GenerateRequest(prompt="hello", seed=12).seed == 12
    assert EvaluationRequest(dataset="general_qa", seed=12).seed == 12
    assert ExperimentConfig(dataset="general_qa", seed=12).seed == 12
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="hello", seed=-1)


def test_failure_classification_is_explicitly_heuristic():
    assert classify_failure({"exact_match": 0, "token_f1": 0, "context_recall": 0}, [], "wrong", "Source passages:", 10) == "retrieval failure"
    assert classify_failure({"exact_match": 1, "token_f1": 1}, [], "right", "prompt", 10) is None


def test_bm25_and_rrf_return_ranked_payloads():
    corpus = [
        {"document_id": "d", "chunk_id": "1", "chunk_text": "vector search uses embeddings", "score": .2},
        {"document_id": "d", "chunk_id": "2", "chunk_text": "BM25 is lexical keyword search", "score": .1},
    ]
    result = bm25_search("BM25 keyword search", corpus, 2)
    assert result[0]["chunk_id"] == "2"
    fused = reciprocal_rank_fusion([result, corpus], 2)
    assert fused[0]["chunk_id"] in {"1", "2"}


def test_remote_qdrant_client_uses_configured_request_timeout(monkeypatch):
    from modelbench_api.rag import vector_store

    captured = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(vector_store, "QdrantClient", FakeQdrantClient)
    monkeypatch.setenv("MODELBENCH_QDRANT_TIMEOUT_SECONDS", "45")

    vector_store.LocalVectorStore(".modelbench/qdrant", url="http://qdrant:6333")

    assert captured == {"url": "http://qdrant:6333", "timeout": 45}


def test_dataset_schema_rejects_unexpected_fields():
    with pytest.raises(ValidationError):
        DatasetExample.model_validate({"id": "x", "question": "q", "ground_truth": "a", "oops": 1})


def test_sample_datasets_are_parseable():
    for name in ("general_qa", "rag_qa", "hallucination"):
        rows = load_dataset(name)
        assert rows and rows[0].question and rows[0].ground_truth
    assert len(load_dataset("general_qa")) >= 20


def test_percentiles_handle_empty_and_ordered_inputs():
    assert percentile([], 95) == 0
    assert percentile([10, 20, 30, 40], 50) == 25
    assert percentile([10, 20, 30, 40], 95) >= 30


def test_selected_default_profile_is_not_registered_twice(tmp_path, monkeypatch):
    config_root = tmp_path / "configs" / "models" / "profiles"
    config_root.mkdir(parents=True)
    settings = ModelSettings(
        name="smollm2-360m-instruct",
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        provider="transformers",
    )
    (config_root / "smollm2-360m.yaml").write_text(
        yaml.safe_dump({"model": settings.model_dump()}), encoding="utf-8"
    )
    monkeypatch.setattr(model_registry, "ROOT", tmp_path)

    registry = model_registry.ModelRegistry(settings, DevelopmentProvider())

    assert len(registry.all()) == 1
    assert registry.resolve(settings.name).settings == settings
    assert registry.resolve(settings.model_id).settings == settings
