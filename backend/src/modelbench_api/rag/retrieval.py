import asyncio
import math
import re
from collections import Counter
from typing import Any


def _terms(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_search(query: str, documents: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Small-corpus BM25 implementation (Okapi defaults k1=1.5, b=0.75)."""
    tokenized = [_terms(str(doc.get("chunk_text", ""))) for doc in documents]
    query_terms = set(_terms(query))
    if not tokenized or not query_terms:
        return []
    frequencies = [Counter(tokens) for tokens in tokenized]
    doc_count = len(tokenized)
    average_length = sum(map(len, tokenized)) / doc_count or 1.0
    document_frequency = Counter(term for counts in frequencies for term in counts)
    scores: list[tuple[float, int]] = []
    for index, counts in enumerate(frequencies):
        length = len(tokenized[index])
        score = 0.0
        for term in query_terms:
            term_frequency = counts.get(term, 0)
            if not term_frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            denominator = term_frequency + 1.5 * (1 - 0.75 + 0.75 * length / average_length)
            score += inverse_frequency * term_frequency * 2.5 / denominator
        if score > 0:
            scores.append((score, index))
    scores.sort(reverse=True)
    return [{**documents[index], "score": float(score)} for score, index in scores[:top_k]]


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]], limit: int, rank_constant: int = 60
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = str(result.get("chunk_id") or result.get("id"))
            current = fused.setdefault(key, {**result, "score": 0.0})
            current["score"] += 1.0 / (rank_constant + rank)
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]


class CrossEncoderReranker:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Reranking dependencies are missing. Install with: "
                "python -m pip install -e 'backend[reranking]'"
            ) from exc
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CrossEncoder(self.model_id, device=device)

    def _rank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._load()
        pairs = [(query, str(candidate.get("chunk_text", ""))) for candidate in candidates]
        scores = self._model.predict(pairs)
        ranked = [
            {**candidate, "score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        return sorted(ranked, key=lambda item: item["score"], reverse=True)

    async def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._rank, query, candidates)
