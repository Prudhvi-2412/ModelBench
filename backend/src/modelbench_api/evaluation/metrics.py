import re
from collections import Counter
from math import sqrt


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for semantic embedding vectors, clamped to [0, 1]."""
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have the same non-zero dimension")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    score = sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    return max(0.0, min(1.0, score))


def semantic_metrics(
    answer: str,
    reference: str,
    contexts: list[str],
    vectors: list[list[float]],
) -> dict[str, float | None]:
    """Add embedding similarity signals; these are proxies, not entailment judgments."""
    if len(vectors) != 2 + len(contexts):
        raise ValueError("Expected answer, reference, and one vector per context")
    result: dict[str, float | None] = {
        "semantic_similarity": cosine_similarity(vectors[0], vectors[1]),
        "semantic_groundedness": None,
        "semantic_hallucination_risk_proxy": None,
    }
    if contexts:
        groundedness = max(cosine_similarity(vectors[0], vector) for vector in vectors[2:])
        result["semantic_groundedness"] = groundedness
        result["semantic_hallucination_risk_proxy"] = 1 - groundedness
    return result


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(normalize(prediction).split())
    expected = Counter(normalize(reference).split())
    overlap = sum((predicted & expected).values())
    if not predicted or not expected or not overlap:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """Token-level ROUGE-L F1; an overlap metric, not a semantic judge."""
    predicted = normalize(prediction).split()
    expected = normalize(reference).split()
    if not predicted or not expected:
        return 0.0
    token_masks: dict[str, int] = {}
    for index, token in enumerate(expected):
        token_masks[token] = token_masks.get(token, 0) | (1 << index)
    state = 0
    for token in predicted:
        matches = token_masks.get(token, 0) | state
        shifted = (state << 1) | 1
        state = matches & ~(matches - shifted)
    lcs = state.bit_count()
    if lcs == 0:
        return 0.0
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return 2 * precision * recall / (precision + recall)


def context_metrics(contexts: list[str], ground_truth: str) -> dict[str, float | None]:
    if not contexts:
        return {"context_recall": None, "context_precision": None, "retrieval_hit": None}
    expected_terms = set(normalize(ground_truth).split())
    if not expected_terms:
        return {"context_recall": None, "retrieval_hit": None}
    context_terms = [set(normalize(context).split()) for context in contexts]
    covered = set().union(*context_terms)
    recall = len(expected_terms & covered) / len(expected_terms)
    relevance = [len(expected_terms & terms) / len(expected_terms) for terms in context_terms]
    hit = any(score >= 0.5 for score in relevance)
    precision = sum(score >= 0.2 for score in relevance) / len(relevance)
    return {"context_recall": recall, "context_precision": precision, "retrieval_hit": float(hit)}


def evaluate_answer(
    prediction: str, reference: str, contexts: list[str] | None = None, question: str | None = None
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge_l": rouge_l(prediction, reference),
    }
    metrics.update(context_metrics(contexts or [], reference))
    if question:
        q_terms = set(normalize(question).split())
        a_terms = set(normalize(prediction).split())
        metrics["answer_relevance"] = len(q_terms & a_terms) / len(q_terms) if q_terms else None
    if contexts:
        answer_terms = set(normalize(prediction).split())
        evidence_terms = set(normalize(" ".join(contexts)).split())
        content_terms = answer_terms - {"the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "of", "in", "on", "for", "it", "this", "that"}
        support = len(content_terms & evidence_terms) / len(content_terms) if content_terms else 1.0
        metrics["faithfulness_proxy"] = support
        metrics["hallucination_risk_proxy"] = 1 - support
    else:
        metrics["faithfulness_proxy"] = None
        metrics["hallucination_risk_proxy"] = None
    return metrics


def aggregate_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    names = sorted({name for row in rows for name in row})
    aggregate = {}
    for name in names:
        values = [row[name] for row in rows if row.get(name) is not None]
        aggregate[name] = sum(values) / len(values) if values else None
    return aggregate


def classify_failure(metrics: dict[str, float | None], sources: list[dict] | list[object], answer: str,
                     prompt: str, max_tokens: int) -> str | None:
    """Deterministic triage labels; these are signals for review, not semantic judgments."""
    if metrics.get("exact_match") == 1 or (metrics.get("token_f1") or 0) >= 0.95:
        return None
    if not sources and "Source passages:" in prompt:
        return "retrieval failure"
    if sources and (metrics.get("context_recall") or 0) < 0.5:
        return "irrelevant context"
    if len(answer.split()) >= max_tokens * 0.9:
        return "token limit"
    if not answer.strip():
        return "inference error"
    if "not provide enough information" in answer.lower() or "don't know" in answer.lower():
        return "context ignored"
    if sources and (metrics.get("hallucination_risk_proxy") or 0) >= 0.5:
        return "hallucination"
    if metrics.get("token_f1", 0) and metrics.get("token_f1", 0) > 0.1:
        return "reasoning failure"
    return "instruction-following failure"
