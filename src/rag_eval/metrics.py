"""Evaluation metrics: retrieval quality, faithfulness, latency."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .retrieval import tokenize

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Retrieval metrics (operate on ranked doc-id lists)
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant docs present in the top-k."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / rank of the first relevant doc; 0.0 when none is retrieved."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalized discounted cumulative gain @ k (binary relevance)."""
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# Faithfulness: what fraction of the answer's sentences are supported by the
# retrieved context? Token-overlap F1 heuristic — fast and deterministic, but
# a heuristic. For production evals, plug an LLM judge into the same shape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceSupport:
    sentence: str
    supported: bool
    best_f1: float
    best_doc_id: str | None


@dataclass(frozen=True)
class FaithfulnessResult:
    score: float  # supported / total, 0.0 when the answer has no sentences
    supported: int
    total: int
    sentences: list[SentenceSupport] = field(default_factory=list)


def _token_f1(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    overlap = len(set_a & set_b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(set_a)
    recall = overlap / len(set_b)
    return 2 * precision * recall / (precision + recall)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def faithfulness(
    answer: str, contexts: list[str], threshold: float = 0.4
) -> FaithfulnessResult:
    """Score each answer sentence against the retrieved contexts.

    Each answer sentence is compared against every *sentence* of every
    context (not the whole document, which would unfairly penalize long
    docs), keeping the best token-overlap F1. A sentence counts as
    supported when that best F1 reaches `threshold`. Tune the threshold on
    labeled data for your domain; the default is a starting point, not a
    standard.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return FaithfulnessResult(score=0.0, supported=0, total=0)

    context_sentences = [
        (doc_idx, tokenize(sent))
        for doc_idx, ctx in enumerate(contexts)
        for sent in _split_sentences(ctx)
    ]
    per_sentence: list[SentenceSupport] = []
    for sentence in sentences:
        sent_tokens = tokenize(sentence)
        best_f1, best_doc = 0.0, None
        for doc_idx, ctx_tokens in context_sentences:
            f1 = _token_f1(sent_tokens, ctx_tokens)
            if f1 > best_f1:
                best_f1, best_doc = f1, doc_idx
        per_sentence.append(
            SentenceSupport(
                sentence=sentence,
                supported=best_f1 >= threshold,
                best_f1=round(best_f1, 4),
                best_doc_id=str(best_doc) if best_doc is not None else None,
            )
        )
    supported = sum(1 for s in per_sentence if s.supported)
    return FaithfulnessResult(
        score=supported / len(per_sentence),
        supported=supported,
        total=len(per_sentence),
        sentences=per_sentence,
    )


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def _percentile(sorted_xs: list[float], pct: float) -> float:
    if not sorted_xs:
        return 0.0
    idx = min(int(math.ceil(pct / 100 * len(sorted_xs))) - 1, len(sorted_xs) - 1)
    return sorted_xs[max(idx, 0)]


def latency_stats(seconds: list[float]) -> dict[str, float]:
    """p50 / p95 / mean over per-item latencies (seconds)."""
    xs = sorted(seconds)
    return {
        "p50_s": _percentile(xs, 50),
        "p95_s": _percentile(xs, 95),
        "mean_s": sum(xs) / len(xs) if xs else 0.0,
    }
