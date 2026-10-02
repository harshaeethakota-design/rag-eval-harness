"""Experiment runner: wire a retriever (+ optional generator) over an EvalSet."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from .datasets import EvalSet
from .generation import Generator
from .metrics import (
    faithfulness,
    latency_stats,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .retrieval import RetrievedDoc, Retriever


@dataclass(frozen=True)
class ItemResult:
    item_id: str
    query: str
    retrieved_ids: list[str]
    answer: str
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    faithfulness: float
    latency_s: float


@dataclass(frozen=True)
class EvalReport:
    k: int
    results: list[ItemResult]
    aggregate: dict[str, float] = field(default_factory=dict)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_eval(
    eval_set: EvalSet,
    retriever: Retriever,
    generator: Generator | None = None,
    k: int = 5,
    faithfulness_threshold: float = 0.4,
) -> EvalReport:
    """Run every item in `eval_set` through retriever -> generator.

    Retrieval metrics use the item's `relevant_doc_ids`. Faithfulness is
    scored only when a generator produced a non-empty answer; otherwise the
    item's faithfulness is 0.0 and the aggregate reflects that honestly.
    """
    results: list[ItemResult] = []
    for item in eval_set.items:
        started = perf_counter()
        retrieved: list[RetrievedDoc] = retriever.retrieve(item.query, k=k)
        answer = generator.generate(item.query, retrieved) if generator else ""
        elapsed = perf_counter() - started

        retrieved_ids = [hit.doc.id for hit in retrieved]
        faith = (
            faithfulness(
                answer,
                [hit.doc.text for hit in retrieved],
                threshold=faithfulness_threshold,
            ).score
            if answer.strip()
            else 0.0
        )
        results.append(
            ItemResult(
                item_id=item.id,
                query=item.query,
                retrieved_ids=retrieved_ids,
                answer=answer,
                recall_at_k=recall_at_k(retrieved_ids, item.relevant_doc_ids, k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, item.relevant_doc_ids),
                ndcg_at_k=ndcg_at_k(retrieved_ids, item.relevant_doc_ids, k),
                faithfulness=faith,
                latency_s=elapsed,
            )
        )

    aggregate = {
        f"recall@{k}": _mean([r.recall_at_k for r in results]),
        "mrr": _mean([r.reciprocal_rank for r in results]),
        f"ndcg@{k}": _mean([r.ndcg_at_k for r in results]),
        "faithfulness": _mean([r.faithfulness for r in results]),
        **latency_stats([r.latency_s for r in results]),
    }
    return EvalReport(k=k, results=results, aggregate=aggregate)
