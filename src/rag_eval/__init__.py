"""rag-eval-harness: a tiny, zero-dependency toolkit for evaluating RAG pipelines.

Public API:
    datasets   - EvalItem / EvalSet, JSONL load/save
    retrieval  - Retriever protocol, Document, TfIdfRetriever (stdlib-only baseline)
    generation - Generator protocol, EchoGenerator (deterministic baseline)
    metrics    - recall@k, reciprocal rank, nDCG@k, faithfulness, latency stats
    runner     - run_eval(): wire a retriever (+ optional generator) over an EvalSet
    report     - to_markdown(): render an EvalReport as Markdown
"""

from .datasets import EvalItem, EvalSet
from .generation import EchoGenerator, Generator
from .metrics import (
    FaithfulnessResult,
    faithfulness,
    latency_stats,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .report import to_markdown
from .retrieval import Document, RetrievedDoc, Retriever, TfIdfRetriever
from .runner import EvalReport, ItemResult, run_eval

__all__ = [
    "EvalItem",
    "EvalSet",
    "EchoGenerator",
    "Generator",
    "FaithfulnessResult",
    "faithfulness",
    "latency_stats",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "to_markdown",
    "Document",
    "RetrievedDoc",
    "Retriever",
    "TfIdfRetriever",
    "EvalReport",
    "ItemResult",
    "run_eval",
]

__version__ = "0.1.0"
