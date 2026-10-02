"""Retriever abstractions plus a stdlib-only TF-IDF baseline."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small stopword list: enough to keep toy corpora honest, not a linguistic claim.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else when where which who whom what this that
    these those am is are was were be been being have has had do does did will
    would should could may might must can of at by for with about into through
    during before after above below to from up down in out on off over under
    again further once here there all any both each few more most other some
    such no nor not only own same so than too very s t just don now
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus stopwords and single chars."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    ]


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class RetrievedDoc:
    doc: Document
    score: float


class Retriever(Protocol):
    """Anything that can rank documents for a query."""

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        ...


class TfIdfRetriever:
    """In-memory TF-IDF retriever with cosine similarity. Stdlib only.

    A deterministic, dependency-free baseline: good enough to exercise the
    harness end to end and to serve as the "beat this" reference when you
    plug in a dense/embedding retriever.
    """

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("TfIdfRetriever needs at least one document")
        self.documents = list(documents)
        doc_freq: Counter[str] = Counter()
        self._doc_tf: list[Counter[str]] = []
        for doc in self.documents:
            tf = Counter(tokenize(doc.text))
            self._doc_tf.append(tf)
            for term in tf:
                doc_freq[term] += 1
        n = len(self.documents)
        # Smoothed IDF so unseen-in-most-docs terms don't explode.
        self._idf = {
            term: math.log((1 + n) / (1 + df)) + 1.0 for term, df in doc_freq.items()
        }

    def _vector(self, tf: Counter[str]) -> dict[str, float]:
        return {
            term: (1.0 + math.log(count)) * self._idf[term]
            for term, count in tf.items()
            if term in self._idf
        }

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(a.get(term, 0.0) * value for term, value in b.items())
        if dot == 0.0:
            return 0.0
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        qvec = self._vector(Counter(tokenize(query)))
        scored = [
            RetrievedDoc(doc, self._cosine(qvec, self._vector(tf)))
            for doc, tf in zip(self.documents, self._doc_tf)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[: max(k, 0)]
