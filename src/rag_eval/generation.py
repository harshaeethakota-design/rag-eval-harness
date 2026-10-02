"""Generator abstractions plus a deterministic baseline."""
from __future__ import annotations

import re
from typing import Protocol

from .retrieval import RetrievedDoc

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class Generator(Protocol):
    """Anything that can produce an answer from retrieved context."""

    def generate(self, query: str, docs: list[RetrievedDoc]) -> str:
        ...


class EchoGenerator:
    """Deterministic baseline: stitches the first sentence of each top chunk.

    Not a real answerer — it exists so the harness runs end to end with no
    API keys, and so real generators have a floor to beat on faithfulness.
    """

    def __init__(self, max_docs: int = 2) -> None:
        self.max_docs = max_docs

    def generate(self, query: str, docs: list[RetrievedDoc]) -> str:
        sentences = []
        for hit in docs[: self.max_docs]:
            parts = _SENTENCE_RE.split(hit.doc.text.strip())
            if parts and parts[0].strip():
                sentences.append(parts[0].strip())
        return " ".join(sentences)
