"""Eval datasets: items, sets, and JSONL serialization."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalItem:
    """One evaluation case.

    Attributes:
        id: Stable identifier for the case.
        query: The user question to evaluate.
        relevant_doc_ids: Doc ids a good retriever should return (may be empty
            when only answer quality is being measured).
        reference_answer: Optional gold answer, used by custom judges.
    """

    id: str
    query: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""


@dataclass
class EvalSet:
    """An ordered collection of EvalItems with JSONL load/save."""

    items: list[EvalItem] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EvalSet":
        items = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                items.append(
                    EvalItem(
                        id=str(obj["id"]),
                        query=str(obj["query"]),
                        relevant_doc_ids=list(obj.get("relevant_doc_ids", [])),
                        reference_answer=str(obj.get("reference_answer", "")),
                    )
                )
        return cls(items)

    def to_jsonl(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for item in self.items:
                fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
