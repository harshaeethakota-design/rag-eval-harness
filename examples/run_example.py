"""End-to-end example: evaluate the TF-IDF baseline + echo generator.

Run from the repo root:
    python examples/run_example.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_eval import (  # noqa: E402
    Document,
    EchoGenerator,
    EvalSet,
    TfIdfRetriever,
    run_eval,
    to_markdown,
)

EXAMPLES = Path(__file__).resolve().parent


def main() -> None:
    corpus = [
        Document(id=obj["id"], text=obj["text"])
        for obj in (
            json.loads(line) for line in open(EXAMPLES / "corpus.jsonl", encoding="utf-8")
        )
    ]
    eval_set = EvalSet.from_jsonl(EXAMPLES / "eval_set.jsonl")

    report = run_eval(
        eval_set,
        retriever=TfIdfRetriever(corpus),
        generator=EchoGenerator(),
        k=3,
    )
    print(to_markdown(report, title="RAG eval report (TF-IDF + echo baseline)"))


if __name__ == "__main__":
    main()
