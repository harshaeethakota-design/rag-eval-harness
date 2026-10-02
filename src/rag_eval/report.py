"""Render an EvalReport as Markdown."""
from __future__ import annotations

from .runner import EvalReport


def to_markdown(report: EvalReport, title: str = "RAG eval report") -> str:
    lines = [f"# {title}", "", f"Items: {len(report.results)} · k={report.k}", ""]
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for name, value in report.aggregate.items():
        lines.append(f"| {name} | {value:.4f} |")
    lines.append("")
    lines.append("## Per item")
    lines.append("")
    lines.append(
        "| item | recall@k | mrr | ndcg@k | faithfulness | latency_s | top doc |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in report.results:
        top = r.retrieved_ids[0] if r.retrieved_ids else "-"
        lines.append(
            f"| {r.item_id} | {r.recall_at_k:.2f} | {r.reciprocal_rank:.2f} | "
            f"{r.ndcg_at_k:.2f} | {r.faithfulness:.2f} | {r.latency_s:.4f} | {top} |"
        )
    lines.append("")
    return "\n".join(lines)
