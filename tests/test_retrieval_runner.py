"""Tests for the TF-IDF retriever and the end-to-end runner."""
import pytest

from rag_eval import (
    Document,
    EchoGenerator,
    EvalItem,
    EvalSet,
    TfIdfRetriever,
    run_eval,
)


@pytest.fixture
def corpus():
    return [
        Document(id="d1", text="The quick brown fox jumps over the lazy dog."),
        Document(id="d2", text="Vector databases store embeddings for similarity search."),
        Document(id="d3", text="Sourdough bread needs a starter, flour, water, and salt."),
    ]


def test_retriever_needs_documents():
    with pytest.raises(ValueError):
        TfIdfRetriever([])


def test_retriever_ranks_relevant_first(corpus):
    retriever = TfIdfRetriever(corpus)
    hits = retriever.retrieve("similarity search over embeddings", k=3)
    assert hits[0].doc.id == "d2"
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_retriever_respects_k(corpus):
    assert len(TfIdfRetriever(corpus).retrieve("fox", k=1)) == 1
    assert len(TfIdfRetriever(corpus).retrieve("fox", k=0)) == 0


def test_runner_end_to_end(corpus):
    eval_set = EvalSet(
        [
            EvalItem(id="q1", query="what is similarity search", relevant_doc_ids=["d2"]),
            EvalItem(id="q2", query="brown fox", relevant_doc_ids=["d1"]),
        ]
    )
    report = run_eval(eval_set, TfIdfRetriever(corpus), EchoGenerator(), k=2)
    assert len(report.results) == 2
    assert report.results[0].retrieved_ids[0] == "d2"
    assert report.results[0].recall_at_k == 1.0
    assert report.results[0].answer  # echo generator produced something
    for key in ("recall@2", "mrr", "ndcg@2", "faithfulness", "p50_s", "p95_s", "mean_s"):
        assert key in report.aggregate
    assert 0.0 <= report.aggregate["faithfulness"] <= 1.0


def test_runner_without_generator_scores_zero_faithfulness(corpus):
    eval_set = EvalSet([EvalItem(id="q1", query="fox", relevant_doc_ids=["d1"])])
    report = run_eval(eval_set, TfIdfRetriever(corpus), generator=None, k=2)
    assert report.results[0].answer == ""
    assert report.results[0].faithfulness == 0.0
