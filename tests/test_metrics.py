"""Tests for retrieval metrics."""
from rag_eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_partial():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert recall_at_k(["x", "y"], ["a", "z"], 2) == 0.0


def test_recall_at_k_respects_k():
    assert recall_at_k(["x", "a"], ["a"], 1) == 0.0
    assert recall_at_k(["x", "a"], ["a"], 2) == 1.0


def test_recall_at_k_no_relevant_is_zero():
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == 1.0


def test_ndcg_penalizes_late_relevant():
    perfect = ndcg_at_k(["a", "b"], ["a", "b"], 2)
    worse = ndcg_at_k(["x", "a"], ["a", "b"], 2)
    assert 0.0 < worse < perfect


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["a"], [], 3) == 0.0
