"""Tests for the faithfulness heuristic and latency stats."""
from rag_eval.metrics import faithfulness, latency_stats


def test_faithfulness_fully_supported():
    ctx = "The Eiffel Tower is in Paris. It was completed in 1889."
    result = faithfulness("The Eiffel Tower is in Paris.", [ctx])
    assert result.total == 1
    assert result.supported == 1
    assert result.score == 1.0


def test_faithfulness_unsupported_sentence():
    ctx = "The Eiffel Tower is in Paris."
    result = faithfulness("Quantum entanglement enables instant messaging.", [ctx])
    assert result.total == 1
    assert result.supported == 0
    assert result.score == 0.0


def test_faithfulness_single_word_flip_is_a_known_limitation():
    # Token-overlap cannot tell "Mars" from "Paris" here; this documents the
    # heuristic's boundary rather than asserting intelligence it lacks.
    ctx = "The Eiffel Tower is in Paris."
    result = faithfulness("The Eiffel Tower is on Mars.", [ctx])
    assert result.total == 1
    assert result.supported == 1  # limitation: use an LLM judge for this


def test_faithfulness_mixed_sentences():
    ctx = "Paris is the capital of France. It is known for the Eiffel Tower."
    answer = "Paris is the capital of France. Penguins run the metro system."
    result = faithfulness(answer, [ctx])
    assert result.total == 2
    assert result.supported == 1
    assert result.score == 0.5


def test_faithfulness_empty_answer():
    result = faithfulness("   ", ["some context"])
    assert result.total == 0
    assert result.score == 0.0


def test_latency_stats():
    stats = latency_stats([0.1, 0.2, 0.3, 0.4])
    assert stats["p50_s"] == 0.2
    assert stats["p95_s"] == 0.4
    assert abs(stats["mean_s"] - 0.25) < 1e-9


def test_latency_stats_empty():
    assert latency_stats([]) == {"p50_s": 0.0, "p95_s": 0.0, "mean_s": 0.0}
