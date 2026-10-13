"""Replay the recorded rag-with-receipts run: the scorer must reproduce its published numbers."""

import pytest

from remember_or_retrieve.data import load_golden_set
from remember_or_retrieve.scoring import abstention, metrics, rag_reference, schema

PUBLISHED_LABELS = {
    "correct": 42,
    "partially_correct": 1,
    "incorrect": 3,
    "incorrectly_abstained": 4,
    "incorrectly_answered": 1,
    "correct_abstention": 4,
}


@pytest.fixture(scope="module")
def reference():
    return rag_reference.load_reference()


@pytest.fixture(scope="module")
def golden():
    return load_golden_set()


@pytest.fixture(scope="module")
def summary(reference, golden):
    return metrics.summarize(rag_reference.reference_scored(reference, golden), golden)


def test_replay_matches_published_accuracy(summary):
    assert summary["question_count"] == 55
    assert summary["correctness_accuracy"] == pytest.approx(46 / 55)
    assert summary["correctness_accuracy"] == pytest.approx(0.8363636363636363)
    assert summary["correctness_by_type"]["single_hop"] == {"accuracy": 1.0, "count": 35}
    assert summary["correctness_by_type"]["multi_hop"]["count"] == 15
    assert summary["correctness_by_type"]["multi_hop"]["accuracy"] == pytest.approx(7 / 15)
    assert summary["correctness_label_counts"] == PUBLISHED_LABELS
    assert summary["error_count"] == 0


def test_replay_matches_published_grounding(summary):
    g = summary["grounding"]
    assert g["claim_count"] == 77
    assert g["answer_count"] == 44
    assert g["grounded_rate"] == pytest.approx(0.948051948051948)
    assert g["contradicted_rate"] == pytest.approx(0.05194805194805195)
    assert g["ungrounded_rate"] == 0.0
    assert g["fully_grounded_answer_rate"] == pytest.approx(0.9090909090909091)


def test_replay_equals_the_recorded_summary(reference, summary):
    recorded = reference["summary"]
    assert summary["correctness_accuracy"] == pytest.approx(recorded["correctness_accuracy"])
    renamed = {"citation_count": "claim_count"}  # a citation is a claim in the upstream run
    for key, val in recorded["grounding"].items():
        assert summary["grounding"][renamed.get(key, key)] == pytest.approx(val)


def test_rebuilt_labels_equal_recorded_labels(reference, golden):
    """Our abstention + judge-verdict logic reproduces upstream's label on every question."""
    expected = {g.id: g.answerable for g in golden}
    for row in reference["results"]:
        got = rag_reference.recorded_label(row, expected[row["question_id"]])
        assert got == row["correctness_label"], row["question_id"]


def test_abstention_counts_for_unanswerable(summary):
    assert summary["abstention"]["unanswerable_count"] == 5
    assert summary["abstention"]["hallucinated_on_unanswerable"] == 1


def test_is_abstention_variants():
    phrase = "I don't have information about that in my Old School RuneScape Wiki knowledge."
    assert abstention.is_abstention(phrase)
    assert abstention.is_abstention(phrase.upper())
    assert abstention.is_abstention(phrase + " Try the wiki.")
    assert not abstention.is_abstention("")
    assert not abstention.is_abstention("I don't know.")
    assert not abstention.is_abstention("The answer is 5. " + phrase)


def test_abstention_label_matrix():
    assert abstention.abstention_label(True, False) is None
    assert abstention.abstention_label(True, True) == schema.INCORRECTLY_ABSTAINED
    assert abstention.abstention_label(False, True) == schema.CORRECT_ABSTENTION
    assert abstention.abstention_label(False, False) == schema.INCORRECTLY_ANSWERED


def test_reference_cites_only_corpus_chunks(reference):
    from remember_or_retrieve.data import load_corpus

    ids = {c.chunk_id for c in load_corpus()}
    assert all(c["chunk_id"] in ids for r in reference["results"] for c in r["citations"])
