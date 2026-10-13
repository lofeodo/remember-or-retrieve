"""Loads the real NLI model (slow, excluded by default): `pytest -m slow`.

Recomputing NLI on the 77 citation pairs rag-with-receipts recorded must give the same labels
and the same probabilities, which is the empirical check that the label order and premise
construction match upstream.
"""

import pytest

from remember_or_retrieve.config import load_config
from remember_or_retrieve.data import load_corpus
from remember_or_retrieve.scoring import rag_reference
from remember_or_retrieve.scoring.grounding import GroundingChecker

pytestmark = pytest.mark.slow


def test_nli_reproduces_the_recorded_citation_verdicts():
    cfg = load_config()["scoring"]
    checker = GroundingChecker.from_config(cfg["grounding"])
    assert checker.scorer.id2label == {0: "contradiction", 1: "entailment", 2: "neutral"}
    corpus = {c.chunk_id: c for c in load_corpus()}
    recorded = [
        v for r in rag_reference.load_reference()["results"] for v in r["citation_verdicts"]
    ]
    assert len(recorded) == 77
    got = checker.check_pairs([(corpus[v["chunk_id"]], v["claim"]) for v in recorded])
    tol = cfg["validation"]["nli_prob_tolerance"]
    for g, r in zip(got, recorded, strict=True):
        assert g.label == r["label"]
        assert abs(g.entailment_prob - r["entailment_prob"]) <= tol
        assert abs(g.contradiction_prob - r["contradiction_prob"]) <= tol
