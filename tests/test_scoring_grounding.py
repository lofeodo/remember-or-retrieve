"""Offline grounding tests with a fake NLI scorer (no model download)."""

import numpy as np
import pytest

from remember_or_retrieve.data import Chunk
from remember_or_retrieve.scoring import grounding
from remember_or_retrieve.scoring.grounding import GroundingChecker

TH = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5, "overlap_threshold": 0.3}


def chunk(cid, text, title="Page", section=""):
    return Chunk(cid, title, "u", section, "", 10, "prose", text)


class FakeNli:
    """Rows keyed by a substring of the hypothesis; default is neutral."""

    def __init__(self, rules):
        self.rules = rules
        self.seen = []

    def score(self, pairs):
        self.seen.extend(pairs)
        out = []
        for premise, hyp in pairs:
            row = (0.0, 0.0, 1.0)
            for (claim_part, premise_part), val in self.rules.items():
                if claim_part in hyp and premise_part in premise:
                    row = val
            out.append(row)
        return np.array(out)


def test_split_claims_keeps_numbers_whole():
    text = "The Flame pit gives 77 XP. It needs 125,000 Coins and 2.5 hours!\nSecond line here"
    assert grounding.split_claims(text) == [
        "The Flame pit gives 77 XP.",
        "It needs 125,000 Coins and 2.5 hours!",
        "Second line here",
    ]
    assert grounding.split_claims("") == []
    assert grounding.split_claims("  ...  ") == []


def test_premise_has_breadcrumb():
    c = chunk("a", "text body", title="Monkey Archer", section="Combat")
    assert grounding.premise(c) == "Monkey Archer > Combat\n\ntext body"
    assert grounding.premise(chunk("a", "t", title="Only")) == "Only\n\nt"


def test_token_overlap():
    assert grounding.token_overlap("a b c d", "a b x y") == 0.5
    assert grounding.token_overlap("", "anything") == 0.0


def test_label_order_and_thresholds():
    assert grounding.label_for(0.9, 0.6, 1.0, TH) == "contradicted"  # contradiction wins
    assert grounding.label_for(0.5, 0.0, 0.0, TH) == "grounded"
    assert grounding.label_for(0.1, 0.0, 0.3, TH) == "grounded"  # overlap rescues
    assert grounding.label_for(0.1, 0.1, 0.29, TH) == "ungrounded"


def test_check_pairs_matches_the_upstream_rule():
    nli = FakeNli({("yes", "Alpha"): (0.9, 0.05, 0.05), ("no", "Alpha"): (0.0, 0.9, 0.1)})
    checker = GroundingChecker(nli, TH)
    out = checker.check_pairs(
        [(chunk("a", "Alpha"), "yes qq zz"), (chunk("a", "Alpha"), "no qq zz")]
    )
    assert [v.label for v in out] == ["grounded", "contradicted"]
    assert out[0].entailment_prob == pytest.approx(0.9)


def test_any_gold_chunk_can_ground_a_claim():
    nli = FakeNli({("claim", "second"): (0.95, 0.0, 0.05)})
    checker = GroundingChecker(nli, TH)
    out = checker.check_answer("A claim xx yy zz.", [chunk("c1", "first"), chunk("c2", "second")])
    assert [v.label for v in out] == ["grounded"]
    assert out[0].chunk_id == "c2"


def test_contradiction_only_when_nothing_grounds():
    nli = FakeNli({("claim", "first"): (0.0, 0.9, 0.1), ("claim", "second"): (0.9, 0.0, 0.1)})
    checker = GroundingChecker(nli, TH)
    both = checker.check_answer("A claim xx yy zz.", [chunk("c1", "first"), chunk("c2", "second")])
    assert both[0].label == "grounded"
    only = checker.check_answer("A claim xx yy zz.", [chunk("c1", "first")])
    assert only[0].label == "contradicted"


def test_nothing_to_check_returns_empty():
    checker = GroundingChecker(FakeNli({}), TH)
    assert checker.check_answer("", [chunk("c", "t")]) == []
    assert checker.check_answer("Some claim here.", []) == []


def test_each_sentence_is_its_own_claim():
    checker = GroundingChecker(FakeNli({}), TH)
    out = checker.check_answer("One xx. Two yy. Three zz.", [chunk("c", "unrelated")])
    assert [v.claim for v in out] == ["One xx.", "Two yy.", "Three zz."]
    assert all(v.label == "ungrounded" for v in out)
