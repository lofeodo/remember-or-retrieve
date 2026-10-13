"""Reward tests: components, bounds, adversarial completions, and checks over the frozen data."""

import json
from collections import defaultdict

import pytest

from remember_or_retrieve.config import REPO_ROOT, load_config
from remember_or_retrieve.data import load_corpus
from remember_or_retrieve.datagen.prompts import ABSTAIN_PHRASE
from remember_or_retrieve.scoring.reward import RewardScorer, facts_present

CFG = load_config()["scoring"]["reward"]
TRAIN = REPO_ROOT / "data" / "training"

ROW = {
    "question": "How much Construction XP does the Flame pit give?",
    "reference_answer": "The Flame pit gives 77 Construction XP and requires 20 Tinderboxes.",
    "reference_facts": ["77", "20"],
    "answerable": True,
    "chunk_ids": ["c1"],
}
UNANSWERABLE = {
    "question": "What are the full rewards for Recipe for Disaster?",
    "reference_answer": ABSTAIN_PHRASE,
    "reference_facts": [],
    "answerable": False,
    "chunk_ids": ["c1"],
}
CHUNKS = {"c1": "Flame pit: 77 Construction XP, 20 Tinderboxes, 125,000 Coins, level 41, 3 slots."}
SCORER = RewardScorer(CFG, CHUNKS)


def total(row, text):
    return SCORER(row, text).total


def test_facts_present_by_value_and_text():
    assert facts_present(["1,200", "2.50"], "It costs 1200 and takes 2.5 hours") == 1.0
    assert facts_present(["77", "20"], "It gives 77 XP") == 0.5
    assert facts_present(["Zamorak"], "Followers of zamorak") == 1.0
    assert facts_present([], "anything") == 0.0


def test_concise_correct_answer_scores_full():
    assert total(ROW, "The Flame pit gives 77 Construction XP and needs 20 Tinderboxes.") == 1.0


def test_reference_answer_scores_full():
    assert total(ROW, ROW["reference_answer"]) == 1.0
    assert total(UNANSWERABLE, ABSTAIN_PHRASE) == 1.0


def test_partial_credit_is_proportional():
    assert total(ROW, "It gives 77 XP.") == pytest.approx(0.5)


def test_abstention_is_right_only_when_unanswerable():
    assert SCORER(UNANSWERABLE, ABSTAIN_PHRASE).correct == 1.0
    b = SCORER(ROW, ABSTAIN_PHRASE)
    assert b.correct == 0.0 and b.abstain == CFG["abstain_on_answerable"]
    assert b.total < 0
    b = SCORER(UNANSWERABLE, "The rewards are 5000 coins.")
    assert b.correct == 0.0 and b.answered_un == CFG["answered_unanswerable"]


def test_bounds():
    junk = " ".join(str(n) for n in range(900, 1000)) * 5
    for row in (ROW, UNANSWERABLE):
        for text in ("", junk, ABSTAIN_PHRASE, "x" * 5000):
            assert -1.0 <= total(row, text) <= 1.0


# Adversarial completions: none may beat a concise correct answer.
GOOD = total(ROW, "The Flame pit gives 77 Construction XP and needs 20 Tinderboxes.")


def test_empty_answer_loses():
    assert total(ROW, "") < GOOD
    assert total(ROW, "") == 0.0


def test_abstain_everything_loses_on_answerable():
    assert total(ROW, ABSTAIN_PHRASE) < total(ROW, "It gives 77 XP.")


def test_confident_wrong_number_loses():
    wrong = "The Flame pit gives 99 Construction XP and needs 7 Tinderboxes."
    assert total(ROW, wrong) < 0 < GOOD
    assert SCORER(ROW, wrong).unsupported > 0


def test_padded_answer_with_right_numbers_loses():
    padded = "That is a great question. " * 30 + "The Flame pit gives 77 XP and needs 20."
    assert total(ROW, padded) < GOOD
    assert SCORER(ROW, padded).verbosity > 0


def test_number_spray_loses():
    spray = "77 20 125000 41 3"  # every number in the evidence, no sentence
    assert total(ROW, spray) < GOOD
    assert SCORER(ROW, spray).stray > 0


def test_stuffing_a_hallucinated_fact_is_charged():
    stuffed = "The Flame pit gives 77 Construction XP, needs 20 Tinderboxes, and was added in 2099."
    assert total(ROW, stuffed) < GOOD


def test_batch_form_matches_single():
    texts = ["It gives 77 XP.", ABSTAIN_PHRASE]
    assert SCORER.rewards([ROW, ROW], texts) == [total(ROW, t) for t in texts]


def _rows(name):
    with open(TRAIN / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def real_scorer():
    return RewardScorer(CFG, {c.chunk_id: c.text for c in load_corpus()})


def test_reference_answer_is_near_max_on_grpo_prompts(real_scorer):
    """4 of 404 frozen rows have a fact their own reference answer omits (or a U+FFFD in the
    fact), so they cannot reach full credit; the frozen data is not edited, the bar allows them."""
    rows = _rows("grpo_prompts")
    low = [r["id"] for r in rows if real_scorer(r, r["reference_answer"]).total < 0.9]
    assert len(low) / len(rows) <= 0.02, low
    for r in rows:
        if not r["answerable"]:
            assert real_scorer(r, r["reference_answer"]).total == 1.0


def test_abstention_phrase_wins_only_on_unanswerable_prompts(real_scorer):
    for row in _rows("grpo_prompts"):
        t = real_scorer(row, ABSTAIN_PHRASE).total
        assert (t == 1.0) == (not row["answerable"]), row["id"]


# Pre-set bars for chosen > rejected, per rejected kind (set before the first run).
DPO_BARS = {"hallucinated": 0.97, "verbose": 0.97, "fabricated_abstention": 1.0}


def test_reward_prefers_dpo_chosen_over_rejected(real_scorer):
    sft = {r["messages"][1]["content"]: r for r in _rows("sft_train") + _rows("sft_val")}
    wins = defaultdict(lambda: [0, 0])
    for d in _rows("dpo"):
        s = sft[d["prompt_messages"][1]["content"]]
        row = {
            "question": d["prompt_messages"][1]["content"],
            "reference_answer": d["chosen"],
            "reference_facts": s["reference_facts"],
            "answerable": s["kind"] != "unanswerable",
            "chunk_ids": s["chunk_ids"],
        }
        k = d["rejected_kind"]
        wins[k][0] += 1
        wins[k][1] += real_scorer(row, d["chosen"]).total > real_scorer(row, d["rejected"]).total
    for kind, bar in DPO_BARS.items():
        n, w = wins[kind]
        assert w / n >= bar, f"{kind}: {w}/{n}"
