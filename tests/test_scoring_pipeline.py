"""Pipeline tests: abstention decides without the judge; the judge only sees answered questions."""

import numpy as np

from remember_or_retrieve.data import Chunk, GoldenQuestion
from remember_or_retrieve.datagen.prompts import ABSTAIN_PHRASE
from remember_or_retrieve.scoring.grounding import GroundingChecker
from remember_or_retrieve.scoring.judge import Verdict
from remember_or_retrieve.scoring.metrics import summarize
from remember_or_retrieve.scoring.pipeline import score_answers
from remember_or_retrieve.scoring.schema import SystemAnswer, read_answers, write_answers

GOLD = [
    GoldenQuestion("q1", "How many?", "single_hop", True, ("c1",), "Five."),
    GoldenQuestion("q2", "What next?", "multi_hop", True, ("c1", "c2"), "Ten."),
    GoldenQuestion("q3", "Unknown thing?", None, False, (), None),
]
CORPUS = {
    "c1": Chunk("c1", "P", "u", "", "", 5, "prose", "There are five of them."),
    "c2": Chunk("c2", "P", "u", "", "", 5, "prose", "Ten follows."),
}


class FakeJudge:
    def __init__(self, verdict="correct", fail=()):
        self.seen, self.verdict, self.fail = [], verdict, fail

    def score_many(self, triples, fresh=False):
        self.seen.extend(triples)
        return [
            RuntimeError("boom") if t[0] in self.fail else Verdict(self.verdict, "ok")
            for t in triples
        ]


class AllEntailed:
    def score(self, pairs):
        return np.array([(0.99, 0.0, 0.01)] * len(pairs))


def test_abstentions_skip_the_judge():
    answers = [
        SystemAnswer("q1", ABSTAIN_PHRASE),
        SystemAnswer("q2", "Ten."),
        SystemAnswer("q3", ABSTAIN_PHRASE),
    ]
    judge = FakeJudge()
    out = score_answers(answers, GOLD, judge)
    assert [s.label for s in out] == ["incorrectly_abstained", "correct", "correct_abstention"]
    assert [t[0] for t in judge.seen] == ["What next?"]


def test_answering_an_unanswerable_question_is_a_hallucination():
    out = score_answers([SystemAnswer("q3", "It is 42.")], GOLD, FakeJudge())
    assert out[2].label == "incorrectly_answered"


def test_the_systems_own_flag_wins_over_text_detection():
    a = SystemAnswer("q1", "I think five.", abstained=True)
    assert score_answers([a], GOLD, FakeJudge())[0].label == "incorrectly_abstained"


def test_missing_answer_and_judge_failure_are_errors_not_silent_drops():
    out = score_answers([SystemAnswer("q1", "Five.")], GOLD, FakeJudge(fail={"How many?"}))
    assert out[0].label == "error" and "boom" in out[0].judge_reasoning
    assert out[1].label == "error" and out[1].judge_reasoning == "no answer provided"
    summary = summarize(out, GOLD)
    assert summary["error_count"] == 3  # judge failure + two missing answers


def test_grounding_runs_only_on_answered_answerable_questions():
    answers = [
        SystemAnswer("q1", "There are five of them. Ten follows."),
        SystemAnswer("q2", ABSTAIN_PHRASE),
        SystemAnswer("q3", "Made up."),
    ]
    th = {"entailment_threshold": 0.5, "contradiction_threshold": 0.5, "overlap_threshold": 0.3}
    checker = GroundingChecker(AllEntailed(), th)
    out = score_answers(answers, GOLD, FakeJudge(), CORPUS, checker)
    assert [v.label for v in out[0].claims] == ["grounded", "grounded"]
    assert out[1].claims is None and out[2].claims is None
    g = summarize(out, GOLD)["grounding"]
    assert g["fully_grounded_answer_rate"] == 1.0 and g["answer_count"] == 1


def test_summary_counts_hallucination_on_unanswerable():
    answers = [SystemAnswer("q1", "Five."), SystemAnswer("q2", "Ten."), SystemAnswer("q3", "Yes.")]
    s = summarize(score_answers(answers, GOLD, FakeJudge()), GOLD)
    assert s["abstention"]["hallucinated_on_unanswerable"] == 1
    assert s["correctness_accuracy"] == 2 / 3


def test_answers_round_trip(tmp_path):
    answers = [SystemAnswer("q1", "Five.", latency_s=1.5, cost_usd=0.01)]
    write_answers(tmp_path / "a.jsonl", answers)
    assert read_answers(tmp_path / "a.jsonl") == answers
