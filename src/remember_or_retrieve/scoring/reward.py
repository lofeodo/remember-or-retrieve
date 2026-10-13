"""Fast local reward for GRPO: no network, no GPU, deterministic.

The reward scores a completion against a training row (`grpo_prompts.jsonl` shape):
  question, reference_answer, reference_facts, answerable, chunk_ids.

Components (breakdown returned so step 6 can log each one):
  correct       fraction of reference facts present (numbers by value, text by substring);
                on an unanswerable row, 1.0 for the exact abstention phrase
  abstain       penalty for declining an answerable question
  answered_un   penalty for answering an unanswerable question
  unsupported   numbers in neither the reference answer, the question, nor the chunks: a cheap
                stand-in for the NLI hallucination checker (running DeBERTa on every rollout
                would cost more than it buys); the full checker scores the final checkpoints
  stray         small per-number cost for numbers in the chunks but not the reference, so
                spraying every number from the evidence cannot buy the correctness credit
  verbosity     smooth penalty past a length budget, targeting the padded DPO rejected answers

reward = correct - abstain - answered_un - unsupported - stray - verbosity, clipped to [-1, 1].
"""

from dataclasses import asdict, dataclass

from remember_or_retrieve.datagen.validate import norm_words, numbers
from remember_or_retrieve.scoring.abstention import is_abstention


@dataclass
class RewardBreakdown:
    correct: float = 0.0
    abstain: float = 0.0
    answered_un: float = 0.0
    unsupported: float = 0.0
    stray: float = 0.0
    verbosity: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _text_credit(fact: str, completion_words: set[str]) -> float:
    """Word recall of a non-numeric fact, so a paraphrase earns most of the credit.

    Recall of 0.6 or more counts as present; below that it is proportional. Words are compared
    after lowercasing, which treats "slams" and "slammed" as different (no stemming, by design).
    """
    words = [w for w in norm_words(fact) if w]
    if not words:
        return 0.0
    recall = sum(w in completion_words for w in words) / len(words)
    return 1.0 if recall >= TEXT_FACT_PRESENT else recall


TEXT_FACT_PRESENT = 0.6


def facts_present(reference_facts: list[str], completion: str) -> float:
    """Fraction of reference facts in the completion: numbers by value, other facts by words."""
    if not reference_facts:
        return 0.0
    nums, words = numbers(completion), set(norm_words(completion))
    credit = 0.0
    for fact in reference_facts:
        fact_nums = numbers(fact)
        credit += float(fact_nums <= nums) if fact_nums else _text_credit(fact, words)
    return credit / len(reference_facts)


class RewardScorer:
    def __init__(self, cfg: dict, chunk_text: dict[str, str]):
        """`cfg` is config['scoring']['reward']; `chunk_text` maps chunk_id -> chunk text."""
        self.cfg = cfg
        self.chunk_text = chunk_text

    def _evidence_numbers(self, row: dict) -> set[str]:
        return numbers(" ".join(self.chunk_text.get(c, "") for c in row.get("chunk_ids", [])))

    def __call__(self, row: dict, completion: str) -> RewardBreakdown:
        c = self.cfg
        completion = completion.strip()
        b = RewardBreakdown()
        abstained = is_abstention(completion)
        if not row["answerable"]:
            b.correct = 1.0 if abstained else 0.0
            b.answered_un = 0.0 if abstained else c["answered_unanswerable"]
        elif abstained:
            b.abstain = c["abstain_on_answerable"]
        else:
            b.correct = facts_present(row["reference_facts"], completion)
            allowed = numbers(row["reference_answer"] + " " + row["question"])
            evidence = self._evidence_numbers(row)
            extra = numbers(completion) - allowed
            unsupported = extra - evidence
            b.unsupported = min(
                c["unsupported_number"] * len(unsupported), c["unsupported_number_cap"]
            )
            b.stray = min(c["stray_number"] * len(extra & evidence), c["stray_number_cap"])
        words = len(completion.split())
        ref_words = len(row["reference_answer"].split())
        budget = max(c["min_words"], c["per_ref_word"] * ref_words)
        if words > budget:
            b.verbosity = c["verbosity"] * min(1.0, (words - budget) / budget)
        total = b.correct - b.abstain - b.answered_un - b.unsupported - b.stray - b.verbosity
        b.total = max(-1.0, min(1.0, total))
        return b

    def rewards(self, rows: list[dict], completions: list[str]) -> list[float]:
        """Batch form for the trainer: one float per (row, completion)."""
        return [self(r, comp).total for r, comp in zip(rows, completions, strict=True)]
