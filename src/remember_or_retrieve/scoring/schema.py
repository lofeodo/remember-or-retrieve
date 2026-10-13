"""The one answer format every system produces, and the scored result.

Steps 9 and 10 write `SystemAnswer` JSONL for the fine-tune, the RAG API, Mistral, and Cohere;
the scorer reads nothing else.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The six labels rag-with-receipts uses, so tables line up. Only the first and last count as right.
CORRECT = "correct"
PARTIAL = "partially_correct"
INCORRECT = "incorrect"
INCORRECTLY_ABSTAINED = "incorrectly_abstained"
INCORRECTLY_ANSWERED = "incorrectly_answered"
CORRECT_ABSTENTION = "correct_abstention"
ERROR = "error"
LABELS = (
    CORRECT,
    PARTIAL,
    INCORRECT,
    INCORRECTLY_ABSTAINED,
    INCORRECTLY_ANSWERED,
    CORRECT_ABSTENTION,
)
RIGHT_LABELS = frozenset({CORRECT, CORRECT_ABSTENTION})

GROUNDED = "grounded"
UNGROUNDED = "ungrounded"
CONTRADICTED = "contradicted"


@dataclass
class SystemAnswer:
    question_id: str
    answer: str
    abstained: bool | None = None  # the system's own flag; None means detect from the text
    citations: list[dict] = field(default_factory=list)  # [{chunk_id, claim}], RAG only
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class ClaimVerdict:
    claim: str
    label: str  # grounded | ungrounded | contradicted
    chunk_id: str | None = None  # the gold chunk that decided it
    entailment_prob: float = 0.0
    contradiction_prob: float = 0.0
    overlap_score: float = 0.0


@dataclass
class ScoredAnswer:
    answer: SystemAnswer
    label: str  # one of LABELS, or ERROR
    judge_reasoning: str = ""
    claims: list[ClaimVerdict] | None = None  # None = grounding not run or not applicable


def read_answers(path: str | Path) -> list[SystemAnswer]:
    with open(path, encoding="utf-8") as f:
        return [SystemAnswer(**json.loads(line)) for line in f if line.strip()]


def write_answers(path: str | Path, answers: list[SystemAnswer]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for a in answers:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
