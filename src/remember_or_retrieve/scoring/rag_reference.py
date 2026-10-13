"""Load the committed rag-with-receipts run as scored answers, to validate the scorer.

The reference holds that project's recorded answers, judge verdicts, and per-citation NLI
verdicts for the 55 golden questions. Replaying them through `metrics.summarize` must give its
published numbers; re-judging and re-running NLI on them checks the rest of the harness.
"""

import json
from pathlib import Path

from remember_or_retrieve.config import load_config, resolve_path
from remember_or_retrieve.data import GoldenQuestion
from remember_or_retrieve.scoring.abstention import abstention_label
from remember_or_retrieve.scoring.schema import (
    ERROR,
    ClaimVerdict,
    ScoredAnswer,
    SystemAnswer,
)


def load_reference(path: str | Path | None = None) -> dict:
    path = Path(path) if path else resolve_path(load_config()["scoring"]["reference_path"])
    return json.loads(path.read_text(encoding="utf-8"))


def reference_answers(reference: dict) -> list[SystemAnswer]:
    """The recorded RAG answers in the shared input format (for re-judging)."""
    return [
        SystemAnswer(
            question_id=r["question_id"],
            answer=r["answer"],
            abstained=not r["answerable"],
            citations=r["citations"],
            input_tokens=r["input_tokens"],
            output_tokens=r["output_tokens"],
        )
        for r in reference["results"]
    ]


def recorded_label(row: dict, expected_answerable: bool) -> str:
    """Rebuild a label from the recorded flag and judge verdict (the scorer's own logic)."""
    if row["error"]:
        return ERROR
    decided = abstention_label(expected_answerable, not row["answerable"])
    return decided or row["judge_verdict"]


def reference_scored(reference: dict, golden: list[GoldenQuestion]) -> list[ScoredAnswer]:
    expected = {g.id: g.answerable for g in golden}
    answers = {a.question_id: a for a in reference_answers(reference)}
    scored = []
    for row in reference["results"]:
        verdicts = [
            ClaimVerdict(
                claim=v["claim"],
                label=v["label"],
                chunk_id=v["chunk_id"],
                entailment_prob=v["entailment_prob"],
                contradiction_prob=v["contradiction_prob"],
                overlap_score=v["overlap_score"],
            )
            for v in row["citation_verdicts"]
        ]
        scored.append(
            ScoredAnswer(
                answer=answers[row["question_id"]],
                label=recorded_label(row, expected[row["question_id"]]),
                claims=verdicts or None,
            )
        )
    return scored
