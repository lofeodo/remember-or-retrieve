"""Aggregate scored answers with rag-with-receipts' formulas so the numbers are comparable."""

import statistics

from remember_or_retrieve.data import GoldenQuestion
from remember_or_retrieve.scoring.schema import (
    CONTRADICTED,
    ERROR,
    GROUNDED,
    INCORRECTLY_ABSTAINED,
    INCORRECTLY_ANSWERED,
    LABELS,
    RIGHT_LABELS,
    UNGROUNDED,
    ScoredAnswer,
)


def _accuracy(labels: list[str]) -> float | None:
    scored = [lab for lab in labels if lab != ERROR]
    return sum(lab in RIGHT_LABELS for lab in scored) / len(scored) if scored else None


def grounding_stats(claim_lists: list[list]) -> dict:
    """Per-claim rates plus the share of answers with every claim grounded.

    Answers with no claims are skipped (grounding is not applicable), as upstream does.
    """
    answers = [c for c in claim_lists if c]
    claims = [v for c in answers for v in c]
    n = len(claims)

    def rate(label: str) -> float:
        return sum(v.label == label for v in claims) / n if n else 0.0

    return {
        "grounded_rate": rate(GROUNDED),
        "contradicted_rate": rate(CONTRADICTED),
        "ungrounded_rate": rate(UNGROUNDED),
        "fully_grounded_answer_rate": (
            sum(all(v.label == GROUNDED for v in c) for c in answers) / len(answers)
            if answers
            else 0.0
        ),
        "claim_count": n,
        "answer_count": len(answers),
    }


def _pctl(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


def summarize(scored: list[ScoredAnswer], golden: list[GoldenQuestion]) -> dict:
    types = {g.id: g.type for g in golden}
    by_id = {s.answer.question_id: s for s in scored}
    labels = [s.label for s in scored]
    out: dict = {
        "question_count": len(scored),
        "correctness_accuracy": _accuracy(labels),
        "correctness_by_type": {},
        "correctness_label_counts": {lab: labels.count(lab) for lab in LABELS if lab in labels},
        "error_count": labels.count(ERROR),
    }
    for t in sorted({t for t in types.values() if t}):
        ls = [by_id[i].label for i, tt in types.items() if tt == t and i in by_id]
        out["correctness_by_type"][t] = {"accuracy": _accuracy(ls), "count": len(ls)}
    unanswerable = [s for s in scored if types.get(s.answer.question_id) is None]
    out["abstention"] = {
        "unanswerable_count": len(unanswerable),
        "hallucinated_on_unanswerable": sum(s.label == INCORRECTLY_ANSWERED for s in unanswerable),
        "incorrectly_abstained": labels.count(INCORRECTLY_ABSTAINED),
    }
    graded = [s.claims for s in scored if s.claims is not None]
    if graded:
        out["grounding"] = grounding_stats(graded)
    lat = [s.answer.latency_s for s in scored if s.answer.latency_s is not None]
    if lat:
        out["latency_s"] = {
            "p50": statistics.median(lat),
            "p95": _pctl(lat, 0.95),
            "mean": statistics.fmean(lat),
            "count": len(lat),
        }
    cost = [s.answer.cost_usd for s in scored if s.answer.cost_usd is not None]
    if cost:
        out["cost_usd"] = round(sum(cost), 6)
    return out
