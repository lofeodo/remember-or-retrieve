"""Score one system's answers on the golden set: abstention, judge, and gold-chunk grounding."""

from remember_or_retrieve.data import Chunk, GoldenQuestion
from remember_or_retrieve.scoring.abstention import abstention_label, is_abstention
from remember_or_retrieve.scoring.grounding import GroundingChecker
from remember_or_retrieve.scoring.judge import Judge
from remember_or_retrieve.scoring.schema import ERROR, ScoredAnswer, SystemAnswer


def score_answers(
    answers: list[SystemAnswer],
    golden: list[GoldenQuestion],
    judge: Judge | None,
    corpus: dict[str, Chunk] | None = None,
    checker: GroundingChecker | None = None,
    fresh: bool = False,
) -> list[ScoredAnswer]:
    """One ScoredAnswer per golden question, in golden order.

    A question the system has no answer for is an `error` (and is excluded from accuracy, so
    a system cannot look better by dropping questions: the summary reports error_count).
    `judge` may be None when every answer is decided by abstention alone.
    """
    by_id = {a.question_id: a for a in answers}
    scored: dict[str, ScoredAnswer] = {}
    to_judge: list[tuple[GoldenQuestion, SystemAnswer]] = []
    for g in golden:
        a = by_id.get(g.id)
        if a is None:
            scored[g.id] = ScoredAnswer(
                SystemAnswer(g.id, ""), ERROR, judge_reasoning="no answer provided"
            )
            continue
        abstained = a.abstained if a.abstained is not None else is_abstention(a.answer)
        label = abstention_label(g.answerable, abstained)
        if label is not None:
            scored[g.id] = ScoredAnswer(a, label)
        else:
            to_judge.append((g, a))
    if to_judge:
        if judge is None:
            raise ValueError("answers need the judge but none was given")
        verdicts = judge.score_many(
            [(g.question, g.gold_answer, a.answer) for g, a in to_judge], fresh=fresh
        )
        for (g, a), v in zip(to_judge, verdicts, strict=True):
            if isinstance(v, Exception):
                scored[g.id] = ScoredAnswer(a, ERROR, judge_reasoning=f"judging failed: {v}")
            else:
                scored[g.id] = ScoredAnswer(a, v.verdict, judge_reasoning=v.reasoning)
    if checker is not None and corpus is not None:
        for g in golden:
            s = scored[g.id]
            if not g.answerable or s.label == ERROR:
                continue
            abstained = (
                s.answer.abstained
                if s.answer.abstained is not None
                else is_abstention(s.answer.answer)
            )
            if abstained:
                continue
            chunks = [corpus[c] for c in g.gold_chunk_ids if c in corpus]
            s.claims = checker.check_answer(s.answer.answer, chunks) or None
    return [scored[g.id] for g in golden]
