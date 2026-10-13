"""Score a system on the golden set, and validate the scorer against rag-with-receipts.

Subcommands (PAID marks calls to the Claude judge; cached verdicts are free):
  replay                          recompute the recorded RAG run's summary (free, deterministic)
  score --answers F --out F       score one system's answers: abstention + judge (PAID) + grounding
  rag-judge [--passes N]          re-judge the recorded RAG answers N times, fresh (PAID)
  rag-nli                         recompute NLI on the recorded citation pairs (local model)
  rag-ground                      gold-chunk grounding on the recorded RAG answers (local model)
  reward-check                    reward sanity over the frozen data and the RAG answers (free)

The API key comes from ANTHROPIC_API_KEY in the environment; it is checked, never printed.
"""

import argparse
import json
import sys
from collections import Counter

from remember_or_retrieve.config import load_config, resolve_path
from remember_or_retrieve.data import load_corpus, load_golden_set
from remember_or_retrieve.datagen import llm
from remember_or_retrieve.scoring import metrics, rag_reference
from remember_or_retrieve.scoring.judge import Judge
from remember_or_retrieve.scoring.pipeline import score_answers
from remember_or_retrieve.scoring.reward import RewardScorer, facts_present
from remember_or_retrieve.scoring.schema import RIGHT_LABELS, read_answers

CFG = load_config()
SC = CFG["scoring"]


def make_judge(max_usd: float | None) -> Judge:
    judge = Judge.from_config(llm.make_client(), SC, resolve_path(SC["cache_dir"]))
    if max_usd is not None:
        judge.max_usd = max_usd
    return judge


def dump(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def cmd_replay(args) -> int:
    golden = load_golden_set()
    ref = rag_reference.load_reference()
    dump(metrics.summarize(rag_reference.reference_scored(ref, golden), golden))
    return 0


def cmd_score(args) -> int:
    golden = load_golden_set()
    answers = read_answers(args.answers)
    judge = make_judge(args.max_usd)
    checker = corpus = None
    if args.ground:
        from remember_or_retrieve.scoring.grounding import GroundingChecker

        checker = GroundingChecker.from_config(SC["grounding"])
        corpus = {c.chunk_id: c for c in load_corpus()}
    scored = score_answers(answers, golden, judge, corpus, checker)
    summary = metrics.summarize(scored, golden)
    summary["judge_spend_usd"] = round(judge.spent_usd, 4)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "summary": summary,
                "results": [
                    {
                        "question_id": s.answer.question_id,
                        "label": s.label,
                        "reasoning": s.judge_reasoning,
                        "claims": None if s.claims is None else [vars(c) for c in s.claims],
                    }
                    for s in scored
                ],
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    dump(summary)
    return 0


def cmd_rag_judge(args) -> int:
    """Re-judge the recorded RAG answers, fresh each pass, and compare to the recorded verdicts."""
    golden = load_golden_set()
    ref = rag_reference.load_reference()
    answers = rag_reference.reference_answers(ref)
    recorded = {r["question_id"]: r for r in ref["results"]}
    judge = make_judge(args.max_usd)
    passes = []
    for n in range(args.passes):
        scored = score_answers(answers, golden, judge, fresh=True)
        passes.append({s.answer.question_id: s for s in scored})
        acc = metrics.summarize(scored, golden)["correctness_accuracy"]
        print(f"pass {n + 1}: accuracy {acc:.4f}  (spent so far ${judge.spent_usd:.4f})")
    judged = [q for q, r in recorded.items() if r["judge_verdict"] is not None]
    tol, min_agree = (
        SC["validation"]["judge_accuracy_tolerance"],
        SC["validation"]["judge_min_agreement"],
    )
    published = ref["summary"]["correctness_accuracy"]
    ok = True
    for n, p in enumerate(passes, 1):
        agree = sum(p[q].label == recorded[q]["correctness_label"] for q in judged) / len(judged)
        acc = metrics.summarize(list(p.values()), golden)["correctness_accuracy"]
        passed = abs(acc - published) <= tol and agree >= min_agree
        ok &= passed
        print(
            f"pass {n}: agreement with recorded {agree:.3f} over {len(judged)} judged answers, "
            f"accuracy delta {acc - published:+.4f} -> {'PASS' if passed else 'FAIL'}"
        )
        for q in judged:
            if p[q].label != recorded[q]["correctness_label"]:
                print(
                    f"   {q}: recorded {recorded[q]['correctness_label']} -> {p[q].label}: "
                    f"{p[q].judge_reasoning[:160]}"
                )
    if len(passes) > 1:
        both = sum(passes[0][q].label == passes[1][q].label for q in judged) / len(judged)
        print(f"run-to-run agreement (pass 1 vs 2): {both:.3f}  <- judge noise floor")
    print(f"judge spend this run: ${judge.spent_usd:.4f}")
    return 0 if ok else 1


def cmd_rag_nli(args) -> int:
    from remember_or_retrieve.scoring.grounding import GroundingChecker

    ref = rag_reference.load_reference()
    corpus = {c.chunk_id: c for c in load_corpus()}
    checker = GroundingChecker.from_config(SC["grounding"])
    print(
        "NLI id2label:", checker.scorer.id2label, "columns (ent, con, neu):", checker.scorer.columns
    )
    tol = SC["validation"]["nli_prob_tolerance"]
    pairs, recorded = [], []
    for row in ref["results"]:
        for v in row["citation_verdicts"]:
            pairs.append((corpus[v["chunk_id"]], v["claim"]))
            recorded.append(v)
    got = checker.check_pairs(pairs)
    label_miss = [
        i for i, (g, r) in enumerate(zip(got, recorded, strict=True)) if g.label != r["label"]
    ]
    worst = max(
        max(
            abs(g.entailment_prob - r["entailment_prob"]),
            abs(g.contradiction_prob - r["contradiction_prob"]),
        )
        for g, r in zip(got, recorded, strict=True)
    )
    print(
        f"{len(pairs)} pairs: label mismatches {len(label_miss)}, max prob difference {worst:.2e}"
    )
    return 0 if not label_miss and worst <= tol else 1


def cmd_rag_ground(args) -> int:
    from remember_or_retrieve.scoring.grounding import GroundingChecker

    golden = load_golden_set()
    ref = rag_reference.load_reference()
    corpus = {c.chunk_id: c for c in load_corpus()}
    checker = GroundingChecker.from_config(SC["grounding"])
    scored = rag_reference.reference_scored(ref, golden)
    answers = [s.answer for s in scored]
    regrounded = score_answers(answers, golden, None, corpus, checker)  # abstention-only labels
    by_type = {}
    for g, s in zip(golden, regrounded, strict=True):
        if s.claims:
            by_type.setdefault(g.type, []).append(s.claims)
    out = {"all": metrics.grounding_stats([s.claims for s in regrounded if s.claims])}
    out |= {t: metrics.grounding_stats(c) for t, c in by_type.items()}
    out["native_citation_grounding_for_comparison"] = ref["summary"]["grounding"]
    dump(out)
    return 0


def cmd_reward_check(args) -> int:
    golden = load_golden_set()
    ref = rag_reference.load_reference()
    scorer = RewardScorer(SC["reward"], {c.chunk_id: c.text for c in load_corpus()})
    from remember_or_retrieve.datagen.validate import numbers

    recorded = {r["question_id"]: r for r in ref["results"]}
    rows = []
    for g in golden:
        r = recorded[g.id]
        if not g.answerable or not r["answerable"] or not numbers(g.gold_answer):
            continue
        row = {
            "question": g.question,
            "reference_answer": g.gold_answer,
            "reference_facts": sorted(numbers(g.gold_answer)),
            "answerable": True,
            "chunk_ids": list(g.gold_chunk_ids),
        }
        label = r["correctness_label"]
        rows.append(
            (
                g.id,
                label,
                facts_present(row["reference_facts"], r["answer"]),
                scorer(row, r["answer"]).total,
            )
        )
    agree = sum((f == 1.0) == (lab in RIGHT_LABELS) for _, lab, f, _ in rows)
    print(f"{len(rows)} RAG answers whose gold answer has numbers")
    print(f"numeric-fact match agrees with the judge on {agree}/{len(rows)}")
    for q, lab, f, t in rows:
        if (f == 1.0) != (lab in RIGHT_LABELS):
            print(f"   {q}: judge {lab}, facts {f:.2f}, reward {t:+.2f}")
    print("label counts:", Counter(lab for _, lab, _, _ in rows))
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("replay").set_defaults(fn=cmd_replay)
    p = sub.add_parser("score")
    p.add_argument("--answers", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ground", action="store_true", help="also run gold-chunk grounding")
    p.add_argument("--max-usd", type=float, default=None)
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("rag-judge")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--max-usd", type=float, default=None)
    p.set_defaults(fn=cmd_rag_judge)
    sub.add_parser("rag-nli").set_defaults(fn=cmd_rag_nli)
    sub.add_parser("rag-ground").set_defaults(fn=cmd_rag_ground)
    sub.add_parser("reward-check").set_defaults(fn=cmd_reward_check)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
