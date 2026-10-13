"""Training-data pipeline CLI. Run stages in order; each resumes from untracked data/raw/.

  estimate            print the pre-submit cost estimate for the generation stage (free)
  pilot               ~18 synchronous calls (<$0.50) written to data/raw/training/pilot.json
  submit gen          submit the generation batch          (PAID, checks max_usd)
  collect gen         wait for it and gate locally
  submit verify       build + submit the verification batch (PAID)
  collect verify      wait for it, then assemble (verify, leakage, dedup, trim)
  submit dpo          submit the DPO rejected-answer batch   (PAID)
  collect dpo         wait for it, then assemble the DPO pairs
  freeze              write data/training/*.jsonl and MANIFEST.json
  report              counts, drops per gate, spend

Needs ANTHROPIC_API_KEY in the environment for pilot, submit and collect.
"""

import argparse
import json
import random
import sys

from remember_or_retrieve.datagen import llm
from remember_or_retrieve.datagen.build import Pipeline


def show_estimate(p: Pipeline, reqs: list[llm.Request], label: str) -> float:
    est = p.estimate(reqs)
    print(f"{label}: {len(reqs)} requests")
    for model, m in est["per_model"].items():
        print(
            f"  {model}: {m['requests']} req, ~{m['input_tokens'] / 1e6:.2f}M in, "
            f"~{m['output_tokens'] / 1e6:.2f}M out, ~${m['usd']:.2f} at batch rates"
        )
    print(f"  estimated total ${est['usd']:.2f} (cap max_usd ${p.td['max_usd']})")
    return est["usd"]


def ledger(p: Pipeline) -> None:
    """Running spend for this step against the project budget ($20 target, $50 cap)."""
    spend = p._stats()["spend_usd"]
    total = sum(spend.values())
    cfg = p.cfg["budget"]
    print("--- spend ledger (step 3, USD) ---")
    for name, usd in spend.items():
        print(f"  {name:8s} ${usd:.3f}")
    print(f"  total    ${total:.3f}  (project target ${cfg['target_usd']}, cap ${cfg['cap_usd']})")


def pilot(p: Pipeline) -> None:
    reqs = p.gen_requests()
    rng = random.Random(7)
    chosen = []
    for prefix, n in (("sh-sft", 8), ("mh-sft", 6), ("un-sft", 4)):
        pool = [r for r in reqs if r.custom_id.startswith(prefix)]
        chosen += rng.sample(pool, n)
    client = llm.make_client()
    out, usd = [], 0.0
    for r in chosen:
        res = llm.sync_call(client, r)
        # synchronous calls are billed at the standard (non-batch) rate
        usd += llm.usage_cost(r.model, res["input_tokens"], res["output_tokens"], p.prices, 1.0)
        out.append({"id": r.custom_id, "meta": r.meta, "result": res})
        print(
            r.custom_id,
            r.meta["chunk_ids"],
            "->",
            json.dumps(res["data"], ensure_ascii=False)[:600],
        )
    path = p.raw / "pilot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    stats = p._stats()
    stats["spend_usd"]["pilot"] = round(usd, 4)
    p._save_stats(stats)
    print(f"pilot spend ${usd:.3f}; full output in {path}")
    ledger(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["estimate", "pilot", "submit", "collect", "freeze", "report"])
    ap.add_argument("stage", nargs="?", choices=["gen", "verify", "dpo"])
    args = ap.parse_args()
    p = Pipeline()

    if args.cmd == "estimate":
        show_estimate(p, p.gen_requests(), "gen")
    elif args.cmd == "pilot":
        pilot(p)
    elif args.cmd == "submit":
        reqs = {"gen": p.gen_requests, "verify": p.verify_requests, "dpo": p.dpo_requests}[
            args.stage
        ]()
        show_estimate(p, reqs, args.stage)
        print("batch id:", p.submit(args.stage, reqs))
    elif args.cmd == "collect":
        print(p.collect(args.stage))
        ledger(p)
        if args.stage == "gen":
            cands = p.gate_generated()
            print(f"{len(cands)} candidates after local gates; see data/raw/training/stats.json")
        elif args.stage == "verify":
            print(json.dumps(p.assemble(), indent=1))
        else:
            print(json.dumps(p.assemble_dpo(), indent=1))
    elif args.cmd == "freeze":
        m = p.freeze()
        print(json.dumps({k: v["rows"] for k, v in m["files"].items()}, indent=1))
    elif args.cmd == "report":
        print(json.dumps(p._stats(), indent=1, sort_keys=True))
        ledger(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
