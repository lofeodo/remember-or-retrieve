"""Hard gate: no frozen training question may match a golden question.

Embeds every question in data/training/*.jsonl and the 55 golden questions, and fails (exit 1)
if any pair is an exact match, has cosine >= the configured threshold, or has word 5-gram
Jaccard >= the configured threshold. `--also "text"` adds an extra training question (used to
confirm the gate fails when a golden question leaks).
"""

import argparse
import json
import sys

from remember_or_retrieve.config import load_config, resolve_path
from remember_or_retrieve.data import load_golden_set
from remember_or_retrieve.datagen.leakage import bge_embedder, find_leaks


def training_questions(out_dir) -> list[str]:
    qs = []
    for name in ("sft_train.jsonl", "sft_val.jsonl", "grpo_prompts.jsonl", "dpo.jsonl"):
        with open(out_dir / name, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "messages" in row:
                    qs.append(row["messages"][1]["content"])
                elif "prompt_messages" in row:
                    qs.append(row["prompt_messages"][1]["content"])
                else:
                    qs.append(row["question"])
    return qs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--also", action="append", default=[], help="extra question to check")
    args = ap.parse_args()
    cfg = load_config()
    lk = cfg["training_data"]["leakage"]
    qs = training_questions(resolve_path(cfg["training_data"]["out_dir"])) + args.also
    gold = [g.question for g in load_golden_set()]
    leaks = find_leaks(qs, gold, bge_embedder(lk["embedding_model"]), lk["cosine"], lk["jaccard"])
    for i, (reason, j, score) in sorted(leaks.items()):
        print(f"LEAK ({reason} {score:.3f}): {qs[i]!r} ~ {gold[j]!r}")
    print(f"{len(qs)} training questions checked against {len(gold)} golden: {len(leaks)} leaks")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
