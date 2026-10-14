"""Render the LoRA-vs-full (and DPO) comparison table from stage `profile.json` files.

  python scripts/compare_stages.py data/checkpoints/run/sft_full data/checkpoints/run/sft_lora ...

Golden-set accuracy is filled in by step 9 once each checkpoint has been scored; this prints the
cost side of the tradeoff: trainable parameters, throughput, peak memory per device, wall-clock.
"""

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    p = path / "profile.json" if path.is_dir() else path
    return json.loads(p.read_text(encoding="utf-8"))


def row(p: dict) -> list[str]:
    peak = max(p["peak_mem_gb_per_rank"]) if p["peak_mem_gb_per_rank"] else 0.0
    return [
        p["stage"],
        p.get("framework", "pytorch"),
        p.get("strategy", "?"),
        str(p["world_size"]),
        f"{p['trainable_params']:,} / {p['total_params']:,}",
        f"{p['tokens_per_s_per_device']:,.0f}",
        f"{peak:.1f} ({p.get('mem_kind', '?')})",
        f"{p['step_time_p50_s']:.2f}",
        f"{p['train_s'] / 60:.1f}",
        "n/a" if p.get("mfu_approx") is None else f"{p['mfu_approx']:.1%}",
    ]


HEADER = [
    "stage",
    "framework",
    "strategy",
    "GPUs",
    "trainable / total params",
    "tokens/s/device",
    "peak GB/device",
    "step p50 (s)",
    "train (min)",
    "MFU~",
]


def table(profiles: list[dict]) -> str:
    rows = [HEADER, *[row(p) for p in profiles]]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|" + "|".join("---" for _ in HEADER) + "|")
    return "\n".join(lines)


ROLLOUT_HEADER = [
    "stage",
    "GPUs",
    "generated tok/s/device",
    "completions/s",
    "rollout p50 (s)",
    "rollout share of step",
    "mean / p95 completion tokens",
]


def rollout_table(profiles: list[dict]) -> str:
    """RL stages only: generation throughput, the usual bottleneck there."""
    rows = [ROLLOUT_HEADER]
    for p in profiles:
        r = p.get("rollout")
        if r:
            rows.append(
                [
                    p["stage"],
                    str(p["world_size"]),
                    f"{r['generated_tokens_per_s_per_device']:,.0f}",
                    f"{r['completions_per_s_global']:.1f}",
                    f"{r['rollout_s_p50']:.2f}",
                    f"{r['rollout_fraction_of_step']:.0%}",
                    f"{r['completion_tokens_mean']:.0f} / {r['completion_tokens_p95']:.0f}",
                ]
            )
    if len(rows) == 1:
        return ""
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|" + "|".join("---" for _ in ROLLOUT_HEADER) + "|")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stages", nargs="+", type=Path, help="stage dirs (or profile.json files)")
    args = ap.parse_args()
    profiles = [load(p) for p in args.stages]
    print(table(profiles))
    if rollouts := rollout_table(profiles):
        print("\nRollouts (RL stages)\n\n" + rollouts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
