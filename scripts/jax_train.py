"""JAX LoRA stage CLI (Python 3.12, `jax` extra). Step 7.

  python scripts/jax_train.py train --num-devices 4          # the real run (step 9)
  python scripts/jax_train.py train --smoke --num-devices 2  # tiny random-init model, CPU
  python scripts/jax_train.py check --num-devices 2          # N-device vs 1-device equivalence
  python scripts/jax_train.py generate --adapter DIR         # answer the 55 golden questions

On CPU, expose virtual devices first: XLA_FLAGS=--xla_force_host_platform_device_count=2
"""

import argparse
import json
import sys
from pathlib import Path

from remember_or_retrieve.config import load_config, resolve_path


def cmd_train(args) -> int:
    from remember_or_retrieve.jax_stage import train

    cfg = load_config()
    out = Path(args.out) if args.out else resolve_path(cfg["jax"]["out_dir"]) / "jax_lora"
    record = train.train(
        cfg, num_devices=args.num_devices, out_dir=out, smoke=args.smoke, resume=args.resume
    )
    print(json.dumps({k: v for k, v in record.items() if k != "global_losses"}, indent=1))
    print(f"final loss {record['global_losses'][-1]:.4f} after {record['steps']} steps -> {out}")
    return 0


def cmd_check(args) -> int:
    from remember_or_retrieve.jax_stage import smoke

    cfg = load_config()
    out = Path(args.out) if args.out else resolve_path(cfg["jax"]["out_dir"]) / "check"
    result = smoke.equivalence(cfg, out, args.num_devices)
    print(json.dumps(result, indent=1))
    return 0


def cmd_generate(args) -> int:
    from remember_or_retrieve.jax_stage import generate

    cfg = load_config()
    n = generate.run(cfg, Path(args.adapter), Path(args.out), smoke=args.smoke)
    print(f"wrote {n} answers to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="LoRA SFT, data parallel over --num-devices")
    t.add_argument("--num-devices", type=int, default=4)
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--out")
    t.set_defaults(fn=cmd_train)
    c = sub.add_parser("check", help="N-device vs 1-device loss-curve equivalence (smoke model)")
    c.add_argument("--num-devices", type=int, default=2)
    c.add_argument("--out")
    c.set_defaults(fn=cmd_check)
    g = sub.add_parser("generate", help="greedy answers for the golden set, scoring format")
    g.add_argument("--adapter", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--smoke", action="store_true")
    g.set_defaults(fn=cmd_generate)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
