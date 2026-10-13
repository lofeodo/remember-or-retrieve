"""Train, check, and sample the PyTorch stages (SFT full/LoRA, DPO) under Ray Train.

Subcommands:
  sft --mode full|lora    one SFT stage (full = FSDP2, lora = DDP, per config)
  dpo --init-from DIR     DPO on top of an SFT checkpoint (LoRA, DDP)
  check --mode M          run a stage on 1 worker and on N, and require the same loss curve
  generate --checkpoint D answer the 55 golden questions, in the scoring answers format

Add --smoke for the tiny randomly initialised CPU model (needs Linux: Ray + gloo; the repo's
docker/smoke.Dockerfile runs it on Windows). Without --smoke this trains the real Qwen3-8B and is
meant for the rented GPU machine. GPUs are used automatically when CUDA is available.
"""

import argparse
import json
import sys
from pathlib import Path

from remember_or_retrieve.config import load_config, resolve_path

CFG = load_config()


def stage_config(args, stage: str, num_workers: int, out_dir: Path) -> dict:
    return {
        "stage": stage,
        "mode": getattr(args, "mode", "lora"),
        "strategy": getattr(args, "strategy", None),
        "cfg": CFG,
        "out_dir": str(out_dir),
        "smoke": args.smoke,
        "init_from": getattr(args, "init_from", None),
        "stop_after": getattr(args, "stop_after", None),
    }


def launch_stage(args, stage: str, num_workers: int, out_dir: Path) -> dict:
    import torch

    from remember_or_retrieve.training import launch

    out_dir.mkdir(parents=True, exist_ok=True)
    launch.run_ray(
        stage_config(args, stage, num_workers, out_dir),
        num_workers=num_workers,
        use_gpu=torch.cuda.is_available() and not args.cpu,
        storage_path=str((out_dir / "ray").resolve()),
        name=stage,
    )
    return json.loads((out_dir / "run.json").read_text(encoding="utf-8"))


def default_out(args, stage: str) -> Path:
    base = Path(args.out) if args.out else resolve_path(CFG["training"]["out_dir"])
    return (base / ("smoke" if args.smoke else "run") / stage).resolve()


def cmd_stage(args) -> int:
    stage = "dpo" if args.cmd == "dpo" else f"sft_{args.mode}"
    out = default_out(args, stage)
    record = launch_stage(args, stage, args.num_workers, out)
    failed = {k: v for k, v in record["checks"].items() if v.get("passed") is False}
    summary = {"out": str(out), "steps": record["steps"], "failed_checks": failed}
    print(json.dumps(summary, indent=1))
    return 1 if failed else 0


def cmd_check(args) -> int:
    from remember_or_retrieve.training import checks

    stage = f"sft_{args.mode}"
    curves = {}
    for workers in (1, args.num_workers):
        out = default_out(args, f"{stage}_w{workers}")
        curves[workers] = launch_stage(args, stage, workers, out)["global_losses"]
    tol = CFG["training"]["checks"]["equivalence_rel_tol"]
    result = checks.check_equivalence(curves[1], curves[args.num_workers], tol)
    print(f"1-worker and {args.num_workers}-worker loss curves agree: {result}")
    return 0


def cmd_generate(args) -> int:
    import torch

    from remember_or_retrieve.data import load_golden_set
    from remember_or_retrieve.scoring.schema import write_answers
    from remember_or_retrieve.training import generate
    from remember_or_retrieve.training import model as model_lib

    tcfg = CFG["training"]
    tok = model_lib.load_tokenizer(tcfg["model"])
    net = model_lib.load_trained(
        tcfg["model"], args.checkpoint, tcfg["smoke"] if args.smoke else None, tcfg["sft"]["seed"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    golden = [{"id": q.id, "question": q.question} for q in load_golden_set()]
    if args.smoke:
        golden = golden[:4]
    answers = generate.generate_answers(net, tok, golden, device)
    write_answers(args.out, answers)
    print(f"wrote {len(answers)} answers to {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, workers=True):
        p.add_argument("--smoke", action="store_true", help="tiny CPU model, a few steps")
        p.add_argument("--cpu", action="store_true", help="ignore GPUs")
        p.add_argument("--out", help="output root (default data/checkpoints)")
        if workers:
            p.add_argument("--num-workers", type=int, default=CFG["compute"]["train_gpu_count"])

    p = sub.add_parser("sft")
    common(p)
    p.add_argument("--mode", choices=["full", "lora"], required=True)
    p.add_argument("--strategy", choices=["ddp", "fsdp"])
    p.add_argument("--stop-after", type=int, help="stop after N steps, leaving a resume checkpoint")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("dpo")
    common(p)
    p.add_argument("--init-from", help="SFT checkpoint dir (model/ or adapter/); omit for the base")
    p.set_defaults(fn=cmd_stage)

    p = sub.add_parser("check")
    common(p)
    p.add_argument("--mode", choices=["full", "lora"], required=True)
    p.add_argument("--strategy", choices=["ddp", "fsdp"])
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("generate")
    common(p, workers=False)
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(fn=cmd_generate)
    p.set_defaults(out_default=None)

    args = ap.parse_args()
    if args.cmd == "generate" and not args.out:
        ap.error("generate needs --out FILE")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
