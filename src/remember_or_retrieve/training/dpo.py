"""DPO: LoRA on DDP, run inside every worker, in the project's own loop.

Written by hand instead of with TRL's `DPOTrainer`: TRL 1.15 always routes DPO through a fused
LM-head kernel that needs Triton, so it cannot run on the CPU smoke test and the stage would reach
the paid GPU window untested. This loop shares the SFT stage's batching, checks, and profiling.

The reference policy is the same network with the LoRA adapter switched off, so there is no second
copy of the 8B model in GPU memory. Both sequences of a pair are scored with the same rendered
prompt as SFT. The loss is summed over a global batch, divided by the global pair count, and scaled
by the world size, so it is independent of how the batch is split across ranks.
"""

import contextlib
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from remember_or_retrieve.config import resolve_path
from remember_or_retrieve.training import checks, data, parallel, profiling
from remember_or_retrieve.training import model as model_lib
from remember_or_retrieve.training.sft import _all_reduce_sum, lr_at


def sequence_logps(model, batch: dict, autocast) -> torch.Tensor:
    """Sum of log-probabilities of each sequence's answer tokens, shape (batch,)."""
    with autocast:
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels != data.IGNORE_INDEX
    logp = torch.gather(
        F.log_softmax(logits, dim=-1), 2, labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair DPO loss and the implicit reward margin (chosen minus rejected)."""
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return -F.logsigmoid(margin), margin


def _pair_batch(pairs: list[dict], pad_id: int, device) -> dict:
    """Chosen sequences first, then rejected, as one padded batch of 2n."""
    seqs = [p["chosen"] for p in pairs] + [p["rejected"] for p in pairs]
    b = data.collate(seqs, pad_id)
    return {k: torch.tensor(v, device=device) for k, v in b.items()}


def _finite(losses: list[float]) -> dict:
    if not losses or not all(math.isfinite(x) for x in losses):
        raise checks.DistributionCheckError("DPO loss is empty or non-finite")
    return {"first": losses[0], "last": losses[-1]}


def train_dpo(
    cfg: dict, *, device, out_dir: Path, init_from: str | None, smoke: bool = False, tokenizer=None
) -> dict:
    t_start = time.perf_counter()
    tcfg = cfg["training"]
    dcfg = tcfg["dpo"]
    rank, world = dist.get_rank(), dist.get_world_size()
    out_dir = Path(out_dir)
    is_cuda = device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if is_cuda else contextlib.nullcontext()

    tok = tokenizer or model_lib.load_tokenizer(tcfg["model"])
    rows = data.read_jsonl(resolve_path(cfg["training_data"]["out_dir"]) / "dpo.jsonl")
    if smoke:
        rows = rows[: tcfg["smoke"]["rows"]]
    pairs = [data.encode_dpo_row(tok, r, dcfg["max_length"]) for r in rows]
    lengths = [max(len(p["chosen"]["input_ids"]), len(p["rejected"]["input_ids"])) for p in pairs]
    gbs, micro = (8, 2) if smoke else (dcfg["global_batch_size"], dcfg["micro_batch_size"])
    accum = data.accum_steps(gbs, micro, world)
    steps_per_epoch = len(pairs) // gbs
    total_steps = dcfg["epochs"] * steps_per_epoch
    if smoke:
        total_steps = min(total_steps, tcfg["smoke"]["steps"])
    warmup = max(1, int(0.1 * total_steps))

    t_load = time.perf_counter()
    if smoke and init_from is None:
        net = model_lib.tiny_model(tcfg["smoke"], len(tok.tok), tcfg["sft"]["seed"])
    elif init_from is None:
        net = model_lib.load_pretrained(tcfg["model"])
    else:
        net = model_lib.load_trained(
            tcfg["model"], init_from, tcfg["smoke"] if smoke else None, tcfg["sft"]["seed"]
        )
    torch.manual_seed(tcfg["sft"]["seed"])
    net = model_lib.build_lora(net, tcfg["lora"])
    if tcfg["sft"]["gradient_checkpointing"]:
        parallel.enable_checkpointing(net)
        net.enable_input_require_grads()
    total_params, trainable_params = model_lib.count_params(net)
    net = parallel.wrap_ddp(net, device)
    load_s = time.perf_counter() - t_load
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=dcfg["lr"], weight_decay=0.0, betas=(0.9, 0.95))

    timer = profiling.StepTimer(rank, tcfg["profiling"]["warmup_steps"], device)
    timer.stats.load_s = load_s
    sampler = checks.GpuUtilSampler(device, tcfg["profiling"]["gpu_util_sample_s"])
    sampler.start()
    results: dict = {}
    failures: list[str] = []

    def run_check(name, fn):
        try:
            results[name] = {"passed": True, "detail": fn()}
        except checks.DistributionCheckError as e:
            results[name] = {"passed": False, "error": str(e)}
            failures.append(f"{name}: {e}")

    run_check(
        "ranks_started",
        lambda: checks.check_ranks(checks.gather(checks.rank_report(rank, world, device)), world),
    )
    net.train()
    global_hist: list[float] = []
    local_hist: list[float] = []
    accuracy_hist: list[float] = []
    margin_hist: list[float] = []
    for step in range(total_steps):
        epoch, i = divmod(step, steps_per_epoch)
        batch_ids = data.global_batches(len(pairs), gbs, tcfg["sft"]["seed"], epoch)[i]
        mine = data.rank_slice(batch_ids, lengths, rank, world)
        micros = data.micro_batches(mine, micro)
        timer.begin()
        loc_loss = loc_correct = loc_margin = 0.0
        step_tokens = 0
        for k, ids in enumerate(micros):
            batch = _pair_batch([pairs[j] for j in ids], tok.pad_token_id, device)
            step_tokens += int(batch["attention_mask"].sum())
            n = len(ids)
            with torch.no_grad(), net.module.disable_adapter():
                ref = sequence_logps(net.module, batch, autocast)
            with parallel.accumulate(net, parallel.DDP, sync=k == len(micros) - 1):
                pol = sequence_logps(net, batch, autocast)
                losses, margin = dpo_loss(pol[:n], pol[n:], ref[:n], ref[n:], dcfg["beta"])
                (losses.sum() / gbs * world).backward()
            loc_loss += float(losses.detach().sum())
            loc_correct += float((margin.detach() > 0).sum())
            loc_margin += float(margin.detach().sum())
        if step == 0:
            run_check(
                "gradients_synced",
                lambda: checks.check_checksums_equal(
                    checks.gather(checks.grad_checksum(net)), "gradient checksums"
                ),
            )
        torch.nn.utils.clip_grad_norm_(params, tcfg["sft"]["max_grad_norm"])
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, warmup, dcfg["lr"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        timer.end(step_tokens)
        g_loss, g_correct, g_margin = _all_reduce_sum([loc_loss, loc_correct, loc_margin], device)
        global_hist.append(g_loss / gbs)
        local_hist.append(loc_loss / max(len(mine), 1))
        accuracy_hist.append(g_correct / gbs)
        margin_hist.append(g_margin / gbs)

    util = sampler.stop()
    all_global = checks.gather(global_hist)
    all_local = checks.gather(local_hist)
    ccfg = tcfg["checks"]
    run_check(
        "weights_identical",
        lambda: checks.check_checksums_equal(
            checks.gather(checks.param_checksum(net)), "adapter weight checksums"
        ),
    )
    run_check(
        "loss_coherent",
        lambda: checks.check_loss_coherent(
            all_global, all_local, ccfg["global_loss_rel_tol"], ccfg["local_loss_band"]
        ),
    )
    run_check("loss_finite", lambda: _finite(global_hist))
    if is_cuda:
        run_check(
            "gpu_busy",
            lambda: checks.check_gpu_util(
                dict(enumerate(checks.gather(util))), tcfg["profiling"]["min_median_gpu_util"]
            ),
        )
    else:
        results["gpu_busy"] = {"skipped": "no GPU (measured in the paid run)"}

    stats = timer.finish()
    all_stats = checks.gather(stats)
    parallel.save_lora(net, parallel.DDP, out_dir / "adapter")
    record = {
        "stage": "dpo",
        "strategy": parallel.DDP,
        "world_size": world,
        "steps": total_steps,
        "global_losses": global_hist,
        "reward_accuracy": accuracy_hist,
        "reward_margin": margin_hist,
        "checks": results,
    }
    if rank == 0:
        tok.tok.save_pretrained(out_dir / "adapter")
        model_lib.write_chain(out_dir / "adapter", init_from)
        profile = profiling.build_profile(
            "dpo",
            all_stats,
            total_params=total_params,
            trainable_params=trainable_params,
            meta={
                "mode": "lora",
                "strategy": parallel.DDP,
                "global_batch_size": gbs,
                "micro_batch_size": micro,
                "grad_accum": accum,
                "smoke": smoke,
            },
            wall_clock_s=time.perf_counter() - t_start,
            peak_tflops=tcfg["profiling"]["a100_bf16_peak_tflops"],
        )
        profiling.write_profile(profile, out_dir)
        (out_dir / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    dist.barrier()
    if failures:
        raise checks.DistributionCheckError("; ".join(failures))
    return record
