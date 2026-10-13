"""SFT training loop for full fine-tuning (FSDP2) and LoRA (DDP), run inside every worker.

Runs under an already-initialised process group (Ray Train sets it up; the tests do too), so it has
no Ray dependency. The loss is normalised by the *global* target-token count of each global batch
and scaled by the world size to cancel DDP/FSDP2's gradient averaging. Gradients are therefore
identical whether the batch is split over 1 rank or 4, which is what lets the equivalence check
compare a 1-rank and a 2-rank run.
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


def resolve_stage(cfg: dict, mode: str, strategy: str | None) -> tuple[str, str]:
    """(stage name, strategy) for an SFT mode; strategy defaults to the config's choice."""
    stage = f"sft_{mode}"
    return stage, strategy or cfg["training"]["strategy"][stage]


def _batch_tensors(examples: list[dict], pad_id: int, device) -> dict:
    b = data.collate(examples, pad_id)
    return {k: torch.tensor(v, device=device) for k, v in b.items()}


def _nll_sum(model, batch: dict, autocast) -> tuple[torch.Tensor, int]:
    """Summed token cross-entropy of the answer tokens, and how many tokens that covers."""
    with autocast:
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=data.IGNORE_INDEX,
        reduction="sum",
    )
    return loss, int((labels != data.IGNORE_INDEX).sum())


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def _all_reduce_sum(values: list[float], device) -> list[float]:
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return t.tolist()


@torch.no_grad()
def evaluate(model, val: list[dict], pad_id: int, micro: int, device, autocast) -> float:
    """Mean answer-token loss on the validation rows, sharded across ranks."""
    rank, world = dist.get_rank(), dist.get_world_size()
    mine = val[rank::world]
    total, count = 0.0, 0
    model.eval()
    for i in range(0, len(mine), micro):
        loss, n = _nll_sum(model, _batch_tensors(mine[i : i + micro], pad_id, device), autocast)
        total += float(loss)
        count += n
    model.train()
    s, c = _all_reduce_sum([total, float(count)], device)
    return s / max(c, 1.0)


def train_sft(
    cfg: dict,
    mode: str,
    strategy: str | None = None,
    *,
    device,
    out_dir: Path,
    smoke: bool = False,
    resume: bool = False,
    stop_after: int | None = None,
    init_from: str | None = None,
    tokenizer=None,
    report=None,
) -> dict:
    """Train one SFT stage on this rank. Returns the run record (identical on every rank).

    `stop_after` ends the run after that optimizer step, as a spot preemption would, leaving a
    resume checkpoint behind. `report(metrics)` is called each step on every rank (Ray hook).
    """
    t_start = time.perf_counter()
    tcfg = cfg["training"]
    sft = tcfg["sft"]
    stage, strategy = resolve_stage(cfg, mode, strategy)
    rank, world = dist.get_rank(), dist.get_world_size()
    is_cuda = device.type == "cuda"
    lora = mode == "lora"
    out_dir = Path(out_dir)
    resume_dir = out_dir / "resume"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if is_cuda else contextlib.nullcontext()

    # ---- data
    tok = tokenizer or model_lib.load_tokenizer(tcfg["model"])
    train_dir = resolve_path(cfg["training_data"]["out_dir"])
    max_len = tcfg["max_length"]
    train_rows = data.read_jsonl(train_dir / "sft_train.jsonl")
    val_rows = data.read_jsonl(train_dir / "sft_val.jsonl")
    if smoke:
        train_rows = train_rows[: tcfg["smoke"]["rows"]]
        val_rows = val_rows[:16]
    train = data.encode_sft(tok, train_rows, max_len)
    val = data.encode_sft(tok, val_rows, max_len)
    lengths = [len(e["input_ids"]) for e in train]
    target_counts = [sum(1 for t in e["labels"] if t != data.IGNORE_INDEX) for e in train]
    gbs, micro = sft["global_batch_size"], sft["micro_batch_size"]
    if smoke:  # small enough to run a handful of steps quickly, still split across ranks
        gbs, micro = 8, 2
    accum = data.accum_steps(gbs, micro, world)
    steps_per_epoch = len(train) // gbs
    total_steps = sft["epochs"] * steps_per_epoch
    if smoke:
        total_steps = min(total_steps, tcfg["smoke"]["steps"])
    warmup = max(1, int(sft["warmup_ratio"] * total_steps))
    peak_lr = sft["lr_lora"] if lora else sft["lr_full"]

    # ---- model
    t_load = time.perf_counter()
    if smoke:
        net = model_lib.tiny_model(tcfg["smoke"], vocab_size=len(tok.tok), seed=sft["seed"])
    else:
        net = model_lib.load_pretrained(tcfg["model"], init_from)
    if lora:
        torch.manual_seed(sft["seed"])
        net = model_lib.build_lora(net, tcfg["lora"])
    if sft["gradient_checkpointing"]:
        parallel.enable_checkpointing(net)
        if lora:
            net.enable_input_require_grads()
    total_params, trainable_params = model_lib.count_params(net)
    compute_dtype = torch.bfloat16 if is_cuda else torch.float32
    if strategy == parallel.FSDP:
        master = torch.float32
        net = parallel.wrap_fsdp2(net, device, master, compute_dtype)
    else:
        net = parallel.wrap_ddp(net, device)
    load_s = time.perf_counter() - t_load
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=peak_lr, weight_decay=sft["weight_decay"], betas=(0.9, 0.95)
    )

    # ---- resume
    start_step, global_hist, local_hist = 0, [], []  # local_hist covers this segment only
    state_file = resume_dir / "trainer_state.json"
    if resume and state_file.exists():
        parallel.load_resume_state(net, optimizer, strategy, resume_dir / "dcp", lora)
        st = json.loads(state_file.read_text(encoding="utf-8"))
        start_step, global_hist = st["step"], st["global"]

    timer = profiling.StepTimer(rank, tcfg["profiling"]["warmup_steps"], device)
    timer.stats.load_s = load_s
    trace_path = out_dir / f"trace_rank{rank}.json"
    trace = profiling.TraceWindow(
        tcfg["profiling"]["trace_steps"], tcfg["profiling"]["warmup_steps"], device, trace_path
    )
    sampler = checks.GpuUtilSampler(device, tcfg["profiling"]["gpu_util_sample_s"])
    sampler.start()
    check_results: dict = {}
    failures: list[str] = []

    def run_check(name, fn):
        try:
            check_results[name] = {"passed": True, "detail": fn()}
        except checks.DistributionCheckError as e:
            check_results[name] = {"passed": False, "error": str(e)}
            failures.append(f"{name}: {e}")

    run_check(
        "ranks_started",
        lambda: checks.check_ranks(checks.gather(checks.rank_report(rank, world, device)), world),
    )
    net.train()
    val_losses: dict[int, float] = {}
    step = start_step
    interrupted = False
    while step < total_steps:
        epoch, i = divmod(step, steps_per_epoch)
        batch_ids = data.global_batches(len(train), gbs, sft["seed"], epoch)[i]
        mine = data.rank_slice(batch_ids, lengths, rank, world)
        global_tokens = sum(target_counts[j] for j in batch_ids)
        micros = data.micro_batches(mine, micro)
        trace.maybe_start(step)
        timer.begin()
        local_nll, local_tok, step_tokens = 0.0, 0, 0
        for k, ids in enumerate(micros):
            ex = [train[j] for j in ids]
            batch = _batch_tensors(ex, tok.pad_token_id, device)
            step_tokens += int(batch["attention_mask"].sum())
            with parallel.accumulate(net, strategy, sync=k == len(micros) - 1):
                nll, n = _nll_sum(net, batch, autocast)
                (nll / global_tokens * world).backward()
            local_nll += float(nll.detach())
            local_tok += n
        if step == start_step and strategy == parallel.DDP:
            run_check(
                "gradients_synced",
                lambda: checks.check_checksums_equal(
                    checks.gather(checks.grad_checksum(net)), "gradient checksums"
                ),
            )
        if step == start_step and strategy == parallel.FSDP:
            run_check("parameters_sharded", lambda: _sharded_check(net))
        torch.nn.utils.clip_grad_norm_(params, sft["max_grad_norm"])
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, total_steps, warmup, peak_lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        timer.end(step_tokens)
        trace.maybe_stop()
        g_nll, _ = _all_reduce_sum([local_nll, float(local_tok)], device)
        global_loss = g_nll / global_tokens
        global_hist.append(global_loss)
        local_hist.append(local_nll / max(local_tok, 1))
        step += 1
        if sft["eval_every"] and not smoke and step % sft["eval_every"] == 0 and val:
            val_losses[step] = evaluate(net, val, tok.pad_token_id, micro, device, autocast)
        if report is not None:
            report({"step": step, "loss": global_loss, "lr": optimizer.param_groups[0]["lr"]})
        stopping = stop_after is not None and step >= stop_after
        if stopping or (sft["checkpoint_every"] and step % sft["checkpoint_every"] == 0):
            _save_resume(net, optimizer, strategy, resume_dir, lora, step, global_hist)
        if stopping:
            interrupted = True
            break

    trace.close()
    util = sampler.stop()
    # ---- run-level checks
    all_global = checks.gather(global_hist[start_step:])
    all_local = checks.gather(local_hist)
    all_util = checks.gather(util)
    if not interrupted:
        if strategy == parallel.DDP:
            run_check(
                "weights_identical",
                lambda: checks.check_checksums_equal(
                    checks.gather(checks.param_checksum(net)), "trainable weight checksums"
                ),
            )
        pcfg, ccfg = tcfg["profiling"], tcfg["checks"]
        run_check(
            "loss_coherent",
            lambda: checks.check_loss_coherent(
                all_global, all_local, ccfg["global_loss_rel_tol"], ccfg["local_loss_band"]
            ),
        )
        run_check("loss_decreasing", lambda: checks.check_loss_decreasing(global_hist))
        if is_cuda:
            run_check(
                "gpu_busy",
                lambda: checks.check_gpu_util(
                    {r: u for r, u in enumerate(all_util)}, pcfg["min_median_gpu_util"]
                ),
            )
        else:
            check_results["gpu_busy"] = {"skipped": "no GPU (measured in the paid run)"}

    # ---- outputs
    stats = timer.finish()
    stats.comm_fraction = trace.fraction
    all_stats = checks.gather(stats)
    record = {
        "stage": stage,
        "mode": mode,
        "strategy": strategy,
        "world_size": world,
        "steps": step,
        "interrupted": interrupted,
        "global_losses": global_hist,
        "local_losses": all_local,
        "val_losses": val_losses,
        "checks": check_results,
    }
    if not interrupted:
        if lora:
            parallel.save_lora(net, strategy, out_dir / "adapter")
        else:
            parallel.save_hf(net, strategy, out_dir / "model")
        if rank == 0:
            tok.tok.save_pretrained(out_dir / ("adapter" if lora else "model"))
    if rank == 0:
        profile = profiling.build_profile(
            stage,
            all_stats,
            total_params=total_params,
            trainable_params=trainable_params,
            meta={
                "mode": mode,
                "strategy": strategy,
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


def _sharded_check(net) -> dict:
    local, total, n_dtensor = checks.shard_report(net)
    result = checks.check_sharded(checks.gather(local), total)
    return {**result, "dtensor_params": n_dtensor}


def _save_resume(net, optimizer, strategy, resume_dir, lora, step, global_hist):
    parallel.save_resume_state(net, optimizer, strategy, resume_dir / "dcp", lora)
    if dist.get_rank() == 0:
        (resume_dir / "trainer_state.json").write_text(
            json.dumps({"step": step, "global": global_hist}),
            encoding="utf-8",
        )
    dist.barrier()
