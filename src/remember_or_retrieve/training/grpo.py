"""GRPO: on-policy RL on top of the DPO model, LoRA on DDP, in the project's own loop.

Group Relative Policy Optimization needs no value network and no learned reward model. Each step,
every rank samples a group of completions for each of its prompts, scores them with the project's
own fast reward (`scoring.reward.RewardScorer`), and reinforces the completions that beat their
group's average. The reference policy for the KL penalty is the same network with the LoRA adapter
switched off (the starting DPO model), so there is no second copy of the 8B model in GPU memory.

Choices, stated rather than hidden:
  * One gradient step per rollout batch, so the policy is exactly the sampling policy, the
    importance ratio is 1, and no old-policy pass or clipping is needed. This is the GRPO
    simplification at this scale, not a PPO implementation.
  * A prompt's whole group lives on one rank, so advantages need no communication. The loss is
    normalised by the *global* completion-token count and scaled by the world size (as in SFT and
    DPO), so the learner is the same training job on 1 rank or 4. Sampling is random per rank, so
    that equivalence is for the learner given fixed rollouts, not for whole runs.
  * Rollouts use Hugging Face `generate` on the unwrapped module, sampling from the unmodified
    policy distribution (temperature 1, top-k and top-p off, since the model's generation config
    would otherwise tilt it).
"""

import contextlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from remember_or_retrieve.config import resolve_path
from remember_or_retrieve.datagen.prompts import SFT_SYSTEM_PROMPT
from remember_or_retrieve.scoring.abstention import is_abstention
from remember_or_retrieve.scoring.reward import RewardBreakdown, RewardScorer
from remember_or_retrieve.training import checks, data, parallel, profiling
from remember_or_retrieve.training import model as model_lib
from remember_or_retrieve.training.sft import _all_reduce_sum, lr_at

COMPONENTS = [f for f in RewardBreakdown.__dataclass_fields__ if f != "total"]


@dataclass
class Rollout:
    """One sampled completion with its score. `prompt_ids` excludes the completion."""

    prompt_ids: list[int]
    completion_ids: list[int]
    truncated: bool
    text: str
    reward: float = 0.0
    advantage: float = 0.0
    breakdown: dict = field(default_factory=dict)


# ----------------------------------------------------------------------- loss


def token_logps(model, batch: dict, autocast) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probabilities of the labelled tokens, shape (batch, T-1), and the loss mask."""
    with autocast:
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels != data.IGNORE_INDEX
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.clamp(min=0).reshape(-1),
        reduction="none",
    ).view_as(labels)
    return -nll, mask


def grpo_loss(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed token-level loss and summed KL over the masked tokens.

    loss = -A * logp + beta * k3, with k3 = exp(ref - logp) - (ref - logp) - 1, the non-negative
    unbiased estimator of KL(policy || reference) from tokens sampled from the policy. The
    caller divides by the global token count.
    """
    m = mask.to(policy_logp.dtype)
    log_ratio = ref_logp - policy_logp
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    per_token = -advantages.unsqueeze(1) * policy_logp + beta * kl
    return (per_token * m).sum(), (kl * m).sum()


def learn(
    net,
    rollouts: list[Rollout],
    *,
    global_tokens: float,
    beta: float,
    micro: int,
    pad_id: int,
    device,
    autocast,
    world: int,
) -> dict:
    """Backward pass over this rank's rollouts (no optimizer step), gradients synced on the last
    micro-batch. Returns local sums: loss, kl, completion tokens, and tokens through the model."""
    order = sorted(
        range(len(rollouts)),
        key=lambda i: len(rollouts[i].prompt_ids) + len(rollouts[i].completion_ids),
    )
    micros = data.micro_batches(order, micro)
    loss_sum = kl_sum = 0.0
    tokens = fwd_tokens = 0
    for k, ids in enumerate(micros):
        rs = [rollouts[j] for j in ids]
        b = data.collate(
            [data.encode_completion(r.prompt_ids, r.completion_ids) for r in rs], pad_id
        )
        batch = {name: torch.tensor(v, device=device) for name, v in b.items()}
        adv = torch.tensor([r.advantage for r in rs], dtype=torch.float32, device=device)
        with torch.no_grad(), net.module.disable_adapter():
            ref, _ = token_logps(net.module, batch, autocast)
        with parallel.accumulate(net, parallel.DDP, sync=k == len(micros) - 1):
            pol, mask = token_logps(net, batch, autocast)
            loss, kl = grpo_loss(pol, ref, adv, mask, beta)
            (loss / global_tokens * world).backward()
        loss_sum += float(loss.detach())
        kl_sum += float(kl.detach())
        tokens += int(mask.sum())
        fwd_tokens += int(batch["attention_mask"].sum())
    return {"loss": loss_sum, "kl": kl_sum, "tokens": tokens, "forward_tokens": fwd_tokens}


# -------------------------------------------------------------------- rollouts


@torch.no_grad()
def sample_completions(
    inner,
    prompts: list[list[int]],
    *,
    pad_id: int,
    im_end: int,
    max_new_tokens: int,
    temperature: float,
    greedy: bool,
    batch_size: int,
    device,
) -> list[tuple[list[int], bool]]:
    """(completion ids, truncated) per prompt. Completions end with `<|im_end|>` when the model
    stopped by itself; `truncated` marks the ones cut at `max_new_tokens`. `inner` is the unwrapped
    (not DDP) module. Restores train mode and the cache setting."""
    was_training, prev_cache = inner.training, inner.config.use_cache
    inner.eval()
    inner.config.use_cache = True
    out: list[tuple[list[int], bool]] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            width = max(len(p) for p in chunk)
            ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in chunk], device=device)
            mask = torch.tensor(
                [[0] * (width - len(p)) + [1] * len(p) for p in chunk], device=device
            )
            kwargs = dict(
                max_new_tokens=max_new_tokens,
                eos_token_id=im_end,
                pad_token_id=pad_id,
                use_cache=True,
            )
            if greedy:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            else:
                kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            gen = inner.generate(input_ids=ids, attention_mask=mask, **kwargs)[:, width:].tolist()
            for seq in gen:
                if im_end in seq:
                    out.append((seq[: seq.index(im_end) + 1], False))
                else:
                    out.append((seq, True))
    finally:
        inner.config.use_cache = prev_cache
        inner.train(was_training)
    return out


def make_rollouts(
    prompts: list[list[int]],
    completions: list[tuple[list[int], bool]],
    rows: list[dict],
    scorer: RewardScorer,
    tok,
    group_size: int,
    im_end: int,
) -> list[Rollout]:
    """Score completions and attach group-normalised advantages. `prompts` and `rows` have one
    entry per prompt; `completions` has `group_size` consecutive entries per prompt."""
    rollouts: list[Rollout] = []
    for n, (ids, truncated) in enumerate(completions):
        j = n // group_size
        body = ids[:-1] if ids and ids[-1] == im_end else ids
        text = tok.tok.decode(body, skip_special_tokens=True).strip()
        b = scorer(rows[j], text)
        rollouts.append(Rollout(prompts[j], ids, truncated, text, b.total, 0.0, b.as_dict()))
    adv = data.group_advantages([r.reward for r in rollouts], group_size)
    for r, a in zip(rollouts, adv, strict=True):
        r.advantage = a
    return rollouts


# ------------------------------------------------------------------- prompts


def load_prompts(cfg: dict, scorer: RewardScorer) -> tuple[list[dict], list[dict]]:
    """(train, heldout) GRPO prompts. Frozen rows whose own reference answer scores below
    `min_reference_reward` (data quirks noted in step 4) are dropped, not edited."""
    gcfg = cfg["training"]["grpo"]
    rows = data.read_jsonl(resolve_path(cfg["training_data"]["out_dir"]) / "grpo_prompts.jsonl")
    rows = [
        r for r in rows if scorer(r, r["reference_answer"]).total >= gcfg["min_reference_reward"]
    ]
    return data.split_heldout(rows, gcfg["heldout"], gcfg["seed"])


def make_scorer(cfg: dict) -> RewardScorer:
    from remember_or_retrieve.data import load_corpus

    return RewardScorer(cfg["scoring"]["reward"], {c.chunk_id: c.text for c in load_corpus()})


# ---------------------------------------------------------------------- loop


def _finite(values: list[float]) -> dict:
    if not values or not all(math.isfinite(x) for x in values):
        raise checks.DistributionCheckError("GRPO loss is empty or non-finite")
    return {"first": values[0], "last": values[-1]}


# Layout of the per-step vector that is summed across ranks (every rank sees the same totals).
_M = [
    "n", "reward", *COMPONENTS, "tokens", "kl", "loss", "truncated", "groups", "zero_var",
    "ans_n", "ans_abstained", "un_n", "un_abstained",
]  # fmt: skip


def _metrics_vector(
    rollouts: list[Rollout], rows: list[dict], group_size: int, learned: dict
) -> list[float]:
    v = dict.fromkeys(_M, 0.0)
    v["n"] = len(rollouts)
    for r in rollouts:
        v["reward"] += r.reward
        for c in COMPONENTS:
            v[c] += r.breakdown[c]
        v["truncated"] += r.truncated
    v["tokens"], v["kl"], v["loss"] = learned["tokens"], learned["kl"], learned["loss"]
    for g in range(0, len(rollouts), group_size):
        group = rollouts[g : g + group_size]
        v["groups"] += 1
        v["zero_var"] += all(r.advantage == 0.0 for r in group)
        answerable = rows[g // group_size]["answerable"]
        abstained = sum(is_abstention(r.text) for r in group)
        v["ans_n" if answerable else "un_n"] += len(group)
        v["ans_abstained" if answerable else "un_abstained"] += abstained
    return [v[k] for k in _M]


def _step_record(totals: list[float], global_tokens: float, step: int, lr: float) -> dict:
    t = dict(zip(_M, totals, strict=True))
    n = max(t["n"], 1.0)
    return {
        "step": step,
        "lr": lr,
        "loss": t["loss"] / global_tokens,
        "kl": t["kl"] / max(t["tokens"], 1.0),
        "reward": t["reward"] / n,
        "components": {c: t[c] / n for c in COMPONENTS},
        "completion_tokens_mean": t["tokens"] / n,
        "truncated": t["truncated"] / n,
        "zero_variance_groups": t["zero_var"] / max(t["groups"], 1.0),
        "abstain_rate_answerable": t["ans_abstained"] / max(t["ans_n"], 1.0),
        "abstain_rate_unanswerable": t["un_abstained"] / max(t["un_n"], 1.0),
    }


def evaluate_heldout(inner, rows, prompts, scorer, tok, gcfg, device, im_end) -> float:
    """Mean reward of greedy answers on the held-out prompts, sharded across ranks."""
    rank, world = dist.get_rank(), dist.get_world_size()
    mine = list(range(rank, len(rows), world))
    total = 0.0
    if mine:
        comps = sample_completions(
            inner,
            [prompts[j] for j in mine],
            pad_id=tok.pad_token_id,
            im_end=im_end,
            max_new_tokens=gcfg["max_new_tokens"],
            temperature=1.0,
            greedy=True,
            batch_size=gcfg["rollout_batch_size"],
            device=device,
        )
        for j, (ids, _) in zip(mine, comps, strict=True):
            body = ids[:-1] if ids and ids[-1] == im_end else ids
            total += scorer(rows[j], tok.tok.decode(body, skip_special_tokens=True).strip()).total
    s, c = _all_reduce_sum([total, float(len(mine))], device)
    return s / max(c, 1.0)


def train_grpo(
    cfg: dict,
    *,
    device,
    out_dir: Path,
    init_from: str | None,
    smoke: bool = False,
    resume: bool = False,
    stop_after: int | None = None,
    tokenizer=None,
    report=None,
) -> dict:
    """Run GRPO on this rank under an initialised process group. Returns the run record."""
    t_start = time.perf_counter()
    tcfg = cfg["training"]
    gcfg = {**tcfg["grpo"], **(tcfg["smoke"]["grpo"] if smoke else {})}
    rank, world = dist.get_rank(), dist.get_world_size()
    out_dir = Path(out_dir)
    resume_dir = out_dir / "resume"
    is_cuda = device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if is_cuda else contextlib.nullcontext()
    G, ppstep = gcfg["group_size"], gcfg["prompts_per_step"]
    seed = gcfg["seed"]

    # ---- data
    tok = tokenizer or model_lib.load_tokenizer(tcfg["model"])
    im_end = tok.tok.convert_tokens_to_ids(data.IM_END)
    scorer = make_scorer(cfg)
    train_rows, heldout_rows = load_prompts(cfg, scorer)
    if smoke:
        train_rows = train_rows[: tcfg["smoke"]["grpo"]["rows"]]
    train_prompts = [data.grpo_prompt_ids(tok, r, SFT_SYSTEM_PROMPT) for r in train_rows]
    held_prompts = [data.grpo_prompt_ids(tok, r, SFT_SYSTEM_PROMPT) for r in heldout_rows]
    lengths = [len(p) for p in train_prompts]
    steps_per_epoch = len(train_rows) // ppstep
    total_steps = gcfg["epochs"] * steps_per_epoch
    if smoke:
        total_steps = min(total_steps, gcfg["steps"])
    warmup = max(1, int(gcfg["warmup_ratio"] * total_steps))
    micro = 2 if smoke else gcfg["micro_batch_size"]
    rollout_bs = 8 if smoke else gcfg["rollout_batch_size"]

    # ---- model: the starting policy is the chain-loaded SFT/DPO model, with a fresh LoRA on top
    t_load = time.perf_counter()
    if init_from is None and smoke:
        net = model_lib.tiny_model(tcfg["smoke"], len(tok.tok), tcfg["sft"]["seed"])
    elif init_from is None:
        net = model_lib.load_pretrained(tcfg["model"])
    else:
        net = model_lib.load_trained(
            tcfg["model"], init_from, tcfg["smoke"] if smoke else None, tcfg["sft"]["seed"]
        )
    torch.manual_seed(tcfg["sft"]["seed"])
    net = model_lib.build_lora(net, {**tcfg["lora"], "dropout": gcfg["lora_dropout"]})
    if tcfg["sft"]["gradient_checkpointing"]:
        parallel.enable_checkpointing(net)
        net.enable_input_require_grads()
    total_params, trainable_params = model_lib.count_params(net)
    net = parallel.wrap_ddp(net, device)
    load_s = time.perf_counter() - t_load
    inner = net.module
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=gcfg["lr"], weight_decay=0.0, betas=(0.9, 0.95))

    # ---- resume
    start_step, log, eval_curve = 0, [], {}
    state_file = resume_dir / "trainer_state.json"
    if resume and state_file.exists():
        parallel.load_resume_state(net, optimizer, parallel.DDP, resume_dir / "dcp", True)
        st = json.loads(state_file.read_text(encoding="utf-8"))
        start_step, log = st["step"], st["log"]
        eval_curve = {int(k): v for k, v in st["eval"].items()}

    timer = profiling.StepTimer(rank, tcfg["profiling"]["warmup_steps"], device)
    timer.stats.load_s = load_s
    timer.step = start_step  # warmup is about this process, so a resumed run re-warms
    timer.warmup = start_step + tcfg["profiling"]["warmup_steps"]
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

    def sync_clock() -> float:
        if is_cuda:
            torch.cuda.synchronize(device)
        return time.perf_counter()

    run_check(
        "ranks_started",
        lambda: checks.check_ranks(checks.gather(checks.rank_report(rank, world, device)), world),
    )
    if start_step == 0 and held_prompts:
        eval_curve[0] = evaluate_heldout(
            inner, heldout_rows, held_prompts, scorer, tok, gcfg, device, im_end
        )

    net.train()
    step, interrupted = start_step, False
    while step < total_steps:
        epoch, i = divmod(step, steps_per_epoch)
        batch_ids = data.global_batches(len(train_rows), ppstep, seed, epoch)[i]
        mine = data.rank_slice(batch_ids, lengths, rank, world)
        rows = [train_rows[j] for j in mine]
        prompts = [train_prompts[j] for j in mine]
        timer.begin()

        # rollouts: sample G completions per prompt, then score them (CPU)
        torch.manual_seed(seed * 1_000_003 + step * world + rank)
        t0 = sync_clock()
        comps = sample_completions(
            inner,
            [p for p in prompts for _ in range(G)],
            pad_id=tok.pad_token_id,
            im_end=im_end,
            max_new_tokens=gcfg["max_new_tokens"],
            temperature=gcfg["temperature"],
            greedy=False,
            batch_size=rollout_bs,
            device=device,
        )
        t1 = sync_clock()
        rollouts = make_rollouts(prompts, comps, rows, scorer, tok, G, im_end)
        t2 = time.perf_counter()

        # learner
        local_tokens = float(sum(len(r.completion_ids) for r in rollouts))
        (global_tokens,) = _all_reduce_sum([local_tokens], device)
        learned = learn(
            net,
            rollouts,
            global_tokens=global_tokens,
            beta=gcfg["beta"],
            micro=micro,
            pad_id=tok.pad_token_id,
            device=device,
            autocast=autocast,
            world=world,
        )
        if step == start_step:
            run_check(
                "gradients_synced",
                lambda: checks.check_checksums_equal(
                    checks.gather(checks.grad_checksum(net)), "gradient checksums"
                ),
            )
        torch.nn.utils.clip_grad_norm_(params, tcfg["sft"]["max_grad_norm"])
        lr = lr_at(step, total_steps, warmup, gcfg["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        timer.record_rollout(t1 - t0, t2 - t1, [len(r.completion_ids) for r in rollouts])
        timer.end(learned["forward_tokens"])

        totals = _all_reduce_sum(_metrics_vector(rollouts, rows, G, learned), device)
        step += 1
        record = _step_record(totals, global_tokens, step, lr)
        log.append(record)
        if gcfg["eval_every"] and step % gcfg["eval_every"] == 0 and held_prompts:
            eval_curve[step] = evaluate_heldout(
                inner, heldout_rows, held_prompts, scorer, tok, gcfg, device, im_end
            )
        if report is not None:
            report({"step": step, "reward": record["reward"], "kl": record["kl"]})
        stopping = stop_after is not None and step >= stop_after
        if stopping or (gcfg["checkpoint_every"] and step % gcfg["checkpoint_every"] == 0):
            _save_resume(net, optimizer, resume_dir, step, log, eval_curve)
        if stopping:
            interrupted = True
            break

    util = sampler.stop()
    losses = [r["loss"] for r in log]
    if not interrupted:
        if held_prompts and total_steps not in eval_curve:
            eval_curve[total_steps] = evaluate_heldout(
                inner, heldout_rows, held_prompts, scorer, tok, gcfg, device, im_end
            )
        run_check(
            "weights_identical",
            lambda: checks.check_checksums_equal(
                checks.gather(checks.param_checksum(net)), "adapter weight checksums"
            ),
        )
        run_check(
            "metrics_identical",
            lambda: checks.check_checksums_equal(
                checks.gather(sum(r["reward"] + 1.0 for r in log)), "reward curves"
            ),
        )
        run_check("loss_finite", lambda: _finite(losses))
        run_check(
            "rollout_health",
            lambda: checks.check_rollout_health(
                [r["reward"] for r in log],
                [r["zero_variance_groups"] for r in log],
                [r["truncated"] for r in log],
            ),
        )
        if is_cuda:
            run_check(
                "gpu_busy",
                lambda: checks.check_gpu_util(
                    dict(enumerate(checks.gather(util))), tcfg["profiling"]["min_median_gpu_util"]
                ),
            )
        else:
            results["gpu_busy"] = {"skipped": "no GPU (measured in the paid run)"}

    # ---- outputs
    stats = timer.finish()
    all_stats = checks.gather(stats)
    record = {
        "stage": "grpo",
        "strategy": parallel.DDP,
        "world_size": world,
        "steps": step,
        "interrupted": interrupted,
        "global_losses": losses,
        "log": log,
        "heldout_reward": {str(k): v for k, v in sorted(eval_curve.items())},
        "train_prompts": len(train_rows),
        "heldout_prompts": len(heldout_rows),
        "checks": results,
    }
    if not interrupted:
        parallel.save_lora(net, parallel.DDP, out_dir / "adapter")
        if rank == 0:
            tok.tok.save_pretrained(out_dir / "adapter")
            model_lib.write_chain(out_dir / "adapter", init_from)
    if rank == 0:
        profile = profiling.build_profile(
            "grpo",
            all_stats,
            total_params=total_params,
            trainable_params=trainable_params,
            meta={
                "mode": "lora",
                "strategy": parallel.DDP,
                "group_size": G,
                "prompts_per_step": ppstep,
                "completions_per_step": G * ppstep,
                "max_new_tokens": gcfg["max_new_tokens"],
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


def _save_resume(net, optimizer, resume_dir: Path, step: int, log: list, eval_curve: dict) -> None:
    parallel.save_resume_state(net, optimizer, parallel.DDP, resume_dir / "dcp", True)
    if dist.get_rank() == 0:
        (resume_dir / "trainer_state.json").write_text(
            json.dumps({"step": step, "log": log, "eval": eval_curve}), encoding="utf-8"
        )
    dist.barrier()
