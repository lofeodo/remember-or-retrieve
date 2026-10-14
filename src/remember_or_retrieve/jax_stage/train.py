"""LoRA SFT of Gemma 4 on N devices with jit + `jax.sharding` (data parallel).

The loss is a summed token cross-entropy divided by the *global* target-token count of each
global batch, exactly as in the PyTorch SFT stage, so a 1-device and an N-device run are the same
training job (checked by `check_equivalence` in the smoke test). A global batch is split into
micro-batches of `micro_batch_size x devices` rows; gradients of the summed loss are accumulated
across them, then normalised once.
"""

import contextlib
import json
import math
import socket
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from remember_or_retrieve.config import resolve_path
from remember_or_retrieve.jax_stage import checks
from remember_or_retrieve.jax_stage import model as model_lib
from remember_or_retrieve.jax_stage.parallel import DataParallel
from remember_or_retrieve.training import data, profiling

IGNORE = data.IGNORE_INDEX


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup then cosine decay to 10% of peak (same schedule as the PyTorch SFT stage)."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def pad_to_multiple(width: int, multiple: int) -> int:
    return -(-width // multiple) * multiple


def make_batch(examples: list[dict], pad_id: int, width_multiple: int) -> dict[str, np.ndarray]:
    """Right-padded int32 arrays. Widths are rounded up to `width_multiple`, so XLA compiles a
    handful of shapes rather than one per distinct batch length."""
    b = data.collate(examples, pad_id)
    width = pad_to_multiple(len(b["input_ids"][0]), width_multiple)
    tokens = np.full((len(examples), width), pad_id, np.int32)
    labels = np.full((len(examples), width), IGNORE, np.int32)
    for i, (ids, lab) in enumerate(zip(b["input_ids"], b["labels"], strict=True)):
        tokens[i, : len(ids)] = ids
        labels[i, : len(lab)] = lab
    return {"tokens": tokens, "labels": labels}


def dummy_example(pad_id: int) -> dict:
    """A row with no targets, used to fill the last micro-batch of an evaluation."""
    return {"input_ids": [pad_id], "labels": [IGNORE]}


def nll_sum(logits, labels):
    """(summed cross-entropy over target tokens, number of target tokens)."""
    logits = logits[:, :-1].astype(jnp.float32)
    targets = labels[:, 1:]
    mask = targets != IGNORE
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.where(mask, targets, 0))
    return (losses * mask).sum(), mask.sum()


def build_steps(lm, par: DataParallel, tx, compute_dtype):
    """The three jitted functions: gradient of a micro-batch, optimizer update, evaluation."""
    rep, bat = par.replicated, par.batch

    def loss_fn(lora, base, tokens, labels):
        variables = model_lib.merge(base, lora, compute_dtype)
        logits = lm.apply(variables, tokens=tokens).logits
        return nll_sum(logits, labels)

    def _grad(lora, base, tokens, labels):
        (nll, n), grads = jax.value_and_grad(loss_fn, has_aux=True)(lora, base, tokens, labels)
        return grads, nll, n

    grad_step = jax.jit(_grad, in_shardings=(rep, rep, bat, bat), out_shardings=(rep, rep, rep))

    eval_step = jax.jit(
        lambda lora, base, tokens, labels: loss_fn(lora, base, tokens, labels),
        in_shardings=(rep, rep, bat, bat),
        out_shardings=(rep, rep),
    )

    @jax.jit
    def add_grads(acc, g):
        return jax.tree.map(jnp.add, acc, g)

    def _apply(lora, opt_state, grads, total_tokens, lr):
        grads = jax.tree.map(lambda g: g / total_tokens, grads)
        updates, opt_state = tx.update(grads, opt_state, lora)
        lora = optax.apply_updates(lora, jax.tree.map(lambda u: -lr * u, updates))
        return lora, opt_state

    apply_step = jax.jit(_apply, in_shardings=(rep, rep, rep, rep, rep), out_shardings=(rep, rep))
    return grad_step, eval_step, add_grads, apply_step


def make_optimizer(sft: dict):
    return optax.chain(
        optax.clip_by_global_norm(sft["max_grad_norm"]),
        optax.scale_by_adam(b1=0.9, b2=0.95),
        optax.add_decayed_weights(sft["weight_decay"]),
    )


def _device_peak_bytes(device) -> int:
    try:
        stats = device.memory_stats()
    except Exception:
        stats = None
    if stats and "peak_bytes_in_use" in stats:
        return int(stats["peak_bytes_in_use"])
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        return 0


def _checkpointer():
    import orbax.checkpoint as ocp

    return ocp.StandardCheckpointer()


def save_resume(out_dir: Path, lora, opt_state, step: int, losses: list[float]) -> None:
    ck = _checkpointer()
    path = (out_dir / "resume" / "state").resolve()
    ck.save(path, {"lora": lora, "opt": opt_state}, force=True)
    ck.wait_until_finished()
    (out_dir / "resume" / "trainer_state.json").write_text(
        json.dumps({"step": step, "global": losses}), encoding="utf-8"
    )


def load_resume(out_dir: Path, lora, opt_state):
    ck = _checkpointer()
    restored = ck.restore(
        (out_dir / "resume" / "state").resolve(), {"lora": lora, "opt": opt_state}
    )
    st = json.loads((out_dir / "resume" / "trainer_state.json").read_text(encoding="utf-8"))
    return restored["lora"], restored["opt"], st["step"], st["global"]


def save_adapter(out_dir: Path, lora) -> None:
    ck = _checkpointer()
    ck.save((out_dir / "adapter").resolve(), lora, force=True)
    ck.wait_until_finished()


def evaluate(eval_step, lora, base, val, pad_id, micro_global, par, multiple) -> float:
    total, count = 0.0, 0
    for i in range(0, len(val), micro_global):
        chunk = val[i : i + micro_global]
        chunk = chunk + [dummy_example(pad_id)] * (micro_global - len(chunk))
        b = make_batch(chunk, pad_id, multiple)
        nll, n = eval_step(lora, base, par.put_batch(b["tokens"]), par.put_batch(b["labels"]))
        total += float(nll)
        count += int(n)
    return total / max(count, 1)


def train(
    cfg: dict,
    *,
    num_devices: int,
    out_dir: Path,
    smoke: bool = False,
    resume: bool = False,
    stop_after: int | None = None,
    tokenizer=None,
    perturb_replica: bool = False,
) -> dict:
    """Run the stage. Returns the run record (also written to `run.json`).

    `stop_after` ends after that optimizer step, leaving a resume checkpoint, as a spot preemption
    would. `perturb_replica` deliberately corrupts one device's adapters at the end so the
    identical-replicas check can be shown to fail (used once, by the broken-run test).
    """
    t_start = time.perf_counter()
    jcfg = cfg["jax"]
    sft = jcfg["sft"]
    pcfg = jcfg["profiling"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compute_dtype = jnp.float32 if smoke else model_lib.DTYPES[jcfg["model"]["dtype"]]

    par = DataParallel(num_devices, jcfg["mesh_axis"])
    from remember_or_retrieve.jax_stage.tokenization import load_tokenizer

    tok = tokenizer or load_tokenizer(cfg)
    pad_id = tok.pad_token_id
    max_len = jcfg["max_length"]
    multiple = jcfg.get("pad_multiple", 64)

    # ---- data
    train_dir = resolve_path(cfg["training_data"]["out_dir"])
    train_rows = data.read_jsonl(train_dir / "sft_train.jsonl")
    val_rows = data.read_jsonl(train_dir / "sft_val.jsonl")
    if smoke:
        train_rows = train_rows[: jcfg["smoke"]["rows"]]
        val_rows = val_rows[:16]
    train_ex = data.encode_sft(tok, train_rows, max_len, tok.end_token)
    val_ex = data.encode_sft(tok, val_rows, max_len, tok.end_token)
    lengths = [len(e["input_ids"]) for e in train_ex]
    target_counts = [sum(1 for t in e["labels"] if t != IGNORE) for e in train_ex]
    gbs, micro = sft["global_batch_size"], sft["micro_batch_size"]
    if smoke:
        gbs, micro = 8, 2
    micro_global = micro * num_devices
    if gbs % micro_global:
        raise ValueError(
            f"global batch {gbs} must be a multiple of {micro} x {num_devices} devices"
        )
    accum = gbs // micro_global
    steps_per_epoch = len(train_ex) // gbs
    total_steps = sft["epochs"] * steps_per_epoch
    if smoke:
        total_steps = min(total_steps, jcfg["smoke"]["steps"])
    warmup = max(1, int(sft["warmup_ratio"] * total_steps))
    peak_lr = jcfg["smoke"]["lr"] if smoke else sft["lr"]

    # ---- model
    t_load = time.perf_counter()
    lm, base_model = model_lib.build_model(cfg, smoke=smoke)
    if smoke:
        base = model_lib.init_base_random(base_model, sft["seed"], dtype=compute_dtype)
    else:
        base = model_lib.load_base_checkpoint(cfg, compute_dtype)
    lora = model_lib.init_lora(lm, sft["seed"])
    total_params = model_lib.count_params(base) + model_lib.count_params(lora)
    trainable_params = model_lib.count_params(lora)
    base, lora = par.replicate(base), par.replicate(lora)
    tx = make_optimizer(sft)
    opt_state = par.replicate(tx.init(lora))
    load_s = time.perf_counter() - t_load
    grad_step, eval_step, add_grads, apply_step = build_steps(lm, par, tx, compute_dtype)

    start_step, global_hist = 0, []
    if resume and (out_dir / "resume" / "trainer_state.json").exists():
        lora, opt_state, start_step, global_hist = load_resume(out_dir, lora, opt_state)

    # ---- checks and bookkeeping
    check_results: dict = {}
    failures: list[str] = []

    def run_check(name, fn):
        try:
            check_results[name] = {"passed": True, "detail": fn()}
        except checks.DistributionCheckError as e:
            check_results[name] = {"passed": False, "error": str(e)}
            failures.append(f"{name}: {e}")

    run_check(
        "devices_started",
        lambda: checks.check_devices([checks.device_report(d) for d in par.devices], num_devices),
    )
    sampler = checks.GpuUtilSampler(par.devices, pcfg.get("gpu_util_sample_s", 1.0))
    sampler.start()

    step_times: list[float] = []
    step_tokens: list[int] = []
    per_device_tokens: list[list[int]] = []
    first_step_s = 0.0
    val_losses: dict[int, float] = {}
    trace_dir = out_dir / "jax_trace"
    tracing = False
    step = start_step
    interrupted = False
    while step < total_steps:
        epoch, i = divmod(step, steps_per_epoch)
        batch_ids = data.global_batches(len(train_ex), gbs, sft["seed"], epoch)[i]
        order = sorted(batch_ids, key=lambda j: (lengths[j], j))
        global_tokens = float(sum(target_counts[j] for j in batch_ids))
        if pcfg["trace_steps"] and step == start_step + pcfg["warmup_steps"] and not tracing:
            jax.profiler.start_trace(str(trace_dir))
            tracing = True
        t0 = time.perf_counter()
        acc, nll_total, tok_in_step = None, 0.0, 0
        dev_tokens = [0] * num_devices
        for k in range(accum):
            ids = order[k * micro_global : (k + 1) * micro_global]
            b = make_batch([train_ex[j] for j in ids], pad_id, multiple)
            tokens_dev = par.put_batch(b["tokens"])
            labels_dev = par.put_batch(b["labels"])
            if step == start_step and k == 0:
                run_check(
                    "batch_sharded",
                    lambda arr=tokens_dev: checks.check_batch_sharded(
                        checks.shard_shapes(arr), tuple(arr.shape)
                    ),
                )
            g, nll, _ = grad_step(lora, base, tokens_dev, labels_dev)
            acc = g if acc is None else add_grads(acc, g)
            nll_total += float(nll)
            real = (b["tokens"] != pad_id) | (b["labels"] != IGNORE)
            for d, rows in enumerate(np.array_split(real, num_devices)):
                dev_tokens[d] += int(rows.sum())
            tok_in_step += int(real.sum())
        lr = lr_at(step, total_steps, warmup, peak_lr)
        lora, opt_state = apply_step(
            lora, opt_state, acc, jnp.float32(global_tokens), jnp.float32(lr)
        )
        jax.block_until_ready(lora)
        dt = time.perf_counter() - t0
        if step == start_step:
            first_step_s = dt
        if step - start_step >= pcfg["warmup_steps"]:
            step_times.append(dt)
            step_tokens.append(tok_in_step)
            per_device_tokens.append(dev_tokens)
        if tracing and step - start_step >= pcfg["warmup_steps"] + pcfg["trace_steps"] - 1:
            jax.profiler.stop_trace()
            tracing = False
        global_hist.append(nll_total / global_tokens)
        step += 1
        if sft["eval_every"] and not smoke and step % sft["eval_every"] == 0 and val_ex:
            val_losses[step] = evaluate(
                eval_step, lora, base, val_ex, pad_id, micro_global, par, multiple
            )
        stopping = stop_after is not None and step >= stop_after
        if stopping or (sft["checkpoint_every"] and step % sft["checkpoint_every"] == 0):
            save_resume(out_dir, lora, opt_state, step, global_hist)
        if stopping:
            interrupted = True
            break
    if tracing:
        with contextlib.suppress(Exception):
            jax.profiler.stop_trace()
    util = sampler.stop()

    # ---- run-level checks
    if perturb_replica and num_devices > 1:
        lora = _perturb_one_replica(lora, par)
    if not interrupted:
        run_check(
            "replicas_identical",
            lambda: {
                "spread": checks.check_checksums_equal(
                    checks.replica_checksums(lora), "LoRA parameter replicas", rel_tol=0.0
                )
            },
        )
        run_check("loss_decreasing", lambda: checks.check_loss_decreasing(global_hist))
        run_check(
            "loss_finite",
            lambda: {"finite": all(math.isfinite(x) for x in global_hist) or _raise_nonfinite()},
        )
        if par.devices[0].platform == "gpu" and sampler.active:
            run_check(
                "gpu_busy",
                lambda: checks.check_gpu_util(util, pcfg.get("min_median_gpu_util", 30)),
            )
        else:
            check_results["gpu_busy"] = {"skipped": "no GPU (measured in the paid run)"}

    # ---- outputs
    record = {
        "stage": "jax_lora",
        "framework": "jax",
        "model": "tiny-gemma4" if smoke else jcfg["model"]["name"],
        "world_size": num_devices,
        "steps": step,
        "interrupted": interrupted,
        "global_losses": global_hist,
        "val_losses": val_losses,
        "checks": check_results,
        "first_step_s": first_step_s,
        "host": socket.gethostname(),
    }
    if not interrupted:
        save_adapter(out_dir, lora)
    ranks = []
    for d, dev in enumerate(par.devices):
        rs = profiling.RankStats(rank=d, device=f"{dev.platform}:{dev.id} {dev.device_kind}")
        rs.step_times_s = list(step_times)
        rs.step_tokens = [row[d] for row in per_device_tokens]
        rs.peak_mem_bytes = _device_peak_bytes(dev)
        rs.mem_kind = "device_peak" if dev.platform == "gpu" else "cpu_rss"
        rs.load_s = load_s
        ranks.append(rs)
    steady = profiling.percentile(step_times, 0.5)
    profile = profiling.build_profile(
        "jax_lora",
        ranks,
        total_params=total_params,
        trainable_params=trainable_params,
        meta={
            "framework": "jax",
            "strategy": "jit_data_parallel",
            "model": record["model"],
            "global_batch_size": gbs,
            "micro_batch_size": micro,
            "grad_accum": accum,
            "smoke": smoke,
            "first_step_s": first_step_s,
            "compile_s_est": max(0.0, first_step_s - steady),
        },
        wall_clock_s=time.perf_counter() - t_start,
        peak_tflops=pcfg["a100_bf16_peak_tflops"],
    )
    profiling.write_profile(profile, out_dir)
    (out_dir / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    if failures:
        raise checks.DistributionCheckError("; ".join(failures))
    return record


def _raise_nonfinite():
    raise checks.DistributionCheckError("non-finite loss in the run")


def _perturb_one_replica(lora, par: DataParallel):
    """Corrupt one device's copy of the adapters (test of the identical-replicas check only)."""
    leaves, treedef = jax.tree.flatten(lora)
    out = []
    for leaf in leaves:
        shards = []
        for s in leaf.addressable_shards:
            arr = np.asarray(s.data)
            if s.device.id == par.devices[-1].id:
                arr = arr + 1e-3
            shards.append(jax.device_put(arr, s.device))
        out.append(jax.make_array_from_single_device_arrays(leaf.shape, par.replicated, shards))
    return jax.tree.unflatten(treedef, out)
