"""The distribution checks: evidence that a run is one real distributed job.

Renting several GPUs and launching an ordinary script is not distributed training, so every run
verifies itself instead of assuming: ranks really started, gradients really sync, weights stay
identical (DDP) or really sharded (FSDP2), every GPU is busy, the loss curve is one curve, and a
multi-rank run reproduces the single-process run on the same global batch.

The `check_*` functions are pure (given gathered numbers) so they are unit-tested, including that a
desynced run fails. The `collect_*` helpers gather those numbers from live ranks.
"""

import math
import os
import socket
import statistics
import threading


class DistributionCheckError(AssertionError):
    """A distribution check failed: the run is not the distributed job it claims to be."""


def _fail(msg: str) -> None:
    raise DistributionCheckError(msg)


# ---------------------------------------------------------------- pure checks


def check_ranks(reports: list[dict], expected_world: int) -> dict:
    """Every expected rank started, agrees on the world size, and has its own process and GPU."""
    ranks = sorted(r["rank"] for r in reports)
    if ranks != list(range(expected_world)):
        _fail(f"expected ranks 0..{expected_world - 1}, got {ranks}")
    for r in reports:
        if r["world_size"] != expected_world:
            _fail(f"rank {r['rank']} sees world_size {r['world_size']}, expected {expected_world}")
    procs = [(r["host"], r["pid"]) for r in reports]
    if len(set(procs)) != len(procs):
        _fail("two ranks share a process: this is not multi-process training")
    gpus = [(r["host"], r["device"]) for r in reports if str(r["device"]).startswith("cuda")]
    if len(set(gpus)) != len(gpus):
        _fail("two ranks share a GPU")
    return {"ranks": ranks, "hosts": sorted({r["host"] for r in reports})}


def check_checksums_equal(values: list[float], what: str, rel_tol: float = 1e-6) -> float:
    """All ranks hold the same value (synced gradients, or identical DDP weights)."""
    spread = max(values) - min(values)
    scale = max(abs(v) for v in values) or 1.0
    if spread / scale > rel_tol:
        _fail(f"{what} differ across ranks (spread {spread:.3e} of {scale:.3e}): not synchronised")
    return spread


def check_sharded(local_numels: list[int], total_numel: int, tol: float = 0.2) -> dict:
    """FSDP: each rank holds about 1/world of the parameters, and together they hold all of them."""
    world = len(local_numels)
    if sum(local_numels) < total_numel * (1 - 1e-9):
        _fail(f"shards hold {sum(local_numels)} of {total_numel} parameters")
    even = total_numel / world
    for rank, n in enumerate(local_numels):
        if abs(n - even) > tol * even:
            _fail(f"rank {rank} holds {n} parameters, expected about {even:.0f}: not sharded")
    return {"local_numels": local_numels, "total_numel": total_numel}


def check_gpu_util(samples_by_rank: dict[int, list[float]], min_median: float) -> dict:
    """Every device, not just rank 0, shows real utilisation during training."""
    medians = {}
    for rank, samples in samples_by_rank.items():
        if not samples:
            _fail(f"rank {rank} recorded no utilisation samples")
        medians[rank] = statistics.median(samples)
        if medians[rank] < min_median:
            _fail(f"GPU of rank {rank} median utilisation {medians[rank]:.0f}% < {min_median}%")
    return {"median_util_pct": medians}


def check_loss_coherent(
    global_losses_by_rank: list[list[float]],
    local_losses_by_rank: list[list[float]],
    global_rel_tol: float,
    local_band: float,
) -> dict:
    """One curve, not several independent ones.

    Every rank must have observed the same all-reduced loss at every step. Each rank's own
    slice loss differs legitimately (different examples) but must stay in a band around it; a
    rank training independently or on garbage falls outside.
    """
    n = min(len(c) for c in global_losses_by_rank)
    for step in range(n):
        vals = [c[step] for c in global_losses_by_rank]
        check_checksums_equal(vals, f"global loss at step {step}", global_rel_tol)
        ref = vals[0]
        for rank, curve in enumerate(local_losses_by_rank):
            if abs(curve[step] - ref) > local_band * max(abs(ref), 1e-8):
                _fail(
                    f"rank {rank} local loss {curve[step]:.4f} far from global {ref:.4f} at {step}"
                )
    return {"steps_checked": n}


def check_loss_decreasing(curve: list[float], min_drop: float = 0.0) -> dict:
    """Mean of the last third is below the mean of the first third."""
    k = max(1, len(curve) // 3)
    head, tail = statistics.mean(curve[:k]), statistics.mean(curve[-k:])
    if not tail < head - min_drop:
        _fail(f"loss did not decrease: first third {head:.4f}, last third {tail:.4f}")
    return {"first_third": head, "last_third": tail}


def check_rollout_health(
    rewards: list[float], zero_variance: list[float], truncated: list[float]
) -> dict:
    """An RL run is learning from something: per-step mean rewards are finite, some groups had
    spread (all-identical rewards give zero advantage, so no gradient), and completions are not
    all cut off at the length limit."""
    if not rewards:
        _fail("no RL steps ran")
    if not all(math.isfinite(r) for r in rewards):
        _fail("non-finite reward")
    if min(zero_variance) >= 1.0:
        _fail("every group had identical rewards at every step: no learning signal")
    if min(truncated) >= 1.0:
        _fail("every completion hit the length limit at every step")
    return {
        "steps": len(rewards),
        "mean_zero_variance_groups": statistics.mean(zero_variance),
        "mean_truncated": statistics.mean(truncated),
    }


def check_equivalence(curve_a: list[float], curve_b: list[float], rel_tol: float) -> dict:
    """Two runs on the same global batches (for example 1 rank vs 2 ranks) trace the same curve."""
    n = min(len(curve_a), len(curve_b))
    if n == 0:
        _fail("empty loss curve")
    worst = 0.0
    for i in range(n):
        d = abs(curve_a[i] - curve_b[i]) / max(abs(curve_a[i]), 1e-8)
        worst = max(worst, d)
        if d > rel_tol:
            _fail(f"step {i}: {curve_a[i]:.5f} vs {curve_b[i]:.5f} (rel diff {d:.2e} > {rel_tol})")
    if not all(math.isfinite(x) for x in curve_a + curve_b):
        _fail("non-finite loss")
    return {"steps": n, "worst_rel_diff": worst}


# ---------------------------------------------------------- collectors (torch)


def rank_report(rank: int, world_size: int, device) -> dict:
    return {
        "rank": rank,
        "world_size": world_size,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": str(device),
    }


def gather(obj) -> list:
    """all_gather_object across the process group (a one-element list without one)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def grad_checksum(model) -> float:
    """Sum of |grad| over this rank's gradients (a local shard under FSDP, so DDP only)."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().double().abs().sum())
    return total


def param_checksum(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total += float(p.detach().double().abs().sum())
    return total


def shard_report(model) -> tuple[int, int, int]:
    """(local numel, global numel, number of DTensor params) over trainable parameters."""
    from torch.distributed.tensor import DTensor

    local = global_ = n_dt = 0
    for p in model.parameters():
        if isinstance(p, DTensor):
            n_dt += 1
            local += p.to_local().numel()
        else:
            local += p.numel()
        global_ += p.numel()
    return local, global_, n_dt


class GpuUtilSampler:
    """Polls NVML utilisation of this rank's GPU in a thread. Inconclusive (empty) off-GPU."""

    def __init__(self, device, period_s: float):
        self.samples: list[float] = []
        self.period = period_s
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        if getattr(device, "type", "") != "cuda":
            return
        try:
            import pynvml
            import torch

            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(
                uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
            )
            self._nvml = pynvml
        except Exception:  # NVML missing or UUID lookup failed: report inconclusive, don't crash
            self._handle = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            self.samples.append(float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu))

    def start(self) -> None:
        if self.active:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> list[float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples
