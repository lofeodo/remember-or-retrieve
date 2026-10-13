"""Per-stage profiling: one `profile.json` schema for every training stage.

Reports tokens/sec (global and per device), step-time percentiles, peak memory per device,
trainable vs total parameters, an approximate MFU, and, over a short window, how much of the
step is communication versus compute. Step 6 adds rollout throughput to the same schema.
"""

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

COMM_MARKERS = ("nccl", "gloo", "all_reduce", "allreduce", "reduce_scatter", "all_gather")


@dataclass
class RankStats:
    """What one rank measured. Gathered to rank 0 with all_gather_object."""

    rank: int
    device: str
    step_times_s: list[float] = field(default_factory=list)  # post-warmup only
    step_tokens: list[int] = field(default_factory=list)  # non-pad target+prompt tokens, this rank
    peak_mem_bytes: int = 0
    mem_kind: str = "cuda_allocated"  # or "cpu_rss" when no GPU
    load_s: float = 0.0
    comm_fraction: float | None = None


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def build_profile(
    stage: str,
    ranks: list[RankStats],
    *,
    total_params: int,
    trainable_params: int,
    meta: dict,
    wall_clock_s: float,
    peak_tflops: float | None = None,
) -> dict:
    """Aggregate per-rank stats into the stage profile. Pure, so it is unit-tested directly.

    A step's duration is the slowest rank's (ranks synchronise every step), and its tokens are
    summed across ranks. MFU uses 6 x params x tokens, which ignores attention and activation
    recompute, so it is a rough number meant for comparing stages, not a hardware benchmark.
    """
    ranks = sorted(ranks, key=lambda r: r.rank)
    n_steps = min((len(r.step_times_s) for r in ranks), default=0)
    step_times = [max(r.step_times_s[i] for r in ranks) for i in range(n_steps)]
    step_tokens = [sum(r.step_tokens[i] for r in ranks) for i in range(n_steps)]
    train_s = sum(step_times)
    tokens = sum(step_tokens)
    world = len(ranks)
    tps = tokens / train_s if train_s else 0.0
    on_gpu = bool(ranks) and ranks[0].mem_kind == "cuda_allocated"
    mfu = None
    if on_gpu and peak_tflops and train_s:
        mfu = (6 * total_params * tokens / train_s) / (peak_tflops * 1e12 * world)
    fractions = [r.comm_fraction for r in ranks if r.comm_fraction is not None]
    return {
        "stage": stage,
        **meta,
        "world_size": world,
        "devices": [r.device for r in ranks],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "measured_steps": n_steps,
        "tokens_measured": tokens,
        "tokens_per_s_global": tps,
        "tokens_per_s_per_device": tps / world if world else 0.0,
        "step_time_p50_s": percentile(step_times, 0.5),
        "step_time_p95_s": percentile(step_times, 0.95),
        "peak_mem_gb_per_rank": [round(r.peak_mem_bytes / 1e9, 3) for r in ranks],
        "mem_kind": ranks[0].mem_kind if ranks else None,
        "mfu_approx": mfu,
        "comm_fraction": statistics.mean(fractions) if fractions else None,
        "load_s": max((r.load_s for r in ranks), default=0.0),
        "train_s": train_s,
        "wall_clock_s": wall_clock_s,
    }


def write_profile(profile: dict, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "profile.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return path


def comm_fraction_from_events(events: list[tuple[str, float]]) -> float | None:
    """Share of profiled time spent in collective-communication ops. `events` is (name, time)."""
    total = sum(t for _, t in events)
    if total <= 0:
        return None
    comm = sum(t for name, t in events if any(m in name.lower() for m in COMM_MARKERS))
    return comm / total


class StepTimer:
    """Times optimizer steps on one rank, excluding warmup, and tracks peak memory."""

    def __init__(self, rank: int, warmup_steps: int, device):
        import torch

        self.torch = torch
        self.warmup = warmup_steps
        self.cuda = device.type == "cuda"
        self.stats = RankStats(rank=rank, device=str(device))
        if self.cuda:
            self.stats.device = f"cuda:{device.index} {torch.cuda.get_device_name(device)}"
            torch.cuda.reset_peak_memory_stats(device)
        else:
            self.stats.mem_kind = "cpu_rss"
        self.device = device
        self.step = 0
        self._t0 = 0.0

    def begin(self) -> None:
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()

    def end(self, tokens: int) -> None:
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - self._t0
        if self.step >= self.warmup:
            self.stats.step_times_s.append(dt)
            self.stats.step_tokens.append(tokens)
        self.step += 1

    def finish(self) -> RankStats:
        if self.cuda:
            self.stats.peak_mem_bytes = int(self.torch.cuda.max_memory_allocated(self.device))
        else:
            try:
                import resource

                self.stats.peak_mem_bytes = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                )
            except ImportError:  # Windows
                self.stats.peak_mem_bytes = 0
        return self.stats


class TraceWindow:
    """torch.profiler over `steps` optimizer steps after warmup; saves a chrome trace and reports
    the communication share of profiled time. A no-op when `steps` is 0."""

    def __init__(self, steps: int, start_step: int, device, trace_path: Path | None):
        self.steps, self.start, self.trace_path = steps, start_step, trace_path
        self.device = device
        self.prof = None
        self.fraction: float | None = None
        self._seen = 0

    def maybe_start(self, step: int) -> None:
        if not self.steps or self.prof is not None or step != self.start:
            return
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            acts.append(ProfilerActivity.CUDA)
        self.prof = profile(activities=acts)
        self.prof.__enter__()

    def close(self) -> None:
        """Stop an open profiler (a run that ends inside the window, for example on preemption)."""
        if self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.prof = None

    def maybe_stop(self) -> None:
        if self.prof is None:
            return
        self._seen += 1
        if self._seen < self.steps:
            return
        self.prof.__exit__(None, None, None)
        key = "self_device_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
        events = [(e.key, float(getattr(e, key, 0.0))) for e in self.prof.key_averages()]
        self.fraction = comm_fraction_from_events(events)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.prof.export_chrome_trace(str(self.trace_path))
        self.prof = None
        self.steps = 0
