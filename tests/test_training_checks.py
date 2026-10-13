"""Fast tests (no torch) for the distribution checks and the profile aggregation."""

import pytest

from remember_or_retrieve.training import checks, profiling


def reports(n=2, **override):
    out = [
        {"rank": r, "world_size": n, "host": "h", "pid": 100 + r, "device": f"cuda:{r}"}
        for r in range(n)
    ]
    for r in out:
        r.update(override)
    return out


def test_ranks_started_passes_for_a_real_job():
    assert checks.check_ranks(reports(4), 4)["ranks"] == [0, 1, 2, 3]


def test_missing_rank_wrong_world_shared_process_or_gpu_all_fail():
    with pytest.raises(checks.DistributionCheckError):
        checks.check_ranks(reports(2)[:1], 2)
    with pytest.raises(checks.DistributionCheckError):
        checks.check_ranks(reports(2, world_size=1), 2)
    with pytest.raises(checks.DistributionCheckError, match="share a process"):
        checks.check_ranks(reports(2, pid=7), 2)
    with pytest.raises(checks.DistributionCheckError, match="share a GPU"):
        checks.check_ranks(
            reports(2, device="cuda:0") and [{**r, "device": "cuda:0"} for r in reports(2)], 2
        )


def test_desynced_gradients_or_weights_fail():
    checks.check_checksums_equal([3.0, 3.0], "grads")
    with pytest.raises(checks.DistributionCheckError, match="not synchronised"):
        checks.check_checksums_equal([3.0, 3.5], "grads")


def test_sharding_check():
    checks.check_sharded([50, 50], 100)
    with pytest.raises(checks.DistributionCheckError, match="not sharded"):
        checks.check_sharded([100, 100], 100)  # every rank holds a full copy
    with pytest.raises(checks.DistributionCheckError):
        checks.check_sharded([40, 40], 100)  # parameters missing


def test_gpu_util_needs_every_device_busy():
    checks.check_gpu_util({0: [80, 90], 1: [70, 75]}, 30)
    with pytest.raises(checks.DistributionCheckError, match="rank 1"):
        checks.check_gpu_util({0: [90, 95], 1: [2, 3]}, 30)  # one idle GPU
    with pytest.raises(checks.DistributionCheckError):
        checks.check_gpu_util({0: [90], 1: []}, 30)


def test_loss_coherence_needs_one_curve():
    glob = [[2.0, 1.5], [2.0, 1.5]]
    local = [[2.1, 1.4], [1.9, 1.6]]
    checks.check_loss_coherent(glob, local, 1e-6, 0.75)
    with pytest.raises(checks.DistributionCheckError, match="global loss"):
        checks.check_loss_coherent([[2.0], [2.2]], local, 1e-6, 0.75)  # ranks disagree
    with pytest.raises(checks.DistributionCheckError, match="far from global"):
        checks.check_loss_coherent(glob, [[2.0, 1.5], [9.0, 1.5]], 1e-6, 0.75)


def test_loss_decreasing_and_equivalence():
    checks.check_loss_decreasing([3, 2.8, 2.5, 2.2, 2.0, 1.8])
    with pytest.raises(checks.DistributionCheckError):
        checks.check_loss_decreasing([2.0, 2.0, 2.0, 2.1, 2.2, 2.3])
    checks.check_equivalence([1.0, 0.9], [1.0001, 0.9], 1e-3)
    with pytest.raises(checks.DistributionCheckError, match="step 1"):
        checks.check_equivalence([1.0, 0.9], [1.0, 0.5], 1e-3)


def rank(r, times, tokens, mem=1_000_000_000):
    return profiling.RankStats(
        rank=r,
        device=f"cuda:{r}",
        step_times_s=times,
        step_tokens=tokens,
        peak_mem_bytes=mem,
        mem_kind="cuda_allocated",
    )


def test_profile_uses_slowest_rank_and_sums_tokens():
    p = profiling.build_profile(
        "sft_full",
        [rank(0, [1.0, 2.0], [100, 100]), rank(1, [1.5, 1.0], [100, 100])],
        total_params=1000,
        trainable_params=1000,
        meta={"mode": "full"},
        wall_clock_s=10.0,
        peak_tflops=312,
    )
    assert p["train_s"] == 3.5  # max(1.0, 1.5) + max(2.0, 1.0)
    assert p["tokens_measured"] == 400
    assert p["tokens_per_s_global"] == pytest.approx(400 / 3.5)
    assert p["tokens_per_s_per_device"] == pytest.approx(400 / 3.5 / 2)
    assert p["peak_mem_gb_per_rank"] == [1.0, 1.0]
    assert 0 < p["mfu_approx"] < 1


def test_profile_without_gpu_has_no_mfu_and_empty_is_safe():
    cpu = profiling.RankStats(rank=0, device="cpu", mem_kind="cpu_rss")
    p = profiling.build_profile(
        "x", [cpu], total_params=1, trainable_params=1, meta={}, wall_clock_s=1, peak_tflops=312
    )
    assert p["mfu_approx"] is None and p["tokens_per_s_global"] == 0.0


def test_comm_fraction_counts_collectives():
    ev = [("nccl:all_reduce", 2.0), ("aten::mm", 6.0), ("gloo:all_gather", 2.0)]
    assert profiling.comm_fraction_from_events(ev) == pytest.approx(0.4)
    assert profiling.comm_fraction_from_events([]) is None
