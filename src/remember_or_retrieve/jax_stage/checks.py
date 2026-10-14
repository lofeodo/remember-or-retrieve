"""Distribution checks for the JAX stage: evidence this is one data-parallel job over N devices.

Same discipline as `training.checks` (assert it, never assume it), expressed for JAX's single
controller model: all devices belong to one process, so "ranks started" becomes "N distinct devices
are in the mesh", and "gradients synced" becomes "the replicated LoRA parameters are bitwise
identical on every device after training". Pure checks are tested without JAX; the collectors take
live arrays.
"""

import threading

from remember_or_retrieve.training.checks import (  # noqa: F401  (re-exported for callers)
    DistributionCheckError,
    _fail,
    check_checksums_equal,
    check_equivalence,
    check_gpu_util,
    check_loss_decreasing,
)


def device_report(device) -> dict:
    return {
        "id": int(device.id),
        "platform": device.platform,
        "kind": getattr(device, "device_kind", ""),
        "process": int(getattr(device, "process_index", 0)),
    }


def check_devices(reports: list[dict], expected: int) -> dict:
    """The mesh really has the requested number of distinct devices."""
    ids = [r["id"] for r in reports]
    if len(reports) != expected:
        _fail(f"expected {expected} devices in the mesh, got {len(reports)}")
    if len(set(ids)) != len(ids):
        _fail(f"mesh repeats a device: {ids}")
    return {"devices": ids, "platform": reports[0]["platform"] if reports else None}


def check_batch_sharded(shard_shapes: list[tuple[int, ...]], global_shape: tuple[int, ...]) -> dict:
    """The batch is split over the devices along its leading axis, evenly, with nothing dropped."""
    n = len(shard_shapes)
    if n == 0:
        _fail("batch has no shards")
    if sum(s[0] for s in shard_shapes) != global_shape[0]:
        _fail(f"shards hold {sum(s[0] for s in shard_shapes)} of {global_shape[0]} rows")
    if n > 1 and any(s[0] != global_shape[0] // n for s in shard_shapes):
        _fail(f"uneven shards {[s[0] for s in shard_shapes]}: data parallelism is not splitting")
    if any(tuple(s[1:]) != tuple(global_shape[1:]) for s in shard_shapes):
        _fail("a shard splits a non-batch axis")
    return {"shard_rows": [s[0] for s in shard_shapes], "global_rows": global_shape[0]}


def replica_checksums(tree) -> list[float]:
    """Per-device sum of |value| over a replicated pytree (one entry per device).

    Reads each device's own copy, not the logical array, so a replica that drifted is visible.
    """
    import jax
    import numpy as np

    totals: dict[int, float] = {}
    for leaf in jax.tree.leaves(tree):
        for shard in leaf.addressable_shards:
            totals[shard.device.id] = totals.get(shard.device.id, 0.0) + float(
                np.abs(np.asarray(shard.data, dtype=np.float64)).sum()
            )
    return [totals[k] for k in sorted(totals)]


def shard_shapes(array) -> list[tuple[int, ...]]:
    return [tuple(s.data.shape) for s in array.addressable_shards]


class GpuUtilSampler:
    """Polls NVML utilisation of the mesh's GPUs in a thread. Empty (inconclusive) off-GPU.

    JAX device ids are the CUDA ordinals visible to the process, which is what NVML indexes.
    """

    def __init__(self, devices, period_s: float):
        self.samples: dict[int, list[float]] = {}
        self.period = period_s
        self._stop = threading.Event()
        self._thread = None
        self._handles: dict[int, object] = {}
        if not devices or devices[0].platform != "gpu":
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            for d in devices:
                self._handles[int(d.id)] = pynvml.nvmlDeviceGetHandleByIndex(int(d.id))
                self.samples[int(d.id)] = []
        except Exception:  # NVML missing: report inconclusive rather than crash the run
            self._handles = {}

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            for i, h in self._handles.items():
                self.samples[i].append(float(self._nvml.nvmlDeviceGetUtilizationRates(h).gpu))

    def start(self) -> None:
        if self.active:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> dict[int, list[float]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.samples
