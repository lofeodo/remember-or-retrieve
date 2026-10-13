"""Launchers: Ray Train for real runs, plain multiprocessing for fast local debugging.

Both end in the same per-worker function, so the smoke test exercises what the paid run runs.
"""

import os
import socket
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_worker(rank: int, world: int, port: int, fn, args, result_dir: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        out = fn(torch.device("cpu"), *args)
        if rank == 0:
            torch.save(out, Path(result_dir) / "result.pt")
    finally:
        dist.destroy_process_group()


def spawn_cpu(fn, world_size: int, *args):
    """Run `fn(device, *args)` on `world_size` CPU processes (gloo). Returns rank 0's result."""
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(
            _spawn_worker,
            args=(world_size, _free_port(), fn, args, d),
            nprocs=world_size,
            join=True,
        )
        return torch.load(Path(d) / "result.pt", weights_only=False)
