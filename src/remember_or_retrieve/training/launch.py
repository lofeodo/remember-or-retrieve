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
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
    )
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


# ------------------------------------------------------------------- Ray Train


def _ray_loop(config: dict) -> None:
    """Per-worker entry point under Ray Train; the process group is already initialised."""
    import ray.train
    from ray.train.torch import get_device

    stage = config["stage"]
    if stage.startswith("sft"):
        from remember_or_retrieve.training import sft

        sft.train_sft(
            config["cfg"],
            config["mode"],
            config.get("strategy"),
            device=get_device(),
            out_dir=Path(config["out_dir"]),
            smoke=config["smoke"],
            resume=True,  # a Ray restart after a preemption continues from the last checkpoint
            stop_after=config.get("stop_after"),
            init_from=config.get("init_from"),
            report=lambda metrics: ray.train.report(metrics),
        )
    elif stage == "dpo":
        from remember_or_retrieve.training import dpo

        dpo.train_dpo(
            config["cfg"],
            device=get_device(),
            out_dir=Path(config["out_dir"]),
            init_from=config["init_from"],
            smoke=config["smoke"],
        )
    elif stage == "grpo":
        from remember_or_retrieve.training import grpo

        grpo.train_grpo(
            config["cfg"],
            device=get_device(),
            out_dir=Path(config["out_dir"]),
            init_from=config["init_from"],
            smoke=config["smoke"],
            resume=True,  # a Ray restart after a preemption continues from the last checkpoint
            stop_after=config.get("stop_after"),
            report=lambda metrics: ray.train.report(metrics),
        )
    else:
        raise ValueError(f"unknown stage {stage}")


def run_ray(
    config: dict, num_workers: int, use_gpu: bool, storage_path: str, name: str
) -> "ray.train.Result":  # noqa: F821
    """Launch `config["stage"]` on `num_workers` Ray Train workers (NCCL on GPU, gloo on CPU)."""
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    # No dashboard, localhost only: nothing is exposed on the network.
    ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=True)
    trainer = TorchTrainer(
        _ray_loop,
        train_loop_config=config,
        torch_config=TorchConfig(backend="nccl" if use_gpu else "gloo"),
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        run_config=RunConfig(
            name=name,
            storage_path=storage_path,
            failure_config=FailureConfig(
                max_failures=config["cfg"]["training"]["checks"]["max_failures"]
            ),
        ),
    )
    try:
        return trainer.fit()
    finally:
        ray.shutdown()
