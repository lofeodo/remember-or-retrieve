"""Per-process jobs for the distributed smoke tests (importable so multiprocessing can pickle)."""

from pathlib import Path

import torch

from remember_or_retrieve.config import load_config
from remember_or_retrieve.training import checks, dpo, sft
from remember_or_retrieve.training import model as model_lib


def sft_job(device, mode: str, strategy: str, out_dir: str, kwargs: dict | None = None) -> dict:
    return sft.train_sft(
        load_config(),
        mode,
        strategy,
        device=device,
        out_dir=Path(out_dir),
        smoke=True,
        **(kwargs or {}),
    )


def dpo_job(device, out_dir: str, init_from: str | None = None) -> dict:
    return dpo.train_dpo(
        load_config(), device=device, out_dir=Path(out_dir), init_from=init_from, smoke=True
    )


def unsynced_job(device) -> dict:
    """A deliberately broken 'distributed' run: every rank trains its own copy with no gradient
    sync. The distribution checks must catch it (they raise, which the test expects)."""
    cfg = load_config()["training"]
    net = model_lib.tiny_model(cfg["smoke"], vocab_size=512, seed=1)
    rank = torch.distributed.get_rank()
    ids = torch.randint(0, 512, (2, 12), generator=torch.Generator().manual_seed(rank))
    net(input_ids=ids, labels=ids).loss.backward()  # no DDP/FSDP: nothing averages gradients
    checks.check_checksums_equal(checks.gather(checks.grad_checksum(net)), "gradient checksums")
    return {}
