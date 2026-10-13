"""Per-process jobs for the distributed smoke tests (importable so multiprocessing can pickle)."""

import contextlib
from pathlib import Path

import torch

from remember_or_retrieve.config import load_config
from remember_or_retrieve.training import checks, dpo, grpo, parallel, sft
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


class SyntheticReward:
    """A deterministic, text-dependent reward in [-1, 1] with spread across samples. The tiny
    random model cannot produce a real fact, so the project's reward would be flat for it; this
    exercises the same machinery (advantages, learner, checks) with a signal."""

    def __call__(self, row, text):
        import zlib

        from remember_or_retrieve.scoring.reward import RewardBreakdown

        return RewardBreakdown(total=zlib.crc32(text.encode()) % 200 / 100.0 - 1.0)


def grpo_job(
    device, out_dir: str, init_from: str | None = None, kwargs: dict | None = None
) -> dict:
    return grpo.train_grpo(
        load_config(),
        reward_fn=SyntheticReward(),
        device=device,
        out_dir=Path(out_dir),
        init_from=init_from,
        smoke=True,
        **(kwargs or {}),
    )


def grpo_learner_job(device, n_rollouts: int = 8) -> dict:
    """The GRPO learner on fixed synthetic rollouts: gradient checksum and loss, for the
    1-rank vs N-rank equivalence check (rollout sampling itself is random per rank)."""
    import torch.distributed as dist

    cfg = load_config()["training"]
    net = model_lib.tiny_model(cfg["smoke"], vocab_size=512, seed=3)
    net = model_lib.build_lora(net, {**cfg["lora"], "dropout": 0.0})
    # LoRA B starts at zero (policy == reference); move it so the KL term has a gradient too.
    gen = torch.Generator().manual_seed(11)
    for p in net.parameters():
        if p.requires_grad:
            p.data.add_(torch.randn(p.shape, generator=gen) * 0.02)
    net = parallel.wrap_ddp(net, device)
    data_gen = torch.Generator().manual_seed(5)
    rollouts = []
    for i in range(n_rollouts):
        prompt = torch.randint(
            0, 512, (int(torch.randint(5, 9, (1,), generator=data_gen)),), generator=data_gen
        ).tolist()
        comp = torch.randint(
            0, 512, (int(torch.randint(2, 6, (1,), generator=data_gen)),), generator=data_gen
        ).tolist()
        rollouts.append(grpo.Rollout(prompt, comp, False, "", 0.0, float((i % 4) - 1.5)))
    rank, world = dist.get_rank(), dist.get_world_size()
    mine = rollouts[rank::world]
    tokens = float(sum(len(r.completion_ids) for r in rollouts))
    out = grpo.learn(
        net,
        mine,
        global_tokens=tokens,
        beta=0.04,
        micro=2,
        pad_id=0,
        device=device,
        autocast=contextlib.nullcontext(),
        world=world,
    )
    sums = grpo._all_reduce_sum([out["loss"], out["kl"]], device)
    return {
        "loss": sums[0] / tokens,
        "kl": sums[1] / tokens,
        "grad_checksum": checks.grad_checksum(net),
        "grad_spread": checks.check_checksums_equal(
            checks.gather(checks.grad_checksum(net)), "gradient checksums"
        ),
    }


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
