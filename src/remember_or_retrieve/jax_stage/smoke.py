"""Smoke-test drivers: the equivalence proof (N devices vs 1) and the broken-run demonstration.

Used by `scripts/jax_train.py check` and by the slow tests. The model is a tiny random-init
Gemma 4 with the real tokenizer and real SFT rows.
"""

from pathlib import Path

from remember_or_retrieve.jax_stage import checks, train


def equivalence(cfg: dict, out_dir: Path, num_devices: int = 2, tokenizer=None) -> dict:
    """Same global batches and seed on 1 device and on `num_devices`: the loss curves must match.

    This is the strongest evidence that the multi-device run is one training job, and the 1-device
    run is also the single-device reference for throughput.
    """
    out_dir = Path(out_dir)
    one = train.train(cfg, num_devices=1, out_dir=out_dir / "d1", smoke=True, tokenizer=tokenizer)
    many = train.train(
        cfg,
        num_devices=num_devices,
        out_dir=out_dir / f"d{num_devices}",
        smoke=True,
        tokenizer=tokenizer,
    )
    tol = cfg["jax"]["checks"]["equivalence_rel_tol"]
    result = checks.check_equivalence(one["global_losses"], many["global_losses"], tol)
    return {"devices": num_devices, "rel_tol": tol, **result}
