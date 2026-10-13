"""Parallelism: DDP and FSDP2 wrappers, gradient-sync control, and checkpoint save/load.

Full fine-tune uses FSDP2 (`fully_shard`) so parameters, gradients and optimizer states are split
across ranks; LoRA and DPO use DDP because the base model fits on one GPU. Both are written out
explicitly rather than hidden behind a launcher, which is the point of the project.
"""

import contextlib
from pathlib import Path

import torch
import torch.distributed as dist

DDP = "ddp"
FSDP = "fsdp"


def enable_checkpointing(model) -> None:
    """Activation checkpointing via transformers (non-reentrant, safe with DDP and FSDP2)."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False


def wrap_ddp(model, device: torch.device):
    from torch.nn.parallel import DistributedDataParallel

    model.to(device)
    ids = [device.index] if device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=ids, find_unused_parameters=False)


def wrap_fsdp2(model, device: torch.device, master_dtype: torch.dtype, compute_dtype: torch.dtype):
    """Shard `model` across all ranks, one FSDP unit per decoder layer plus the root.

    Parameters are held in `master_dtype` (fp32, so small updates are not lost to bf16 rounding),
    computed in `compute_dtype`, and gradients are reduced in fp32.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    model.to(master_dtype)
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=compute_dtype, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)
    return model


@contextlib.contextmanager
def accumulate(model, strategy: str, sync: bool):
    """Skip the gradient all-reduce / reduce-scatter on all but the last micro-batch."""
    if sync:
        yield
    elif strategy == DDP:
        with model.no_sync():
            yield
    else:
        model.set_requires_gradient_sync(False)
        try:
            yield
        finally:
            model.set_requires_gradient_sync(True)


def unwrap(model, strategy: str):
    return model.module if strategy == DDP else model


# --------------------------------------------------------------- checkpointing


def save_lora(model, strategy: str, out_dir: Path) -> None:
    """Adapters only (rank 0). The base model is not copied."""
    if dist.get_rank() == 0:
        unwrap(model, strategy).save_pretrained(out_dir)
    dist.barrier()


def save_hf(model, strategy: str, out_dir: Path, dtype: torch.dtype = torch.bfloat16) -> None:
    """A loadable Hugging Face checkpoint of the full model, gathered to rank 0 in `dtype`.
    Every rank must call this: gathering FSDP2 shards is collective."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    inner = unwrap(model, strategy)
    sd = get_model_state_dict(
        inner, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if dist.get_rank() == 0:
        sd = {k: v.to(dtype) for k, v in sd.items()}
        inner.save_pretrained(out_dir, state_dict=sd, safe_serialization=True)
    dist.barrier()


def _resume_options(trainable_only: bool):
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # LoRA: the frozen base is reloaded from the pretrained weights, so only adapters are saved.
    # strict=False because the saved dict omits the frozen base: loading must not demand its keys.
    return StateDictOptions(ignore_frozen_params=trainable_only, strict=not trainable_only)


def save_resume_state(
    model, optimizer, strategy: str, ckpt_dir: Path, trainable_only: bool = False
) -> None:
    """Sharded distributed checkpoint (model + optimizer) for resuming after a preemption."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    msd, osd = get_state_dict(
        unwrap(model, strategy), optimizer, options=_resume_options(trainable_only)
    )
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(ckpt_dir))
    dist.barrier()


def load_resume_state(
    model, optimizer, strategy: str, ckpt_dir: Path, trainable_only: bool = False
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_state_dict,
        set_state_dict,
    )

    inner = unwrap(model, strategy)
    options = _resume_options(trainable_only)
    msd, osd = get_state_dict(inner, optimizer, options=options)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=str(ckpt_dir))
    set_state_dict(
        inner,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optim"],
        options=options,
    )
    dist.barrier()
