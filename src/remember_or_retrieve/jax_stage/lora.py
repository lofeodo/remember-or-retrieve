"""LoRA for Gemma 4, written here because `gm.nn.LoRA` is a silent no-op on it.

`gm.nn.LoRA` swaps in adapters only for `nn.Dense`, `nn.Einsum` and `gm.nn._layers.Einsum`. Gemma 4
uses its own `gemma4._layers.Einsum` and `ClippedEinsum`, so wrapping a Gemma 4 model adds zero
parameters and trains nothing (verified in step 7, see docs/roadmap.md). This wrapper intercepts
those two layer types and adds the same adapter (`gemma.peft.LoRAEinsumAdapter`, scale fixed at 1).
Base weights keep their checkpoint paths; adapters live in sibling `lora` collections.
"""

import dataclasses
import functools

import flax.linen as nn
import jax.numpy as jnp
from gemma import peft
from gemma.gm.nn.gemma4 import _layers


class LoRAEinsum(nn.Module):
    """A Gemma 4 einsum layer plus a low-rank adapter."""

    _: dataclasses.KW_ONLY
    rank: int
    dtype: jnp.dtype
    wrapped: nn.Module

    @nn.compact
    def __call__(self, eqn, x):
        adapter = peft.LoRAEinsumAdapter(
            name="lora", rank=self.rank, dtype=self.dtype, einsum_str=eqn, shape=self.wrapped.shape
        )
        return self.wrapped(eqn, x) + adapter(x).astype(x.dtype)


def _replace(module: nn.Module, *, rank: int, dtype) -> nn.Module:
    if isinstance(module, _layers.Einsum | _layers.ClippedEinsum):
        # FeedForward calls two einsums that share a scope: wrappers need distinct names.
        name = None if module.weight_name == "w" else f"LoRAEinsum_{module.weight_name}"
        return LoRAEinsum(name=name, rank=rank, dtype=dtype, wrapped=module)
    return module


class LoRA(nn.Module):
    """Wraps a Gemma 4 transformer so every einsum gets an adapter of the given rank."""

    _: dataclasses.KW_ONLY
    rank: int
    model: nn.Module
    dtype: jnp.dtype = jnp.float32

    def __post_init__(self):
        super().__post_init__()
        if self.scope is not None:  # transparent wrapper: params stay at the model's own paths
            nn.share_scope(self, self.model)

    @nn.compact
    def __call__(self, *args, **kwargs):
        fn = functools.partial(_replace, rank=self.rank, dtype=self.dtype)
        with peft.ModuleInterceptor(fn):
            return self.model(*args, **kwargs)


def is_lora_path(path: tuple) -> bool:
    return any(getattr(k, "key", k) == "lora" for k in path)
