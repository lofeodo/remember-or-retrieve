"""Gemma 4 model construction: the real E4B, or a tiny random-init one for the smoke test.

The model is a `jax_stage.lora.LoRA` wrapper around the library's Gemma 4 transformer. Parameters
are kept as two trees: the frozen base (bf16, checkpoint paths) and the trainable LoRA adapters
(fp32 masters), merged only inside the jitted step.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gemma import gm, peft
from gemma.gm.nn.gemma4 import _config as gemma_config
from gemma.gm.nn.gemma4 import _transformer as gemma_transformer

from remember_or_retrieve.jax_stage.lora import LoRA

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}


def tiny_config(smoke: dict) -> Any:
    """Gemma 4 E4B's structure at toy size: same attention pattern, per-layer inputs and KV
    sharing, no vision or audio tower. The 262,144-token vocabulary is kept (real tokenizer)."""
    big = gm.nn.Gemma4_E4B.config
    attn = gemma_config.make_attention_layers_types(
        gm.nn.Gemma4_E4B.attention_pattern, num_layers=smoke["num_layers"]
    )
    return dataclasses.replace(
        big,
        embed_dim=smoke["embed_dim"],
        hidden_dim=smoke["hidden_dim"],
        num_heads=smoke["num_heads"],
        num_kv_heads=smoke["num_kv_heads"],
        head_dim=smoke["head_dim"],
        per_layer_input_dim=smoke["per_layer_input_dim"],
        attention_types=attn,
        vision_encoder=None,
        audio_encoder=None,
    )


def build_model(cfg: dict, *, smoke: bool, rank: int | None = None):
    """(LoRA-wrapped model, plain model). The plain one shares weights and is for reference."""
    jcfg = cfg["jax"]
    if smoke:
        base = gemma_transformer.Transformer(
            config=tiny_config(jcfg["smoke"]), return_last_only=False
        )
        rank = rank or jcfg["smoke"]["rank"]
    else:
        base = gm.nn.Gemma4_E4B(return_last_only=False)
        rank = rank or jcfg["lora"]["rank"]
    return LoRA(rank=rank, model=base, dtype=DTYPES[jcfg["lora"]["dtype"]]), base


def init_lora(lm, seed: int, seq_len: int = 8) -> Any:
    """Initial adapters only. Jitting the full init and keeping just the LoRA subtree lets XLA drop
    the base-weight initialisation, so the real 7.5B model is never materialised randomly."""
    tokens = jnp.ones((1, seq_len), jnp.int32)

    @jax.jit
    def init(key):
        return peft.split_params(lm.init(key, tokens=tokens))[1]

    return init(jax.random.key(seed))


def init_base_random(base_model, seed: int, seq_len: int = 8, dtype=jnp.float32) -> Any:
    """Smoke only: random base weights for the tiny model."""
    tokens = jnp.ones((1, seq_len), jnp.int32)
    params = base_model.init(jax.random.key(seed), tokens=tokens)
    return jax.tree.map(lambda x: x.astype(dtype), params)


def load_base_checkpoint(cfg: dict, dtype) -> Any:
    """The pretrained Gemma 4 instruction-tuned weights from the public bucket (no token)."""
    path = gm.ckpts.CheckpointPath(cfg["jax"]["model"]["checkpoint"])
    params = gm.ckpts.load_params(path)
    return jax.tree.map(lambda x: x.astype(dtype), {"params": params})


def count_params(tree) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree.leaves(tree)))


def merge(base, lora, compute_dtype=None):
    """Full variables for `lm.apply` from frozen base and trainable adapters."""
    if compute_dtype is not None:
        lora = jax.tree.map(lambda x: x.astype(compute_dtype), lora)
    return peft.merge_params(base, lora)
