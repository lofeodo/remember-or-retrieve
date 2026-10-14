"""JAX/XLA LoRA fine-tune of Gemma 4 on the SFT data, data-parallel with `jax.sharding` (step 7).

Runs in its own Python 3.12 environment (the `jax` extra); nothing here imports torch or Ray.
"""
