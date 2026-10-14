"""Data parallelism with `jax.sharding`: one 1-D mesh, batch split on its leading axis.

Everything else (frozen base weights, LoRA adapters, optimizer state) is replicated. The training
step is a single `jax.jit` over sharded inputs; XLA inserts the cross-device reduction of the
summed loss and gradients itself, so there is no hand-written all-reduce and no `pmap`.
"""

import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class DataParallel:
    def __init__(self, num_devices: int, axis: str = "data"):
        import jax

        devices = jax.devices()
        if len(devices) < num_devices:
            raise RuntimeError(
                f"asked for {num_devices} devices but JAX sees {len(devices)} "
                f"({devices[0].platform})"
            )
        self.devices = devices[:num_devices]
        self.n = num_devices
        self.axis = axis
        self.mesh = Mesh(np.array(self.devices), (axis,))
        self.batch = NamedSharding(self.mesh, PartitionSpec(axis))
        self.replicated = NamedSharding(self.mesh, PartitionSpec())

    def put_batch(self, array):
        import jax

        return jax.device_put(array, self.batch)

    def replicate(self, tree):
        import jax

        return jax.device_put(tree, self.replicated)
