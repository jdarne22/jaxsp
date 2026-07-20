
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)




import sys
sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses')



import numpy as np
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import Stellar_sim_funcs as SSF

import importlib
importlib.reload(SSF)






class ShardingManager:
    """Helper class to manage sharding of large arrays across multiple GPUs.

    This is used by `Analytic_t_dep_sim_mem_saver` to shard the large
    `(Nr, L, 2L-1)` arrays across the L axis, and the `(Nr, Nj)` arrays
    across the Nj axis. When only one device is visible these become no-ops,
    so behaviour is identical to the single-GPU original.
    """
    def __init__(self, use_multi_gpu=True):
        self.use_multi_gpu = use_multi_gpu
        self._setup_sharding()
    # ----------------------------------------------------------------------
    # Sharding helpers (change 7: shard the heavy arrays across both GPUs).
    # When only one device is visible these become no-ops, so behaviour is
    # identical to the single-GPU original.
    # ----------------------------------------------------------------------
    def _setup_sharding(self):
        devs = jax.devices()
        self.devices = devs
        if self.use_multi_gpu and len(devs) > 1:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec
            from jax.experimental import mesh_utils
            device_mesh = mesh_utils.create_device_mesh((len(devs),))
            self.mesh = Mesh(device_mesh, axis_names=('x',))
            # Shard `Nj` axis (last) for arrays like (Nr, Nj) and (N_unique, Nj).
            self.shard_nj  = NamedSharding(self.mesh, PartitionSpec(None, 'x'))
            # Shard `J` axis for 1-D arrays of shape (J,) e.g. optimizer params.
            self.shard_j   = NamedSharding(self.mesh, PartitionSpec('x'))
            # Shard `L`  axis (middle) for arrays of shape (Nr, L, 2L-1).
            self.shard_l   = NamedSharding(self.mesh, PartitionSpec(None, 'x', None))
            self.shard_rep = NamedSharding(self.mesh, PartitionSpec())
            print(f"[mem_saver] Multi-GPU sharding enabled across {len(devs)} devices.")
        else:
            self.mesh = None
            self.shard_nj = None
            self.shard_j = None
            self.shard_l = None
            self.shard_rep = None

    def shard_nj_arr(self, arr):
        """Place a 2-D array (..., Nj) sharded along Nj. No-op if single-GPU.
        Pads the last axis to the nearest multiple of n_dev if needed so sharding
        is never silently skipped. Callers that pre-pad (e.g. the simulation) will
        see pad=0 here and are unaffected.
        """
        if self.shard_nj is None:
            return arr
        n_dev = len(self.devices)
        pad = (-arr.shape[-1]) % n_dev
        if pad:
            print(f"[ShardingManager] shard_nj_arr: padding last axis by {pad} "
                  f"({arr.shape[-1]} → {arr.shape[-1] + pad}) for {n_dev}-way sharding.")
            arr = jnp.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)])
        return jax.device_put(arr, self.shard_nj)

    def shard_j_arr(self, arr):
        """Place a 1-D array (J,) sharded along J. No-op if single-GPU.
        Pads to the nearest multiple of n_dev if needed so sharding is never
        silently skipped.
        """
        if self.shard_j is None:
            return arr
        n_dev = len(self.devices)
        pad = (-arr.shape[0]) % n_dev
        if pad:
            print(f"[ShardingManager] shard_j_arr: padding axis-0 by {pad} "
                  f"({arr.shape[0]} → {arr.shape[0] + pad}) for {n_dev}-way sharding.")
            arr = jnp.pad(arr, [(0, pad)])
        return jax.device_put(arr, self.shard_j)

    def shard_l_arr(self, arr):
        """Place a 3-D array (Nr, L, 2L-1) sharded along L. No-op if single-GPU,
        or if the L axis isn't divisible by the number of devices.

        `device_put` requires exact divisibility; `with_sharding_constraint`
        inside a JIT does not (GSPMD pads internally). So this method is
        strictly weaker than the in-JIT constraint — prefer that path. For
        L_out = 2L-1 (always odd), this falls back to replicated.
        """
        if self.shard_l is None:
            return arr
        n_dev = len(self.devices)
        if arr.ndim >= 2 and arr.shape[1] % n_dev != 0:
            return arr
        return jax.device_put(arr, self.shard_l)

    def build_sparse_au_j(self, lm_idx, parent_j, aj, N_unique, Nj):
        """Build the dense `(N_unique, Nj)` scatter `a_u_j[lm_idx, parent_j] = aj`
        without peak-doubling on the GPU.

        `jnp.zeros((N_unique, Nj)).at[...].add(aj)` allocates the zeros + a
        new array for the scatter result, peaking at 2x the final size — at
        m22=10 with complex64 that's ~26 GB which overflows a 25 GB GPU.
        Here we materialise the scatter in CPU RAM via numpy, then transfer
        (sharded if multi-GPU). CPU RAM is plentiful and the GPU only ever
        holds the final, single-copy array.
        """
        np_dtype = np.zeros((), dtype=aj.dtype).dtype  # jax dtype -> numpy dtype
        a_u_j_np = np.zeros((N_unique, Nj), dtype=np_dtype)
        a_u_j_np[np.asarray(lm_idx), np.asarray(parent_j)] = np.asarray(aj)
        if self.shard_nj is not None:
            return jax.device_put(a_u_j_np, self.shard_nj)
        return jnp.asarray(a_u_j_np)
