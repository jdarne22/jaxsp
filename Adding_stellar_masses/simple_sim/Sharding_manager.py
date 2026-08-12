
import jax
jax.config.update("jax_enable_x64", True)




import jax.numpy as jnp






class ShardingManager:
    """Decides where the big arrays live across the visible GPUs.

    Two placements are used:
      - the (n_radii, L, 2L-1) rho_lm / phi_lm arrays are split on the L axis;
      - everything indexed by radial mode is replicated, because the density
        kernels slice that axis per l and an l's modes do not line up with
        device boundaries.

    When only one device is visible these become no-ops, so behaviour is
    identical to the single-GPU original.
    """
    def __init__(self, use_multi_gpu=True):
        self.use_multi_gpu = use_multi_gpu
        self._setup_sharding()

    def _setup_sharding(self):
        devs = jax.devices()
        self.devices = devs
        if self.use_multi_gpu and len(devs) > 1:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec
            from jax.experimental import mesh_utils
            device_mesh = mesh_utils.create_device_mesh((len(devs),))
            self.mesh = Mesh(device_mesh, axis_names=('x',))
            # Shard the leading axis, replicating the rest.
            self.shard_j   = NamedSharding(self.mesh, PartitionSpec('x'))
            # Shard `L`  axis (middle) for arrays of shape (Nr, L, 2L-1).
            self.shard_l   = NamedSharding(self.mesh, PartitionSpec(None, 'x', None))
            self.shard_rep = NamedSharding(self.mesh, PartitionSpec())
            print(f"[mem_saver] Multi-GPU sharding enabled across {len(devs)} devices.")
        else:
            self.mesh = None
            self.shard_j = None
            self.shard_l = None
            self.shard_rep = None

    def shard_leading_axis_arr(self, arr):
        """Place an array sharded along its leading axis, every other axis
        replicated. No-op if single-GPU.

        Never pads. The axis is the radial-mode index and its length has to
        keep matching the arrays it gets contracted against, so a length that
        does not divide evenly falls back to a single device rather than
        silently growing by a mode.
        """
        if self.shard_j is None or arr.shape[0] % len(self.devices) != 0:
            return jnp.asarray(arr)
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

    def replicate_arr(self, arr):
        """Place an array replicated on every device. No-op if single-GPU.

        For arrays that are indexed with *dynamic* indices along the axis
        `shard_l_arr` would shard. The SPMD partitioner cannot keep such a
        gather device-local, so it all-gathers the whole array anyway -
        replicating up front costs the same per-device memory but moves the
        one collective out of the consuming kernel (or, if the consumer is a
        scan, stops it happening once per iteration).
        """
        if self.shard_rep is None:
            # jnp.asarray, not `arr`: callers pass host numpy arrays, and
            # returning one unchanged would make every downstream jit
            # re-upload it on each call.
            return jnp.asarray(arr)
        return jax.device_put(arr, self.shard_rep)
