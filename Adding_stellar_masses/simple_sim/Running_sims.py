
import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import jax

jax.config.update("jax_enable_x64", True)


import Master_sim as MS


import numpy as np
import jax.numpy as jnp


import importlib
importlib.reload(MS)



dt_override = 0.1
ramp_time = 0

# 32
l_band_size = 32
use_multi_gpu = True
# 262144
sparse_k_batch=262144

#128
r_chunk_size = 128

compute_dtype = jnp.complex64

L_out_frac = 0.3

m22 = 50
R0 = 0.19

chunk_batch_size = None


sim = MS.StellarSimTDep(m22 = m22, r_half = R0, r_half_width = 0.05, no_of_particles = 100, total_evolve_time = 10, r_min = 20, 
                            r_max_enclosing_frac = 0.99, no_radius_bins = 1000, dt_override = dt_override, ramp_time=ramp_time, 
                            sparse_k_batch=sparse_k_batch, r_chunk_size=r_chunk_size, l_band_size=l_band_size, compute_dtype=compute_dtype, 
                            use_multi_gpu=use_multi_gpu, L_out_frac=L_out_frac, chunk_batch_size=chunk_batch_size)

sim.run_simulation(checkpoint_every=50, 
checkpoint_dir=f'/gpfs/home/jd925/Adding_stellar_masses/Checkpoints/checkpoints_m22_{m22}_r0_{R0}_Lout_{L_out_frac}')


