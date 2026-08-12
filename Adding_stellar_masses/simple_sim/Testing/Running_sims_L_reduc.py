
import os

#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import jax

jax.config.update("jax_enable_x64", True)

import sys
sys.path.append('/gpfs/home/jd925/Adding_stellar_masses/simple_sim')

import Master_sim as MS


import numpy as np
import jax.numpy as jnp


import importlib
importlib.reload(MS)



dt_override = 2
ramp_time = 0

# Modes per chunk in the Poisson solve's (l, m) loop. 
l_band_size = 128
use_multi_gpu = False


#128
r_chunk_size = 128

compute_dtype = jnp.complex64

m22 = 10
R0 = 0.19

# (l, m) chunks batched per lax.map step in the merged solver. 
chunk_batch_size = 128

# Particles per streamed chunk inside rho_lm_at_particles. 
particle_chunk_size = 100

# Particles per pass through construct_acc_master_func - the Poisson solve AND
# the angular contraction that follows it, which now run in the same batch.
particle_batch_size = 100

# shortens the grid the density lives on and the Poisson integrals run over.
r_cut_kpc = None


# Both False is a normal time-dependent, anisotropic run.
frozen = False
sph_sym = False

L_out_fracs = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01]


for L_out_frac in L_out_fracs:
    sim = MS.StellarSimTDep(m22 = m22, r_half = R0, r_half_width = 0.05, no_of_particles = 1000, total_evolve_time = 10, r_min = 20,
                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, dt_override = dt_override, ramp_time=ramp_time,
                                r_chunk_size=r_chunk_size, l_band_size=l_band_size, compute_dtype=compute_dtype,
                                use_multi_gpu=use_multi_gpu, L_out_frac=L_out_frac, chunk_batch_size=chunk_batch_size,
                                frozen=frozen, sph_sym=sph_sym, r_cut_kpc=r_cut_kpc,
                                particle_chunk_size=particle_chunk_size,
                                particle_batch_size=particle_batch_size)

    # Keep frozen / sph_sym / truncated runs in their own checkpoint directory -
    # otherwise they resume from, and overwrite, a normal run's checkpoints.
    mode_tag = (('frozen_' if frozen else '') + ('sphsym_' if sph_sym else '')
                + ('' if r_cut_kpc is None else f'rcut{r_cut_kpc:g}_'))

    sim.run_simulation(checkpoint_every=50,
    checkpoint_dir=f'/gpfs/home/jd925/Adding_stellar_masses/Checkpoints/L_reduc/checkpoints_{mode_tag}m22_{m22}_r0_{R0}_Lout_{L_out_frac}')


