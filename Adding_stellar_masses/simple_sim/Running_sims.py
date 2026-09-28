
import os

#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import jax

jax.config.update("jax_enable_x64", True)


import Master_sim as MS


import numpy as np
import jax.numpy as jnp


import importlib
importlib.reload(MS)

#Either 'SegII' or 'LeoII'
gal_name = "LeoII"

#LeoII: 20pc (or 1pc) , SegII: 0.05pc
r_min = 20  # pc

#Minimum dt_override = 2
dt_override = [2, 10, 20]
#dt_override = 2

ramp_time = 0

L_out_frac = [0.1, 0.3, 0.5, 1]
#L_out_frac = 0.125

m22 = 10
R0 = [0.05, 0.1, 0.2]  #Kpc

#m22_list = [3000, 5000]
#R0 = 0.001

# shortens the grid the density lives on and the Poisson integrals run over.
r_cut_kpc = [None, 10]

#r_cut_kpc = 1 #kpc

#no_of_particles = 3000
no_of_particles = 1000

no_radius_bins = 1000

# Things to loop over:
# R0, dt_override, L_out_frac, r_cut_kpc

from itertools import product

# the four axes, named so the tuple unpacking below can't silently transpose
axes = dict(R0=R0, dt_override=dt_override, L_out_frac=L_out_frac, r_cut_kpc=r_cut_kpc)
combos = list(product(*axes.values()))          # 3 x 3 x 4 x 2 = 72 runs



# Both False is a normal time-dependent, anisotropic run.
frozen = False
sph_sym = False

# Modes per chunk in the Poisson solve's (l, m) loop. 
l_band_size = 128
use_multi_gpu = True

#128
r_chunk_size = 32


# (l, m) chunks batched per lax.map step in the merged solver. 
chunk_batch_size = 32

# Particles per streamed chunk inside rho_lm_at_particles. 
particle_chunk_size = 50

# Particles per pass through construct_acc_master_func - the Poisson solve AND
# the angular contraction that follows it, which now run in the same batch.
particle_batch_size = 100

compute_dtype = jnp.complex64



for R0, dt_override, L_out_frac, r_cut_kpc in combos:
#for m22 in m22_list:

    print(f'Completing sim with: m22 = {m22}, R0 = {R0}, dt = {dt_override}, L_out_frac = {L_out_frac}, r_cut = {r_cut_kpc}')

    sim = MS.StellarSimTDep(gal_name = gal_name, m22 = m22, r_half = R0, r_half_width = 0.05, no_of_particles = no_of_particles, total_evolve_time = 10, r_min = r_min,
                                r_max_enclosing_frac = 0.99, no_radius_bins = no_radius_bins, dt_override = dt_override, ramp_time=ramp_time,
                                r_chunk_size=r_chunk_size, l_band_size=l_band_size, compute_dtype=compute_dtype,
                                use_multi_gpu=use_multi_gpu, L_out_frac=L_out_frac, chunk_batch_size=chunk_batch_size,
                                frozen=frozen, sph_sym=sph_sym, r_cut_kpc=r_cut_kpc,
                                particle_chunk_size=particle_chunk_size,
                                particle_batch_size=particle_batch_size)

    # Keep frozen / sph_sym / truncated runs in their own checkpoint directory -
    # otherwise they resume from, and overwrite, a normal run's checkpoints.
    mode_tag = (('frozen_' if frozen else '') + ('sphsym_' if sph_sym else '')
                + ('' if r_cut_kpc is None else f'rcut{r_cut_kpc:g}_'))

    sim.run_simulation(checkpoint_every=100,
    checkpoint_dir=f'/gpfs/home/jd925/Adding_stellar_masses/Checkpoints/{gal_name}/m22_10_analysis/checkpoints_{mode_tag}m22_{m22}_r0_{R0}_Lout_{L_out_frac}_dt_{dt_override}')


