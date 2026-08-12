"""
Determinism probe, round 4: does the solver's output depend on GPU runtime
state rather than on its inputs?

Everything cheap has now been ruled out. The force pipeline is bit-identical
across repeats, across processes, across two instances in one process, across
XLA_PYTHON_CLIENT_PREALLOCATE, and - round 3 arm 1 - across a genuine
compile-then-deserialise asymmetry (395 s compile, 14 s load), which is the
one the reproduc job actually had.

Yet run_0 and run_1 differ after their FIRST macro step, on all 1000
particles, by an implied |da|/|a| of up to 1.64e-10.

The single thing every probe so far has held constant is how much work the GPU
had already done. The reproduc runs were `for i in range(2)`, and instance 1
began after instance 0 had run 1295 macro steps - roughly five hours of
allocate/free churn against a pool running with
XLA_PYTHON_CLIENT_PREALLOCATE=false, i.e. one that grows and fragments as it
goes. Every probe so far ran 3-5 steps.

That matters because a fixed executable is not the same thing as a fixed
kernel. XLA hands cuBLAS/cuFFT a scratch workspace out of the same allocator,
and those libraries pick algorithms based on how much workspace they are
given. Less contiguous memory -> a different algorithm -> a different
summation order -> a different last bit. Same code, same inputs, same
executable.

This probe holds the INPUTS exactly fixed - one set of particle positions and
one rho_lms, reused for every measurement - and varies only how much the GPU
has been made to do in between. If the acceleration moves, the dependence is
on runtime state, and no XLA flag or code change fixes it; what fixes it is
making the two runs' allocator histories match.

Usage:  python determinism_probe4.py
"""

import gc
import hashlib
import os
import sys
from time import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.append('/gpfs/home/jd925/Adding_stellar_masses/simple_sim')

import Master_sim as MS

# Exactly Running_sims_reproduc.py's configuration.
dt_override = 2
ramp_time = 0
l_band_size = 128
use_multi_gpu = False
r_chunk_size = 128
compute_dtype = jnp.complex64
m22 = 10
R0 = 0.19
chunk_batch_size = 128
particle_chunk_size = 100
particle_batch_size = 100
r_cut_kpc = None
frozen = False
sph_sym = False
L_out_frac = 1

# How much churn to put between measurements. Cumulative: the run does 5 real
# macro steps, measures, another 15, measures, and so on, so the last
# measurement follows 100 steps. The reproduc gap was 1295, but if the effect
# is there at all it should be visible long before that.
CHURN_SCHEDULE = [0, 'alloc', 5, 15, 30, 50]


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def device_mem():
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
        return (f"in_use {stats.get('bytes_in_use', 0) / 1024**3:.2f} GiB, "
                f"pool {stats.get('pool_bytes', 0) / 1024**3:.2f} GiB, "
                f"largest_free {stats.get('largest_free_block_bytes', 0) / 1024**3:.2f} GiB, "
                f"peak {stats.get('peak_bytes_in_use', 0) / 1024**3:.2f} GiB")
    except Exception as exc:
        return f"unavailable ({exc})"


sim = MS.StellarSimTDep(
    m22=m22, r_half=R0, r_half_width=0.05, no_of_particles=1000,
    total_evolve_time=10, r_min=20, r_max_enclosing_frac=0.99,
    no_radius_bins=1000, dt_override=dt_override, ramp_time=ramp_time,
    r_chunk_size=r_chunk_size, l_band_size=l_band_size,
    compute_dtype=compute_dtype, use_multi_gpu=use_multi_gpu,
    L_out_frac=L_out_frac, chunk_batch_size=chunk_batch_size,
    frozen=frozen, sph_sym=sph_sym, r_cut_kpc=r_cut_kpc,
    particle_chunk_size=particle_chunk_size,
    particle_batch_size=particle_batch_size)

banner("initialisation")
start = time()
sim.sim_init.Run_initialisation()
sim.phi_builder.initialise()
sim.rho_builder.n_ramp_steps = sim.sim_init.no_ramp_steps
print(f"init {time() - start:.1f} s", flush=True)

sim_init = sim.sim_init
particles = sim_init.particles
for particle in particles:
    particle.Create_V_array(sim_init.no_time_steps + 10)

# THE fixed input. Captured once, never recomputed, so every measurement below
# is the same function of the same numbers.
POS_SPH = jnp.array([p.r_pos_sph for p in particles])
POS_CART = np.array([p.r_pos for p in particles])
VEL_CART = np.array([p.v for p in particles])


def measure():
    """rho_lm for step 0 and the acceleration at POS_SPH. Pure function of
    fixed inputs - anything that moves, moves because of runtime state."""
    rho = sim.rho_builder.build_rho_lms_for_timestep(0).astype(compute_dtype)
    sim.rho_builder.rho_lms = sim._place_rho_lms(rho)
    rho_hash = digest(np.asarray(sim.rho_builder.rho_lms))

    a_r, a_theta, a_phi, phi = sim.acc_calculator.construct_acc_master_func(
        POS_SPH, poten=True)
    a_r, a_theta, a_phi, phi = jax.block_until_ready((a_r, a_theta, a_phi, phi))
    acc = np.stack([np.asarray(a_r), np.asarray(a_theta), np.asarray(a_phi)], axis=1)
    return rho_hash, acc


def churn_alloc():
    """Grow and release a few large device buffers, to fragment the pool
    without doing any physics."""
    for size_gb in (2, 4, 1, 3):
        n = int(size_gb * 1024**3 / 8)
        buf = jnp.ones((n,), dtype=jnp.float64)
        buf = jax.block_until_ready(buf * 1.000001)
        del buf
    gc.collect()


def churn_steps(n_steps, step_offset):
    """n_steps of the real macro loop. Moves the particles - which is fine,
    because `measure` never reads their current state, only POS_SPH."""
    for step in range(step_offset, step_offset + n_steps):
        rho = sim.rho_builder.build_rho_lms_for_timestep(step).astype(compute_dtype)
        sim.rho_builder.rho_lms = sim._place_rho_lms(rho)
        for i, particle in enumerate(particles):
            p = sim_init.sim_particles[i]
            p.x, p.y, p.z = float(particle.r_pos[0]), float(particle.r_pos[1]), float(particle.r_pos[2])
            p.vx, p.vy, p.vz = float(particle.v[0]), float(particle.v[1]), float(particle.v[2])
        sim_init.sim.integrate(sim_init.sim.t + sim_init.dt)
        for i, particle in enumerate(particles):
            p = sim_init.sim_particles[i]
            particle.update_state([p.x, p.y, p.z], [p.vx, p.vy, p.vz],
                                  record_energy=False)


banner("measurements  (identical inputs throughout)")
reference = None
steps_done = 0
rows = []

for entry in CHURN_SCHEDULE:
    if entry == 'alloc':
        label = 'after 8 GiB alloc/free churn'
        churn_alloc()
    elif entry == 0:
        label = 'baseline (no churn)'
    else:
        label = f'after {steps_done + entry} macro steps'
        start = time()
        churn_steps(entry, steps_done)
        steps_done += entry
        print(f"  ({entry} steps in {time() - start:.0f} s)", flush=True)

    rho_hash, acc = measure()

    if reference is None:
        reference = acc
        rho_reference = rho_hash
        verdict, worst = 'reference', 0.0
    else:
        same = np.array_equal(acc.view(np.uint8), reference.view(np.uint8))
        scale = np.max(np.abs(reference))
        worst = float(np.max(np.abs(acc - reference))) / scale if scale > 0 else 0.0
        verdict = 'identical' if same else 'DIFFERS'

    rho_note = 'rho same' if rho_hash == rho_reference else 'RHO ALSO DIFFERS'
    print(f"  {label:<34} {verdict:<10} worst rel {worst:.3e}   {rho_note}")
    print(f"      mem: {device_mem()}", flush=True)
    rows.append((label, verdict, worst))


banner("VERDICT")
diverged = [r for r in rows if r[1] == 'DIFFERS']
if not diverged:
    print(f"""The acceleration did not move once, through {steps_done} macro steps and
8 GiB of allocate/free churn. GPU runtime state does not reach the result
either, so this probe does not explain the reproduc difference.

What is left is what differed between run_0 and run_1 and NOT between any pair
tested so far. The remaining candidate is scale: instance 1 there began after
1295 steps, not {steps_done}.""")
else:
    first = diverged[0]
    print(f"""The acceleration moved with runtime state alone: first at
"{first[0]}", worst rel {first[2]:.3e}, against an unchanged rho_lm and
byte-identical inputs.

Compare: the two reproduc runs implied |da|/|a| up to 1.64e-10, and round 1
measured a deliberate kernel swap at 1.11e-10.

This is not fixable by an XLA flag or by changing the code - the executable and
the inputs were already identical. What fixes it is giving the runs matching
allocator histories: XLA_PYTHON_CLIENT_PREALLOCATE=true so the pool is fixed
from the start, and one run per process rather than `for i in range(2)`.""")
