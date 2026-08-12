"""
Determinism probe.

Answers one question: with the executable and the GPU held fixed, does the
same input produce the same bits twice?

Running_sims_reproduc.py ran its two runs inside ONE process (`for i in
range(2)`), and the second run's first acceleration call took 15 s against the
first's 232 s - i.e. the compilation cache hit and both runs executed the same
executable on the same card. So whatever made their trajectories differ has to
be per-*launch*, not per-compile. That is exactly what repeating a call inside
a single process tests, and it costs one initialisation instead of two.

Three stages are probed separately, so a difference can be localised rather
than just detected:

  psi_lm  - the per-l matmuls (cuBLAS)                    [complex64]
  rho_lm  - psi_lm + the s2fft round trip (cuFFT + more)  [complex64]
  acc     - merged Poisson solve + angular contraction    [complex128]

The dtypes matter for reading the result. compute_dtype is complex64, so a
difference originating in psi_lm/rho_lm shows up at ~1e-7 relative. But the
merged solver accumulates in `jnp.result_type(float64, complex64)` =
complex128, and combine_acc contracts against float64 Y_lm, so a difference
originating there shows up at ~1e-13. The reproduc runs differed by 4.4e-13 at
step 1, which points at the acc stage - this confirms or refutes that.

Writes <out_dir>/<tag>.npz so runs under different XLA flags, and different
processes, can be compared afterwards by compare_determinism.py.

Usage:  python determinism_probe.py <tag> [out_dir]
"""

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
import Memory_speed_savers as MSS


TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else (
    '/gpfs/home/jd925/Adding_stellar_masses/simple_sim/Testing/determinism_out')

# How many times each stage is repeated. The acceleration call is the cheap
# one and the prime suspect, so it gets the most repeats: a difference that
# appears intermittently (a race) rather than every time still has to be
# caught, and one extra call is ~10 s.
N_REPEAT_PSI = 3
N_REPEAT_RHO = 3
N_REPEAT_ACC = 5


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def digest(arr):
    """sha256 of the raw bytes - a bit-level identity, unlike ==, which
    would call every NaN unequal to itself."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def compare_repeats(name, arrays):
    """Bit-compare every repeat against the first. Returns a summary dict."""
    hashes = [digest(a) for a in arrays]
    identical = all(h == hashes[0] for h in hashes)

    reference = arrays[0].astype(np.complex128 if np.iscomplexobj(arrays[0]) else np.float64)
    scale = np.max(np.abs(reference))
    worst_abs = 0.0
    for other in arrays[1:]:
        other = other.astype(reference.dtype)
        worst_abs = max(worst_abs, float(np.max(np.abs(other - reference))))
    worst_rel = worst_abs / scale if scale > 0 else 0.0

    print(f"[{name}] dtype={arrays[0].dtype} shape={arrays[0].shape}")
    for i, h in enumerate(hashes):
        print(f"[{name}]   repeat {i}: sha256 {h}{'' if i == 0 or h == hashes[0] else '   <-- DIFFERS'}")
    print(f"[{name}] BIT-IDENTICAL ACROSS REPEATS: {identical}")
    print(f"[{name}] worst abs diff {worst_abs:.6e}, worst rel diff {worst_rel:.6e}", flush=True)

    return {'identical': identical, 'hashes': hashes,
            'worst_abs': worst_abs, 'worst_rel': worst_rel}


# ----------------------------------------------------------------------
# Exactly the configuration Running_sims_reproduc.py used. Do not "tidy"
# these - the whole point is that the kernels are the ones that produced
# the two runs being explained.
# ----------------------------------------------------------------------
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

banner(f"determinism probe: tag={TAG}")
print(f"XLA_FLAGS                   = {os.environ.get('XLA_FLAGS', '(unset)')}")
print(f"XLA_PYTHON_CLIENT_PREALLOCATE = {os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', '(unset)')}")
print(f"JAX_COMPILATION_CACHE_DIR   = {os.environ.get('JAX_COMPILATION_CACHE_DIR', '(unset)')}")
print(f"jax {jax.__version__}, devices: {jax.devices()}", flush=True)

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
# run_simulation does this before its first rho build; ramp_frac_for_step
# reads n_ramp_steps, so it has to be set even though ramp_time is 0.
sim.rho_builder.n_ramp_steps = sim.sim_init.no_ramp_steps
print(f"initialisation took {time() - start:.1f} s", flush=True)

results = {}


# ----------------------------------------------------------------------
# Stage: rho_lm. build_rho_lms_for_timestep(0), repeated.
# Covers psi_lm_at_rows + the streamed s2fft round trip.
# ----------------------------------------------------------------------
banner(f"stage rho_lm  ({N_REPEAT_RHO} repeats of build_rho_lms_for_timestep(0))")
rho_arrays = []
for i in range(N_REPEAT_RHO):
    start = time()
    rho = sim.rho_builder.build_rho_lms_for_timestep(0).astype(compute_dtype)
    rho = jax.block_until_ready(rho)
    rho_host = np.asarray(rho)
    del rho
    print(f"  repeat {i} built in {time() - start:.1f} s", flush=True)
    rho_arrays.append(rho_host)

results['rho_lm'] = compare_repeats('rho_lm', rho_arrays)
rho_reference = rho_arrays[0]
del rho_arrays


# ----------------------------------------------------------------------
# Stage: acceleration. The full fused solve, repeated on identical inputs.
#
# rho_lms is pinned to the FIRST build for every repeat, so this stage is
# testing the solver alone - if the rho build were itself nondeterministic
# it would otherwise leak in here and be indistinguishable.
# ----------------------------------------------------------------------
banner(f"stage acc  ({N_REPEAT_ACC} repeats of construct_acc_master_func)")
sim.rho_builder.rho_lms = sim._place_rho_lms(jnp.asarray(rho_reference))

particles = sim.sim_init.particles
r_pos_sphs = jnp.array([p.r_pos_sph for p in particles])

acc_arrays = {'a_r': [], 'a_theta': [], 'a_phi': [], 'phi': []}
for i in range(N_REPEAT_ACC):
    start = time()
    a_r, a_theta, a_phi, phi = sim.acc_calculator.construct_acc_master_func(
        r_pos_sphs, poten=True)
    a_r, a_theta, a_phi, phi = jax.block_until_ready((a_r, a_theta, a_phi, phi))
    print(f"  repeat {i} took {time() - start:.1f} s"
          f"{'  (includes compile)' if i == 0 else ''}", flush=True)
    acc_arrays['a_r'].append(np.asarray(a_r))
    acc_arrays['a_theta'].append(np.asarray(a_theta))
    acc_arrays['a_phi'].append(np.asarray(a_phi))
    acc_arrays['phi'].append(np.asarray(phi))

for name, arrays in acc_arrays.items():
    results[name] = compare_repeats(name, arrays)


# ----------------------------------------------------------------------
# Stage: psi_lm on its own. Only useful if rho_lm came out nondeterministic
# - it splits "the matmuls" from "the s2fft round trip". Skipped otherwise,
# because jitting it separately costs a compile of its own.
#
# Guarded: this is a diagnostic refinement, and it must not be able to lose
# the results above.
# ----------------------------------------------------------------------
if not results['rho_lm']['identical']:
    banner(f"stage psi_lm  ({N_REPEAT_PSI} repeats) - rho_lm differed, splitting matmul from s2fft")
    try:
        groups = sim.rho_builder.l_groups
        psi_jit = jax.jit(
            lambda amps, ph, R: MSS.psi_lm_at_rows(amps, ph, R, groups))

        phase_c = sim.rho_builder.phase_for_step(0).astype(compute_dtype)
        psi_arrays = []
        for i in range(N_REPEAT_PSI):
            psi = psi_jit(sim.rho_builder.amplitudes, phase_c,
                          sim.rho_builder.R_j_r_fixed)
            psi = jax.block_until_ready(psi)
            psi_arrays.append(np.asarray(psi))
            del psi
        results['psi_lm'] = compare_repeats('psi_lm', psi_arrays)
        del psi_arrays
    except Exception as exc:
        print(f"[psi_lm] skipped: {exc}", flush=True)
else:
    print("\n[psi_lm] skipped - rho_lm was bit-identical, so the matmuls and "
          "the s2fft round trip both were too.", flush=True)


# ----------------------------------------------------------------------
# Save for the cross-process / cross-config comparison.
#
# rho_lm is ~930 MB, far too big to keep one copy per config, so only its
# digest is stored. The acceleration arrays are (1000,) and are stored in
# full - that is what actually drives the trajectories, and having the
# values (not just a hash) lets compare_determinism report how big a
# cross-config difference is, not merely that there is one.
# ----------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f'{TAG}.npz')

payload = {
    'tag': TAG,
    'pid': os.getpid(),
    'xla_flags': os.environ.get('XLA_FLAGS', ''),
    'preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', ''),
    'rho_lm_digest': results['rho_lm']['hashes'][0],
    'rho_lm_identical': results['rho_lm']['identical'],
}
for name in ('a_r', 'a_theta', 'a_phi', 'phi'):
    payload[name] = acc_arrays[name][0]
    payload[f'{name}_identical'] = results[name]['identical']
    payload[f'{name}_worst_rel'] = results[name]['worst_rel']

np.savez(out_path, **payload)


banner(f"SUMMARY  tag={TAG}")
for name in ('psi_lm', 'rho_lm', 'a_r', 'a_theta', 'a_phi', 'phi'):
    if name in results:
        r = results[name]
        verdict = 'bit-identical' if r['identical'] else f"DIFFERS (rel {r['worst_rel']:.3e})"
        print(f"  {name:<10} {verdict}")
print(f"\nwritten: {out_path}", flush=True)
