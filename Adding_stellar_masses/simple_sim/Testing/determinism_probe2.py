"""
Determinism probe, round 2.

Round 1 result: the whole force pipeline is bit-reproducible. rho_lm and all
four acceleration outputs were identical across repeats inside one process,
across two separate processes, and with XLA_PYTHON_CLIENT_PREALLOCATE flipped.
So the 4.4e-13 that separated the two reproduc runs at step 1 is NOT
per-launch kernel nondeterminism, and it is not allocator state.

That leaves the one condition round 1 did not reproduce: Running_sims_reproduc
ran `for i in range(2)`, i.e. TWO StellarSimTDep instances inside a single
process, each running the full rebound integration loop. Round 1 built one
instance and called the solver directly, never touching rebound.

So this probe builds two instances the same way that script does, runs the
actual macro-timestep loop from Master_sim.run_simulation on each, and
bit-compares them at every intermediate point - which localises the first
divergence to one of:

  dt / t     - the integrator's clock and step size
  rho_lm     - the density built for that step
  acc        - the acceleration at the synced positions, before integrating
  pos/vel    - the state after rebound has integrated one macro step

If everything up to `acc` matches and `pos/vel` does not, the divergence is
inside rebound rather than in any JAX kernel, and no XLA flag will ever fix
it. If `acc` already differs, then something about being the second instance
in a process changes the solver's inputs.

Usage:  python determinism_probe2.py [n_steps]
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


N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

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


def banner(text):
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def build_sim():
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
    sim.sim_init.Run_initialisation()
    sim.phi_builder.initialise()
    sim.rho_builder.n_ramp_steps = sim.sim_init.no_ramp_steps
    return sim


def run_instance(index):
    """
    One instance, run through N_STEPS of Master_sim.run_simulation's loop.

    The loop body is copied rather than called, because run_simulation also
    checkpoints, evaluates energies and runs to completion. Everything that
    touches the state - the rebound sync, the rho build, integrate(), the
    read-back - is kept in the same order it happens there.
    """
    banner(f"instance {index}")
    start = time()
    sim = build_sim()
    print(f"  init {time() - start:.1f} s", flush=True)

    sim_init = sim.sim_init
    particles = sim_init.particles

    for particle in particles:
        particle.Create_V_array(sim_init.no_time_steps + 10)

    record = {
        'dt_bits': np.float64(sim_init.sim.dt).tobytes().hex(),
        'dt': float(sim_init.sim.dt),
        # Initial state, straight out of the ICs.
        'pos_0': np.array([p.r_pos for p in particles]),
        'vel_0': np.array([p.v for p in particles]),
    }

    for step in range(N_STEPS):
        rho_start = time()
        rho = sim.rho_builder.build_rho_lms_for_timestep(step).astype(compute_dtype)
        sim.rho_builder.rho_lms = sim._place_rho_lms(rho)
        record[f'rho_{step}'] = digest(np.asarray(sim.rho_builder.rho_lms))
        rho_seconds = time() - rho_start

        # Sync particle state into rebound, exactly as run_simulation does.
        for i, particle in enumerate(particles):
            p = sim_init.sim_particles[i]
            p.x, p.y, p.z = float(particle.r_pos[0]), float(particle.r_pos[1]), float(particle.r_pos[2])
            p.vx, p.vy, p.vz = float(particle.v[0]), float(particle.v[1]), float(particle.v[2])

        # The acceleration at exactly those synced positions, computed BEFORE
        # integrating. Not part of run_simulation - it is the probe point that
        # separates "the force differs" from "the integration differs".
        #
        # Its wall time is also the compile probe. In the reproduc job run_0's
        # first acceleration call took 232 s and run_1's took 15 s: run_0
        # compiled, run_1 deserialised the executable from
        # JAX_COMPILATION_CACHE_DIR. That asymmetry is the thing under test
        # here, so it has to be visible in the output rather than assumed.
        acc_start = time()
        r_pos_sphs = jnp.array([p.r_pos_sph for p in particles])
        a_r, a_theta, a_phi, _ = sim.acc_calculator.construct_acc_master_func(
            r_pos_sphs, poten=True)
        a_r, a_theta, a_phi = jax.block_until_ready((a_r, a_theta, a_phi))
        acc_seconds = time() - acc_start
        record[f'acc_{step}'] = np.stack(
            [np.asarray(a_r), np.asarray(a_theta), np.asarray(a_phi)], axis=1)

        record[f't_before_{step}'] = float(sim_init.sim.t)
        target_time = sim_init.sim.t + sim_init.dt
        sim_init._force_call_count = 0
        sim_init.sim.integrate(target_time)
        record[f'force_calls_{step}'] = sim_init._force_call_count
        record[f't_after_{step}'] = float(sim_init.sim.t)

        for i, particle in enumerate(particles):
            p = sim_init.sim_particles[i]
            particle.update_state([p.x, p.y, p.z], [p.vx, p.vy, p.vz],
                                  record_energy=False)

        record[f'pos_{step + 1}'] = np.array([p.r_pos for p in particles])
        record[f'vel_{step + 1}'] = np.array([p.v for p in particles])
        print(f"  step {step}: {record[f'force_calls_{step}']} force calls, "
              f"t {record[f't_before_{step}']:.10e} -> {record[f't_after_{step}']:.10e}, "
              f"rho {rho_seconds:.1f} s, acc {acc_seconds:.1f} s"
              f"{'   <-- compile happens here on a cold cache' if step == 0 else ''}",
              flush=True)

    return record


print(f"jax {jax.__version__}, devices {jax.devices()}")
print(f"XLA_FLAGS = {os.environ.get('XLA_FLAGS', '(unset)')}")
print(f"steps per instance: {N_STEPS}", flush=True)

records = [run_instance(0), run_instance(1)]


banner("INSTANCE 0  vs  INSTANCE 1")

first_divergence = None


def compare(label, a, b):
    global first_divergence
    if isinstance(a, (str, int)):
        same = a == b
        detail = '' if same else f"   {a!r} vs {b!r}"
    elif isinstance(a, float):
        same = np.float64(a).tobytes() == np.float64(b).tobytes()
        detail = '' if same else f"   {a!r} vs {b!r}  (diff {b - a:.6e})"
    else:
        same = np.array_equal(a.view(np.uint8), b.view(np.uint8))
        if same:
            detail = ''
        else:
            scale = np.max(np.abs(a))
            worst = float(np.max(np.abs(a - b))) / scale if scale > 0 else 0.0
            detail = f"   worst rel {worst:.6e}"
    if not same and first_divergence is None:
        first_divergence = label
    print(f"  {label:<18} {'identical' if same else 'DIFFERS'}{detail}")
    return same


compare('dt', records[0]['dt_bits'], records[1]['dt_bits'])
compare('pos_0', records[0]['pos_0'], records[1]['pos_0'])
compare('vel_0', records[0]['vel_0'], records[1]['vel_0'])

for step in range(N_STEPS):
    print(f"  --- step {step} ---")
    compare(f'rho_{step}', records[0][f'rho_{step}'], records[1][f'rho_{step}'])
    compare(f'acc_{step}', records[0][f'acc_{step}'], records[1][f'acc_{step}'])
    compare(f't_before_{step}', records[0][f't_before_{step}'], records[1][f't_before_{step}'])
    compare(f'force_calls_{step}', records[0][f'force_calls_{step}'], records[1][f'force_calls_{step}'])
    compare(f't_after_{step}', records[0][f't_after_{step}'], records[1][f't_after_{step}'])
    compare(f'pos_{step + 1}', records[0][f'pos_{step + 1}'], records[1][f'pos_{step + 1}'])
    compare(f'vel_{step + 1}', records[0][f'vel_{step + 1}'], records[1][f'vel_{step + 1}'])


banner("VERDICT")
if first_divergence is None:
    print(f"""No divergence in {N_STEPS} steps. Two instances in one process are
bit-identical, so `for i in range(2)` alone does not reproduce it either. The
cause is something that differed between the two reproduc runs but not here -
next suspects are the checkpoint round trip (run_0 and run_1 wrote and could
have resumed from different files) and anything that changed on the node
between them.""")
else:
    print(f"""First divergence: {first_divergence}

  acc_* first   -> the solver's output depends on being the second instance in
                   the process. Look at what the second construction inherits.
  pos_*/vel_* first, with acc_* identical
                -> identical forces, different integrated state: the
                   divergence is inside rebound, not in any JAX kernel, and no
                   XLA flag can fix it.""")
