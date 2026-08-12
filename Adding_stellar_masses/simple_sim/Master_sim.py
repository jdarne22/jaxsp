
import jax
jax.config.update("jax_enable_x64", True)

import os
from time import time

import numpy as np

import sys
sys.path.append('/gpfs/home/jd925/Adding_stellar_masses')

import jaxsp as jsp
import jax.numpy as jnp
from jaxsp.constants import GN

import Sharding_manager as SM
import Sim_init
import Building_rho_lms
import Building_phi_lms
import Calculate_acc
import Checkpoint_manager

import importlib
importlib.reload(SM)
importlib.reload(Sim_init)
importlib.reload(Building_rho_lms)
importlib.reload(Building_phi_lms)
importlib.reload(Calculate_acc)
importlib.reload(Checkpoint_manager)


#--------------------------------------------------------------------------------------------------------------------


def log_host_rss(label):
    """
    Current and peak host RSS, in GB.

    Host memory, not device memory: the m22 = 100 run segfaulted with 250 GB
    of host RSS while XLA was still compiling the merged Poisson solver, so
    the useful thing to watch is how much of that is already spent before the
    compile starts. Reads /proc, so it costs nothing and cannot itself fail
    the run.
    """
    try:
        with open('/proc/self/status') as f:
            fields = dict(
                line.split(':', 1) for line in f if line.startswith(('VmRSS', 'VmHWM')))
        current_gb = int(fields['VmRSS'].split()[0]) / 1024**2
        peak_gb = int(fields['VmHWM'].split()[0]) / 1024**2
        print(f"[host mem] {label}: {current_gb:.1f} GB now, {peak_gb:.1f} GB peak", flush=True)
    except Exception as exc:
        print(f"[host mem] {label}: unavailable ({exc})", flush=True)


def log_device_mem(label):
    """
    Per-device GPU memory, in GiB.

    Host RSS says nothing about the thing that is actually killing runs now:
    jobs 661302 and 661306 both died at timestep 1 with

      Failed to load in-memory CUBIN ... CUDA_ERROR_OUT_OF_MEMORY
      [executable_name='jit__combine_acc_chunked']

    and then the same for 'jit_gather', a trivial op - so it is not one
    executable being too big, it is the device having no room left to load
    ANY module. Loading a CUBIN draws on driver memory OUTSIDE the BFC pool,
    so what matters is bytes_in_use against bytes_limit, and how much of the
    card the pool has already claimed. largest_free_block_bytes separates
    "genuinely full" from "full enough that nothing contiguous is left".

    Best effort - memory_stats() is backend-specific and returns None on
    some platforms, so this must never be able to fail a run.
    """
    try:
        for device in jax.local_devices():
            stats = device.memory_stats()
            if not stats:
                print(f"[dev mem] {label}: {device} reports no stats", flush=True)
                continue
            # Every field, not a chosen four. The four I picked first time
            # (in_use / peak / limit / largest_free_block) did not include
            # whatever the executable loader actually draws on, so they could
            # not distinguish "pool is full" from "pool has headroom but the
            # driver does not". pool_bytes and bytes_reservable_limit do.
            rendered = ', '.join(
                f"{key} {value / 1024**3:.2f}" if isinstance(value, (int, float)) else f"{key} {value}"
                for key, value in sorted(stats.items())
            )
            print(f"[dev mem] {label}: {device.id} (GiB) {rendered}", flush=True)
    except Exception as exc:
        print(f"[dev mem] {label}: unavailable ({exc})", flush=True)


class StellarSimTDep:

    '''
    Stellar simulation which controls how everything is done. All the actual
    work happens on the SimInit / Rho_lm_Builder / Phi_lm_Builder /
    Acceleration_Calculator / Checkpoint_manager instances built here -
    this class just wires them together and drives them in order.
    '''

    def __init__(self, m22, r_half, r_half_width, no_of_particles, total_evolve_time, r_min, r_max_enclosing_frac,
                 no_radius_bins, dt_override, ramp_time, r_chunk_size, l_band_size,
                 compute_dtype, use_multi_gpu=True, L_out_frac=1.0,
                 use_merged_solver=True, chunk_batch_size=None, frozen=False, sph_sym=False,
                 r_cut_kpc=None, particle_chunk_size=None, particle_batch_size=None,
                 poten_every=10):

        self.m22 = m22
        self.r_half = r_half
        self.time_step = 0

        # How often the potential energy at the particles is evaluated, in
        # timesteps. This is a *second* full pass of the fused solver on top of
        # the one rebound's force callback already does - same cost, and it
        # buys nothing the dynamics need, only the energy-conservation
        # diagnostic. 1 = every step; at the default 10 it costs a tenth of
        # that, which takes the per-step cost from two solver passes to 1.1.
        #
        # kinetic_energy and ang_mom are recorded on the same cadence (see
        # Particles.update_state), so all three energy histories have the same
        # length and share one time axis: entry k is timestep
        # k * poten_every. Every other history - r_values, positions_xyz,
        # velocities, stellar_v_disp - is still one entry per step.
        self.poten_every = max(1, int(poten_every))

        # Outer radius (kpc) of the background rho_lm / phi_lm grid; None
        # keeps the full r_max_enclosing_frac grid. See
        # Sim_init.Truncate_radial_grid.
        self.r_cut_kpc = r_cut_kpc

        # frozen : wavefunction held at t = dt, so rho is the same every step.
        # sph_sym: only the l = 0 coefficient of rho kept, i.e. a
        #          spherically symmetric density.
        # Both are consumed by Rho_lm_Builder; both False is a normal run.
        self.frozen = bool(frozen)
        self.sph_sym = bool(sph_sym)

        self.sharding = SM.ShardingManager(use_multi_gpu)

        self.sim_init = Sim_init.SimInit(m22, r_min, r_max_enclosing_frac, no_radius_bins,
                                         r_cut_kpc=r_cut_kpc)

        # SimInit's own methods (Run_initialisation, Setup_rebound, Particle_ICs, ...)
        # expect these on self - set them here rather than in SimInit.__init__
        # so SimInit stays agnostic of the wider simulation config.
        self.sim_init.u = jsp.set_schroedinger_units(m22)
        self.sim_init.G = GN.value * (self.sim_init.u.from_cm**3) / (self.sim_init.u.from_g * self.sim_init.u.from_s**2)
        self.sim_init.r_half = r_half
        self.sim_init.r_half_width = r_half_width
        self.sim_init.no_of_particles = no_of_particles
        self.sim_init.total_evolve_time = total_evolve_time
        self.sim_init.dt_override = dt_override
        self.sim_init.ramp_time = ramp_time
        self.sim_init.L_out_frac = L_out_frac
        self.sim_init.sharding = self.sharding
        # Truncate_radial_grid rounds the kept bin count up to a whole
        # r_chunk_size chunk, so it needs to know the chunk size.
        self.sim_init.r_chunk_size = r_chunk_size

        self.rho_builder = Building_rho_lms.Rho_lm_Builder(
            self.sim_init, self.sharding, compute_dtype, r_chunk_size, ramp_time,
            frozen=self.frozen, sph_sym=self.sph_sym,
            particle_chunk_size=particle_chunk_size,
        )
        # Particle_ICs() builds the static density via self.Rho_lm_builder.
        self.sim_init.Rho_lm_builder = self.rho_builder

        self.phi_builder = Building_phi_lms.Phi_lm_Builder(
            self.sim_init, self.rho_builder, l_band_size,
            use_merged_solver=use_merged_solver,
            chunk_batch_size=chunk_batch_size,
            particle_batch_size=particle_batch_size,
        )

        self.acc_calculator = Calculate_acc.Acceleration_Calculator(
            self.sim_init, self.rho_builder, self.phi_builder)
        # rebound's force callback (built in Setup_rebound) calls self.construct_acc_master_func.
        self.sim_init.construct_acc_master_func = self.acc_calculator.construct_acc_master_func

        self.checkpoint_manager = None

    def _place_rho_lms(self, rho_lms):
        """Put the (Nr, L, 2L-1) density onto the GPU(s).

        The merged solver gathers `rho_lms[:, l_vals, m_inds]` - dynamic
        indices on the L axis, i.e. the axis `shard_l_arr` shards - inside
        the (l, m) `lax.map`. GSPMD can't make that gather device-local, so
        it all-gathers the full array, and it does so once per scan
        iteration (n_modes / l_band_size chunks). That is where the 2-GPU
        run deadlocked on its first NCCL clique acquire. Handing the solver
        a replicated array costs the same per-device memory - XLA was
        materialising the whole thing anyway - with no collective inside
        the scan.

        The per-particle solver has the same gather pattern, but it is the
        long-standing path and has run fine L-sharded, so it is left alone.
        If it ever hangs the same way, replicate here unconditionally.
        """
        if self.phi_builder.use_merged_solver:
            return self.sharding.replicate_arr(rho_lms)
        return self.sharding.shard_l_arr(rho_lms)

    def run_simulation(self, checkpoint_every=50, checkpoint_dir=None):

        if checkpoint_dir is None:
            # frozen / sph_sym get their own directory so they never resume
            # from - or overwrite - a normal run's checkpoints.
            # r_cut changes the potential the particles move in, so it has to
            # be part of the tag - otherwise a truncated run resumes from, and
            # overwrites, a full-grid one.
            mode_tag = (('_frozen_' if self.frozen else '') + ('_sphsym_' if self.sph_sym else '')
                        + ('' if self.r_cut_kpc is None else f'_rcut{self.r_cut_kpc:g}_'))
            checkpoint_dir = os.path.join(
                '/gpfs/home/jd925/Adding_stellar_masses', "Checkpoints", f"checkpoints_{mode_tag}m22_{self.m22:g}_r0_{self.r_half:g}_Lout_{self.sim_init.L_out_frac:g}"
            )
            
        self.checkpoint_manager = Checkpoint_manager.Checkpoint_manager(checkpoint_dir)

        # Which timestep the rho_lms currently on the GPU were built for -
        # None until the first build. Only the frozen path reads it.
        self._rho_lms_step = None

        print(f"Run mode: frozen={self.frozen}, sph_sym={self.sph_sym}, r_cut_kpc={self.r_cut_kpc}")

        start = time()
        self.sim_init.Run_initialisation()  # also triggers self.rho_builder.initialise() via Particle_ICs
        end = time()
        print(f"Initialisation completed in {end - start:.2f} seconds")
        log_host_rss('after initialisation')

        self.phi_builder.initialise()

        # Ramp phase: linearly switch on the off-diagonal cross-terms over
        # ramp_time, then run for total_evolve_time in the full potential.
        self.rho_builder.n_ramp_steps = self.sim_init.no_ramp_steps
        self.sim_init.no_time_steps += self.sim_init.no_ramp_steps
        print(f"Ramp phase: {self.rho_builder.n_ramp_steps} steps")
        print(f"Total: {self.sim_init.no_time_steps} steps")

        self.rho_builder.rho_lms = self._place_rho_lms(
            self.rho_builder.build_rho_lms_for_timestep(0).astype(self.rho_builder.compute_dtype)
        )
        self._rho_lms_step = 0
        print('completed rho_lms precomputation')
        log_host_rss('after rho_lms precomputation')
        log_device_mem('after rho_lms precomputation')

        particles = self.sim_init.particles

        # Initial potential energy for each particle. This is where the first
        # (and by far the largest) Poisson-solver compile happens, so bracket
        # it: with JAX_LOG_COMPILES=1 in job.sh, anything between these two
        # lines is compile time, and the host RSS afterwards is the real cost
        # of holding that executable.
        r_pos_sphs = jnp.array([p.r_pos_sph for p in particles])
        start = time()
        _, _, _, phi_at_parts = self.acc_calculator.construct_acc_master_func(r_pos_sphs, poten=True)

        # Block here, deliberately. JAX dispatch is asynchronous and nothing
        # below forces a transfer - `phi_at_parts[i]` builds another device
        # array rather than reading a value - so without this the timing above
        # measures compile-and-queue, not compile-and-run, and any execution
        # failure surfaces at whatever unrelated line blocks first. In job
        # 661306 that was the force callback at timestep 1, which is why the
        # CUBIN-load error appeared to come from rebound.
        phi_at_parts = jax.block_until_ready(phi_at_parts)
        print(f"First acceleration call (incl. compile) took {time() - start:.2f} seconds", flush=True)
        log_host_rss('after first acceleration call')
        log_device_mem('after first acceleration call')

        # One device -> host transfer for the whole array. Indexing the device
        # array per particle instead builds 1000 single-element gathers and
        # blocks on each of them.
        phi_at_parts = np.asarray(phi_at_parts)
        for i, particle in enumerate(particles):
            particle.potential_energy.append(float(phi_at_parts[i]))
            particle.Create_V_array(self.sim_init.no_time_steps)

        self.maximum_rho_00 = [jnp.max(jnp.abs(self.rho_builder.rho_lms[:, 0, self.sim_init.L_max_out - 1]))]

        loaded = self.checkpoint_manager.load(particles, self.sim_init.sim)
        if loaded is not None:
            self.time_step, self.sim_init.no_time_steps, self.rho_builder.n_ramp_steps = loaded
            print(f"Resumed from step {self.time_step} / {self.sim_init.no_time_steps}", flush=True)

            # A checkpoint written at a different cadence has energy histories
            # this run cannot extend consistently - entry k would no longer be
            # timestep k * poten_every. Fail here rather than silently produce
            # an unusable energy series; use a fresh checkpoint_dir instead.
            restored_every = getattr(particles[0], 'energy_every', 1)
            if restored_every != self.poten_every:
                raise ValueError(
                    f"Checkpoint was written with poten_every={restored_every}, this run "
                    f"has poten_every={self.poten_every}. Either match it or start a new "
                    f"checkpoint directory.")
        else:
            print("No checkpoint found, starting from step 0.", flush=True)

        for particle in particles:
            particle.energy_every = self.poten_every

        sim_init = self.sim_init

        while self.time_step < sim_init.no_time_steps:

            print(f"Time step {self.time_step + 1} / {sim_init.no_time_steps}")

            # frozen holds the wavefunction at t = dt, so rho_lms only
            # changes while the ramp is still blending it against the static
            # background. Once both this step and the step the current
            # rho_lms was built for are past the ramp, the array on the GPU
            # (and rho_builder's stashed phase / ramp_frac) is already right.
            rho_is_unchanging = (
                self.frozen
                and self.rho_builder.rho_lms is not None
                and self._rho_lms_step is not None
                and self.time_step >= self.rho_builder.n_ramp_steps
                and self._rho_lms_step >= self.rho_builder.n_ramp_steps
            )

            if rho_is_unchanging:
                print('frozen: reusing rho_lms from the previous timestep')
            else:
                print('Building rho_lms for this timestep...')
                start = time()
                self.rho_builder.rho_lms = None
                self.rho_builder.rho_lms = self._place_rho_lms(
                    self.rho_builder.build_rho_lms_for_timestep(self.time_step).astype(self.rho_builder.compute_dtype)
                )
                self._rho_lms_step = self.time_step
                end = time()
                print(f"rho_lms built in {end - start:.2f} seconds")
                log_device_mem(f'after rho_lms build, step {self.time_step}')

            self.maximum_rho_00.append(jnp.max(jnp.abs(self.rho_builder.rho_lms[:, 0, sim_init.L_max_out - 1])))

            # Synchronise all particle states into rebound, integrate one
            # macro timestep, then read back and update each particle.
            start = time()
            for i, particle in enumerate(particles):
                p = sim_init.sim_particles[i]
                p.x,  p.y,  p.z  = float(particle.r_pos[0]), float(particle.r_pos[1]), float(particle.r_pos[2])
                p.vx, p.vy, p.vz = float(particle.v[0]),     float(particle.v[1]),     float(particle.v[2])

            sim_init._force_call_count = 0
            target_time = sim_init.sim.t + sim_init.dt
            sim_init.sim.integrate(target_time)
            print(f"  Force calls this timestep: {sim_init._force_call_count}")

            record_energy = (self.time_step + 1) % self.poten_every == 0
            for i, particle in enumerate(particles):
                p = sim_init.sim_particles[i]
                particle.update_state([p.x, p.y, p.z], [p.vx, p.vy, p.vz],
                                      record_energy=record_energy)
            end = time()
            print(f"Time stepping all particles completed in {end - start:.2f} seconds")

            # Potential energy at the updated particle positions. A full extra
            # pass of the fused solver - as expensive as the force call above -
            # so it runs on the poten_every cadence, matching the
            # kinetic_energy / ang_mom appends update_state just made.
            if record_energy:
                r_pos_sphs_new = jnp.array([p.r_pos_sph for p in particles])
                _, _, _, phi_at_parts = self.acc_calculator.construct_acc_master_func(r_pos_sphs_new, poten=True)
                phi_at_parts = np.asarray(phi_at_parts)
                for i, particle in enumerate(particles):
                    particle.potential_energy.append(float(phi_at_parts[i]))

            self.time_step += 1

            if self.time_step % checkpoint_every == 0:
                self.checkpoint_manager.save(
                    particles, sim_init.sim, self.time_step, sim_init.no_time_steps, self.rho_builder.n_ramp_steps
                )

        self.checkpoint_manager.save(
            particles, sim_init.sim, self.time_step, sim_init.no_time_steps, self.rho_builder.n_ramp_steps, final=True
        )
