
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)

import os
from time import time

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


class StellarSimTDep:

    '''
    Stellar simulation which controls how everything is done. All the actual
    work happens on the SimInit / Rho_lm_Builder / Phi_lm_Builder /
    Acceleration_Calculator / Checkpoint_manager instances built here -
    this class just wires them together and drives them in order.
    '''

    def __init__(self, m22, r_half, r_half_width, no_of_particles, total_evolve_time, r_min, r_max_enclosing_frac,
                 no_radius_bins, dt_override, ramp_time, sparse_k_batch, r_chunk_size, l_band_size,
                 compute_dtype, use_multi_gpu=True, L_out_frac=1.0,
                 use_merged_solver=True, chunk_batch_size=None):

        self.m22 = m22
        self.r_half = r_half
        self.time_step = 0

        self.sharding = SM.ShardingManager(use_multi_gpu)

        self.sim_init = Sim_init.SimInit(m22, r_min, r_max_enclosing_frac, no_radius_bins)

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

        self.rho_builder = Building_rho_lms.Rho_lm_Builder(
            self.sim_init, self.sharding, compute_dtype, sparse_k_batch, r_chunk_size, ramp_time
        )
        # Particle_ICs() builds the static density via self.Rho_lm_builder.
        self.sim_init.Rho_lm_builder = self.rho_builder

        self.phi_builder = Building_phi_lms.Phi_lm_Builder(
            self.sim_init, self.rho_builder, l_band_size,
            use_merged_solver=use_merged_solver,
            chunk_batch_size=chunk_batch_size,
        )

        self.acc_calculator = Calculate_acc.Acceleration_Calculator(self.sim_init, self.rho_builder, self.phi_builder)
        # rebound's force callback (built in Setup_rebound) calls self.construct_acc_master_func.
        self.sim_init.construct_acc_master_func = self.acc_calculator.construct_acc_master_func

        self.checkpoint_manager = None

    def run_simulation(self, checkpoint_every=50, checkpoint_dir=None):

        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(
                '/gpfs/home/jd925/Adding_stellar_masses', "Checkpoints", f"checkpoints_m22_{self.m22:g}_r0_{self.r_half:g}_Lout_{self.sim_init.L_out_frac:g}"
            )
            
        self.checkpoint_manager = Checkpoint_manager.Checkpoint_manager(checkpoint_dir)

        start = time()
        self.sim_init.Run_initialisation()  # also triggers self.rho_builder.initialise() via Particle_ICs
        end = time()
        print(f"Initialisation completed in {end - start:.2f} seconds")

        self.phi_builder.initialise()

        # Ramp phase: linearly switch on the off-diagonal cross-terms over
        # ramp_time, then run for total_evolve_time in the full potential.
        self.rho_builder.n_ramp_steps = self.sim_init.no_ramp_steps
        self.sim_init.no_time_steps += self.sim_init.no_ramp_steps
        print(f"Ramp phase: {self.rho_builder.n_ramp_steps} steps")
        print(f"Total: {self.sim_init.no_time_steps} steps")

        self.rho_builder.rho_lms = self.sharding.shard_l_arr(
            self.rho_builder.build_rho_lms_for_timestep(0).astype(self.rho_builder.compute_dtype)
        )
        print('completed rho_lms precomputation')

        particles = self.sim_init.particles

        # Initial potential energy for each particle.
        r_pos_sphs = jnp.array([p.r_pos_sph for p in particles])
        _, _, _, phi_lm_at_r, Y_lm_all = self.acc_calculator.construct_acc_master_func(r_pos_sphs, poten=True)
        phi_at_parts = jnp.sum(phi_lm_at_r * Y_lm_all.T, axis=1)

        for i, particle in enumerate(particles):
            particle.potential_energy.append(phi_at_parts[i].real)
            particle.Create_V_array(self.sim_init.no_time_steps)

        self.maximum_rho_00 = [jnp.max(jnp.abs(self.rho_builder.rho_lms[:, 0, self.sim_init.L_max_out - 1]))]

        loaded = self.checkpoint_manager.load(particles, self.sim_init.sim)
        if loaded is not None:
            self.time_step, self.sim_init.no_time_steps, self.rho_builder.n_ramp_steps = loaded
            print(f"Resumed from step {self.time_step} / {self.sim_init.no_time_steps}", flush=True)
        else:
            print("No checkpoint found, starting from step 0.", flush=True)

        sim_init = self.sim_init

        while self.time_step < sim_init.no_time_steps:

            print(f"Time step {self.time_step + 1} / {sim_init.no_time_steps}")

            print('Building rho_lms for this timestep...')
            start = time()
            self.rho_builder.rho_lms = None
            self.rho_builder.rho_lms = self.sharding.shard_l_arr(
                self.rho_builder.build_rho_lms_for_timestep(self.time_step).astype(self.rho_builder.compute_dtype)
            )
            end = time()
            print(f"rho_lms built in {end - start:.2f} seconds")

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

            for i, particle in enumerate(particles):
                p = sim_init.sim_particles[i]
                particle.update_state([p.x, p.y, p.z], [p.vx, p.vy, p.vz])
            end = time()
            print(f"Time stepping all particles completed in {end - start:.2f} seconds")

            # Potential energy at the updated particle positions.
            r_pos_sphs_new = jnp.array([p.r_pos_sph for p in particles])
            _, _, _, phi_lm_new, Ylm_new = self.acc_calculator.construct_acc_master_func(r_pos_sphs_new, poten=True)
            phi_at_parts = jnp.sum(phi_lm_new * Ylm_new.T, axis=1).real
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
