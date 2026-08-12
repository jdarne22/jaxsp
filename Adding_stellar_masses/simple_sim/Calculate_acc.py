"""
Acceleration_Calculator computes the acceleration on every particle at its
current position: rho_lm and phi_lm at each particle's exact radius
(Phi_lm_Builder, JIT'd), the spherical harmonics and their angular
derivatives at each particle's (theta, phi) (Memory_speed_savers), and the
contraction of the two into (a_r, a_theta, a_phi). This is the function
rebound's force callback calls every force sub-step.
"""

import functools

import jax
import jax.numpy as jnp

import Memory_speed_savers as MSS


class Acceleration_Calculator:

    def __init__(self, sim_init, rho_builder, phi_builder):

        self.sim_init = sim_init
        self.rho_builder = rho_builder
        self.phi_builder = phi_builder

        # There is no separate chunk size for the angular half of the
        # calculation. It runs inside the Poisson solver's own particle batch,
        # so particle_batch_size sizes both - see per_batch_reduce in
        # Building_phi_lms and _fused_solve_and_combine below.

        # Built on first call to construct_acc_master_func, once - JIT
        # compiling here rather than at construction time keeps __init__
        # cheap and avoids compiling before the objects it closes over
        # (rho_builder, phi_builder) are fully initialised.
        self._fused_jit = None

    @staticmethod
    def combine_acc(dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
                    particle_r, particle_theta):
        """
        JIT-compilable: contracts the radial outputs (phi_lm, dphi_lm/dr)
        with the angular terms (Y_lm and its derivatives) to get the
        acceleration, in spherical coordinates, at every particle, plus the
        potential at each particle, sum_lm phi_lm Y_lm.

        The potential is the same contraction over modes, done here rather
        than by the caller so the two (n_particles, n_modes) arrays it needs
        never have to leave the device.

        It is computed unconditionally, and callers that only want the
        acceleration simply drop it. Making it a static flag instead splits
        this into two executables - one per value of the flag - and job
        661302 died on exactly that: the poten=True variant loaded at
        startup, then rebound's first force call needed the poten=False
        variant loaded onto an already-full GPU and got
        "Failed to load in-memory CUBIN ... CUDA_ERROR_OUT_OF_MEMORY".
        One extra reduction per chunk is far cheaper than a second
        resident executable.
        """
        dphi_lm_dr_T = dphi_lm_dr_at_parts.T   # (n_modes, n_particles)
        phi_lm_T = phi_lm_at_parts.T           # (n_modes, n_particles)

        a_r = jnp.sum(-dphi_lm_dr_T * Ylm_all, axis=0).real
        a_theta = jnp.sum(-phi_lm_T * dY_dtheta / particle_r[None, :], axis=0).real
        a_phi = jnp.sum(
            -phi_lm_T * dY_dphi / (particle_r[None, :] * jnp.sin(particle_theta[None, :])), axis=0
        ).real

        # Master_sim used to do this itself as
        #   jnp.sum(phi_lm_at_r * Y_lm_all.T, axis=1).real
        # on the returned arrays. Same reduction, same result.
        phi_at_parts = jnp.sum(phi_lm_T * Ylm_all, axis=0).real

        return a_r, a_theta, a_phi, phi_at_parts

    @staticmethod
    def angular_terms_and_combine(dphi_batch, phi_batch, positions_batch,
                                  output_lm_pairs, n_max):
        """
        The whole angular half for one particle batch: Y_lm and its two
        derivatives at this batch's angles, contracted against this batch's
        radial arrays.

        Built to be handed to a solver as per_batch_reduce, so it runs inside
        the solver's particle batch. Everything it touches is
        (n_modes, batch) - 1.27 GB at L_max_out = 892 and batch = 100, against
        11.86 GiB apiece if the same thing is done at 1000 particles.
        """
        Ylm_all, dY_dtheta = MSS.compute_Ylm_and_dtheta_jit(
            output_lm_pairs, positions_batch[:, 1], positions_batch[:, 2], n_max,
        )                                                  # (n_modes, batch) each

        m_values = output_lm_pairs[:, 1, None]              # (n_modes, 1)
        dY_dphi = 1j * m_values * Ylm_all

        return Acceleration_Calculator.combine_acc(
            dphi_batch, phi_batch, Ylm_all, dY_dtheta, dY_dphi,
            positions_batch[:, 0], positions_batch[:, 1],
        )

    def construct_acc_master_func(self, positions_sph, poten=False):
        """
        Computes the acceleration (and, if poten=True, the potential too)
        at every particle's current position, in one batched call rather
        than once per particle:

          1. rho_lm + phi_lm at every particle's exact radius (Phi_lm_Builder, JIT'd)
          2. Y_lm and its theta-derivative at every particle's (theta, phi)
          3. Combine the two into (a_r, a_theta, a_phi)

        All three run together, particle_batch_size particles at a time: step 1
        already batched, and steps 2 and 3 are now handed to it as
        per_batch_reduce so they run inside the same batch. The
        (n_particles, n_modes) phi_lm / dphi_lm_dr arrays that used to sit
        between step 1 and step 2 no longer exist - see per_batch_reduce in
        Building_phi_lms for why that matters.

        Returns (a_r, a_theta, a_phi), plus phi_at_parts when poten=True.
        Every element is (n_particles,).
        """
        if self._fused_jit is None:
            # phi_builder.use_merged_solver picks between solving every
            # particle from one shared radial grid (merged) and inserting
            # one particle at a time (original). Same signature either way.
            solver = (self.phi_builder.calc_rho_lm_at_parts_and_call_insert_merged
                      if self.phi_builder.use_merged_solver
                      else self.phi_builder.calc_rho_lm_at_parts_and_call_insert)

            # l ranges over [0, L_max_out - 1] in output_lm_pairs. Static, so
            # closing over it is free.
            n_max = int(self.sim_init.L_max_out - 1)

            def fused(positions_sph, current_phase, rho_lms, ramp_frac,
                      amplitudes, eigenmode_params, output_lm_pairs):
                def per_batch_reduce(dphi_batch, phi_batch, positions_batch):
                    return Acceleration_Calculator.angular_terms_and_combine(
                        dphi_batch, phi_batch, positions_batch, output_lm_pairs, n_max,
                    )

                return solver(positions_sph, current_phase, rho_lms, ramp_frac,
                              amplitudes, eigenmode_params,
                              per_batch_reduce=per_batch_reduce)

            # output_lm_pairs is an argument rather than closed over, for the
            # same reason `amplitudes` is: closure arrays lower to module
            # constants. It is only a few MB here, but the habit is cheap.
            self._fused_jit = jax.jit(fused)

        rho_builder = self.rho_builder

        args = (
            positions_sph,
            rho_builder.current_phase,
            rho_builder.rho_lms,
            rho_builder.current_ramp_frac,
            # One complex amplitude per k-mode, 5.7 GB at m22 = 100. An
            # argument, not a closure, for the reason below.
            rho_builder.amplitudes,
            # The radial eigenmode library, 20.3 GB at m22 = 100 and the
            # largest array in the run. An argument, not a closure: closed
            # over, it lowers into the module as a constant, which is
            # replicated on every device at load time and is what "generated
            # code 26826.8 MiB" was in job 661xxx - 28.13 GB to the byte. See
            # Building_rho_lms.R_j_at_radii.
            self.sim_init.radial_eigenmode_params,
            self.phi_builder.output_lm_pairs,
        )

        # One compiled executable regardless of poten - the potential always
        # comes back and is dropped here when unwanted, rather than being a
        # static flag that would compile a second copy of the whole fused
        # solve. See combine_acc.
        a_r, a_theta, a_phi, phi_at_parts = self._fused_jit(*args)

        if poten:
            return a_r, a_theta, a_phi, phi_at_parts

        return a_r, a_theta, a_phi

