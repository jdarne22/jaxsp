"""
Acceleration_Calculator computes the acceleration on every particle at its
current position: rho_lm and phi_lm at each particle's exact radius
(Phi_lm_Builder, JIT'd), the spherical harmonics and their angular
derivatives at each particle's (theta, phi) (Memory_speed_savers), and the
contraction of the two into (a_r, a_theta, a_phi). This is the function
rebound's force callback calls every force sub-step.
"""

import jax
import jax.numpy as jnp

import Memory_speed_savers as MSS


class Acceleration_Calculator:

    def __init__(self, sim_init, rho_builder, phi_builder):

        self.sim_init = sim_init
        self.rho_builder = rho_builder
        self.phi_builder = phi_builder

        # Built on first call to construct_acc_master_func, once - JIT
        # compiling here rather than at construction time keeps __init__
        # cheap and avoids compiling before the objects it closes over
        # (rho_builder, phi_builder) are fully initialised.
        self.calc_rho_lm_at_parts_and_call_insert_jit = None
        self.combine_acc_jit = None

    @staticmethod
    def combine_acc(dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi, particle_r, particle_theta):
        """
        JIT-compilable: contracts the radial outputs (phi_lm, dphi_lm/dr)
        with the angular terms (Y_lm and its derivatives) to get the
        acceleration, in spherical coordinates, at every particle.
        """
        dphi_lm_dr_T = dphi_lm_dr_at_parts.T   # (n_modes, n_particles)
        phi_lm_T = phi_lm_at_parts.T           # (n_modes, n_particles)

        a_r = jnp.sum(-dphi_lm_dr_T * Ylm_all, axis=0).real
        a_theta = jnp.sum(-phi_lm_T * dY_dtheta / particle_r[None, :], axis=0).real
        a_phi = jnp.sum(
            -phi_lm_T * dY_dphi / (particle_r[None, :] * jnp.sin(particle_theta[None, :])), axis=0
        ).real

        return a_r, a_theta, a_phi

    def construct_acc_master_func(self, positions_sph, poten=False):
        """
        Computes the acceleration (and, if poten=True, the potential too)
        at every particle's current position, in one batched call rather
        than once per particle:

          1. rho_lm + phi_lm at every particle's exact radius (Phi_lm_Builder, JIT'd)
          2. Y_lm and its theta-derivative at every particle's (theta, phi)
          3. Combine the two into (a_r, a_theta, a_phi)
        """
        if self.calc_rho_lm_at_parts_and_call_insert_jit is None:
            # phi_builder.use_merged_solver picks between solving every
            # particle from one shared radial grid (merged) and inserting
            # one particle at a time (original). Same signature either way.
            solver = (self.phi_builder.calc_rho_lm_at_parts_and_call_insert_merged
                      if self.phi_builder.use_merged_solver
                      else self.phi_builder.calc_rho_lm_at_parts_and_call_insert)
            self.calc_rho_lm_at_parts_and_call_insert_jit = jax.jit(solver)
            self.combine_acc_jit = jax.jit(Acceleration_Calculator.combine_acc)

        rho_builder = self.rho_builder

        dphi_lm_dr_at_parts, phi_lm_at_parts = self.calc_rho_lm_at_parts_and_call_insert_jit(
            positions_sph,
            rho_builder.current_phase,
            rho_builder.rho_lms,
            rho_builder.current_ramp_frac,
            (rho_builder.aj_sorted, rho_builder.parent_j_sorted, rho_builder.bin_blocks.as_arrays()),
        )

        # On-GPU Y_lm + dY_lm/dtheta. l ranges over [0, L_max_out - 1] in
        # output_lm_pairs.
        Ylm_all, dY_dtheta = MSS.compute_Ylm_and_dtheta_jit(
            self.phi_builder.output_lm_pairs,
            positions_sph[:, 1],
            positions_sph[:, 2],
            int(self.sim_init.L_max_out - 1),
        )   # (n_modes, n_particles) each

        m_values = self.phi_builder.output_lm_pairs[:, 1, None]   # (n_modes, 1)
        dY_dphi = 1j * m_values * Ylm_all                          # (n_modes, n_particles)

        if poten:
            a_r, a_theta, a_phi = self.combine_acc_jit(
                dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
                positions_sph[:, 0], positions_sph[:, 1],
            )
            return a_r, a_theta, a_phi, phi_lm_at_parts, Ylm_all

        return self.combine_acc_jit(
            dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
            positions_sph[:, 0], positions_sph[:, 1],
        )

