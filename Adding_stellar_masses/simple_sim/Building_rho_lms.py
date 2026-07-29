"""
Rho_lm_Builder builds the density rho(r), expressed as a spherical-harmonic
expansion rho_lm(r), from the wavefunction data SimInit loaded. It hands the
result to Master_sim ready for s2fft: an (n_radii, L_max_out, 2*L_max_out - 1)
complex array each timestep, plus a way to evaluate the same expansion at an
arbitrary particle radius.

The heavy numerical kernels - the sparse scatter over k-modes and the
streamed s2fft round-trip - live in Memory_speed_savers.py (imported below
as MSS). This file owns the state those kernels need (aj, R_j_r_fixed,
bin_blocks, ...) and the simpler, non-streamed pieces of the calculation.
"""

import jax
import jax.numpy as jnp
import jaxsp as jsp

import Memory_speed_savers as MSS


class Rho_lm_Builder:

    def __init__(self, sim_init, sharding, compute_dtype,
                 sparse_k_batch, r_chunk_size, ramp_time):


        self.sim_init = sim_init


        self.sharding = sharding

        # ---- compute dtype ----
        # The density arrays are the largest thing in memory, so they run
        # in complex64 / float32 by default. complex128 / float64 is there
        # for when precision matters more than memory.
        self.compute_dtype = jnp.dtype(compute_dtype)
        if self.compute_dtype == jnp.complex64:
            self.compute_real_dtype = jnp.float32
        elif self.compute_dtype == jnp.complex128:
            self.compute_real_dtype = jnp.float64
        else:
            raise ValueError(
                f"compute_dtype must be complex64 or complex128, got {compute_dtype}")

        # memory-saving chunk sizes, passed straight through to MSS
        self.sparse_k_batch = int(sparse_k_batch)
        self.r_chunk_size = int(r_chunk_size)


        self.ramp_time = ramp_time
        self.n_ramp_steps = None

        # filled in by the methods below, in the order initialise() calls them 
        self.aj = None                  # random-phase amplitude per k-mode
        self.R_j_r_fixed = None         # radial basis functions, cast + padded + sharded
        self.nj_pad = 0                 # how many zero radial-mode columns were added
        self.l_values_padded = None     # l, padded to match R_j_r_fixed's radial-mode axis
        self.eigen_energies_padded = None
        self.weight_j = None            # per-radial-mode weight in the static density
        self.rho_static_r_l00 = None    # static density's (l=0, m=0) coefficient
        self.bin_blocks = None          # k-mode grouping plan, see precompute_bin_blocks
        self.aj_sorted = None
        self.parent_j_sorted = None
        self._eval_library = None       # vmapped radial-eigenmode evaluator, see R_j_at_radii

        # ---- current-timestep state ----
        # Set by build_rho_lms_for_timestep every macro step; read by
        # whatever computes phi_lm/accelerations at each force sub-step
        # within that same macro step (Phi_lm_Builder, Acceleration_Calculator).
        self.current_phase = None
        self.current_ramp_frac = None

    def initialise(self):
        """
        Runs every one-time setup step, in the order it needs to happen.
        """
        self.build_initial_aj()
        self.prepare_R_j_r()
        self.precompute_bin_blocks()
        self.sort_modes_for_sphht()
        rho_static_r = self.compute_diagonal_rho_expansion()

        # Pure function of (r, radial_eigenmode_params) - vmapped once here
        # so R_j_at_radii doesn't rebuild it on every call.
        self._eval_library = jax.vmap(
            jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0, None))

        return rho_static_r



    def build_initial_aj(self, seed=42):
        """
        Gives every k-mode (a basis function labelled by radial mode j and
        angular numbers l, m) a random starting phase.
        """
        parent_j = self.sim_init.parent_j   # which radial mode j each k-mode belongs to
        aj_2 = self.sim_init.aj_2           # |a_j|^2 target, one value per radial mode j

        number_of_k_modes = len(parent_j)

        random_phase = jax.random.uniform(
            jax.random.PRNGKey(seed),
            shape=(number_of_k_modes,),
            minval=0.0,
            maxval=2.0 * jnp.pi,
        )

        amplitude = jnp.sqrt(aj_2[parent_j])   # (number_of_k_modes,) real
        self.aj = (amplitude * jnp.exp(1j * random_phase)).astype(self.compute_dtype)

    def prepare_R_j_r(self):
        """
        R_j_r is the radial part of every basis function, evaluated on the
        background radial grid: shape (n_radii, n_radial_modes).

        Need to cast into the dtype set

        Need to pad nl modes so that it can be divided across multiple GPUs
        """

        R_j_r = jnp.asarray(self.sim_init.R_j_r, dtype=self.compute_real_dtype)

        l_values = jnp.asarray(self.sim_init.l)

        eigen_energies = self.sim_init.eigen_energies

        self.nj_pad = 0

        if self.sharding.shard_nj is not None:
            n_devices = len(self.sharding.devices)
            n_radial_modes = R_j_r.shape[1]
            pad_amount = (-n_radial_modes) % n_devices

            if pad_amount:
                R_j_r = jnp.pad(R_j_r, ((0, 0), (0, pad_amount)))
                l_values = jnp.pad(l_values, (0, pad_amount))
                eigen_energies = jnp.pad(eigen_energies, (0, pad_amount))
                self.nj_pad = pad_amount

        # Padded radial modes are never pointed to by parent_j, so they
        # never contribute to any sum - the padding is inert, purely there
        # to make the array shape divide evenly across devices.

        self.l_values_padded = l_values
        self.eigen_energies_padded = eigen_energies
        self.R_j_r_fixed = self.sharding.shard_nj_arr(R_j_r)


    def precompute_bin_blocks(self):
        """
        The density calculation needs every k-mode grouped by which (l, m)
        pair it belongs to so the sparse matmul can be done in fixed-size blocks.
        """
        lm_idx_per_mode = self.sim_init.lm_idx_per_mode
        n_unique_lm_pairs = self.sim_init.lm_pairs.shape[0]

        self.bin_blocks = MSS.precompute_bin_blocks(
            lm_idx_per_mode, n_unique_lm_pairs, self.sparse_k_batch)

    def sort_modes_for_sphht(self):
        """
        Reorders aj and parent_j to match the grouping worked out in
        precompute_bin_blocks, and pads the tail so the last group is
        always a fixed size.
        """

        permutation = self.bin_blocks.perm
        pad_length = self.bin_blocks.k_block

        parent_j = self.sim_init.parent_j

        self.aj_sorted = jnp.concatenate([
            self.aj[permutation],
            jnp.zeros(pad_length, dtype=self.aj.dtype),
        ])
        self.parent_j_sorted = jnp.concatenate([
            parent_j[permutation],
            jnp.zeros(pad_length, dtype=parent_j.dtype),
        ])

    def compute_diagonal_rho_expansion(self):
        """
        The time-averaged ("static") density profile rho_static(r).

            rho_static(r) = total_mass * sum_j  weight_j * |R_j(r)|^2
            weight_j        = |a_j|^2 * (2 l_j + 1) / (4 pi)

        This is the background the particles' circular-velocity initial
        conditions are built from, and the baseline the ramp schedule
        blends away from at the start of the simulation.
        """

        total_mass = self.sim_init.total_mass
        parent_j = self.sim_init.parent_j
        l_values = self.l_values_padded

        n_radial_modes = self.R_j_r_fixed.shape[1]

        # |a_j|^2 is shared by every k-mode with the same j, so recover it
        # by scattering the (already-squared) per-k-mode amplitudes back
        # onto the radial-mode axis.
        amplitude_sq_per_k_mode = jnp.abs(self.aj) ** 2
        amplitude_sq_per_radial_mode = (
            jnp.zeros(n_radial_modes, dtype=jnp.float64).at[parent_j].set(amplitude_sq_per_k_mode)
        )

        weight_j = amplitude_sq_per_radial_mode * (2.0 * l_values.astype(jnp.float64) + 1.0) / (4.0 * jnp.pi)
        self.weight_j = weight_j

        R_j_r_squared = (jnp.abs(self.R_j_r_fixed) ** 2).astype(jnp.float64)
        rho_static_r = total_mass * (R_j_r_squared @ weight_j)

        # Y_00 = 1 / sqrt(4 pi), so the (l=0, m=0) coefficient of an
        # isotropic profile is the profile itself times sqrt(4 pi).
        self.rho_static_r_l00 = (rho_static_r * jnp.sqrt(4.0 * jnp.pi)).astype(self.compute_dtype)

        return rho_static_r


    def ramp_frac_for_step(self, time_step):
        """
        Fraction of the time-dependent part of rho to include at this
        step: ramps linearly from ~0 up to 1 over the first
        n_ramp_steps, then stays at 1 for the rest of the simulation.
        """

        if time_step < self.n_ramp_steps:
            return (time_step + 1) / self.n_ramp_steps
        else:
            return 1.0


    def build_rho_lms_for_timestep(self, time_step):
        """
        Builds rho_lm(r) at this timestep:

            rho = (1 - ramp_frac) * rho_static + ramp_frac * rho_full(t)
        """

        dt = self.sim_init.dt
        eigen_energies = self.eigen_energies_padded

        # e^{-i E_j t}: how far each radial mode's wavefunction has rotated
        # in phase by this timestep. Stays in complex128 - small phase
        # errors here compound badly over a long simulation.
        phase = jnp.exp(-1j * eigen_energies * time_step * dt)
        phase_in_compute_dtype = phase.astype(self.compute_dtype)

        ramp_frac = self.ramp_frac_for_step(time_step)
        ramp_frac_in_real_dtype = jnp.asarray(ramp_frac, dtype=self.compute_real_dtype)

        # Stashed so Phi_lm_Builder / Acceleration_Calculator can read the
        # same phase and ramp fraction for every force sub-step of this
        # macro timestep, without recomputing them. Kept in float64 here
        # (unlike the real_dtype-cast copy above, which exists only to
        # avoid promoting the c64 rho_lms path to c128).
        self.current_phase = phase
        self.current_ramp_frac = jnp.float64(ramp_frac)

        n_unique_lm_pairs = self.sim_init.lm_pairs.shape[0]

        return MSS.build_sphht_rho_lms_jit(
            self.R_j_r_fixed, phase_in_compute_dtype,
            self.aj_sorted, self.parent_j_sorted, self.bin_blocks.as_arrays(),
            self.sim_init.lm_pairs, self.sim_init.total_mass,
            ramp_frac_in_real_dtype, self.rho_static_r_l00,
            int(self.sim_init.L_max_out),
            n_unique_lm_pairs, self.sparse_k_batch,
            self.r_chunk_size,
            out_sharding=self.sharding.shard_l,
        )


    def R_j_at_radii(self, radii):
        """
        Evaluates every radial basis function R_j at an arbitrary set of
        radii - e.g. particle positions, which move every timestep -
        rather than the fixed background grid prepare_R_j_r used. Cast
        and padded the same way as R_j_r_fixed, so it can be contracted
        against the same per-radial-mode arrays (weight_j, aj, ...).
        """
        R_j_at_radii = self._eval_library(radii, self.sim_init.radial_eigenmode_params)
        R_j_at_radii = R_j_at_radii.astype(self.compute_real_dtype)

        if self.nj_pad:
            R_j_at_radii = jnp.pad(R_j_at_radii, ((0, 0), (0, self.nj_pad)))

        return R_j_at_radii

    def rho_lm_at_particles_diagonal_only(self, R_j_at_particles):
        """
        rho_lm evaluated at a set of particle radii, using only the static
        (diagonal) density. Because that density is spherically symmetric,
        only the (l=0, m=0) coefficient is nonzero, so no spherical
        harmonic transform is needed - just a dot product against
        weight_j.

        R_j_at_particles: (n_particles, n_radial_modes), the radial basis
        functions evaluated at each particle's radius. Must already be
        padded by self.nj_pad to line up with weight_j.
        """
        R_j_at_particles_squared = (jnp.abs(R_j_at_particles) ** 2).astype(self.compute_real_dtype)
        rho_at_particles = self.sim_init.total_mass * (
            R_j_at_particles_squared @ self.weight_j.astype(self.compute_real_dtype)
        )

        n_particles = rho_at_particles.shape[0]
        L_max_out = self.sim_init.L_max_out
        m_zero_index = L_max_out - 1   # column of m=0 in the (2*L_max_out - 1)-wide m axis

        rho_lm_at_particles = jnp.zeros(
            (n_particles, L_max_out, 2 * L_max_out - 1), dtype=self.compute_dtype)

        return rho_lm_at_particles.at[:, 0, m_zero_index].set(
            rho_at_particles * jnp.sqrt(4.0 * jnp.pi))

    def rho_lm_at_particles(self, R_j_at_particles, phase_in_compute_dtype):
        """
        rho_lm evaluated at a set of particle radii, using the full
        time-dependent density. The caller blends this against
        rho_lm_at_particles_diagonal_only's result using the ramp
        fraction, the same way build_rho_lms_for_timestep does for the
        background grid.
        """
        n_unique_lm_pairs = self.sim_init.lm_pairs.shape[0]

        return MSS.compute_rho_lm_at_particles_sphht_jit(
            R_j_at_particles, phase_in_compute_dtype,
            self.aj_sorted, self.parent_j_sorted, self.bin_blocks.as_arrays(),
            self.sim_init.lm_pairs, self.sim_init.total_mass,
            int(self.sim_init.L_max_out),
            n_unique_lm_pairs, self.sparse_k_batch,
        )





