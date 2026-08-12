"""
Rho_lm_Builder builds the density rho(r), expressed as a spherical-harmonic
expansion rho_lm(r), from the wavefunction data SimInit loaded. It hands the
result to Master_sim ready for s2fft: an (n_radii, L_max_out, 2*L_max_out - 1)
complex array each timestep, plus a way to evaluate the same expansion at an
arbitrary particle radius.

The heavy numerical kernels - the per-l matmuls that build psi_lm and the
streamed s2fft round trip - live in Memory_speed_savers.py (imported below as
MSS). This file owns the state those kernels need (amplitudes, R_j_r_fixed,
l_groups, ...) and the simpler pieces of the calculation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import jaxsp as jsp

import Memory_speed_savers as MSS


class Rho_lm_Builder:

    def __init__(self, sim_init, sharding, compute_dtype, r_chunk_size, ramp_time,
                 frozen=False, sph_sym=False, particle_chunk_size=None):

        self.sim_init = sim_init
        self.sharding = sharding

        # ---- run modes ----
        # frozen : hold the wavefunction at t = dt, so every timestep sees
        #          the same rho (no phase evolution of the eigenmodes).
        # sph_sym: keep only the (l=0, m=0) coefficient of rho, giving a
        #          spherically symmetric density.
        # Both False is the normal time-dependent, fully anisotropic run.
        self.frozen = bool(frozen)
        self.sph_sym = bool(sph_sym)

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

        # Radii per chunk of the s2fft round trip inside
        # build_sphht_rho_lms_jit. Bounds that transform's working set, which
        # is the only part of the build that needs bounding.
        self.r_chunk_size = int(r_chunk_size)

        # Same knob for the particle path. None = all particles at once.
        self.particle_chunk_size = (
            None if particle_chunk_size is None else int(particle_chunk_size))

        self.ramp_time = ramp_time
        self.n_ramp_steps = None

        # filled in by the methods below, in the order initialise() calls them
        self.amplitudes = None          # one complex amplitude per k-mode
        self.l_groups = None            # where each l's modes/amplitudes live
        self.R_j_r_fixed = None         # radial basis functions on the background grid
        self.sht_precomputes = None     # s2fft recursion coefficients, built once
        self.weight_j = None            # per-radial-mode weight in the static density
        self.rho_static_r_l00 = None    # static density's (l=0, m=0) coefficient
        self._eval_library = None       # vmapped radial-eigenmode evaluator

        # ---- current-timestep state ----
        # Set by build_rho_lms_for_timestep every macro step; read by
        # whatever computes phi_lm/accelerations at each force sub-step
        # within that same macro step (Phi_lm_Builder, Acceleration_Calculator).
        self.current_phase = None
        self.current_ramp_frac = None

    def initialise(self):
        """
        Runs every one-time setup step, in the order it needs to happen.

        The ordering is memory-driven. `amplitudes` is one complex number per
        k-mode - 5.7 GB at m22 = 100 - so it is built on the CPU device and
        stays in host RAM until the one device_put at the end.
        """
        self.build_amplitudes()
        self.prepare_R_j_r()

        # Which radial modes belong to which l. Pure numpy, and the only
        # bookkeeping the per-timestep kernels need.
        self.l_groups = MSS.group_radial_modes_by_l(self.sim_init.l)

        # s2fft would otherwise rebuild these inside every transform call.
        self.sht_precomputes = MSS.build_sht_precomputes(int(self.sim_init.L_max_out))

        rho_static_r = self.compute_diagonal_rho_expansion()

        # Pure function of (r, radial_eigenmode_params) - vmapped once here
        # so R_j_at_radii doesn't rebuild it on every call.
        self._eval_library = jax.vmap(
            jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0, None))

        # Replicated, not sharded: psi_lm_at_rows slices it by l, and both
        # devices need every l. It is the same 5.7 GB per device that the old
        # sorted copy cost device 0, and it replaces *two* arrays - the sorted
        # amplitudes and the per-k-mode parent index that went with them.
        # Straight from the host array build_amplitudes left behind - going
        # via jnp.asarray first would land the whole 5.7 GB on GPU 0 before
        # the replicate, and with XLA_PYTHON_CLIENT_PREALLOCATE=false the
        # allocator never gives that region back.
        self.amplitudes = self.sharding.replicate_arr(self.amplitudes)

        return rho_static_r

    def build_amplitudes(self, seed=42):
        """
        Gives every k-mode (a radial mode j paired with an m in -l_j .. l_j) a
        random starting phase, on top of the amplitude |a_j| its radial mode
        already fixes.

        Runs on the CPU device so the result - 5.7 GB at m22 = 100 - is built
        in host RAM rather than costing a GPU that much before the run starts.
        threefry is bit-identical across backends, so the random stream is the
        same as a device build would give.

        The whole chain runs under one jit so XLA fuses it and writes only the
        final complex64 array. Eagerly, every step materialises its own
        full-length temporary - two of them in complex128 - which is ~68 GB of
        intermediates to produce a 5.7 GB result.
        """
        l = np.asarray(self.sim_init.l)
        k_modes_per_radial_mode = 2 * l.astype(np.int64) + 1
        n_k_modes = int(k_modes_per_radial_mode.sum())

        # |a_j|^2 is a property of the radial mode, so every one of its 2l+1
        # k-modes shares it. np.repeat spells that out directly; the old code
        # did the same thing as a gather through a per-k-mode parent index,
        # which meant carrying that 2.9 GB index array around for no other
        # reason.
        amplitude_sq_per_k_mode = np.repeat(
            np.asarray(self.sim_init.aj_2), k_modes_per_radial_mode)

        @jax.jit
        def build(amplitude_sq):
            random_phase = jax.random.uniform(
                jax.random.PRNGKey(seed), shape=(n_k_modes,),
                minval=0.0, maxval=2.0 * jnp.pi)
            return (jnp.sqrt(amplitude_sq) * jnp.exp(1j * random_phase)).astype(
                self.compute_dtype)

        with jax.default_device(jax.devices('cpu')[0]):
            self.amplitudes = np.asarray(build(amplitude_sq_per_k_mode))

    def prepare_R_j_r(self):
        """
        R_j_r is the radial part of every basis function, evaluated on the
        background radial grid: shape (n_radii, n_radial_modes).

        Cast on the HOST. Doing it with jnp would put the whole array on GPU 0
        first, and because XLA_PYTHON_CLIENT_PREALLOCATE is false the allocator
        keeps every region it ever grows, so that transient would permanently
        shrink the largest contiguous block left for everything after it.

        Replicated rather than sharded on the radial-mode axis. psi_lm_at_rows
        takes a contiguous column slice per l, and an l's modes do not line up
        with device boundaries, so sharding that axis would make every one of
        those slices a cross-device read. At 3.2 GB replicating is the cheaper
        answer, and it removes the padding the sharded layout needed.
        """
        self.R_j_r_fixed = self.sharding.replicate_arr(
            np.asarray(self.sim_init.R_j_r).astype(
                np.dtype(self.compute_real_dtype), copy=False))

        # One value per radial mode, ~7 MB and ~14 MB - small enough that
        # keeping them as device arrays costs nothing.
        self.l_values = jnp.asarray(self.sim_init.l)
        self.eigen_energies = jnp.asarray(self.sim_init.eigen_energies)

    def compute_diagonal_rho_expansion(self):
        """
        The time-averaged ("static") density profile rho_static(r).

            rho_static(r) = total_mass * sum_j  weight_j * |R_j(r)|^2
            weight_j      = |a_j|^2 * (2 l_j + 1) / (4 pi)

        This is the background the particles' circular-velocity initial
        conditions are built from, and the baseline the ramp schedule
        blends away from at the start of the simulation.

        |a_j|^2 is read straight off aj_2, which is what it is. The old code
        recovered the same numbers by squaring all 713 million per-k-mode
        amplitudes and scattering them back down onto the radial-mode axis -
        an exact round trip, since every k-mode of j carries the same |a_j|,
        but one that cost a full-length temporary. Reading aj_2 is also
        slightly more accurate: the round trip went through the complex64
        amplitudes, so it returned float32-rounded values.
        """
        total_mass = self.sim_init.total_mass
        l_values = self.l_values

        amplitude_sq_per_radial_mode = jnp.asarray(self.sim_init.aj_2, dtype=jnp.float64)

        weight_j = (amplitude_sq_per_radial_mode
                    * (2.0 * l_values.astype(jnp.float64) + 1.0) / (4.0 * jnp.pi))
        self.weight_j = weight_j

        # Squaring R_j_r whole means a full (n_radii, n_radial_modes) float32
        # temporary plus its float64 widening - 3.2 + 6.4 GB at m22 = 100.
        # Walk the radial grid in chunks instead. Each output radius is an
        # independent dot over j, so slicing rows changes nothing about what is
        # summed or in what order.
        rho_static_r = jnp.concatenate([
            (jnp.abs(self.R_j_r_fixed[r_start:r_start + self.r_chunk_size]) ** 2)
            .astype(jnp.float64) @ weight_j
            for r_start in range(0, self.R_j_r_fixed.shape[0], self.r_chunk_size)
        ])
        rho_static_r = total_mass * rho_static_r

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

    def phase_for_step(self, time_step):
        """
        e^{-i E_j t}: how far each radial mode's wavefunction has rotated in
        phase by this timestep. Stays in complex128 - small phase errors here
        compound badly over a long simulation.

        frozen pins t at dt, so the wavefunction - and hence rho - is the same
        at every step.
        """
        dt = self.sim_init.dt
        t = dt if self.frozen else time_step * dt
        return jnp.exp(-1j * self.eigen_energies * t)

    def build_rho_lms_for_timestep(self, time_step):
        """
        Builds rho_lm(r) at this timestep:

            rho = (1 - ramp_frac) * rho_static + ramp_frac * rho_full(t)
        """
        phase = self.phase_for_step(time_step)
        ramp_frac = self.ramp_frac_for_step(time_step)

        # Stashed so Phi_lm_Builder / Acceleration_Calculator can read the
        # same phase and ramp fraction for every force sub-step of this macro
        # timestep, without recomputing them.
        self.current_phase = phase
        self.current_ramp_frac = jnp.float64(ramp_frac)

        return MSS.build_sphht_rho_lms_jit(
            self.R_j_r_fixed, phase.astype(self.compute_dtype),
            self.amplitudes,
            self.sim_init.lm_pairs, self.sim_init.total_mass,
            jnp.asarray(ramp_frac, dtype=self.compute_real_dtype),
            self.rho_static_r_l00, self.sht_precomputes,
            groups=self.l_groups,
            L_out=int(self.sim_init.L_max_out),
            r_chunk=self.r_chunk_size,
            out_sharding=self.sharding.shard_l,
            sph_sym=self.sph_sym,
        )

    def R_j_at_radii(self, radii, eigenmode_params=None):
        """
        Evaluates every radial basis function R_j at an arbitrary set of
        radii - e.g. particle positions, which move every timestep - rather
        than the fixed background grid prepare_R_j_r used.

        eigenmode_params : the spline-table pytree to evaluate, or None to
        read it off self.sim_init.

        Pass it. Reading it off self inside a jitted caller closes over it,
        and a closed-over jax.Array lowers as a module *constant* rather than
        an argument. This library is 20.3 GB at m22 = 100, and constants are
        materialised on every device when the executable loads - which is what
        "generated code 26826.8 MiB" was in job 661xxx. Passed as an argument
        it stays an argument.
        """
        if eigenmode_params is None:
            eigenmode_params = self.sim_init.radial_eigenmode_params

        R_j_at_radii = self._eval_library(radii, eigenmode_params)
        R_j_at_radii = R_j_at_radii.astype(self.compute_real_dtype)

        # The eigenmode params may be sharded on the radial-mode axis, so this
        # comes back sharded on its column axis. psi_lm_at_rows slices that
        # axis per l, so replicate here - it is only (n_particles,
        # n_radial_modes), and n_particles is order 100.
        return self.sharding.replicate_arr(R_j_at_radii)

    def rho_lm_at_particles_diagonal_only(self, R_j_at_particles):
        """
        rho_lm evaluated at a set of particle radii, using only the static
        (diagonal) density. Because that density is spherically symmetric,
        only the (l=0, m=0) coefficient is nonzero, so no spherical
        harmonic transform is needed - just a dot product against weight_j.
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

    def rho_lm_at_particles(self, R_j_at_particles, phase_in_compute_dtype,
                            amplitudes=None):
        """
        rho_lm evaluated at a set of particle radii, using the full
        time-dependent density. The caller blends this against
        rho_lm_at_particles_diagonal_only's result using the ramp fraction,
        the same way build_rho_lms_for_timestep does for the background grid.

        amplitudes : the per-k-mode amplitudes, or None to read them off self.

        Pass them, for the same reason R_j_at_radii asks for eigenmode_params:
        a closed-over jax.Array lowers into the HLO module as a constant
        rather than an argument. This array is 5.7 GB at m22 = 100, which is
        past protobuf's hard 2 GB message limit - that is where jobs
        661286/661288/661290/661291 segfaulted, inside backend_compile_and_load
        and before XLA logged a single pass on the module.
        """
        if amplitudes is None:
            amplitudes = self.amplitudes

        # sht_precomputes is read off self rather than passed, unlike
        # `amplitudes`. This one is ~49 MB, so lowering it into the enclosing
        # module as a constant is harmless - the rule exists for the multi-GB
        # arrays, which blow past protobuf's 2 GB message limit.
        return MSS.compute_rho_lm_at_particles_sphht_jit(
            R_j_at_particles, phase_in_compute_dtype, amplitudes,
            self.sim_init.lm_pairs, self.sim_init.total_mass,
            self.sht_precomputes,
            groups=self.l_groups,
            L_out=int(self.sim_init.L_max_out),
            p_chunk=self.particle_chunk_size,
            sph_sym=self.sph_sym,
        )
