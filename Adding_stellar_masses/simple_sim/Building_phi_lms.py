"""
Phi_lm_Builder solves for the potential phi_lm(r) sourced by rho_lm(r).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import s2fft
import functools


def _affine_prefix_scan(weights, addends, axis=1):
    """
    Solves the linear recurrence  x[i+1] = weights[i] * x[i] + addends[i],
    with x[0] = 0, and returns x[1] .. x[n] - one value per interval.

    Each step is an affine map  x -> w*x + b, and composing two affine maps
    gives another affine map, so the recurrence is an associative scan
    rather than a sequential one: O(log n) depth on the GPU instead of the
    O(n) sequential dependency chain a `lax.scan` would produce.

    Composing "apply `earlier`, then apply `later`" gives
        x -> w_later * (w_earlier * x + b_earlier) + b_later
           = (w_earlier * w_later) * x + (w_later * b_earlier + b_later)
    which is what `compose` implements.

    Only the addend component of the result is returned: with x[0] = 0 the
    accumulated weight multiplies nothing, so x[i+1] is exactly the
    accumulated addend. (That accumulated weight is a running product of
    factors <= 1, so it underflows to zero at high l - harmless, since it
    is discarded.)
    """
    def compose(earlier, later):
        weight_earlier, addend_earlier = earlier
        weight_later, addend_later = later
        return (weight_earlier * weight_later,
                weight_later * addend_earlier + addend_later)

    _, running = jax.lax.associative_scan(compose, (weights, addends), axis=axis)
    return running


class Phi_lm_Builder:

    def __init__(self, sim_init, rho_builder, l_band_size, plot=False,
                 use_merged_solver=True, chunk_batch_size=None,
                 particle_batch_size=None):

        self.sim_init = sim_init
        self.rho_builder = rho_builder

        # Which Poisson solver Acceleration_Calculator picks up:
        #   True  -> calc_rho_lm_at_parts_and_call_insert_merged
        #            all particles solved from one shared radial grid
        #   False -> calc_rho_lm_at_parts_and_call_insert
        #            original: one particle inserted and integrated at a time
        # The two are physically equivalent but not bit-identical - the
        # merged grid carries n_particles extra quadrature knots.
        self.use_merged_solver = bool(use_merged_solver)

        # How many (l, m) chunks the merged solver batches per lax.map step.
        # None -> l_band_size, matching the per-particle solver. Lower it if
        # the merged solve runs out of memory; see the note at the lax.map
        # call for what one batch slot costs.
        self.chunk_batch_size = chunk_batch_size

        # How many particles are solved per pass. The whole per-particle
        # working set scales with this: rho_lm_at_particles alone is
        # (n_particles, L_max_out, 2*L_max_out - 1), 12.7 GB at 1000
        # particles and L_max_out = 892, and three of them are live while the
        # static and full densities are blended. Batching bounds all of that
        # at the batch size instead of the particle count.
        #
        # None = one pass over every particle, i.e. exactly the behaviour
        # before this knob existed. Note the trade: the merged solver inserts
        # each pass's particle radii into the shared radial grid as extra
        # quadrature knots, so a batched run integrates on a slightly coarser
        # grid than an unbatched one and the results shift a little. That is
        # the same class of difference as merged-vs-per-particle (see
        # use_merged_solver above), not a correctness break - but it does mean
        # results depend on this value, so keep it fixed across a run and its
        # restarts.
        self.particle_batch_size = (
            None if particle_batch_size is None else int(particle_batch_size))

        # Chunk size for the (l, m) loop inside the Poisson solve - caps
        # its per-particle intermediate at (l_band_size, n_radii + 1)
        # instead of holding all L_max_out modes at once.
        self.l_band_size = int(l_band_size)

        # If True, insert_particle_rholm_and_get_philm plots rho_lm/rho(r)
        # around the inserted particle whenever it runs eagerly (outside
        # a JIT trace) with a fully-ramped-in density.
        self.plot = plot

        # Filled in by initialise().
        self.n_radii = None
        self.all_radial_indices = None
        self.output_lm_pairs = None

    def initialise(self):
        """
        Call once, after sim_init has finished setting up the background
        radial grid and L_max_out.
        """
        self.n_radii = len(self.sim_init.r)
        self.all_radial_indices = jnp.arange(self.n_radii + 1)   # indices 0 .. n_radii

        # Every (l, m) mode the Poisson solve / Y_lm evaluation produces
        # output for, l = 0 .. L_max_out - 1.
        L_max_out = self.sim_init.L_max_out
        lm_pairs = [(l, m) for l in range(L_max_out) for m in range(-l, l + 1)]
        self.output_lm_pairs = jnp.array(lm_pairs)

    def insert_particle_rholm_and_get_philm(self, r_pos_sph, rho_lm_at_particle, rho_lms):
        """
        Inserts one particle's own rho_lm into the background radial grid
        at its exact radius, then solves for phi_lm and dphi_lm/dr there.

        Rather than splicing the particle into a new (n_radii + 1, L,
        2L-1) copy of rho_lms, this gathers from the unpadded rho_lms with
        an index shift: every background radius below the particle keeps
        its own row, everything at or above shifts down by one, and the
        particle's own row is substituted at the insertion point. That
        avoids ever materialising a second full-size copy of rho_lms.
        """
        sim_init = self.sim_init
        particle_r = r_pos_sph[0]

        insert_idx = jnp.searchsorted(sim_init.r, particle_r)

        # gather_idx: for i  < insert_idx -> i;  for i  >= insert_idx -> i - 1
        # (clipped so the i == insert_idx slot is safe; that slot is then
        # overridden by particle_r / rho_lm_at_particle below).
        gather_idx = jnp.clip(
            self.all_radial_indices - (self.all_radial_indices > insert_idx).astype(jnp.int32),
            0,
            self.n_radii - 1,
        )

        r_updated = jnp.where(
            self.all_radial_indices == insert_idx,
            particle_r,
            sim_init.r[gather_idx],
        )

        insert_mask = self.all_radial_indices == insert_idx
        mask_int = jnp.arange(self.n_radii) < insert_idx
        mask_ext = jnp.arange(self.n_radii) < (self.n_radii - insert_idx)

        dphi_lm_dr_at_r, phi_lm_at_r = self.compute_phi_lm_and_deriv(
            rho_lms, rho_lm_at_particle, gather_idx, insert_mask,
            r_updated, self.output_lm_pairs,
            mask_int, mask_ext, int(sim_init.L_max_out), sim_init.G, particle_r,
            self.l_band_size,
        )

        if self.plot and not isinstance(particle_r, jax.core.Tracer):
            if self.rho_builder.current_ramp_frac == 1.0:
                self._plot_inserted_rho(particle_r, r_updated, gather_idx, insert_mask, rho_lm_at_particle, rho_lms)

        return dphi_lm_dr_at_r, phi_lm_at_r   # (n_modes,), (n_modes,)




    @staticmethod
    @functools.partial(jax.jit, static_argnums=(8, 11))
    def compute_phi_lm_and_deriv(rho_lms, rho_lm_at_particle, gather_idx, insert_mask,
                                r_updated, output_lm_pairs, mask_int, mask_ext,
                                L_max_out, G, particle_r, l_band_size):

        """Streamed computation of dphi_dr and phi_lm for all (l,m) pairs.

        The (l,m) modes are processed in chunks of size `l_band_size` via
        `jax.lax.map`, so the intermediate `(l_band_size, Nr+1)` slice
        `f_at_lm_band` replaces the naive `(Nmodes, Nr+1)` materialisation —
        AND replaces the old `(Nr+1, L, 2L-1)` pre-built `rho_lm_updated`
        which was the dominant per-particle peak at high m22.

        Per chunk we do ONE bulk gather of `(l_band_size, Nr+1)` from
        `rho_lms` (vs `l_band_size` scalar gathers per chunk if the gather
        were inlined into the inner vmap — that pattern hits XLA's per-op
        launch overhead hard at Nmodes ~ L²).

        The radial integrands are *ratio-folded*: the naive multipole Green's
        function forms `r'^(l+2)` and `r_p^(-l-1)` (and `r'^(1-l)`, `r_p^l`)
        as separate factors. At l ~ O(50+) in Schroedinger units (where r is
        not O(1)) those span hundreds of orders of magnitude and destroy
        float64 precision in the high-l modes — the dominant cause of
        spurious forces / particle ejection at higher m22. Folding the
        particle-radius prefactor into the integrand leaves every factor a
        ratio <= 1, so nothing over/underflows:

            r_p^(-l-1) r'^(l+2)  ->  r' (r'/r_p)^(l+1)   [interior, r' <= r_p]
            r_p^l      r'^(1-l)  ->  r' (r_p/r')^l       [exterior, r' >= r_p]

        The folded interior/exterior integrals are shared between `phi_lm`
        and `dphi_lm_dr` (only the scalar per-l prefactor differs). The
        `(L_max_out, Nr+1)` ratio-power tables are still precomputed ONCE
        per call and gathered per chunk by `l_vals`; because `output_lm_pairs`
        is constructed in l-sorted order, l-coherent chunks let XLA collapse
        the per-l gather toward a single-row broadcast.
        """

        dr     = jnp.diff(r_updated)
        dr_rev = jnp.diff(r_updated[::-1])

        cdtype = rho_lms.dtype  # complex compute dtype (c64 or c128)
        zero_c = jnp.asarray(0.0, dtype=cdtype)

        # ---- Precomputed per-l tables (built once, gathered per chunk) ----
        ell_f = jnp.arange(L_max_out, dtype=r_updated.dtype)  # 0..L_max_out-1
        rp    = particle_r

        # Ratio-folded radial tables. `ratio_int` (`r'/r_p`) is <= 1 in the
        # interior region and `ratio_ext` (`r_p/r'`) is <= 1 in the exterior
        # region; out-of-region entries are clipped to 1 (harmless — they are
        # masked out of the integrals below) so the powers never overflow.
        # Real arrays — promoted to `cdtype` only when multiplied with f_at_lm.
        ratio_int = jnp.clip(r_updated / rp, 0.0, 1.0)                                    # (Nr+1,)
        ratio_ext = jnp.clip(rp / r_updated, 0.0, 1.0)                                    # (Nr+1,)
        r_fold_int = r_updated[None, :] * jnp.power(ratio_int[None, :], (ell_f + 1.0)[:, None])  # (L_max, Nr+1)
        r_fold_ext = r_updated[None, :] * jnp.power(ratio_ext[None, :], ell_f[:, None])          # (L_max, Nr+1)

        # Per-l scalar prefactors. With the r_p prefactor folded into the
        # integrand, the folded interior/exterior integrals are shared by
        # `phi_lm` and `dphi_lm_dr`; only these per-l scalars differ.
        prefix_arr     = -4.0 * jnp.pi * G / (2.0 * ell_f + 1.0)   # (L_max,)
        dphi_ext_scale = ell_f / rp                                # l / r_p
        dphi_int_scale = (ell_f + 1.0) / rp                        # (l+1) / r_p


        n_modes = output_lm_pairs.shape[0]
        n_chunks = (n_modes + l_band_size - 1) // l_band_size
        pad_to = n_chunks * l_band_size
        pad_amount = pad_to - n_modes

        if pad_amount > 0:
            pairs_padded = jnp.concatenate(
                [output_lm_pairs,
                jnp.zeros((pad_amount, output_lm_pairs.shape[1]), dtype=output_lm_pairs.dtype)],
                axis=0,
            )
        else:
            pairs_padded = output_lm_pairs

        pairs_chunked = pairs_padded.reshape(n_chunks, l_band_size, 2)

        chunk_idx_arr = jnp.arange(n_chunks, dtype=jnp.int32)


        def chunk_compute(args):
            i, chunk = args


            # chunk: (l_band_size, 2)
            l_vals = chunk[:, 0]                             # (l_band_size,)
            m_inds = chunk[:, 1] + L_max_out - 1              # (l_band_size,)

            # ONE bulk gather per chunk: pairs of axes (1, 2) at once →
            # `(Nr, l_band_size)`; then gather axis 0 with `gather_idx` →
            # `(Nr+1, l_band_size)`. XLA compiles this to two gather kernels
            # per chunk regardless of l_band_size, vs one tiny gather per
            # vmap'd mode in the previous (slow) version.

            rho_band     = rho_lms[:, l_vals, m_inds]         # (Nr, l_band_size)
            f_orig_band  = rho_band[gather_idx].T             # (l_band_size, Nr+1)
            f_ins_band   = rho_lm_at_particle[l_vals, m_inds]  # (l_band_size,)
            f_at_lm_band = jnp.where(insert_mask[None, :],
                                        f_ins_band[:, None],
                                        f_orig_band)            # (l_band_size, Nr+1)


            # Gather precomputed ratio-folded radial tables — single XLA gather
            # per chunk along axis 0. Collapses to a broadcast for l-coherent
            # chunks.
            r_fold_ext_band = r_fold_ext[l_vals]              # (l_band_size, Nr+1)
            r_fold_int_band = r_fold_int[l_vals]

            integrand_ext = r_fold_ext_band * f_at_lm_band    # (l_band_size, Nr+1)
            integrand_int = r_fold_int_band * f_at_lm_band


            # Masked trapezoidal sums along the radial axis.
            avg_int = 0.5 * (integrand_int[:, 1:] + integrand_int[:, :-1])
            integral_int = jnp.sum(
                jnp.where(mask_int[None, :], avg_int * dr[None, :], zero_c),
                axis=1,
            )

            integrand_ext_rev = integrand_ext[:, ::-1]
            avg_ext = 0.5 * (integrand_ext_rev[:, 1:] + integrand_ext_rev[:, :-1])
            integral_ext = -jnp.sum(
                jnp.where(mask_ext[None, :], avg_ext * dr_rev[None, :], zero_c),
                axis=1,
            )

            # Per-l scalar prefactors gathered for this chunk. The folded
            # integrals already carry the r_p prefactor, so `phi` is just
            # their sum and `dphi/dr` weights them by l/r_p and (l+1)/r_p.
            pref      = prefix_arr[l_vals]                    # (l_band_size,)
            ext_scale = dphi_ext_scale[l_vals]
            int_scale = dphi_int_scale[l_vals]

            dphi_band = pref * (ext_scale * integral_ext - int_scale * integral_int)
            phi_band  = pref * (integral_ext + integral_int)


            # `jax.debug.print` is the runtime-safe print inside jit/scan/map.
            # `ordered=True` serialises execution so the prints arrive in
            # chunk order — useful for tracking, *will* slow the kernel.
            # Switch to `ordered=False` (async) or remove for production runs.
            #jax.debug.print("phi_lm chunk {i}/{n}", i=i, n=n_chunks, ordered=False)

            return dphi_band, phi_band

        # Batch up to l_band_size chunks together per vmap'd step instead of
        # looping over chunks one at a time. Capping the batch at l_band_size
        # (rather than n_chunks) keeps the per-step working set bounded even
        # as m22 grows n_chunks — n_chunks scales with L_max_out^2, but the
        # cap doesn't, so this can't reintroduce the full (Nmodes, Nr+1)
        # materialisation the chunking was built to avoid.
        dphi_chunks, phi_chunks = jax.lax.map(
            chunk_compute, (chunk_idx_arr, pairs_chunked),
            batch_size=min(n_chunks, l_band_size),
        )


        dphi = dphi_chunks.reshape(pad_to)[:n_modes]
        phi  = phi_chunks.reshape(pad_to)[:n_modes]
    
        return dphi, phi



    def _solve_in_particle_batches(self, positions_sph, solve_batch):
        """
        Runs `solve_batch` over the particles `particle_batch_size` at a time
        and stitches the (n_particles, n_modes) results back together.

        Everything on the per-particle path scales linearly with how many
        particles are in flight at once - the s2fft working set inside
        rho_lm_at_particles, the (n_particles, L_max_out, 2*L_max_out - 1)
        densities, R_j_at_particles at (n_particles, n_radial_modes) - and
        none of it is sharded. Batching bounds all of it at once.

        particle_batch_size = None runs a single batch, which is exactly what
        this path did before the knob existed.

        Whatever solve_batch returns is stitched back along the particle axis,
        shape-agnostically. That matters for per_batch_reduce (see the
        solvers): a solve_batch that has already contracted the mode axis away
        returns (batch,) per output and stitches to (n_particles,), and the
        full-width (batch, n_modes) arrays never exist at (n_particles,
        n_modes). Those are 11.86 GiB each at L_max_out = 892 and 1000
        particles, and holding both is what left no room on the device in job
        661385.
        """
        n_particles, n_columns = positions_sph.shape

        batch = self.particle_batch_size
        batch = n_particles if not batch else max(1, min(int(batch), n_particles))

        n_batches = (n_particles + batch - 1) // batch
        if n_batches == 1:
            return solve_batch(positions_sph)

        n_particles_padded = n_batches * batch
        pad_amount = n_particles_padded - n_particles

        # Pad by repeating the last particle rather than with zeros: a zero
        # radius would divide by zero in the solvers' 1/r factors, and would
        # put a spurious quadrature knot at the origin of the merged grid. A
        # duplicated radius only adds a zero-width interval, which both
        # solvers already tolerate.
        if pad_amount > 0:
            positions_sph = jnp.concatenate(
                [positions_sph,
                 jnp.repeat(positions_sph[-1:], pad_amount, axis=0)], axis=0)

        # Default batch_size (None) makes this a scan, so one batch's working
        # set is live at a time - the point of the exercise.
        results = jax.lax.map(
            solve_batch, positions_sph.reshape(n_batches, batch, n_columns))

        return jax.tree.map(
            lambda a: a.reshape((n_particles_padded,) + a.shape[2:])[:n_particles],
            results,
        )

    def calc_rho_lm_at_parts_and_call_insert(self, positions_sph, current_phase, rho_lms, ramp_frac,
                                             amplitudes=None, eigenmode_params=None,
                                             per_batch_reduce=None):
        """
        JIT-compilable: evaluates rho_lm at every particle's exact radius
        (blended between the static and full time-dependent density), then
        inserts each particle and solves for phi_lm in one vmapped call.

        Particles are processed `particle_batch_size` at a time. Each one is
        inserted into the background grid on its own here, so the batching
        changes nothing about the result - only how many are in flight.

        per_batch_reduce, if given, is called as
        (dphi_batch, phi_batch, positions_batch) inside the batch loop and its
        result is what gets stitched together instead. See the note on the
        merged solver's copy of this argument.

        eigenmode_params is forwarded to R_j_at_radii, and a jitted caller
        should pass it - see that method for what closing over it costs.
        """
        rho_builder = self.rho_builder
        phase_c = current_phase.astype(rho_builder.compute_dtype)

        def solve_batch(positions_batch):
            particle_radii = positions_batch[:, 0]
            R_j_at_particles = rho_builder.R_j_at_radii(particle_radii, eigenmode_params)

            # Compute rho_lm at every particle's radius OUTSIDE the vmap over
            # particles, so the expensive sparse matmul + scatter happen once
            # per batch, not once per particle.
            rho_lm_at_particles_full = rho_builder.rho_lm_at_particles(
                R_j_at_particles, phase_c, amplitudes)
            rho_lm_at_particles_static = rho_builder.rho_lm_at_particles_diagonal_only(R_j_at_particles)

            rho_lm_at_particles = (
                rho_lm_at_particles_static
                + ramp_frac * (rho_lm_at_particles_full - rho_lm_at_particles_static)
            )
            # rho_lm_at_particles : (batch, L_max_out, 2*L_max_out - 1)

            # Vmap over the batch's particles rather than looping over them
            # one at a time.
            dphi_batch, phi_batch = jax.lax.map(
                lambda inp: self.insert_particle_rholm_and_get_philm(inp[0], inp[1], rho_lms),
                (positions_batch, rho_lm_at_particles),
                batch_size=positions_batch.shape[0],
            )
            # each: (batch, n_modes)

            if per_batch_reduce is None:
                return dphi_batch, phi_batch

            return per_batch_reduce(dphi_batch, phi_batch, positions_batch)

        return self._solve_in_particle_batches(positions_sph, solve_batch)

    # ------------------------------------------------------------------
    # Merged-grid Poisson solve (experimental alternative to the two
    # methods above). Same physics, but every particle is solved from one
    # shared pass over a single merged radial grid.
    # ------------------------------------------------------------------

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(5, 7, 8))
    def compute_phi_lm_and_deriv_merged(rho_lms, rho_lm_at_particles,
                                        background_radii, particle_radii,
                                        output_lm_pairs, L_max_out, G,
                                        l_band_size, chunk_batch_size=None):
        """
        Solves for phi_lm and dphi_lm/dr at EVERY particle in one pass.

        The per-particle version inserts one particle into the background
        grid, integrates the whole radial range, and throws the
        intermediate results away - so with n_particles particles the same
        radial integral is redone n_particles times. Here every particle
        radius is inserted into the grid at once (length n_radii +
        n_particles), the interior/exterior integrals are accumulated once
        along that shared grid, and each particle simply reads its own
        value out at its own insertion index. Work drops from
        n_particles * n_radii to (n_radii + n_particles).

        This is legitimate because the particles are massless test
        particles: an inserted rho_lm value is just the true global
        rho_lm evaluated at an extra radius, not a particle-specific
        source. Every particle sees the same field, only sampled on a
        finer grid.

        Why the accumulation is not a plain `cumsum`
        --------------------------------------------
        The per-particle solver keeps the multipole integrand numerically
        safe by folding the particle radius into it, so every factor is a
        ratio <= 1:

            r_p^(-l-1) r'^(l+2)  ->  r' (r'/r_p)^(l+1)   [interior]
            r_p^l      r'^(1-l)  ->  r' (r_p/r')^l       [exterior]

        That fold is tied to *which* particle is being evaluated, so it
        cannot be hoisted out of a shared cumsum. Folding against a single
        shared reference radius instead does not work: at l ~ O(500) the
        integrand for an inner particle underflows to exactly zero long
        before the compensating (r_ref/r_p)^(l+1) factor - which itself
        overflows - could be applied.

        The fix is to renormalise the running integral to the *current*
        radius at every interval, which turns the accumulation into a
        damped (affine) scan rather than a plain sum. Writing

            A_i = r_i^(-l-1) * integral(0 -> r_i) of r'^(l+2) rho dr'

        the trapezoidal step is

            A_{i+1} = w_i A_i + (dr_i/2) (r_i w_i f_i + r_{i+1} f_{i+1})
            w_i     = (r_i / r_{i+1})^(l+1)   <= 1

        and every factor is again a ratio <= 1, for every particle
        simultaneously. When w_i underflows to zero at high l that is
        physically correct: mass far inside r_i genuinely cannot influence
        that multipole. The exterior integral is the mirror image,
        accumulated inward with w_i = (r_i/r_{i+1})^l.

        A_i is already in physical units at r_i, so - unlike a
        shared-reference fold - no correction factor is applied afterwards.

        Returns
        -------
        (n_particles, n_modes) complex arrays (dphi_lm_dr, phi_lm), the
        same layout `calc_rho_lm_at_parts_and_call_insert` returns.
        """
        n_radii = background_radii.shape[0]
        n_particles = particle_radii.shape[0]

        # ---- merged radial grid --------------------------------------
        # Background radii and particle radii sorted into one grid, with a
        # record of where every merged slot came from. `argsort` is stable,
        # so a particle sitting exactly on a background radius resolves
        # deterministically (background first).
        radii_concatenated = jnp.concatenate([background_radii, particle_radii])
        sort_order = jnp.argsort(radii_concatenated)
        r_merged = radii_concatenated[sort_order]

        slot_is_particle = sort_order >= n_radii
        background_source = jnp.clip(sort_order, 0, n_radii - 1)
        particle_source = jnp.clip(sort_order - n_radii, 0, n_particles - 1)

        # Inverse permutation, restricted to the particle block: where each
        # particle ended up in the merged grid. This is the read-out index.
        particle_slot = jnp.argsort(sort_order)[n_radii:]     # (n_particles,)

        cdtype = rho_lms.dtype
        real_dtype = r_merged.dtype
        # The real radial tables promote the complex density (matching the
        # per-particle solver's behaviour, which promotes c64 -> c128 the
        # same way).
        accumulate_dtype = jnp.result_type(real_dtype, cdtype)

        # ---- per-interval geometry -----------------------------------
        r_left = r_merged[:-1]
        r_right = r_merged[1:]
        half_dr = 0.5 * (r_right - r_left)

        # Ratio of adjacent radii, <= 1 by construction since r_merged is
        # sorted ascending. Equal radii give a ratio of 1 and dr of 0, so
        # duplicate points contribute nothing - harmless.
        step_ratio = jnp.clip(r_left / r_right, 0.0, 1.0)

        ell_f = jnp.arange(L_max_out, dtype=real_dtype)

        # Damping weights per (l, interval). Built once here and gathered
        # per (l, m) chunk below - this table is only
        # (L_max_out, n_radii + n_particles) reals, a few MB.
        interior_weight_table = jnp.power(step_ratio[None, :], (ell_f + 1.0)[:, None])
        exterior_weight_table = jnp.power(step_ratio[None, :], ell_f[:, None])

        prefix_arr = -4.0 * jnp.pi * G / (2.0 * ell_f + 1.0)      # (L_max_out,)

        # ---- (l, m) chunking (unchanged from the per-particle solver) --
        n_modes = output_lm_pairs.shape[0]
        n_chunks = (n_modes + l_band_size - 1) // l_band_size
        pad_to = n_chunks * l_band_size
        pad_amount = pad_to - n_modes

        if pad_amount > 0:
            pairs_padded = jnp.concatenate(
                [output_lm_pairs,
                 jnp.zeros((pad_amount, output_lm_pairs.shape[1]), dtype=output_lm_pairs.dtype)],
                axis=0,
            )
        else:
            pairs_padded = output_lm_pairs

        pairs_chunked = pairs_padded.reshape(n_chunks, l_band_size, 2)

        zero_column = jnp.zeros((l_band_size, 1), dtype=accumulate_dtype)

        def chunk_compute(chunk):
            # chunk: (l_band_size, 2)
            l_vals = chunk[:, 0]
            m_inds = chunk[:, 1] + L_max_out - 1

            # rho_lm on the merged grid: each slot takes either its
            # background row or its particle row. Two bulk gathers, then a
            # select - the merged (n_radii + n_particles, L, 2L-1) array is
            # never materialised.
            rho_background_band = rho_lms[:, l_vals, m_inds]                # (n_radii, l_band)
            rho_particle_band = rho_lm_at_particles[:, l_vals, m_inds]      # (n_particles, l_band)

            f = jnp.where(
                slot_is_particle[None, :],
                rho_particle_band[particle_source].T,
                rho_background_band[background_source].T,
            )                                              # (l_band, n_radii + n_particles)

            interior_weight = interior_weight_table[l_vals]     # (l_band, n_intervals)
            exterior_weight = exterior_weight_table[l_vals]

            # Interior: accumulate outward from r_min. Every factor is a
            # ratio <= 1 (see docstring), so nothing over/underflows.
            interior_addend = half_dr[None, :] * (
                r_left[None, :] * interior_weight * f[:, :-1]
                + r_right[None, :] * f[:, 1:]
            )
            interior_tail = _affine_prefix_scan(interior_weight, interior_addend)
            # Prepend A_0 = 0: the integral from 0 to the innermost radius,
            # which the per-particle solver also neglects.
            interior_integral = jnp.concatenate([zero_column, interior_tail], axis=1)

            # Exterior: same recurrence accumulated inward. Reverse both
            # arrays, run the same forward scan, then reverse the result -
            # cheaper to reason about than a reversed scan operator.
            exterior_addend = half_dr[None, :] * (
                r_left[None, :] * f[:, :-1]
                + r_right[None, :] * exterior_weight * f[:, 1:]
            )
            exterior_tail = _affine_prefix_scan(
                exterior_weight[:, ::-1], exterior_addend[:, ::-1])
            exterior_integral = jnp.concatenate(
                [zero_column, exterior_tail], axis=1)[:, ::-1]

            # Each particle reads its own value out of the shared integrals.
            interior_at_particles = interior_integral[:, particle_slot]   # (l_band, n_particles)
            exterior_at_particles = exterior_integral[:, particle_slot]

            prefix = prefix_arr[l_vals][:, None]                  # (l_band, 1)
            l_column = l_vals.astype(real_dtype)[:, None]
            inverse_particle_r = (1.0 / particle_radii)[None, :]  # (1, n_particles)

            dphi_band = prefix * inverse_particle_r * (
                l_column * exterior_at_particles
                - (l_column + 1.0) * interior_at_particles
            )
            phi_band = prefix * (interior_at_particles + exterior_at_particles)

            return dphi_band, phi_band

        # n_chunks scales as L_max_out^2 / l_band_size - ~5000 at
        # L_max_out=481, l_band_size=46 - and each chunk's kernels are small,
        # so mapping one chunk at a time is launch-latency bound on the GPU.
        # Batching chunks amortises that, at the cost of holding
        # chunk_batch_size working sets at once.
        #
        # Unlike the per-particle solver, a chunk here also carries an
        # n_particles-wide output, so the batch costs
        #   chunk_batch_size * (l_band_size * (n_radii + n_particles) * ~6
        #                       + l_band_size * n_particles * 2) * itemsize
        # Default matches the per-particle solver; drop it if n_particles is
        # large enough that the output term stops being negligible.
        if chunk_batch_size is None:
            chunk_batch_size = l_band_size
        batch_size = max(1, min(int(chunk_batch_size), n_chunks))

        dphi_chunks, phi_chunks = jax.lax.map(
            chunk_compute, pairs_chunked, batch_size=batch_size)

        dphi = dphi_chunks.reshape(pad_to, n_particles)[:n_modes]
        phi = phi_chunks.reshape(pad_to, n_particles)[:n_modes]

        # (n_modes, n_particles) -> (n_particles, n_modes), matching the
        # per-particle solver's return layout.
        return dphi.T, phi.T

    def calc_rho_lm_at_parts_and_call_insert_merged(self, positions_sph, current_phase,
                                                    rho_lms, ramp_frac, amplitudes=None,
                                                    eigenmode_params=None,
                                                    per_batch_reduce=None):
        """
        Drop-in alternative to `calc_rho_lm_at_parts_and_call_insert` that
        solves every particle from one merged radial grid.

        The rho_lm-at-particles half is identical - it was already batched
        across particles. Only the Poisson solve changes: instead of
        mapping a single-particle insert over the particle axis, all the
        particle radii go into one grid and
        `compute_phi_lm_and_deriv_merged` handles them together.

        Results will not be bit-identical to the per-particle path: the
        merged grid carries n_particles extra quadrature knots, so the
        trapezoidal sums differ slightly. It should be the more accurate
        of the two.

        Particles are processed `particle_batch_size` at a time. Here that is
        not free: only the current batch's radii go into the merged grid, so a
        batched run integrates on a coarser grid than an unbatched one. The
        difference is the same kind as merged-vs-per-particle above (a
        quadrature refinement, not a different physical field), but it does
        mean the batch size is part of the run's configuration.

        per_batch_reduce
        ----------------
        If given, called as (dphi_batch, phi_batch, positions_batch) at the end
        of each batch, and its result is stitched along the particle axis
        instead of the raw (batch, n_modes) arrays.

        Acceleration_Calculator passes the angular contraction here, so the
        (n_particles, n_modes) arrays are never built: they are 11.86 GiB each
        in complex128 at L_max_out = 892 and 1000 particles, they were only
        ever an intermediate on the way to (n_particles,) accelerations, and
        holding both is what left the device with no room to load a kernel in
        job 661385. It also means one knob - particle_batch_size - now sizes
        both halves of the calculation, which is the intent.

        eigenmode_params
        ----------------
        Forwarded to R_j_at_radii. A jitted caller should pass it rather than
        let it be read off self: it is the single largest array in the run
        (28.13 GB at m22 = 100) and closing over it bakes the whole thing
        into the executable as a constant. See R_j_at_radii.
        """
        rho_builder = self.rho_builder
        phase_c = current_phase.astype(rho_builder.compute_dtype)

        def solve_batch(positions_batch):
            particle_radii = positions_batch[:, 0]
            R_j_at_particles = rho_builder.R_j_at_radii(particle_radii, eigenmode_params)

            rho_lm_at_particles_full = rho_builder.rho_lm_at_particles(
                R_j_at_particles, phase_c, amplitudes)
            rho_lm_at_particles_static = rho_builder.rho_lm_at_particles_diagonal_only(R_j_at_particles)

            rho_lm_at_particles = (
                rho_lm_at_particles_static
                + ramp_frac * (rho_lm_at_particles_full - rho_lm_at_particles_static)
            )

            dphi_batch, phi_batch = self.compute_phi_lm_and_deriv_merged(
                rho_lms, rho_lm_at_particles,
                self.sim_init.r, particle_radii,
                self.output_lm_pairs, int(self.sim_init.L_max_out),
                self.sim_init.G, self.l_band_size,
                self.chunk_batch_size,
            )

            if per_batch_reduce is None:
                return dphi_batch, phi_batch

            return per_batch_reduce(dphi_batch, phi_batch, positions_batch)

        return self._solve_in_particle_batches(positions_sph, solve_batch)
