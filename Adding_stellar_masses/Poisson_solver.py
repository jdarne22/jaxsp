
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)

import functools

import sys
sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses')


import jax.numpy as jnp

import Stellar_sim_funcs as SSF

import importlib
importlib.reload(SSF)


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

    dphi_chunks, phi_chunks = jax.lax.map(chunk_compute, (chunk_idx_arr, pairs_chunked))


    dphi = dphi_chunks.reshape(pad_to)[:n_modes]
    phi  = phi_chunks.reshape(pad_to)[:n_modes]
 
    return dphi, phi

