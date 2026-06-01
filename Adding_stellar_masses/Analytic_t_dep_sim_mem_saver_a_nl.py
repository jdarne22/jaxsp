
import functools
import os
import pickle
from time import time
import matplotlib.pyplot as plt
import rebound
import s2fft
from scipy.special import sph_harm_y
from collections import defaultdict

import sys
sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses')

import jaxsp as jsp

import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxsp.constants import GN

import PhD_year_1.jaxsp.Adding_stellar_masses.gaunt_funcs as gf
import Stellar_sim_funcs as SSF

import importlib
importlib.reload(SSF)
importlib.reload(gf)

# k mode is defined as a unique nlm pair
# j mode is defined as a unique nl pair (multiple j modes can have the same l, but different n)
# u is index of unique lm pair

def precompute_lm_pairs(l):
    '''Precompute (l, m) bookkeeping for spherical harmonics.
    '''

    l_for_kmode = []   
    m_for_kmode = []      
    ind_for_jmode_over_all_k = []  
    lm_pairs_dict = defaultdict(int)

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            l_for_kmode.append(ell)
            m_for_kmode.append(m)
            ind_for_jmode_over_all_k.append(j_idx)
            lm_pairs_dict[(ell, m)] += 1

    lm_pairs_list = list(lm_pairs_dict.keys())      
    lm_pairs = jnp.array(lm_pairs_list)  

    lm_pair_to_idx = {pair: i for i, pair in enumerate(lm_pairs_list)} # {(l, m): idx in unique lm pairs}

    lm_pairs_idx_for_kmode = jnp.array(
        [lm_pair_to_idx[(ell, m)] for ell, m in zip(l_for_kmode, m_for_kmode)], dtype=jnp.int32) # way of going from each lm for k to its index in lm pairs

    L = int(max(l)) + 1
    L_max_out = 2 * L - 1
    n_theta = L_max_out
    n_phi = 2 * L_max_out - 1

    i = np.arange(n_theta)
    theta_np = (np.pi * (2 * i + 1)) / (2 * L_max_out - 1)
    j = np.arange(n_phi)
    phi_np = (2 * np.pi * j) / (2 * L_max_out - 1)

    return (jnp.array(ind_for_jmode_over_all_k), lm_pairs,
            jnp.array(l_for_kmode), jnp.array(m_for_kmode),
            jnp.asarray(theta_np), jnp.asarray(phi_np),
            lm_pairs_idx_for_kmode)



def _sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R, N_unique, k_batch):
    """Sparse equivalent of `(a_u_j_dense @ (R * phase[None, :]).T)`.

    The dense `a_u_j` is a one-hot scatter: each k-mode (n, l, m) writes
    `aj[k]` to row `lm_idx[k]`, column `parent_j[k]`. So `a_u_j` has
    `Nmodes_k = sum_j (2 l_j + 1)` nonzeros in `N_unique * Nj ≈ m22**5`
    slots — typically <1% dense. We never materialise it; instead we
    stream over k-modes in batches of `k_batch`, gathering the needed
    columns of `R` and scatter-adding into the output `S`.

    Memory: `O(M * k_batch)` per batch instead of `O(N_unique * Nj)` for
    the dense matrix. At m22=10 this swaps a 13 GB standing array for a
    ~100 MB per-batch transient.

    Inputs
    ------
    aj         : (Nmodes_k,) complex amplitudes per k-mode
    parent_j   : (Nmodes_k,) int — j-index per k-mode
    lm_idx     : (Nmodes_k,) int — unique-(l,m) index per k-mode
    phase_c    : (Nj,) complex — e^{-i E_j t}, already cast to cdtype
    R          : (M, Nj) — `R_j_r_fixed` (M = Nr) or `R_j_at_parts` (M = Np)
    N_unique   : int (static) — number of distinct (l,m) pairs
    k_batch    : int (static) — k-modes per scan iteration

    Output: `S` of shape (N_unique, M), equal to
            Σ_k δ(u, lm_idx[k]) · aj[k] · phase[parent_j[k]] · R[:, parent_j[k]]
    """
    Nmodes_k = aj.shape[0]
    n_chunks = (Nmodes_k + k_batch - 1) // k_batch
    pad_to = n_chunks * k_batch
    pad_n = pad_to - Nmodes_k

    if pad_n > 0:
        # Dummy entries have aj=0 so their scatter-add contribution is 0.
        aj = jnp.concatenate([aj, jnp.zeros((pad_n,), dtype=aj.dtype)])
        parent_j = jnp.concatenate([parent_j, jnp.zeros((pad_n,), dtype=parent_j.dtype)])
        lm_idx = jnp.concatenate([lm_idx, jnp.zeros((pad_n,), dtype=lm_idx.dtype)])

    aj_chunks = aj.reshape(n_chunks, k_batch)
    pj_chunks = parent_j.reshape(n_chunks, k_batch)
    u_chunks  = lm_idx.reshape(n_chunks, k_batch)

    M = R.shape[0]
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R.dtype)
    S_init = jnp.zeros((N_unique, M), dtype=cdtype)

    def body(S, batched):
        aj_b, pj_b, u_b = batched
        aj_phased = aj_b * phase_c[pj_b]                  # (k_batch,)
        R_cols    = R[:, pj_b]                            # (M, k_batch)
        contrib   = aj_phased[None, :] * R_cols           # (M, k_batch)
        S = S.at[u_b, :].add(contrib.T)                   # scatter-add along u-axis
        return S, None

    S, _ = jax.lax.scan(body, S_init, (aj_chunks, pj_chunks, u_chunks))
    return S


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch", "r_chunk"))
def _build_sphht_rho_rtp_jit(R_j_r_fixed, phase_c, aj, parent_j, lm_idx,
                              lm_pairs, total_mass, L_out, N_unique, k_batch, r_chunk):
    """JIT'd: sparse-matmul → scatter → inverse-SHT → rho_rtp, streamed
    over chunks of `r_chunk` radial bins.

    Without r-chunking, the SHT round-trip transient is
    `(Nr, L_out, 2L_out-1)` complex — 3.7 GB at m22=10 — plus s2fft's
    internal working set, easily blowing past 13 GB. We stream so the
    peak per chunk is `(r_chunk, L_out, 2L_out-1)` complex.
    """
    Nr = R_j_r_fixed.shape[0]
    Nj = R_j_r_fixed.shape[1]
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_r_fixed.dtype)
    rdtype = R_j_r_fixed.dtype

    n_chunks = (Nr + r_chunk - 1) // r_chunk
    pad_to = n_chunks * r_chunk
    pad_n = pad_to - Nr

    if pad_n > 0:
        R_padded = jnp.concatenate(
            [R_j_r_fixed, jnp.zeros((pad_n, Nj), dtype=rdtype)], axis=0)
    else:
        R_padded = R_j_r_fixed

    R_chunks = R_padded.reshape(n_chunks, r_chunk, Nj)

    def chunk_body(R_chunk):
        # R_chunk: (r_chunk, Nj)
        S_chunk = _sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R_chunk,
                                         N_unique, k_batch)                # (N_unique, r_chunk)

        flm_chunk = jnp.zeros((r_chunk, L_out, 2 * L_out - 1), dtype=cdtype)
        flm_chunk = flm_chunk.at[:, lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(S_chunk.T)

        psi_chunk = jax.vmap(
            lambda f: s2fft.inverse(f, L_out, sampling='mw', method='jax')
        )(flm_chunk)

        return total_mass * (jnp.abs(psi_chunk) ** 2)                       # (r_chunk, n_theta, n_phi)

    rho_rtp_chunks = jax.lax.map(chunk_body, R_chunks)                      # (n_chunks, r_chunk, ...)
    # MW sampling: n_theta = L_out, n_phi = 2 L_out - 1.
    rho_rtp_padded = rho_rtp_chunks.reshape(pad_to, L_out, 2 * L_out - 1)
    return rho_rtp_padded[:Nr]


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch", "r_chunk", "out_sharding"))
def _build_sphht_rho_lms_jit(R_j_r_fixed, phase_c, aj, parent_j, lm_idx,
                              lm_pairs, total_mass,
                              ramp_c, rho_static_r_l00,
                              L_out, N_unique, k_batch, r_chunk,
                              out_sharding=None):
    """JIT'd: full SHT round-trip → rho_lms, streamed over `r_chunk`
    radial bins. Per-chunk transient ~ `(r_chunk, L_out, 2L_out-1)`.

    The ramp / static blend is folded in *per chunk* before the write to
    `out`. Equivalent to (outside the JIT):
        rho_lms = ramp_c * rho_lms_full
        rho_lms = rho_lms.at[:, 0, L_out - 1].add((1 - ramp_c) * rho_static_r_l00)
    but without materialising two extra full-size `(Nr, L, 2L-1)` copies.

    Sharding strategy (when `out_sharding` is given, typically
    `NamedSharding(mesh, PartitionSpec(None, 'x', None))`):

    * The output is pre-allocated as a sharded `(pad_to, L_out, 2L_out-1)`
      accumulator and written to via `dynamic_update_slice` inside a
      `lax.scan`. Each device therefore only ever holds its own L-shard
      of the accumulator (~ `pad_to * L_out/N_dev * (2L_out-1)`).
    * Each chunk's `(r_chunk, L_out, 2L_out-1)` is constrained to the same
      L-sharding so the inverse/forward SHT result is sharded on output
      and not gathered before the slice-write.

    Avoiding `jax.lax.map` matters: its stacked output `(n_chunks, r_chunk,
    L_out, 2L_out-1)` would be allocated replicated on each device, with
    the same total size as the final output, defeating the sharding.
    """
    Nr = R_j_r_fixed.shape[0]
    Nj = R_j_r_fixed.shape[1]
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_r_fixed.dtype)
    rdtype = R_j_r_fixed.dtype

    n_chunks = (Nr + r_chunk - 1) // r_chunk
    pad_to = n_chunks * r_chunk
    pad_n = pad_to - Nr

    if pad_n > 0:
        R_padded = jnp.concatenate(
            [R_j_r_fixed, jnp.zeros((pad_n, Nj), dtype=rdtype)], axis=0)
        static_padded = jnp.concatenate(
            [rho_static_r_l00, jnp.zeros((pad_n,), dtype=rho_static_r_l00.dtype)], axis=0)
    else:
        R_padded = R_j_r_fixed
        static_padded = rho_static_r_l00

    def _sht_roundtrip(S_T):
        flm = jnp.zeros((S_T.shape[0], L_out, 2 * L_out - 1), dtype=cdtype)
        flm = flm.at[:, lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(S_T)
        psi = jax.vmap(
            lambda f: s2fft.inverse(f, L_out, sampling='mw', method='jax')
        )(flm)
        rho = total_mass * (jnp.abs(psi) ** 2)
        return jax.vmap(
            lambda r: s2fft.forward(r, L_out, sampling='mw', method='jax')
        )(rho).astype(cdtype)

    def chunk_body(R_chunk):
        S_chunk = _sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R_chunk,
                                         N_unique, k_batch)                 # (N_unique, r_chunk)
        S_T = S_chunk.T                                                      # (r_chunk, N_unique)

        if out_sharding is not None:
            mesh = out_sharding.mesh
            S_T = jax.lax.with_sharding_constraint(
                S_T, NamedSharding(mesh, P('x', None))
            )
            rho_lms_chunk = shard_map(
                _sht_roundtrip, mesh=mesh,
                in_specs=P('x', None),
                out_specs=P('x', None, None),
                check_rep=False,
            )(S_T)
            rho_lms_chunk = jax.lax.with_sharding_constraint(rho_lms_chunk, out_sharding)
        else:
            rho_lms_chunk = _sht_roundtrip(S_T)
        return rho_lms_chunk

    out = jnp.zeros((pad_to, L_out, 2 * L_out - 1), dtype=cdtype)
    if out_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, out_sharding)

    one_minus_ramp = jnp.asarray(1.0, dtype=ramp_c.dtype) - ramp_c

    def step(out_acc, i):
        R_chunk = jax.lax.dynamic_slice_in_dim(R_padded, i * r_chunk, r_chunk, axis=0)
        rho_lms_chunk = chunk_body(R_chunk)

        rho_lms_chunk = ramp_c * rho_lms_chunk
        static_chunk = jax.lax.dynamic_slice_in_dim(
            static_padded, i * r_chunk, r_chunk, axis=0)
        static_add = (one_minus_ramp * static_chunk).astype(cdtype)
        rho_lms_chunk = rho_lms_chunk.at[:, 0, L_out - 1].add(static_add)

        out_acc = jax.lax.dynamic_update_slice(out_acc, rho_lms_chunk, (i * r_chunk, 0, 0))
        if out_sharding is not None:
            out_acc = jax.lax.with_sharding_constraint(out_acc, out_sharding)
        return out_acc, None

    out, _ = jax.lax.scan(step, out, jnp.arange(n_chunks))
    return out[:Nr]


# Private JAX helper that builds the associated-Legendre table; we call it
# directly so the table sizes as (n_max+1, n_max+1, N_eval) at *N_p* unique
# θ values rather than at (n_max+1)² × Nmodes (the shape jax.scipy.special
# .sph_harm_y forces because it requires (n, m, θ, φ) to share a flat shape).
from jax._src.scipy.special import _gen_associated_legendre


@functools.partial(jax.jit, static_argnames=("n_max",))
def _compute_Ylm_and_dtheta_jit(lm_pairs, theta_arr, phi_arr, n_max):
    """Y_lm and dY_lm/dθ at every particle, fully on-GPU.

    Replaces the previous scipy.special.sph_harm_y(diff_n=1) call, which
    ran single-threaded on CPU and forced a GPU↔host round-trip per
    timestep.

    Why this isn't just `jax.scipy.special.sph_harm_y`: that wrapper
    requires `(n, m, θ, φ)` to share a flat shape, so feeding it our
    `(Nmodes,)` (n, m) and `(Np,)` (θ, φ) forces broadcasting θ up to
    `(Nmodes,)` per particle. Internally it then builds an associated
    Legendre cube of shape `(n_max+1, n_max+1, Nmodes)` — at L_max=481
    that's ~430 GiB per particle (~2 TiB total under the outer vmap).
    Since every entry of that broadcast θ is identical for a given
    particle, 99.999% of the cube is duplicated work.

    Fix: call `_gen_associated_legendre` ourselves with cos(θ) shape
    `(Np,)`. Table size is `(n_max+1, n_max+1, Np)` ≈ 9 MiB. Then gather
    `legendre[|m|, l, p]` for every (mode, particle) pair, multiply by
    `exp(i |m| φ_p)`, and apply the m<0 sign correction — same formula
    `_sph_harm` uses internally. JVP w.r.t. θ gives dY/dθ in one pass.

    Inputs
    ------
    lm_pairs : (Nmodes, 2) int — [l, m] per output mode
    theta_arr, phi_arr : (Np,) float — particle colat/azimuth
    n_max    : int (static) — max l in `lm_pairs[:, 0]`

    Returns
    -------
    Y, dY_dtheta : (Nmodes, Np) complex
    """
    n_arr = lm_pairs[:, 0]
    m_arr = lm_pairs[:, 1]
    abs_m = jnp.abs(m_arr)

    # exp(i |m| φ_p) — (Nmodes, Np). Independent of θ, computed once
    # outside the JVP so it isn't part of the differentiation graph.
    angle  = abs_m[:, None] * phi_arr[None, :]
    vander = jnp.cos(angle) + 1j * jnp.sin(angle)
    sign_neg_m = ((-1.0) ** abs_m)[:, None]   # (Nmodes, 1)
    m_neg_mask = (m_arr < 0)[:, None]         # (Nmodes, 1)

    def Y_of_theta(theta):
        # theta: (Np,) → legendre: (n_max+1, n_max+1, Np)
        cos_theta = jnp.cos(theta)
        legendre = _gen_associated_legendre(n_max, cos_theta, True)
        # Gather P_l^|m|(cos θ_p) for every (mode, particle).
        # Result shape: (Nmodes, Np), real-valued.
        leg_vals = legendre[abs_m, n_arr, :]
        Y_pos = leg_vals * vander                                       # (Nmodes, Np) complex
        Y     = jnp.where(m_neg_mask, sign_neg_m * jnp.conjugate(Y_pos), Y_pos)
        return Y

    # JVP with tangent=1 in θ returns (Y, dY/dθ) without a second SHT pass.
    Y, dY_dtheta = jax.jvp(Y_of_theta, (theta_arr,), (jnp.ones_like(theta_arr),))
    return Y, dY_dtheta


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch"))
def _compute_rho_lm_at_particles_sphht_jit(R_j_at_parts, phase_c, aj, parent_j,
                                            lm_idx, lm_pairs, total_mass,
                                            L_out, N_unique, k_batch):
    """JIT'd per-particle rho_lm via sparse a_u_j × R_j_at_parts.

    Equivalent to the old dense `einsum('uj,j,pj->pu', a_u_j_sphht, ...)`
    followed by per-particle inverse-SHT, |.|^2, forward-SHT.
    """
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_at_parts.dtype)

    # (N_unique, Np) — same kernel, just M = N_particles.
    S_up = _sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R_j_at_parts,
                                  N_unique, k_batch)
    S_pu = S_up.T                                                          # (Np, N_unique)

    def single(S_u):
        flm = jnp.zeros((L_out, 2 * L_out - 1), dtype=cdtype)
        flm = flm.at[lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(S_u)
        psi_at_r = s2fft.inverse(flm, L_out, sampling='mw', method='jax')
        rho_at_r = total_mass * jnp.abs(psi_at_r) ** 2
        return s2fft.forward(rho_at_r, L_out, sampling='mw', method='jax')

    return jax.vmap(single)(S_pu)


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



class Simulation_Particle:
    """
    Stores the state (position + velocity) and history for a single stellar particle.
    """

    def __init__(self, particle_id, init_pos_cart, init_vel_cart, u):

        self.id = particle_id
        self.u = u

        # Current Cartesian state
        self.r_pos = np.array(init_pos_cart)   # (3,)
        self.v     = np.array(init_vel_cart)    # (3,)

        # Convert to spherical for initial record
        self.r_pos_sph = SSF.Cartesian_to_sph(self.r_pos[0], self.r_pos[1], self.r_pos[2])
        self.v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2],self.v[0], self.v[1], self.v[2])


        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]
        self.stellar_v_disp = [0]
        self.r_values       = [float(self.r_pos_sph[0])]
        self.average_r      = [float(self.r_pos_sph[0])]
        self.positions_xyz  = [[float(self.r_pos[0]), float(self.r_pos[1]), float(self.r_pos[2])]]

        self.potential_energy = []
        self.kinetic_energy = [1/2 * np.sum(self.v**2)]
        self.ang_mom = [np.linalg.norm(np.cross(self.r_pos, self.v))]

        self.time_step = 0


    def Change_to_new_vel(self, v_corrected):

        self.v = np.array(v_corrected)
        self.v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2], v_corrected[0], v_corrected[1], v_corrected[2])
        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]

        self.kinetic_energy = [1/2 * np.sum(self.v**2)]
        self.ang_mom = [np.linalg.norm(np.cross(self.r_pos, self.v))]

    def Create_V_array(self, no_time_steps):
        # Preallocate (no_time_steps + 1, 3) so row 0 holds the initial v_sph
        # and rows 1..no_time_steps hold the values written by update_state.
        self.velocities_arr = np.zeros((no_time_steps + 1, 3))
        self.velocities_arr[0] = np.asarray(self.v_sph)


    def update_state(self, new_pos_cart, new_vel_cart):
        """
        Called after each rebound integration step to update this particle's
        Cartesian and spherical state and append to history arrays.

        """
        x, y, z    = float(new_pos_cart[0]), float(new_pos_cart[1]), float(new_pos_cart[2])
        vx, vy, vz = float(new_vel_cart[0]), float(new_vel_cart[1]), float(new_vel_cart[2])

        self.r_pos = np.array([x, y, z])
        self.v     = np.array([vx, vy, vz])

        r, theta, phi      = SSF.Cartesian_to_sph_np(x, y, z)
        vr, vtheta, vphi   = SSF.Cartesian_to_sph_vel_np(x, y, z, vx, vy, vz)
        self.r_pos_sph     = np.array([r, theta, phi])
        self.v_sph         = np.array([vr, vtheta, vphi])

        self.velocities.append(self.v_sph)
        self.velocities_cart.append(self.v)

        # In-place write into preallocated array; row 0 is the initial v_sph,
        # so the k-th update writes at row k.
        self.velocities_arr[self.time_step + 1] = self.v_sph
        valid = self.velocities_arr[:self.time_step + 2]

        new_vel_disp = (
            np.std(valid[:, 0])**2
            + np.std(valid[:, 1])**2
            + np.std(valid[:, 2])**2
        ) ** 0.5

        self.stellar_v_disp.append(new_vel_disp)

        self.r_values.append(r)
        self.positions_xyz.append([x, y, z])
        self.kinetic_energy.append(0.5 * (vx*vx + vy*vy + vz*vz))
        self.ang_mom.append(np.linalg.norm(np.cross(self.r_pos, self.v)))

        self.time_step += 1


#--------------------------------------------------------------------------------------------------------------------


class StellarSimTDep:

    '''
    Stellar simulation which controls how everything is done and calls the particle
    class to update particle states.
    '''

    def __init__(self, m22, r_half, r_half_width, no_of_particles, no_time_steps, total_evolve_time, r_min, r_max_enclosing_frac,
                 no_radius_bins, SphHT, integrator, plot, dt_override, ramp_time, sparse_k_batch, r_chunk_size, l_band_size,
                 compute_dtype, a_j_threshold, use_multi_gpu=True):

        self.stellar_v_disp = []
        self.average_r = []
        self.time_step = 0
        self.SphHT = SphHT
        self.integrator = integrator
        self.plot = plot
        self.dt_override = dt_override
        self.ramp_time = ramp_time
        self.a_j_threshold = a_j_threshold

        # ---------- Memory-saving knobs ----------
        # complex64 / float32 for the heavy density / R_j_r path. Eigenenergies
        # and the per-step phase stay in float64 — they multiply by t and need
        # the precision for long-time stability.
        self.compute_dtype = jnp.dtype(compute_dtype)
        if self.compute_dtype == jnp.complex64:
            self.compute_real_dtype = jnp.float32
        elif self.compute_dtype == jnp.complex128:
            self.compute_real_dtype = jnp.float64
        else:
            raise ValueError(f"compute_dtype must be complex64 or complex128, got {compute_dtype}")

        # Chunk size for streaming the (l,m) integration loop in
        # `compute_phi_lm_and_deriv` — caps the per-particle intermediate at
        # (l_band_size, Nr+1) instead of (L_max_out**2, Nr+1).
        self.l_band_size = int(l_band_size)

        # Chunk size for the streamed sparse-a_u_j scatter-add. The dense
        # `(N_unique, Nj)` matrix scales as m22**5 and is ~99% zeros at
        # high m22; we never build it. Instead we stream `sparse_k_batch`
        # k-modes per scan iteration. Lower = less memory but more scan
        # iterations; higher = fewer iterations but larger per-batch peak.
        self.sparse_k_batch = int(sparse_k_batch)

        # Chunk size for streaming the SHT round-trip over the radial
        # axis. Caps the in-flight transient at (r_chunk, L_out, 2L_out-1)
        # complex per chunk instead of the full (Nr, ...) tensor. At m22=10
        # with c64 and Nr=1000, r_chunk=64 gives ~240 MB per chunk vs
        # 3.7 GB without chunking — plus s2fft's internal working set.
        self.r_chunk_size = int(r_chunk_size)

        # Shard the big (Nr, Nj) and (N_unique, Nj) arrays across all visible
        # CUDA devices when more than one is present.
        self.use_multi_gpu = bool(use_multi_gpu)
        self._setup_sharding()

        if self.shard_l is not None:
            n_dev = len(self.devices)
            if self.r_chunk_size % n_dev != 0:
                raise ValueError(
                    f"r_chunk_size ({self.r_chunk_size}) must be divisible by "
                    f"number of devices ({n_dev}): shard_map splits the SHT "
                    f"chunk along the r axis."
                )

        self.m22 = m22
        self.u = jsp.set_schroedinger_units(self.m22)

        self.no_of_particles = no_of_particles

        # List of Simulation_Particle instances — populated in initialising_simulation()
        self.particles = []

        self.r_half = r_half
        self.r_half_width = r_half_width
        self.no_time_steps = no_time_steps
        self.total_evolve_time = total_evolve_time
        self.dt = (self.total_evolve_time * self.u.from_Gyr) / self.no_time_steps

        self.r_min = r_min
        self.r_max_enclosing_frac = r_max_enclosing_frac

        self.no_radius_bins = no_radius_bins

        self.G = GN.value * (self.u.from_cm**3) / (self.u.from_g * self.u.from_s**2)


        self.current_phase = None
        self.R_j_r_phased = None
        self.eigen_energies = None
        self.lm_pairs_np = None

    # ----------------------------------------------------------------------
    # Sharding helpers (change 7: shard the heavy arrays across both GPUs).
    # When only one device is visible these become no-ops, so behaviour is
    # identical to the single-GPU original.
    # ----------------------------------------------------------------------
    def _setup_sharding(self):
        devs = jax.devices()
        self.devices = devs
        if self.use_multi_gpu and len(devs) > 1:
            from jax.sharding import Mesh, NamedSharding, PartitionSpec
            from jax.experimental import mesh_utils
            device_mesh = mesh_utils.create_device_mesh((len(devs),))
            self.mesh = Mesh(device_mesh, axis_names=('x',))
            # Shard `Nj` axis (last) for arrays like (Nr, Nj) and (N_unique, Nj).
            self.shard_nj  = NamedSharding(self.mesh, PartitionSpec(None, 'x'))
            # Shard `L`  axis (middle) for arrays of shape (Nr, L, 2L-1).
            self.shard_l   = NamedSharding(self.mesh, PartitionSpec(None, 'x', None))
            self.shard_rep = NamedSharding(self.mesh, PartitionSpec())
            print(f"[mem_saver] Multi-GPU sharding enabled across {len(devs)} devices.")
        else:
            self.mesh = None
            self.shard_nj = None
            self.shard_l = None
            self.shard_rep = None

    def _shard_nj(self, arr):
        """Place a 2-D array (..., Nj) sharded along Nj. No-op if single-GPU."""
        if self.shard_nj is not None:
            return jax.device_put(arr, self.shard_nj)
        return arr

    def _shard_l(self, arr):
        """Place a 3-D array (Nr, L, 2L-1) sharded along L. No-op if single-GPU,
        or if the L axis isn't divisible by the number of devices.

        `device_put` requires exact divisibility; `with_sharding_constraint`
        inside a JIT does not (GSPMD pads internally). So this method is
        strictly weaker than the in-JIT constraint — prefer that path. For
        L_out = 2L-1 (always odd), this falls back to replicated.
        """
        if self.shard_l is None:
            return arr
        n_dev = len(self.devices)
        if arr.ndim >= 2 and arr.shape[1] % n_dev != 0:
            return arr
        return jax.device_put(arr, self.shard_l)

    def _build_sparse_au_j(self, lm_idx, parent_j, aj, N_unique, Nj):
        """Build the dense `(N_unique, Nj)` scatter `a_u_j[lm_idx, parent_j] = aj`
        without peak-doubling on the GPU.

        `jnp.zeros((N_unique, Nj)).at[...].add(aj)` allocates the zeros + a
        new array for the scatter result, peaking at 2x the final size — at
        m22=10 with complex64 that's ~26 GB which overflows a 25 GB GPU.
        Here we materialise the scatter in CPU RAM via numpy, then transfer
        (sharded if multi-GPU). CPU RAM is plentiful and the GPU only ever
        holds the final, single-copy array.
        """
        np_dtype = np.zeros((), dtype=aj.dtype).dtype  # jax dtype -> numpy dtype
        a_u_j_np = np.zeros((N_unique, Nj), dtype=np_dtype)
        a_u_j_np[np.asarray(lm_idx), np.asarray(parent_j)] = np.asarray(aj)
        if self.shard_nj is not None:
            return jax.device_put(a_u_j_np, self.shard_nj)
        return jnp.asarray(a_u_j_np)
    

    def Reduce_small_a_j_modes(self, aj_2):

        # Zero out a_j below the threshold to save memory and speed up the SHT.
        # The sparse-a_u_j scatter-add kernel will skip these modes entirely
        # instead of scattering tiny aj values across all particles.
        small_mask = aj_2 >= self.a_j_threshold
        return small_mask




    def initialising_simulation(self):

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_wf")
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"m22_{float(self.m22):.6g}_rbins_{int(self.no_radius_bins)}"
        r_j_r_fname = os.path.join(cache_dir, f"precomputed_R_j_r_{cache_suffix}.npz")
        pkl_fname   = os.path.join(cache_dir, f"precomputed_objs_{cache_suffix}.pkl")

        cache_params = {
            'm22': float(self.m22),
            'r_min': float(self.r_min),
            'r_max_enclosing_frac': float(self.r_max_enclosing_frac),
            'no_radius_bins': int(self.no_radius_bins),
        }

        def _cache_valid(data, expected):
            for k, v in expected.items():
                if k not in data.files:
                    return False
                cached = data[k].item() if data[k].shape == () else data[k]
                if isinstance(v, float):
                    if not np.isclose(cached, v):
                        return False
                elif cached != v:
                    return False
            return True

        R_j_r = None
        if os.path.isfile(r_j_r_fname) and os.path.isfile(pkl_fname):
            data = np.load(r_j_r_fname)
            if _cache_valid(data, cache_params):
                print(f"Loading precomputed R_j_r from {r_j_r_fname}...")
                R_j_r = data['R_j_r']
                rmin = data['rmin'].item()
                rmax = data['rmax'].item()
                with open(pkl_fname, 'rb') as f:
                    objs = pickle.load(f)
                eigenstate_lib    = objs['eigenstate_lib']
                wavefunction_params = objs['wavefunction_params']
                eval_library = jax.vmap(jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0, None))

            else:
                print(f"Cached {r_j_r_fname} stale (parameter mismatch); recomputing.")

        if R_j_r is None:

            cNFWtides_params = jnp.array([
            357964808.148399 * self.u.from_Msun,
            25.690207,
            0.407461,
            0.012670 * self.u.from_Kpc,
            1.857991 * self.u.from_Kpc,
            3.729259
            ])

            density_params = jsp.init_core_NFW_tides_params_from_sample(cNFWtides_params)

            N = 512
            rmin = .1 * self.u.from_pc
            rmax = jsp.enclosing_radius(0.999, density_params)
            potential_params = jsp.init_potential_params(density_params, rmin, rmax, N)

            eval_library = jax.vmap(jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0,None))

            N = 1024
            a = 1
            b = 10

            rmax = jsp.enclosing_radius(self.r_max_enclosing_frac, density_params)
            eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin, rmax, a, b, N)


            rmin = self.r_min * self.u.from_pc

            tol = 1e-7
            wavefunction_params = jsp.init_wavefunction_params(eigenstate_lib, density_params, rmin, rmax, tol)


            r = jnp.logspace(jnp.log10(rmin), jnp.log10(rmax), self.no_radius_bins)
            R_j_r = eval_library(r, eigenstate_lib.radial_eigenmode_params)  # (Nr, Nj)

            np.savez(r_j_r_fname, R_j_r=np.array(R_j_r), rmin=rmin, rmax=rmax, **cache_params)
            with open(pkl_fname, 'wb') as f:
                pickle.dump({'eigenstate_lib': eigenstate_lib, 'wavefunction_params': wavefunction_params}, f)

                
        l = eigenstate_lib.radial_eigenmode_params.l

        print('l max from jaxsp:', max(l))
        L = int(max(l) + 1)

        self.L = L

        self.L_max_out = 2 * L - 1

        self.rmin = rmin
        self.rmax = rmax
        

        r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)
        self.r = r

        total_mass = wavefunction_params.total_mass
        self.total_mass = total_mass
        aj_2_full = wavefunction_params.aj_2

        keep = self.Reduce_small_a_j_modes(aj_2_full)

        keep_np = np.asarray(keep)

        aj_2 = aj_2_full[keep]

        # Integer indices of kept modes in the FULL library. Needed wherever
        # R_j is evaluated against the unfiltered eigenstate_lib (e.g. at
        # particle radii) so it can be sliced down to the same filtered space
        # everything else (a_u_j, parent_j, weight_j, ...) lives in.
        kept_idx_np = np.flatnonzero(keep_np)
        self.kept_indices = jnp.asarray(kept_idx_np)

        l = np.asarray(eigenstate_lib.radial_eigenmode_params.l)[keep_np]
        eigen_energies = eigenstate_lib.radial_eigenmode_params.E[keep]
        R_j_r = R_j_r[:, keep_np]                             # filter the Nj axis
        print(f"Kept {int(keep.sum())} / {keep.size} radial modes "
            f"above threshold {self.a_j_threshold:.3g}")



        # Heavy array — cast to compute_real_dtype (float32 by default) and
        # shard along Nj across all visible GPUs. At m22=50, R_j_r is ~30 GB
        # in float64; this halves it and splits it across devices.
        R_j_r_cast = jnp.asarray(R_j_r, dtype=self.compute_real_dtype)

        self.Nj_pad = 0
        if self.shard_nj is not None:
            n_dev = len(self.devices)
            pad = (-R_j_r_cast.shape[1]) % n_dev
            if pad:
                R_j_r_cast = jnp.pad(R_j_r_cast, ((0, 0), (0, pad)))
                # Sibling per-j arrays must broadcast against R_j_r_fixed; pad
                # with zeros so the padded slot contributes nothing (R_j_r is
                # zero there, parent_j never indexes into it).
                l = np.concatenate([l, np.zeros(pad, dtype=l.dtype)])
                eigen_energies = jnp.concatenate(
                    [jnp.asarray(eigen_energies),
                     jnp.zeros(pad, dtype=eigen_energies.dtype)])
                self.Nj_pad = int(pad)

        self.l = jnp.asarray(l)
        self.eigen_energies = eigen_energies


        self.R_j_r_fixed = self._shard_nj(R_j_r_cast)
        del R_j_r_cast, R_j_r

        # NOTE: R_j_r_phased is no longer materialised. Downstream code uses
        # `R_j_r_fixed` and a per-step `phase` (or pre-phased aj) directly.

        (parent_j, lm_pairs, lm_l_per_mode, lm_m_per_mode, theta, phi, lm_idx_per_mode) = precompute_lm_pairs(l)

        Nmodes = len(parent_j)
        rand_phase_per_mode = jax.random.uniform(jax.random.PRNGKey(42), shape=(Nmodes,), minval=0.0, maxval=2 * jnp.pi)
        aj = (jnp.sqrt(aj_2[parent_j]) * jnp.exp(1j * rand_phase_per_mode)).astype(self.compute_dtype)


        self.parent_j = parent_j
        self.lm_l = lm_pairs[:, 0]
        self.lm_m = lm_pairs[:, 1]
        self.lm_l_per_mode = lm_l_per_mode
        self.lm_m_per_mode = lm_m_per_mode
        self.lm_idx_per_mode = lm_idx_per_mode
        self.lm_pairs_jax = jnp.asarray(lm_pairs, dtype=jnp.int32)
        self.theta = theta
        self.phi = phi

        # Sparse representation of a_u_j: we keep the three (Nmodes_k,)
        # triplet arrays — `aj`, `parent_j`, `lm_idx_per_mode` — instead of
        # ever materialising the dense `(N_unique, Nj)` matrix. The dense
        # form scales as m22**5 and is ~99% zeros; the sparse form scales
        # as ~m22**4. At m22=10 that swaps 13 GB for ~100 MB.
        # The (sparse) "a_u_j" is now just `(self.aj, self.parent_j, self.lm_idx_per_mode)`.
        self.N_unique_sphht = int(self.lm_pairs_jax.shape[0])
        self.Nj_total = int(len(self.eigen_energies))

        # Set self.aj here so `construct_rho_rtp` (called next) can access
        # it via the sparse triplet. run_simulation will overwrite this with
        # the returned aj — the value is identical.
        self.aj = jnp.asarray(aj)

        # Constructing initial conditions based on Andrew paper

        # The static background is the time-averaged (diagonal) density — the
        # smooth profile the halo is built to reproduce. Both arXiv:2510.17079
        # (Eq. 8) and arXiv:2604.26393 (§III) set the orbit ICs in this mean
        # field and treat the granular fluctuations as a perturbation ramped
        # on top; the instantaneous granule snapshot is NOT the equilibrium.
        # compute_diagonal_rho_expansion also sets self.weight_j (reused
        # per-particle during the ramp).
        rho_diag = self.compute_diagonal_rho_expansion()

        # (l=0, m=0) coefficient of the same diagonal density — the ramp
        # baseline consumed by Build_rho_lms_for_timestep. Y00 = 1/sqrt(4π)
        # so the coefficient is rho_diag · sqrt(4π).
        self.rho_static_r_l00 = (
            rho_diag * jnp.sqrt(4.0 * jnp.pi)).astype(self.compute_dtype)

        # Cumulative enclosed mass M_enc(r) on the radial grid; interpolated
        # per particle below. SSF.Enclosed_mass applies the 4π r² factor.
        M_enc_arr = SSF.Enclosed_mass(self.r, rho_diag)

        # M_enc_tot = M_enc_arr[-1]

        # print(f"Total enclosed mass at rmax: {M_enc_tot:.3e}")
        # print(f"Total mass from wavefunction: {total_mass:.3e}")

        # multiply_factor = total_mass / M_enc_tot

        # print(f"Scaling density and mass by factor {multiply_factor} to match total mass")

        # self.total_mass *= multiply_factor

        if self.plot:

            plt.plot(self.r * self.u.to_Kpc, rho_diag * self.u.to_Msun / (self.u.to_Kpc)**3)

            plt.xlabel('r (kpc)')
            plt.ylabel(r'$\rho$ [$M_\odot / kpc^3$]')
            plt.title(f'Time-averaged (diagonal) density profile with m22 = {self.m22}')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid()
            plt.show()



        #------------------------------------------------------------------
        # SIMULATION

        sim = rebound.Simulation()


        if self.integrator == 'ias15':
        
            sim.integrator = "ias15"
            sim.force_is_velocity_dependent = False
            sim.integrator.ri_ias15.epsilon = 1e-5
        
        elif self.integrator == 'leapfrog':

            sim.integrator = "leapfrog"
            sim.dt = self.dt

        init_vels = []
        r_orbit_mean = self.r_half * self.u.from_Kpc
        r_orbit_min = r_orbit_mean - self.r_half_width/2 * self.u.from_Kpc
        r_orbit_max = r_orbit_mean + self.r_half_width/2 * self.u.from_Kpc


        self.particles = []
        for i in range(self.no_of_particles):

            r_orbit = jax.random.uniform(jax.random.PRNGKey(i), shape=(), minval=r_orbit_min, maxval=r_orbit_max)


            X1 = jax.random.normal(jax.random.PRNGKey(i+1000), shape=(), dtype=jnp.float64)
            X2 = jax.random.normal(jax.random.PRNGKey(i+2000), shape=(), dtype=jnp.float64)
            X3 = jax.random.normal(jax.random.PRNGKey(i+3000), shape=(), dtype=jnp.float64)

            mag = jnp.sqrt(X1**2 + X2**2 + X3**2)

            r_i = r_orbit * jnp.array([X1, X2, X3]) / mag

            r_i_unit = r_i / r_orbit

            #avoid degeneracy near z-axis
            ref = jnp.where(jnp.abs(r_i_unit[2]) < 0.9, 
                            jnp.array([0., 0., 1.]), 
                            jnp.array([1., 0., 0.]))
            o_i_unit = jnp.cross(r_i_unit, ref)
            o_i_unit = o_i_unit / jnp.linalg.norm(o_i_unit)


            t_i_unit = jnp.cross(r_i_unit, o_i_unit)

            b_i_unit = jnp.cross(t_i_unit, r_i_unit)

            rand_theta = jax.random.uniform(jax.random.PRNGKey(i+4000), shape=(), minval=0.0, maxval=2 * jnp.pi,)

            v_i_unit = t_i_unit * jnp.sin(rand_theta) + b_i_unit * jnp.cos(rand_theta)

            # Compute circular velocity from spherically-averaged enclosed mass
            M_enc_at_r = jnp.interp(r_orbit, self.r, M_enc_arr)
            v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / r_orbit)

            init_pos = r_i
            init_vel = v_circ_mag * v_i_unit

            init_vels.append(v_circ_mag)

            print(f"Particle {i}: v_circ = {v_circ_mag * self.u.to_kms:.3f} km/s")

            particle = Simulation_Particle(i, init_pos, init_vel, self.u)
            self.particles.append(particle) # Adding instances of particles to simulation class

            sim.add(
                m=0.0,
                x=float(init_pos[0]), y=float(init_pos[1]), z=float(init_pos[2]),
                vx=float(init_vel[0]), vy=float(init_vel[1]), vz=float(init_vel[2])
            )

        sim_particles = sim.particles

        autodiff_data = {'eval_library': eval_library, 'eigenstate_lib': eigenstate_lib}
        self.autodiff_data = autodiff_data

        self._force_call_count = 0

        def additional_forces_step(_reb_sim):
            """
            IAS15 calls this multiple times per timestep at different positions.
            All particle accelerations are computed in a single batched JAX call
            (vmap over the radial integrals + one vectorised scipy angular call),
            then written back to each rebound particle.
            """
            N = self.no_of_particles

            # Pull rebound Cartesian state in one pass and do the Cartesian->spherical
            # transform batched in numpy. Avoids N+1 separate jnp.array dispatches.
            xyz = np.empty((N, 3))
            for i in range(N):
                p = sim_particles[i]
                xyz[i, 0] = p.x
                xyz[i, 1] = p.y
                xyz[i, 2] = p.z
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            r, theta, phi = SSF.Cartesian_to_sph_np(x, y, z)

            positions_sph = jnp.asarray(np.stack([r, theta, phi], axis=1))

            self._force_call_count += 1

            # Single batched acceleration computation — parallel over all particles
            a_r_all, a_theta_all, a_phi_all = self.construct_acc_master_func(
                positions_sph,
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib']
            )

            # Pull accs back to host once, then do the spherical->Cartesian
            # rotation batched in numpy.
            a_x, a_y, a_z = SSF.acceleration_spherical_to_cartesian_np(
                np.asarray(a_r_all), np.asarray(a_theta_all), np.asarray(a_phi_all),
                theta, phi,
            )


            for i in range(N):
                sim_particles[i].ax += float(a_x[i])
                sim_particles[i].ay += float(a_y[i])
                sim_particles[i].az += float(a_z[i])

        sim.additional_forces = additional_forces_step

        self.sim = sim
        self.sim_particles = sim_particles

        r_orbits = jnp.array([p.r_values[0] for p in self.particles])

        r_orbit_mean = jnp.mean(r_orbits)

        print(f"Mean r: {r_orbit_mean * self.u.to_Kpc:.3f} kpc")

        if self.dt_override is not None:

            orbital_P = 2 * jnp.pi * r_orbits / jnp.array(init_vels)

            min_orbital_P = jnp.min(orbital_P)

            mean_init_vel = jnp.mean(jnp.array(init_vels))

            lambda_db_kpc = 19.15 / (self.m22 * mean_init_vel * self.u.to_kms)
            T_c = lambda_db_kpc / (mean_init_vel * self.u.to_Kpc) 

            print(f"Min T_orb: {min_orbital_P * self.u.to_Myr:.3f} Myr")
            print(f"T_c: {T_c * self.u.to_Myr:.3f} Myr")

            new_dt_orb = min_orbital_P / self.dt_override

            new_dt_c = T_c / self.dt_override

            new_dt = min(new_dt_orb, new_dt_c)

            self.sim.dt = float(new_dt)

            self.dt = new_dt

            self.no_time_steps = int(self.total_evolve_time * self.u.from_Gyr / new_dt)

            print(f"dt: {self.dt * self.u.to_Gyr:.3f} Gyr")
            print(f"Number of time steps: {self.no_time_steps}")

        return aj


    def ramp_frac_for_step(self, time_step):

        """Scalar in [0, 1] giving the fraction of the off-diagonal (j != j') 
        cross-terms switched on.
        Linear from ~0 to 1 over n_ramp_steps
        """

        if time_step < self.n_ramp_steps:
            return (time_step + 1) / self.n_ramp_steps
        else:
            return 1.0

    def compute_diagonal_rho_expansion(self):

        """Time-averaged (diagonal) density ρ_static(r) — the smooth static
        profile the halo is built to reproduce, and the background that BOTH
        the circular-velocity ICs and the ramp baseline are built from
        (cf. arXiv:2510.17079 Eq. 8, arXiv:2604.26393 §III).

        Because |a_{j,m}|² is m-independent (isotropic random phases), the
        addition theorem Σ_m |Y_{lm}(θ,φ)|² = (2l+1)/4π collapses the angular
        sum to a constant, giving a spherically symmetric static density:

            ρ_static(r) = total_mass · Σ_j  weight_j · |R_j(r)|²
            weight_j     = |a_j|² · (2l_j + 1) / (4π)

        Returns the (Nr,) real density ρ_static(r). Side effect: sets
        self.weight_j, reused by `compute_rho_lm_at_particles_diagonal_only`.
        The (l=0, m=0) spherical-harmonic coefficient — the only nonzero slot,
        used as the ramp baseline — is ρ_static(r)·sqrt(4π).
        """

        Nj = self.R_j_r_fixed.shape[1]

        # Recover |a|² per j-mode: all k-modes sharing the same j carry the same value.
        aj_sq_k = jnp.abs(self.aj) ** 2                                       # (Nk,)
        aj_sq_j = jnp.zeros(Nj, dtype=jnp.float64).at[self.parent_j].set(aj_sq_k)  # (Nj,)

        weight_j = aj_sq_j * (2.0 * self.l.astype(jnp.float64) + 1.0) / (4.0 * jnp.pi)
        self.weight_j = weight_j                                               # (Nj,) — reused by static rho_lm calls

        R_sq = (jnp.abs(self.R_j_r_fixed) ** 2).astype(jnp.float64)
        rho_static_r = self.total_mass * (R_sq @ weight_j)                    # (Nr,)

        return rho_static_r

    def compute_rho_lm_at_particles_diagonal_only(self, R_j_at_particles):
        """Time-averaged (diagonal-only) rho_lm at each particle's exact r.

        Because ρ_static is spherically symmetric, only (l=0, m=0) is nonzero:
            ρ_lm(r_p) = ρ_static(r_p) · sqrt(4π) · δ_{l0} δ_{m0}
        No SHT needed — just a dot product against weight_j.
        """
        R_sq_all = (jnp.abs(R_j_at_particles) ** 2).astype(self.compute_real_dtype)  # (N_p, Nj)
        rho_r_p  = self.total_mass * (R_sq_all @ self.weight_j.astype(self.compute_real_dtype))  # (N_p,)
        N_p      = rho_r_p.shape[0]
        L_out    = self.L_max_out
        out = jnp.zeros((N_p, L_out, 2 * L_out - 1), dtype=self.compute_dtype)
        return out.at[:, 0, L_out - 1].set(rho_r_p * jnp.sqrt(4.0 * jnp.pi))

    def construct_rho_lms_gaunt(self, aj, parent_j, R_j_r_phased):

        '''
        Construct rho_lms using the Gaunt kernel.
        '''

        rho_lm_gaunt = gf.compute_rho_lm_gaunt(
            aj, R_j_r_phased, parent_j, self.lm_idx_sorted_per_mode,
            self.total_mass,
            L_max_out=self.L_max_out,
            gaunt_table=self.gaunt_table,
            batch_size = 100_000
        )

        return rho_lm_gaunt


    def compute_rho_lms_s2fft(self, rho_rtp):

        '''Forward s2fft of 3d density to get rho_lm(r)
        '''

        #Parallel forward SHT over all radii
        def forward_sht_single_r(rho_at_r):
            return s2fft.forward(rho_at_r, self.L_max_out, sampling='mw', method='jax')

        flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)


        return flm_r

    def Build_rho_lms_for_timestep(self, time_step):
        """Build rho_lms at `time_step`, applying the two-phase schedule.

        Phase 1 (ramp, time_step < n_ramp_steps):
            rho = rho_static + ramp_frac * (rho_full(t) - rho_static).
            `rho_static` is the time-averaged (diagonal) density — the same
            smooth background the circular-velocity ICs were built from. The
            interference terms (the full time-dependent piece, both the
            monopole's breathing and the l>=1 asphericity) are linearly
            switched on from 0 to 1 over `ramp_time`.

        Phase 2 (main, time_step >= n_ramp_steps):
            rho = rho_full(t). Full instantaneous ULDM density, all terms.
        """

        # Phase stays in float64/complex128 for long-time stability; cast to
        # the compute dtype only when multiplying R_j_r (which is c64/f32).
        phase = jnp.exp(-1j * self.eigen_energies * time_step * self.dt / 1)
        self.current_phase = phase
        phase_c = phase.astype(self.compute_dtype)

        ramp_frac = self.ramp_frac_for_step(time_step)
        self.current_ramp_frac = jnp.float64(ramp_frac)
        # Cast ramp_frac to compute_real_dtype so multiplying it against a
        # c64 array doesn't promote to c128 (which then breaks the
        # c64 sharded output contract downstream).
        ramp_c = jnp.asarray(ramp_frac, dtype=self.compute_real_dtype)

        if self.SphHT:
            # Fused JIT: sparse-matmul -> scatter -> inv-SHT -> |.|^2 -> fwd-SHT
            # -> per-chunk ramp blend + static (l=0,m=0) add. The ramp / static
            # combine happens inside the scan so no second full-size
            # (Nr, L, 2L-1) tensor is ever materialised. XLA frees flm_r /
            # psi_rtp / rho_rtp transiently rather than holding them all at once.
            return _build_sphht_rho_lms_jit(
                self.R_j_r_fixed, phase_c,
                self.aj, self.parent_j, self.lm_idx_per_mode,
                self.lm_pairs_jax, self.total_mass,
                ramp_c, self.rho_static_r_l00,
                int(self.L_max_out),
                int(self.N_unique_sphht), int(self.sparse_k_batch),
                int(self.r_chunk_size),
                out_sharding=self.shard_l,
            )

        # Gaunt path uses an external helper (`gf.compute_rho_lm_gaunt`)
        # that expects an already-phased R_j_r; keep that contract but
        # build the phased array transiently.
        R_j_r_phased = self.R_j_r_fixed * phase_c[None, :]
        rho_lms_full = self.construct_rho_lms_gaunt(self.aj, self.parent_j, R_j_r_phased)
        del R_j_r_phased

        # rho_lms = (1 - ramp_frac) * rho_static + ramp_frac * rho_full.
        L_out = self.L_max_out
        rho_lms = ramp_c * rho_lms_full
        rho_lms = rho_lms.at[:, 0, L_out - 1].add(
            ((1.0 - ramp_c) * self.rho_static_r_l00).astype(rho_lms.dtype)
        )
        return rho_lms


    def construct_rho_rtp(self, R_j_r_fixed, phase_c, lm_pairs):

        '''
        Construct rho_rtp without using Y_lms.
        Inverse SHT to get psi on the dense grid, then square.

        Delegates to the module-level JIT'd helper using the SPARSE a_u_j
        representation — `(self.aj, self.parent_j, self.lm_idx_per_mode)` —
        so the `(N_unique, Nj)` dense matrix is never materialised. The
        scatter-add streams k-modes in batches of `self.sparse_k_batch`.
        '''
        return _build_sphht_rho_rtp_jit(
            R_j_r_fixed, phase_c,
            self.aj, self.parent_j, self.lm_idx_per_mode,
            lm_pairs, self.total_mass, int(self.L_max_out),
            int(self.N_unique_sphht), int(self.sparse_k_batch),
            int(self.r_chunk_size),
        )


    def compute_rho_lm_at_particles_gaunt(self, R_j_at_parts, phase_c, a_u_j, all_i, all_j, all_G, all_Lf):

        """Batched Gaunt path: compute rho_lm at every particle's radius in
        ONE call instead of once per particle inside a vmap.

        Change 9: fold the phase into the matmul rather than materialising
        an (N_particles, Nj) R_j_phased_at_parts copy.
        """

        # F_all[p, u] = Σ_j a_u_j[u, j] · phase[j] · R_j_at_parts[p, j]
        F_all = jnp.einsum('uj,j,pj->pu', a_u_j, phase_c, R_j_at_parts,
                           optimize='optimal')
        return gf.compute_rho_lm_gaunt_F(
            F_all, self.total_mass, self.L_max_out,
            all_i, all_j, all_G, all_Lf,
            batch_size=100_000,
        )
    

    def compute_rho_lm_at_particles_sphht(self, R_j_at_parts, phase_c):

        """SphHT path: rho_lm per particle via s2fft round-trip, using the
        sparse a_u_j scatter-add (no dense `(N_unique, Nj)` matrix).
        """
        return _compute_rho_lm_at_particles_sphht_jit(
            R_j_at_parts, phase_c,
            self.aj, self.parent_j, self.lm_idx_per_mode,
            self.lm_pairs_jax, self.total_mass, int(self.L_max_out),
            int(self.N_unique_sphht), int(self.sparse_k_batch),
        )

    def insert_particle_rholm_and_get_philm(self, r_pos_sph, rho_lm_at_particle, rho_lms):

        """
        Per-particle insertion into the background radial grid + call to
        _compute_all_phi. Safe to vmap. rho_lm_at_particle is supplied by
        the caller.

        Change 2: instead of carrying two pre-padded `(Nr+1, L, 2L-1)`
        copies of rho_lms (`rho_lms_below` and `rho_lms_above`), we gather
        from the unpadded `(Nr, L, 2L-1)` array using an index shift —
        2x less standing rho_lms memory.

        Pattern: for i in [0, Nr]:
            if i  < insert_idx: take rho_lms[i]
            if i == insert_idx: insert rho_lm_at_particle
            if i  > insert_idx: take rho_lms[i-1]  (shifted; particle sits between)
        which is rho_lms[i - (i > insert_idx)] with override at i == insert_idx.
        """

        particle_r = r_pos_sph[0]

        insert_idx = jnp.searchsorted(self.r, particle_r)

        # gather_idx: for i  < insert_idx -> i;   for i  >= insert_idx -> i-1
        # (clipped so the i == insert_idx slot is safe; that slot is then
        # overwritten by particle_r / rho_lm_at_particle in the where below).
        gather_idx = jnp.clip(
            self.all_idx - (self.all_idx > insert_idx).astype(jnp.int32),
            0,
            self.Nr - 1,
        )

        r_updated = jnp.where(
            self.all_idx == insert_idx,
            particle_r,
            self.r[gather_idx],
        )

        # Change 3: do NOT materialise `rho_lm_updated = (Nr+1, L, 2L-1)`.
        # `compute_phi_lm_and_deriv` builds each `(Nr+1,)` (l,m) slice
        # lazily from `rho_lms + rho_lm_at_particle + gather_idx + insert_mask`,
        # saving a ~3.7 GiB per-particle transient at m22~5, Nr=1000, L_out=481.
        insert_mask = self.all_idx == insert_idx

        mask_int = jnp.arange(self.Nr) < insert_idx
        mask_ext = jnp.arange(self.Nr) < (self.Nr - insert_idx)

        dphi_lm_dr_at_r, phi_lm_at_r = compute_phi_lm_and_deriv(
            rho_lms, rho_lm_at_particle, gather_idx, insert_mask,
            r_updated, self.output_lm_pairs,
            mask_int, mask_ext, int(self.L_max_out), self.G, particle_r,
            int(self.l_band_size),
        )

        # if self.plot and not isinstance(particle_r, jax.core.Tracer):

        #     if self.current_ramp_frac == 1.0:
        #         for l in range(3):
        #             plt.plot(r_updated * self.u.to_Kpc, rho_lm_updated[:, l, self.L_max_out - 1] * self.u.to_Msun / (self.u.to_Kpc)**3, label=f'l={l}')
        #         plt.xlabel('r (kpc)')
        #         plt.ylabel(r'$\rho_{lm}$ [$M_\odot / kpc^3$]')
        #         plt.title(f'Updated rho_lm with inserted particle at r={particle_r * self.u.to_Kpc:.3f} kpc')
        #         plt.xscale('log')
        #         plt.yscale('log')
        #         plt.grid()
        #         plt.legend()
        #         plt.show()

        #         def inverse_s2fft(rho_lm):
        #             return s2fft.inverse(rho_lm, int(self.L_max_out), sampling='mw', method='jax')
                
        #         rho_rtp = jax.vmap(inverse_s2fft)(rho_lm_updated)  # (Nr, n_theta, n_phi)

        #         plotting_theta = int(self.theta.shape[0] / 2)
        #         plotting_phi = 0

        #         rho_r = rho_rtp[:, plotting_theta, plotting_phi]
        #         plt.plot(r_updated * self.u.to_Kpc, rho_r * self.u.to_Msun / (self.u.to_Kpc)**3, label='Updated rho(r)')
        #         plt.xlabel('r (kpc)')
        #         plt.ylabel(r'$\rho$ [$M_\odot / kpc^3$]')
        #         plt.title(f'Updated rho(r) with inserted particle at r={particle_r * self.u.to_Kpc:.3f} kpc')
        #         plt.xscale('log')
        #         plt.yscale('log')
        #         plt.grid()
        #         plt.legend()
        #         plt.show()


        return dphi_lm_dr_at_r, phi_lm_at_r  # (Nmodes,), (Nmodes,)

    def calc_rho_lm_at_parts_and_call_insert(self, positions_sph, current_phase, radial_eigenmode_params, a_u_j, all_i, all_j,
                              all_G, all_Lf, rho_lms, ramp_frac):

        """JIT-compilable: radial basis evaluation + batched rho_lm + vmap
        over the per-particle insertion step.

        Changes 2 & 9: takes unpadded `rho_lms` (gather is done inside
        `insert_particle_rholm_and_get_philm`) and passes the per-step
        phase through to the SphHT/Gaunt density helpers without
        materialising a phased R_j table.
        """

        # Cast phase down to compute_dtype just before contracting with R_j.
        # `current_phase` is c128 (precision needed in the exp); cast loses
        # negligible info for the contraction.
        phase_c = current_phase.astype(self.compute_dtype)

        particle_rs = positions_sph[:, 0]
        R_j_at_particles = self._eval_library(particle_rs, radial_eigenmode_params)
        # eval_library uses the FULL unfiltered radial_eigenmode_params, so
        # the output is (N_particles, Nj_full). Slice to the kept modes and
        # zero-pad to the shard-aligned Nj so it matches a_u_j / parent_j /
        # weight_j / phase_c.
        R_j_at_particles = R_j_at_particles[:, self.kept_indices]
        if self.Nj_pad:
            R_j_at_particles = jnp.pad(
                R_j_at_particles, ((0, 0), (0, self.Nj_pad)))
        R_j_at_particles = R_j_at_particles.astype(self.compute_real_dtype)

        # R_j_at_particles : (N_particles, Nj)  (real)

        # Compute rho_lm at each particle's radius OUTSIDE the vmap so the
        # expensive a_u_j matmul + Gaunt reduction happen once per call,
        # not once per particle.

        #print('computing rho_lm at particle positions...')
        if self.SphHT:
            # `a_u_j_sphht` arg is ignored — the sparse JIT pulls
            # (aj, parent_j, lm_idx_per_mode) directly from `self`.
            rho_lm_at_particles_full = self.compute_rho_lm_at_particles_sphht(
                R_j_at_particles, phase_c)
            
       
        else:
            rho_lm_at_particles_full = self.compute_rho_lm_at_particles_gaunt(
                R_j_at_particles, phase_c, a_u_j, all_i, all_j, all_G, all_Lf,
            )

        # Time-averaged (diagonal-only) rho_lm at the same radii. Used as
        # the baseline for the ramp.
        rho_lm_at_particles_static = self.compute_rho_lm_at_particles_diagonal_only(R_j_at_particles)

        rho_lm_at_particles = (rho_lm_at_particles_static +
                               ramp_frac * (rho_lm_at_particles_full - rho_lm_at_particles_static))
        # rho_lm_at_particles : (N_particles, L_max_out, 2*L_max_out-1)

        #print('computing per-particle insertions and phi_lm...')

        dphi_lm_dr_at_parts, phi_lm_at_parts = jax.lax.map(
            lambda inp: self.insert_particle_rholm_and_get_philm(
                inp[0], inp[1], rho_lms),
            (positions_sph, rho_lm_at_particles),
        )
        # dphi_dr_all : (N_particles, Nmodes)
        # phi_lm_all  : (N_particles, Nmodes)


        return dphi_lm_dr_at_parts, phi_lm_at_parts

    @staticmethod
    def combine_acc(dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi, particle_r, particle_theta):

        """
        JIT-compilable: contract radial outputs with angular terms to get accelerations.
        """

        dphi_lm_dr_T = dphi_lm_dr_at_parts.T   # (Nmodes, N_particles)
        phi_lm_T  = phi_lm_at_parts.T   # (Nmodes, N_particles)

        a_r     = jnp.sum(-dphi_lm_dr_T * Ylm_all, axis=0).real                                                      # (N_particles,)
        a_theta = jnp.sum(-phi_lm_T  * dY_dtheta / particle_r[None, :], axis=0).real                             # (N_particles,)
        a_phi   = jnp.sum(-phi_lm_T  * dY_dphi   / (particle_r[None, :] * jnp.sin(particle_theta[None, :])), axis=0).real  # (N_particles,)

        return a_r, a_theta, a_phi

    def construct_acc_master_func(self, positions_sph, eval_library, eigenstate_lib, poten = False):

        '''
        Constructing acc vectors for all particles master function.
        With JIT-compilable radial eval + rho_lm + insertion, followed by a non-JIT angular contraction to get accs.
        '''

        if not hasattr(self, 'calc_rho_lm_at_parts_and_call_insert_jit'):
            self._eval_library = eval_library
            self.calc_rho_lm_at_parts_and_call_insert_jit = jax.jit(self.calc_rho_lm_at_parts_and_call_insert)
            self.combine_acc_jit = jax.jit(StellarSimTDep.combine_acc)

        #print("Compiling JIT for per-particle rho_lm insertion and acc combination...")
        dphi_lm_dr_at_parts, phi_lm_at_parts = self.calc_rho_lm_at_parts_and_call_insert_jit(
            positions_sph,
            self.current_phase,
            eigenstate_lib.radial_eigenmode_params,
            self.a_u_j,
            self._jit_all_i,
            self._jit_all_j,
            self._jit_all_G,
            self._jit_all_Lf,
            self.rho_lms,
            self.current_ramp_frac,
        )


        #print('got dphi_lm_dr_at_parts and phi_lm_at_parts from JIT-compiled function')

        # if self.plot:
        #     # Eagerly recompute rho_lm for particle 0 so matplotlib gets concrete
        #     # arrays. _construct_acc_radial is called directly here (outside JIT /
        #     # lax.map), so all arrays are concrete and plt.plot works normally.
        #     _p0_r = positions_sph[0:1, 0]
        #     _R_j_p0 = self._eval_library(_p0_r, eigenstate_lib.radial_eigenmode_params)
        #     _R_j_phased_p0 = _R_j_p0 * self.current_phase[None, :]
        #     if self.SphHT:
        #         _rho_full_p0 = self._compute_rho_lm_at_particles_sphht(_R_j_phased_p0, self.a_u_j_sphht)
        #     else:
        #         _rho_full_p0 = self._compute_rho_lm_at_particles_gaunt(
        #             _R_j_phased_p0, self.a_u_j, self._jit_all_i, self._jit_all_j,
        #             self._jit_all_G, self._jit_all_Lf,
        #         )
        #     _rho_static_p0 = self._compute_rho_lm_at_particles_static(_R_j_p0, self.M_j_t)
        #     _rho_p0 = (_rho_static_p0[0]
        #                + self.current_ramp_frac * (_rho_full_p0[0] - _rho_static_p0[0]))
        #     self._construct_acc_radial(positions_sph[0], _rho_p0,
        #                                self.rho_lms_below, self.rho_lms_above)




        #print('computing Y_lm and derivatives with jax.scipy (GPU)...')

        # On-GPU Y_lm + dY_lm/dθ via jax.scipy.special.sph_harm_y + JVP.
        # No more GPU↔host round-trip or single-threaded scipy CPU call.
        # n_max must be static for sph_harm_y under JIT; l ranges over
        # [0, L_max_out-1] in output_lm_pairs.
        Ylm_all, dY_dtheta = _compute_Ylm_and_dtheta_jit(
            self.output_lm_pairs,
            positions_sph[:, 1],
            positions_sph[:, 2],
            int(self.L_max_out - 1),
        )                                            # (Nmodes, N_particles) each

        m_vals  = self.output_lm_pairs[:, 1, None]  # (Nmodes, 1)
        dY_dphi = 1j * m_vals * Ylm_all             # (Nmodes, N_particles)

        #print('combining accs with JIT-compiled contraction...')
        if poten:
            accs = self.combine_acc_jit(
                dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
                positions_sph[:, 0], positions_sph[:, 1],
            )
            return *accs, phi_lm_at_parts, Ylm_all


        return self.combine_acc_jit(
            dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
            positions_sph[:, 0], positions_sph[:, 1],
        )


    def time_step_particle(self):

        """
        Synchronise all Simulation_Particle states into rebound, integrate
        one macro timestep, then read back and update each particle instance.
        """

        # Write current state of every particle into the rebound simulation
        for i, particle in enumerate(self.particles):
            p = self.sim_particles[i]
            p.x,  p.y,  p.z  = float(particle.r_pos[0]), float(particle.r_pos[1]), float(particle.r_pos[2])
            p.vx, p.vy, p.vz = float(particle.v[0]),     float(particle.v[1]),     float(particle.v[2])

        self._force_call_count = 0   # reset counter for this macro step
        target_time = self.sim.t + self.dt
        self.sim.integrate(target_time)
        print(f"  Force calls this timestep: {self._force_call_count}")

        # Read back and update each Simulation_Particle
        for i, particle in enumerate(self.particles):
            p = self.sim_particles[i]
            particle.update_state(
                [p.x,  p.y,  p.z],
                [p.vx, p.vy, p.vz]
            )
            #print(f"  Particle {i}: r = {float(particle.r_pos_sph[0]) * self.u.to_Kpc:.4f} kpc")


    def run_simulation(self):

        start = time()
        aj = self.initialising_simulation()
        end = time()
        self.aj = aj


        Nr = len(self.r)        # number of background radial bins
        all_idx = jnp.arange(Nr + 1)   # indices 0 .. Nr
        self.Nr = Nr
        self.all_idx = all_idx


        # r_below / r_above are no longer needed — `self.r[gather_idx]`
        # inside the JIT'd insert handles the radial-grid shift directly.

        L_max_out = 2 * self.L - 1  # captures all density harmonics up to l1+l2 <= 2*(L-1)
        self.L_max_out = L_max_out


        if self.SphHT == False:

            # Precompute Gaunt table ONCE — reuse this across all time steps
            gaunt_table = gf.precompute_gaunt_table(self.lm_l, self.lm_m, L_max_out)
            self.gaunt_table = gaunt_table

            _, _, _, _, unique_lm = gaunt_table

            self.lm_idx_sorted_per_mode = gf.make_lm_idx_sorted_per_mode(
                self.lm_l_per_mode, self.lm_m_per_mode, unique_lm)


            Nj = len(self.eigen_energies)
            N_unique = len(unique_lm)
            # CPU-numpy scatter (avoids GPU peak-doubling) + optional Nj shard.
            self.a_u_j = self._build_sparse_au_j(
                self.lm_idx_sorted_per_mode, self.parent_j, self.aj,
                N_unique, Nj,
            )


            self._jit_all_i, self._jit_all_j, self._jit_all_G, self._jit_all_Lf, _ = gaunt_table

        else:

            # SphHT branch is fully sparse: (aj, parent_j, lm_idx_per_mode)
            # live on `self` directly. Gaunt-path placeholders still need to
            # be present because they appear in the JIT signature.
            self.a_u_j       = jnp.zeros((1, 1), dtype=self.aj.dtype)
            self._jit_all_i  = jnp.zeros(1, dtype=jnp.int32)
            self._jit_all_j  = jnp.zeros(1, dtype=jnp.int32)
            self._jit_all_G  = jnp.zeros(1, dtype=jnp.float64)
            self._jit_all_Lf = jnp.zeros(1, dtype=jnp.int32)


        # Pre-convert lm_pairs to numpy once so scipy sph_harm_y receives a
        # plain numpy array and avoids a GPU to CPU device transfer every sub-step.
        out_lm = [(L, M) for L in range(L_max_out) for M in range(-L, L+1)]
        output_lm_pairs = jnp.array(out_lm)

        self.output_lm_pairs = output_lm_pairs
        self.lm_pairs_np = np.array(output_lm_pairs)

        # Ramp phase: linearly switch on the off-diagonal cross-terms — i.e.
        # the time-dependent piece — over `ramp_time` so particles are not
        # abruptly exposed to the full fluctuating spectrum. After the ramp
        # the system evolves for `total_evolve_time` in the full ULDM
        # potential.
        ramp_time = self.ramp_time * self.u.from_Gyr
        self.n_ramp_steps = int(jnp.ceil(ramp_time / self.dt).item())

        # Update no_time_steps to include ramp steps on top of the original
        # main-phase steps.
        self.no_time_steps = self.n_ramp_steps + self.no_time_steps
        print(f"Ramp phase: {self.n_ramp_steps} steps "
              f"({self.n_ramp_steps * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Main phase: {self.no_time_steps - self.n_ramp_steps} steps "
              f"({(self.no_time_steps - self.n_ramp_steps) * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Total: {self.no_time_steps} steps")


        # rho_static_r_l00 (the time-averaged diagonal monopole, used as the
        # ramp baseline) was precomputed in initialising_simulation;
        # Build_rho_lms_for_timestep needs it in self. Result is cast to
        # compute_dtype and sharded along L.
        self.rho_lms = self._shard_l(self.Build_rho_lms_for_timestep(0).astype(self.compute_dtype))
        print('completed rho_lms precomputation')
        # Change 2: rho_lms_below / rho_lms_above are no longer pre-built.
        # The gather inside `insert_particle_rholm_and_get_philm` consumes
        # `self.rho_lms` directly, saving 2x the standing rho_lms memory.


        ###################################

        # if self.plot == True:

        #     def inverse_s2fft(single_rho_lm_r):
        #         return s2fft.inverse(single_rho_lm_r, self.L_max_out, sampling='mw', method='jax')

        #     rho_rtp = jax.vmap(inverse_s2fft)(self.rho_lms)  # (Nr, L, 2*L-1)


        #     import yt
        #     from yt.visualization.volume_rendering.api import (
        #         Scene, 
        #         Camera, 
        #         TransferFunctionHelper, 
        #         create_volume_source
        #     )

        #     def rho_rtp_to_cart(rho_rtp, r, theta, phi, Ncart=None):
        #         import numpy as np
        #         from scipy.interpolate import RegularGridInterpolator

        #         if Ncart is None:
        #             Ncart = len(r)

        #         r = np.asarray(r)
        #         theta = np.asarray(theta)
        #         phi = np.asarray(phi)
        #         rho_rtp = np.asarray(rho_rtp)

        #         r_max = r[-1]
        #         r_min = r[0]

        #         # Interpolate in log-r space since r is logspaced
        #         log_r = np.log10(r)
        #         interp = RegularGridInterpolator(
        #             (log_r, theta, phi), rho_rtp,
        #             bounds_error=False, fill_value=0.0
        #         )

        #         # Cartesian grid
        #         x = np.linspace(-r_max, r_max, Ncart)
        #         y = np.linspace(-r_max, r_max, Ncart)
        #         z = np.linspace(-r_max, r_max, Ncart)
        #         X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        #         # Cartesian -> spherical
        #         R = np.sqrt(X**2 + Y**2 + Z**2)
        #         Theta = np.arccos(np.clip(Z / np.clip(R, 1e-30, None), -1, 1))
        #         Phi = np.arctan2(Y, X) % (2 * np.pi)

        #         # Interpolate in log-r space
        #         log_R = np.log10(np.clip(R, r_min, None))
        #         pts = np.stack([log_R.ravel(), Theta.ravel(), Phi.ravel()], axis=-1)
        #         rho_xyz = interp(pts).reshape(X.shape)

        #         return rho_xyz, x, y, z

        #     rho_xyz, x, y, z = rho_rtp_to_cart(rho_rtp, self.r, self.theta, self.phi)


        #     ds = yt.load_uniform_grid(
        #     dict(density=np.asarray(rho_xyz) * float(self.u.to_Msun)/float(self.u.to_Kpc)**3),
        #     [1000,1000,1000],
        #     bbox=np.array([[-self.rmax, self.rmax], [-self.rmax, self.rmax], [-self.rmax, self.rmax]]) * float(self.u.to_Kpc),
        #     length_unit="kpc",
        #     mass_unit="Msun"
        #     )

        #     ds_section = ds.sphere(ds.domain_center,((self.rmax * self.u.to_Kpc).item(),"kpc"))
        #     sc = yt.create_scene(ds_section, ("stream", "density"), "perspective")
        #     source = sc.get_source()
        #     source.set_log(True)
        #     bounds=(1e-2, 3e5)
                
        #     tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=False)

        #     def quadramp(vals, minval, maxval):
        #         return ((vals - vals.min()) / (vals.max() - vals.min()))**0.5

        #     tf.map_to_colormap(
        #         np.log10(bounds[0]), np.log10(bounds[1]), 
        #         colormap="gist_stern", 
        #         scale_func=quadramp
        #     )

            
        #     tf.add_layers(8,
        #                 colormap="gist_stern", 
        #                 alpha=np.geomspace(1, 6, 8))

        #     source.tfh.tf = tf
        #     source.tfh.bounds = bounds

        #     camera = sc.camera
        #     camera.position = [1.,0,0]
        #     camera.resolution = (900,900)
        #     camera.zoom(1.)

        #     camera.switch_orientation()
        #     import matplotlib.pyplot as plt
        #     import matplotlib.colors as mcolors

        #     # Render the scene to an image array
        #     im = sc.render()

        #     # Plot with matplotlib so we can add a colorbar
        #     fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        #     ax.imshow(im[:, :, :3] / im[:, :, :3].max(), origin="lower")
        #     ax.set_axis_off()

        #     # Add colorbar matching your transfer function bounds
        #     norm = mcolors.LogNorm(vmin=bounds[0], vmax=bounds[1])
        #     sm = plt.cm.ScalarMappable(cmap="gist_stern", norm=norm)
        #     sm.set_array([])
        #     cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        #     cbar.set_label(r"Density [M$_\odot$ / kpc$^3$]")

        #     plt.tight_layout()
        #     plt.show()

        ######################################


        # Compute initial potential energy for each particle
                
        r_pos_sphs = jnp.array([particle.r_pos_sph for particle in self.particles])  # (N_particles, 3)

        a_r, _, _, phi_lm_at_r, Y_lm_all = self.construct_acc_master_func(
            r_pos_sphs,
            self.autodiff_data['eval_library'],
            self.autodiff_data['eigenstate_lib'],
            poten = True
        )


        #v_circ_true = jnp.sqrt(jnp.abs(a_r) * r_pos_sphs[:, 0])

        phi_at_parts = jnp.sum(phi_lm_at_r * Y_lm_all.T, axis=1)  # (N_particles,)

        for i, particle in enumerate(self.particles):

            #v_old = particle.v
            #v_dir = v_old / jnp.linalg.norm(v_old)
            #v_new = v_dir * v_circ_true[i]
            #p = self.ps_step[i]
            #p.vx, p.vy, p.vz = float(v_new[0]), float(v_new[1]), float(v_new[2])
            particle.potential_energy.append(phi_at_parts[i].real)

            #particle.Change_to_new_vel(v_new)

        # Preallocate the per-particle velocity history array now that
        # no_time_steps is final (after dt_override + ramp adjustments).
        for particle in self.particles:
            particle.Create_V_array(self.no_time_steps)

        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")


            # Cast to compute_dtype and (optionally) shard along L. The
            # gather-based insert in `insert_particle_rholm_and_get_philm`
            # consumes this directly — no rho_lms_below / rho_lms_above.
            # Free the previous step's rho_lms *before* building the new one —
            # otherwise the RHS allocates the full new (Nr, L, 2L-1) array
            # while the old one is still live, doubling peak memory.
            print('Building rho_lms for this timestep...')
            start = time()
            self.rho_lms = None
            self.rho_lms = self._shard_l(
                self.Build_rho_lms_for_timestep(self.time_step).astype(self.compute_dtype)
            )
            end = time()
            print(f"rho_lms built in {end - start:.2f} seconds")


            # Time step all particles (IAS15 calls additional_forces_step ~8× internally,
            # which loops over every particle each call)
            start = time()
            self.time_step_particle()
            end = time()
            print(f"Time stepping all particles completed in {end - start:.2f} seconds")


            self.current_phase = jnp.exp(-1j * self.eigen_energies * (self.time_step + 1) * self.dt / 1)
            # Compute phi at updated particle positions for potential energy tracking
            r_pos_sphs_new = jnp.array([p.r_pos_sph for p in self.particles])
            _, _, _, phi_lm_new, Ylm_new = self.construct_acc_master_func(
                r_pos_sphs_new,
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib'],
                poten=True
            )

            phi_at_parts = jnp.sum(phi_lm_new * Ylm_new.T, axis=1).real  # (N_particles,)
            for i, particle in enumerate(self.particles):
                particle.potential_energy.append(float(phi_at_parts[i]))


            self.time_step += 1


    def run_simulation_profiled(self, time_output='profile_time.prof',
                                memory_output='profile_memory.txt', top_n=30):
        """Run `run_simulation` under cProfile (time) and tracemalloc (memory).

        cProfile output goes to `time_output` (binary). Inspect with e.g.
            python -m pstats profile_time.prof
            # or interactively: snakeviz profile_time.prof
        A console summary of the top `top_n` functions is printed for both
        cumulative-time and self-time sorts.

        tracemalloc takes a single snapshot at the end of the run and writes
        the top `top_n` allocation sites (by total size) to `memory_output`.
        Peak/current allocated bytes are also reported.

        Caveat for JAX: most heavy work here is JIT'd and dispatched
        asynchronously, so cProfile timings reflect Python-side overhead +
        time blocking on async results. For pure kernel timing, use
        `jax.profiler` or insert `jax.block_until_ready(...)` at key points.
        """
        import cProfile
        import pstats
        import tracemalloc

        tracemalloc.start()
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            self.run_simulation()
        finally:
            profiler.disable()
            snapshot = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        profiler.dump_stats(time_output)

        stats = pstats.Stats(profiler).strip_dirs()
        print(f"\n===== cProfile: top {top_n} by cumulative time =====")
        stats.sort_stats('cumulative').print_stats(top_n)
        print(f"\n===== cProfile: top {top_n} by self (tottime) =====")
        stats.sort_stats('tottime').print_stats(top_n)

        top_stats = snapshot.statistics('lineno')
        header = (f"tracemalloc: peak = {peak / 1e6:.2f} MB, "
                  f"current = {current / 1e6:.2f} MB")
        print(f"\n===== {header} =====")
        print(f"Top {top_n} allocation sites (by size):")
        with open(memory_output, 'w') as f:
            f.write(header + "\n\n")
            f.write(f"Top {top_n} allocation sites (by size):\n")
            for stat in top_stats[:top_n]:
                line = str(stat)
                print(line)
                f.write(line + "\n")

        print(f"\nFull cProfile data written to: {time_output}")
        print(f"Memory summary written to:     {memory_output}")
