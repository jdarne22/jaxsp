
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)


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

import numpy as np
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jaxsp.constants import GN

import gaunt_funcs as gf
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


def sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R, N_unique, k_batch):
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
    # Deterministic segment-sum over the (l,m) bins, replacing the previous
    # atomic scatter-add (`S.at[u_b, :].add(...)` inside a k-batched scan).
    #
    # Why the change: the atomic scatter-add accumulates many radial modes
    # into each (l,m) bin via GPU atomics, whose ordering is not reproducible
    # across kernel launches. At complex64 that seeds ~1e-5 run-to-run noise,
    # which the chaotic stellar dynamics amplify to O(1) by late times (worse
    # at high m22, where there are more modes per bin and the orbits are more
    # chaotic). `jax.ops.segment_sum` does NOT fix this here — even with
    # `indices_are_sorted=True` it still lowers to atomics (nondeterministic),
    # and on this jaxlib it segfaults for complex inputs.
    #
    # Instead: sort modes by their bin, take a prefix sum (cumsum — fixed
    # reduction order, bit-reproducible), and difference at the segment
    # boundaries. This is deterministic by construction (verified diff=0) and
    # ~3x FASTER than the atomic scatter. `bincount` is integer-valued so its
    # accumulation order is irrelevant.
    #
    # Memory: this materialises `contrib` of shape `(M, Nmodes_k)` rather than
    # the old `(M, k_batch)` per-scan-step. `M` is the caller's r_chunk (or
    # N_particles), so lower `r_chunk_size` to bound the transient at very
    # high m22. `k_batch` is retained for call-site compatibility (unused).
    del k_batch
    M = R.shape[0]
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R.dtype)

    # contrib[:, k] = aj[k] * phase[parent_j[k]] * R[:, parent_j[k]]
    contrib = (aj * phase_c[parent_j]).astype(cdtype)[None, :] * R[:, parent_j]  # (M, Nmodes_k)

    order = jnp.argsort(lm_idx)                                    # fixed permutation
    cs = jnp.cumsum(contrib[:, order], axis=1)                     # (M, Nmodes_k)
    cs = jnp.concatenate([jnp.zeros((M, 1), cs.dtype), cs], axis=1)  # prepend 0 -> index by end+1
    counts  = jnp.bincount(lm_idx, length=N_unique)                # modes per (l,m) bin
    seg_end = jnp.cumsum(counts)                                   # prefix-end index into `cs`
    totals  = cs[:, seg_end]                                       # (M, N_unique) cumulative incl bin u
    S = jnp.diff(totals, axis=1, prepend=jnp.zeros((M, 1), cs.dtype))  # (M, N_unique)
    return S.T                                                     # (N_unique, M)


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch", "r_chunk"))
def build_sphht_rho_rtp_jit(R_j_r_fixed, phase_c, aj, parent_j, lm_idx,
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
        S_chunk = sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R_chunk,
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
def build_sphht_rho_lms_jit(R_j_r_fixed, phase_c, aj, parent_j, lm_idx,
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
        S_chunk = sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R_chunk,
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
def compute_Ylm_and_dtheta_jit(lm_pairs, theta_arr, phi_arr, n_max):
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
def compute_rho_lm_at_particles_sphht_jit(R_j_at_parts, phase_c, aj, parent_j,
                                            lm_idx, lm_pairs, total_mass,
                                            L_out, N_unique, k_batch):
    """JIT'd per-particle rho_lm via sparse a_u_j × R_j_at_parts.

    Equivalent to the old dense `einsum('uj,j,pj->pu', a_u_j_sphht, ...)`
    followed by per-particle inverse-SHT, |.|^2, forward-SHT.
    """
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_at_parts.dtype)

    # (N_unique, Np) — same kernel, just M = N_particles.
    S_up = sparse_a_u_j_matmul(aj, parent_j, lm_idx, phase_c, R_j_at_parts,
                                  N_unique, k_batch)
    S_pu = S_up.T                                                          # (Np, N_unique)

    def single(S_u):
        flm = jnp.zeros((L_out, 2 * L_out - 1), dtype=cdtype)
        flm = flm.at[lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(S_u)
        psi_at_r = s2fft.inverse(flm, L_out, sampling='mw', method='jax')
        rho_at_r = total_mass * jnp.abs(psi_at_r) ** 2
        return s2fft.forward(rho_at_r, L_out, sampling='mw', method='jax')

    return jax.vmap(single)(S_pu)

