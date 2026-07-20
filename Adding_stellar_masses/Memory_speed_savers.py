
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
from typing import NamedTuple

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

    Vectorized over numpy. For large eigenbases (l_max ~ 1e3, total
    k-modes sum(2l+1) ~ 1e9) the original pure-Python loop + dict boxed
    every (l, m, j) triple as a separate Python int/tuple object, which
    blew past 256GB of host RAM and took ~1hr before OOM-killing the job.
    This does the same bookkeeping with bulk array ops in seconds and a
    few GB.
    '''
    l = np.asarray(l)
    n_j = l.shape[0]
    n_per_j = (2 * l + 1).astype(np.int64)
    total_k = int(n_per_j.sum())

    # j-index and l for every k-mode (each j contributes 2l+1 k-modes,
    # m = -l..l).
    ind_for_jmode_over_all_k = np.repeat(np.arange(n_j, dtype=np.int32), n_per_j)
    l_for_kmode = np.repeat(l, n_per_j).astype(np.int32)

    block_start = np.cumsum(n_per_j) - n_per_j
    within_block = np.arange(total_k, dtype=np.int64) - np.repeat(block_start, n_per_j)
    m_for_kmode = (within_block - l_for_kmode).astype(np.int32)

    # Unique (l, m) pairs only depend on which l values occur (<= l_max+1
    # of them), not on how many j's share each l, so build them directly
    # instead of de-duplicating the full total_k-length k-mode table.
    unique_l = np.unique(l).astype(np.int64)
    n_per_unique_l = 2 * unique_l + 1
    pair_block_start = np.cumsum(n_per_unique_l) - n_per_unique_l
    lm_pairs_l = np.repeat(unique_l, n_per_unique_l)
    within_pair_block = np.arange(lm_pairs_l.shape[0], dtype=np.int64) - np.repeat(pair_block_start, n_per_unique_l)
    lm_pairs_m = within_pair_block - lm_pairs_l
    lm_pairs_np = np.stack([lm_pairs_l, lm_pairs_m], axis=1).astype(np.int32)

    # Index into lm_pairs_np for every k-mode: that l's block start, plus
    # position within the block (m - (-l) = m + l).
    l_to_block_start = np.zeros(int(unique_l.max()) + 1, dtype=np.int64)
    l_to_block_start[unique_l] = pair_block_start
    lm_pairs_idx_for_kmode = (
        l_to_block_start[l_for_kmode] + (m_for_kmode.astype(np.int64) + l_for_kmode)
    ).astype(np.int32)

    L = int(l.max()) + 1
    L_max_out = 2 * L - 1
    n_theta = L_max_out
    n_phi = 2 * L_max_out - 1

    i = np.arange(n_theta)
    theta_np = (np.pi * (2 * i + 1)) / (2 * L_max_out - 1)
    j = np.arange(n_phi)
    phi_np = (2 * np.pi * j) / (2 * L_max_out - 1)

    return (jnp.asarray(ind_for_jmode_over_all_k), jnp.asarray(lm_pairs_np),
            jnp.asarray(l_for_kmode), jnp.asarray(m_for_kmode),
            jnp.asarray(theta_np), jnp.asarray(phi_np),
            jnp.asarray(lm_pairs_idx_for_kmode))


class BinBlocks(NamedTuple):
    """Setup-time bin-packing plan for `sparse_a_u_j_matmul`.

    `perm` sorts the k-modes by their (l,m) bin. The remaining arrays
    describe a partition of the *sorted* mode axis into `n_blocks`
    contiguous slices, each holding at most `k_block` modes and, crucially,
    only *whole* (l,m) bins — no bin ever straddles a block boundary.

    perm       : (Nmodes_k,) int64  — argsort(lm_idx), applied at setup
    block_start: (n_blocks,)  int32 — first sorted-mode index of each block
    bin_lo/hi  : (n_blocks, B) int32 — bin bounds *local* to the block, as
                 offsets into the block's prepended-zero cumsum
    bin_id     : (n_blocks, B) int32 — destination row in `S`; padding
                 entries point at the dead row `N_unique` with lo == hi == 0
    k_block    : int — static block width (== `sparse_k_batch`)
    """
    perm: np.ndarray
    block_start: jnp.ndarray
    bin_lo: jnp.ndarray
    bin_hi: jnp.ndarray
    bin_id: jnp.ndarray
    k_block: int

    def as_arrays(self):
        """The four device arrays, in the order `sparse_a_u_j_matmul` wants."""
        return (self.block_start, self.bin_lo, self.bin_hi, self.bin_id)


def precompute_bin_blocks(lm_idx, N_unique, k_block, max_bins_per_block=4096):
    """Build the `BinBlocks` plan. Pure numpy — runs once, at setup.

    Everything here used to happen *inside* the jit as `jnp.argsort(lm_idx)`
    and `jnp.bincount(lm_idx)`. Because `lm_idx` reaches the trace as a
    closed-over constant (`self.lm_idx_per_mode` under a bound-method jit),
    XLA constant-folded the sort on the host: a single-threaded stable sort
    of 58.9M elements at m22=50, which took over an hour of compile time
    while the other ranks sat in the collective rendezvous. Hoisting it here
    costs one numpy argsort at startup and removes the op from the graph.

    Blocks are packed greedily and aligned to bin boundaries, so each (l,m)
    bin is written exactly once across the whole scan. That preserves the
    bit-reproducibility the cumsum approach was chosen for, without needing
    a cumsum over the full `Nmodes_k` axis.
    """
    lm_idx = np.asarray(lm_idx)
    Nmodes = lm_idx.shape[0]

    perm = np.argsort(lm_idx, kind='stable')
    counts = np.bincount(lm_idx[perm], minlength=N_unique)
    if counts.max() > k_block:
        raise ValueError(
            f"k_block={k_block} is smaller than the largest (l,m) bin "
            f"({counts.max()} modes). Blocks must hold whole bins; raise "
            f"`sparse_k_batch` to at least {int(counts.max())}.")

    starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)  # (N_unique+1,)

    # Greedy pack: extend the block while it holds whole bins, stays within
    # `k_block` modes, and stays within `max_bins_per_block` bins.
    spans = []
    u0 = 0
    while u0 < N_unique:
        u1 = u0
        base = starts[u0]
        while (u1 < N_unique
               and starts[u1 + 1] - base <= k_block
               and u1 - u0 < max_bins_per_block):
            u1 += 1
        assert u1 > u0, "empty block — bin larger than k_block slipped through"
        spans.append((u0, u1))
        u0 = u1

    n_blocks = len(spans)
    B = max(u1 - u0 for u0, u1 in spans)

    block_start = np.zeros(n_blocks, np.int32)
    bin_lo = np.zeros((n_blocks, B), np.int32)
    bin_hi = np.zeros((n_blocks, B), np.int32)
    bin_id = np.full((n_blocks, B), N_unique, np.int32)   # dead row for padding

    for i, (u0, u1) in enumerate(spans):
        st = starts[u0]
        nb = u1 - u0
        block_start[i] = st
        bin_lo[i, :nb] = starts[u0:u1] - st
        bin_hi[i, :nb] = starts[u0 + 1:u1 + 1] - st
        bin_id[i, :nb] = np.arange(u0, u1)

    print(f"[sparse_a_u_j] {Nmodes} k-modes → {n_blocks} blocks of ≤{k_block} "
          f"modes / ≤{B} bins. Peak transient per block: M × {k_block}.")

    return BinBlocks(
        perm=perm,
        block_start=jnp.asarray(block_start),
        bin_lo=jnp.asarray(bin_lo),
        bin_hi=jnp.asarray(bin_hi),
        bin_id=jnp.asarray(bin_id),
        k_block=int(k_block),
    )


def fold_phase_into_aj(aj_sorted, parent_j_sorted, phase_c, R_dtype):
    """Compute `sparse_a_u_j_matmul`'s `aj_phase` once, for callers that invoke
    it repeatedly with the same `(aj, parent_j, phase_c)` but different `R`.

    `aj_phase` is loop-invariant across an r-chunk loop: it depends only on the
    amplitudes, the per-j phase, and `cdtype` — never on `R`'s *values*, only on
    its dtype. Recomputing it per chunk costs a random gather over all Nmodes_k
    (201M elements / ~2.4 GB of traffic at m22=50) for nothing, and XLA will not
    reliably hoist an intermediate that large out of a `lax.scan` body.

    `cdtype` is derived from exactly the same three dtypes as the in-function
    path, so passing the result back into `sparse_a_u_j_matmul(..., aj_phase=…)`
    is bit-for-bit identical to letting it recompute.
    """
    cdtype = jnp.result_type(aj_sorted.dtype, phase_c.dtype, R_dtype)
    return (aj_sorted * phase_c[parent_j_sorted]).astype(cdtype)


def sparse_a_u_j_matmul(aj_sorted, parent_j_sorted, blocks, phase_c, R,
                        N_unique, k_batch, aj_phase=None):
    """Sparse equivalent of `(a_u_j_dense @ (R * phase[None, :]).T)`.

    The dense `a_u_j` is a one-hot scatter: each k-mode (n, l, m) writes
    `aj[k]` to row `lm_idx[k]`, column `parent_j[k]`. So `a_u_j` has
    `Nmodes_k = sum_j (2 l_j + 1)` nonzeros in `N_unique * Nj ≈ m22**5`
    slots — typically <1% dense. We never materialise it.

    Inputs
    ------
    aj_sorted       : (Nmodes_k + k_batch,) complex — amplitudes per k-mode,
                      permuted by `blocks.perm` and zero-padded by `k_batch`
    parent_j_sorted : (Nmodes_k + k_batch,) int — j-index per k-mode, same
                      permutation, zero-padded (padded amplitudes are 0, so
                      the column they gather is irrelevant)
    blocks          : 4-tuple `(block_start, bin_lo, bin_hi, bin_id)` from
                      `precompute_bin_blocks(...).as_arrays()`
    phase_c         : (Nj,) complex — e^{-i E_j t}, already cast to cdtype
    R               : (M, Nj) — `R_j_r_fixed` (M = Nr) or `R_j_at_parts` (M = Np)
    N_unique        : int (static) — number of distinct (l,m) pairs
    k_batch         : int (static) — block width; must equal `blocks.k_block`

    Output: `S` of shape (N_unique, M), equal to
            Σ_k δ(u, lm_idx[k]) · aj[k] · phase[parent_j[k]] · R[:, parent_j[k]]
    """
    # Deterministic segment-sum over the (l,m) bins, replacing the original
    # atomic scatter-add (`S.at[u_b, :].add(...)` inside a k-batched scan).
    #
    # Why not atomics: the scatter-add accumulates many radial modes into each
    # (l,m) bin via GPU atomics, whose ordering is not reproducible across
    # kernel launches. At complex64 that seeds ~1e-5 run-to-run noise, which
    # the chaotic stellar dynamics amplify to O(1) by late times (worse at
    # high m22, where there are more modes per bin and the orbits are more
    # chaotic). `jax.ops.segment_sum` does NOT fix this — even with
    # `indices_are_sorted=True` it still lowers to atomics, and on this jaxlib
    # it segfaults for complex inputs.
    #
    # Instead: with modes pre-sorted by bin (done at setup, see
    # `precompute_bin_blocks`), scan over bin-aligned blocks of `k_batch`
    # modes. Per block, take a prefix sum — fixed reduction order, hence
    # bit-reproducible — and difference it at the bin boundaries. Because
    # blocks never split a bin, every bin is written exactly once, so the
    # `.at[].add()` below never has two live contributions racing for the
    # same row; only the dead padding row `N_unique` sees collisions, and it
    # is discarded.
    #
    # Memory: peak transient is `(M, k_batch)`, not `(M, Nmodes_k)`. The
    # latter is 47 GB at m22=50 with 100 particles — it is what produced the
    # `Failed to allocate 12.39GiB` in the m22=50 run.
    #
    # Accuracy: differencing a cumsum over ≤ k_batch terms rather than over
    # all 58.9M is also strictly better conditioned — the old form subtracted
    # two large prefix sums to recover a small bin sum.
    if not isinstance(blocks, (tuple, list)) or len(blocks) != 4:
        raise TypeError(
            "sparse_a_u_j_matmul now takes a bin-block plan, not `lm_idx`. "
            "Pass `precompute_bin_blocks(lm_idx, N_unique, k_batch).as_arrays()` "
            "and permute `aj`/`parent_j` by its `.perm` (see `_sort_modes_for_sphht`).")

    M = R.shape[0]
    cdtype = jnp.result_type(aj_sorted.dtype, phase_c.dtype, R.dtype)

    # Fold the per-j phase into the amplitudes once, outside the scan. Callers
    # that loop over r-chunks should hoist this further still, via
    # `fold_phase_into_aj`, and pass the result in — it does not vary with `R`.
    if aj_phase is None:
        aj_phase = fold_phase_into_aj(aj_sorted, parent_j_sorted, phase_c, R.dtype)

    zeros_col = jnp.zeros((M, 1), cdtype)

    def block_body(S, blk):
        st, lo, hi, bid = blk
        a_b = jax.lax.dynamic_slice(aj_phase, (st,), (k_batch,))        # (k_batch,)
        p_b = jax.lax.dynamic_slice(parent_j_sorted, (st,), (k_batch,))  # (k_batch,)

        # contrib[:, k] = aj[k] * phase[parent_j[k]] * R[:, parent_j[k]]
        contrib = a_b[None, :] * R[:, p_b]                               # (M, k_batch)

        cs = jnp.cumsum(contrib, axis=1)
        cs = jnp.concatenate([zeros_col, cs], axis=1)                    # (M, k_batch+1)
        sums = cs[:, hi] - cs[:, lo]                                     # (M, B)

        return S.at[bid, :].add(sums.T), None

    S0 = jnp.zeros((N_unique + 1, M), cdtype)                            # +1 = dead row
    S, _ = jax.lax.scan(block_body, S0, blocks)
    return S[:N_unique]


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch", "r_chunk"))
def build_sphht_rho_rtp_jit(R_j_r_fixed, phase_c, aj, parent_j, blocks,
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

    # Loop-invariant: hoist out of the per-chunk map (see `fold_phase_into_aj`).
    aj_phase = fold_phase_into_aj(aj, parent_j, phase_c, rdtype)

    def chunk_body(R_chunk):
        # R_chunk: (r_chunk, Nj)
        S_chunk = sparse_a_u_j_matmul(aj, parent_j, blocks, phase_c, R_chunk,
                                         N_unique, k_batch, aj_phase=aj_phase)  # (N_unique, r_chunk)

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
def build_sphht_rho_lms_jit(R_j_r_fixed, phase_c, aj, parent_j, blocks,
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

    # Loop-invariant: hoist out of the per-chunk scan (see `fold_phase_into_aj`).
    # At m22=50 this was a 201M-element gather redone once per chunk.
    aj_phase = fold_phase_into_aj(aj, parent_j, phase_c, rdtype)

    def chunk_body(R_chunk):
        S_chunk = sparse_a_u_j_matmul(aj, parent_j, blocks, phase_c, R_chunk,
                                         N_unique, k_batch, aj_phase=aj_phase)  # (N_unique, r_chunk)
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
                                            blocks, lm_pairs, total_mass,
                                            L_out, N_unique, k_batch):
    """JIT'd per-particle rho_lm via sparse a_u_j × R_j_at_parts.

    Equivalent to the old dense `einsum('uj,j,pj->pu', a_u_j_sphht, ...)`
    followed by per-particle inverse-SHT, |.|^2, forward-SHT.
    """
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_at_parts.dtype)

    # (N_unique, Np) — same kernel, just M = N_particles.
    S_up = sparse_a_u_j_matmul(aj, parent_j, blocks, phase_c, R_j_at_parts,
                                  N_unique, k_batch)
    S_pu = S_up.T                                                          # (Np, N_unique)

    def single(S_u):
        flm = jnp.zeros((L_out, 2 * L_out - 1), dtype=cdtype)
        flm = flm.at[lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(S_u)
        psi_at_r = s2fft.inverse(flm, L_out, sampling='mw', method='jax')
        rho_at_r = total_mass * jnp.abs(psi_at_r) ** 2
        return s2fft.forward(rho_at_r, L_out, sampling='mw', method='jax')

    return jax.vmap(single)(S_pu)

