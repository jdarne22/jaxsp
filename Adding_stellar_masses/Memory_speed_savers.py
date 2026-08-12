
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)

import functools
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
import s2fft
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P



# Vocabulary used throughout this file:
#   k-mode : one basis function, labelled by a radial mode j and angular
#            numbers (l, m). Each radial mode j contributes 2l+1 k-modes.
#   (l, m) bin : all the k-modes (across every radial mode j) that share
#            the same l and m.


def precompute_lm_pairs(l):
    """
    Works out, for every k-mode, which radial mode j it belongs to and
    what its l and m are - plus a table of every distinct (l, m) pair
    that appears, and an index from each k-mode into that table.

    Vectorised with numpy. A pure-Python loop over this (roughly 1e9
    k-modes at high l_max) used to box every (l, m, j) triple as a
    separate Python object and blew past 256GB of RAM; this does the same
    bookkeeping with bulk array operations in seconds and a few GB.

    Returns
    -------
    parent_j_per_k_mode        : (n_k_modes,) int - radial mode j of each k-mode
    lm_pairs                   : (n_unique_lm_pairs, 2) int - every distinct [l, m]
    l_per_k_mode, m_per_k_mode : (n_k_modes,) int - l and m of each k-mode
    theta, phi                 : McEwen-Wiaux angular grid sized for the
                                  largest l present
    lm_index_per_k_mode        : (n_k_modes,) int - row of `lm_pairs` each
                                  k-mode maps to
    """
    l = np.asarray(l)
    n_radial_modes = l.shape[0]
    k_modes_per_radial_mode = (2 * l + 1).astype(np.int64)
    n_k_modes = int(k_modes_per_radial_mode.sum())

    # Radial mode 0's k-modes come first, then radial mode 1's, and so on.
    parent_j_per_k_mode = np.repeat(np.arange(n_radial_modes, dtype=np.int32), k_modes_per_radial_mode)
    l_per_k_mode = np.repeat(l, k_modes_per_radial_mode).astype(np.int32)

    # Within radial mode j's block of 2l+1 k-modes, m runs from -l to l in order.
    radial_mode_block_start = np.cumsum(k_modes_per_radial_mode) - k_modes_per_radial_mode
    position_in_radial_block = (
        np.arange(n_k_modes, dtype=np.int64) - np.repeat(radial_mode_block_start, k_modes_per_radial_mode)
    )
    m_per_k_mode = (position_in_radial_block - l_per_k_mode).astype(np.int32)

    # The distinct (l, m) pairs only depend on which l values occur (at
    # most l_max + 1 of them), not on how many radial modes share each l -
    # so build the table directly instead of de-duplicating every k-mode.
    distinct_l_values = np.unique(l).astype(np.int64)
    pairs_per_distinct_l = 2 * distinct_l_values + 1
    lm_block_start = np.cumsum(pairs_per_distinct_l) - pairs_per_distinct_l
    lm_pairs_l = np.repeat(distinct_l_values, pairs_per_distinct_l)
    position_in_lm_block = (
        np.arange(lm_pairs_l.shape[0], dtype=np.int64) - np.repeat(lm_block_start, pairs_per_distinct_l)
    )
    lm_pairs_m = position_in_lm_block - lm_pairs_l
    lm_pairs = np.stack([lm_pairs_l, lm_pairs_m], axis=1).astype(np.int32)

    # Row in `lm_pairs` for every k-mode: that l's block start, plus
    # position within the block (m - (-l) = m + l).
    l_to_lm_block_start = np.zeros(int(distinct_l_values.max()) + 1, dtype=np.int64)
    l_to_lm_block_start[distinct_l_values] = lm_block_start
    lm_index_per_k_mode = (
        l_to_lm_block_start[l_per_k_mode] + (m_per_k_mode.astype(np.int64) + l_per_k_mode)
    ).astype(np.int32)

    # McEwen-Wiaux ("mw") sampling grid, sized for the largest l present.
    L = int(l.max()) + 1
    L_max_out = 2 * L - 1
    n_theta = L_max_out
    n_phi = 2 * L_max_out - 1
    theta = (np.pi * (2 * np.arange(n_theta) + 1)) / n_phi
    phi = (2 * np.pi * np.arange(n_phi)) / n_phi

    return (jnp.asarray(parent_j_per_k_mode), jnp.asarray(lm_pairs),
            jnp.asarray(l_per_k_mode), jnp.asarray(m_per_k_mode),
            jnp.asarray(theta), jnp.asarray(phi),
            jnp.asarray(lm_index_per_k_mode))


class BinBlocks(NamedTuple):
    """
    A plan for summing k-modes into (l,m) bins without GPU atomics (see
    precompute_bin_blocks for why atomics are a problem here).

    perm: sorts all k-modes so that modes sharing a bin sit next to each
    other. The rest describes how that sorted list is cut into
    fixed-size blocks, where every block holds only *whole* bins - never
    part of one:

    block_start : (n_blocks,)       - where each block begins, in the sorted order
    bin_lo/hi   : (n_blocks, max_bins_in_a_block) - each bin's [start, end)
                  *within its block*
    bin_id      : (n_blocks, max_bins_in_a_block) - which output row each
                  bin belongs to (unused slots point at a harmless "dead" row)
    k_block     : how many modes wide each block is (== sparse_k_batch)
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
    """
    Works out how to sum every k-mode into its (l,m) bin, deterministically
    and without ever materialising a huge dense array. Pure numpy - this
    runs once at setup, not every timestep.

    Why not just do `S.at[lm_idx, :].add(contribution)`? Many k-modes share
    a bin, so that scatter-add resolves via GPU atomics, and atomics don't
    complete in a fixed order between runs. That seeds tiny (~1e-5)
    run-to-run differences - normally harmless, but this simulation is
    chaotic, so those differences blow up to O(1) by late times. Two runs
    of the "same" simulation would silently diverge.

    The fix: sort the k-modes by bin, then a bin's sum is just
    `cumsum[bin_end] - cumsum[bin_start]` - a fixed reduction order, so
    it's reproducible. That only works per-chunk if a chunk never splits
    a bin, so this function's job is to pack the sorted k-modes into
    fixed-size blocks that only ever contain whole bins.
    """
    lm_idx = np.asarray(lm_idx)
    number_of_k_modes = lm_idx.shape[0]

    # Step 1: sort the k-modes by which (l,m) bin they belong to, so that
    # every bin's modes become one contiguous run.
    sort_order = np.argsort(lm_idx, kind='stable')
    modes_per_bin = np.bincount(lm_idx[sort_order], minlength=N_unique)

    if modes_per_bin.max() > k_block:
        raise ValueError(
            f"k_block={k_block} is smaller than the largest (l,m) bin "
            f"({modes_per_bin.max()} modes). Blocks must hold whole bins; "
            f"raise sparse_k_batch to at least {int(modes_per_bin.max())}.")

    # bin_start[u] is where bin u begins in the sorted order; bin u runs
    # from bin_start[u] up to (not including) bin_start[u + 1].
    bin_start = np.concatenate([[0], np.cumsum(modes_per_bin)]).astype(np.int64)

    # Step 2: greedily pack consecutive whole bins into blocks, each no
    # wider than k_block modes. A bin is never split across two blocks -
    # that's what lets a block's own local cumsum recover an exact sum
    # for each of its bins.
    block_bin_ranges = []
    first_bin_in_block = 0
    while first_bin_in_block < N_unique:
        last_bin_in_block = first_bin_in_block
        block_start_offset = bin_start[first_bin_in_block]

        while (last_bin_in_block < N_unique
               and bin_start[last_bin_in_block + 1] - block_start_offset <= k_block
               and last_bin_in_block - first_bin_in_block < max_bins_per_block):
            last_bin_in_block += 1

        assert last_bin_in_block > first_bin_in_block, (
            "a single (l,m) bin is larger than k_block - raise sparse_k_batch")

        block_bin_ranges.append((first_bin_in_block, last_bin_in_block))
        first_bin_in_block = last_bin_in_block

    # Step 3: for every block, record where each of its bins starts/ends
    # *within that block* (bin_lo/bin_hi) and which output row it belongs
    # to (bin_id). Blocks can hold different numbers of bins, so pad every
    # block's row out to the widest one (max_bins_in_a_block) using a
    # "dead" row index (N_unique) that gets discarded at the end - it just
    # needs somewhere harmless to write zero-length ranges.
    number_of_blocks = len(block_bin_ranges)
    max_bins_in_a_block = max(last - first for first, last in block_bin_ranges)

    block_start = np.zeros(number_of_blocks, np.int32)
    bin_lo = np.zeros((number_of_blocks, max_bins_in_a_block), np.int32)
    bin_hi = np.zeros((number_of_blocks, max_bins_in_a_block), np.int32)
    bin_id = np.full((number_of_blocks, max_bins_in_a_block), N_unique, np.int32)

    for block_index, (first_bin, last_bin) in enumerate(block_bin_ranges):
        block_start_offset = bin_start[first_bin]
        n_bins_in_this_block = last_bin - first_bin

        block_start[block_index] = block_start_offset
        bin_lo[block_index, :n_bins_in_this_block] = bin_start[first_bin:last_bin] - block_start_offset
        bin_hi[block_index, :n_bins_in_this_block] = bin_start[first_bin + 1:last_bin + 1] - block_start_offset
        bin_id[block_index, :n_bins_in_this_block] = np.arange(first_bin, last_bin)

    print(f"[precompute_bin_blocks] {number_of_k_modes} k-modes -> "
          f"{number_of_blocks} blocks of at most {k_block} modes / "
          f"at most {max_bins_in_a_block} bins each.")

    return BinBlocks(
        perm=sort_order,
        block_start=jnp.asarray(block_start),
        bin_lo=jnp.asarray(bin_lo),
        bin_hi=jnp.asarray(bin_hi),
        bin_id=jnp.asarray(bin_id),
        k_block=int(k_block),
    )


def fold_phase_into_aj(aj_sorted, parent_j_sorted, phase_c, R_dtype):
    """
    Multiplies each k-mode's amplitude by its radial mode's phase,
    e^{-i E_j t}, once. Callers that repeat sparse_a_u_j_matmul over many
    r-chunks with the same (aj, parent_j, phase) - see
    build_sphht_rho_lms_jit - can compute this once up front instead of
    redoing the same gather-and-multiply inside every chunk.
    """
    cdtype = jnp.result_type(aj_sorted.dtype, phase_c.dtype, R_dtype)
    phase_per_k_mode = phase_c[parent_j_sorted]
    return (aj_sorted * phase_per_k_mode).astype(cdtype)


def sparse_a_u_j_matmul(aj_sorted, parent_j_sorted, blocks, phase_c, R,
                        N_unique, k_batch, aj_phase=None):
    """
    For every (l, m) bin u, sums `aj[k] * phase[parent_j[k]] * R[:, parent_j[k]]`
    over every k-mode k that belongs to that bin. Equivalent to the dense
    matrix product `a_u_j_dense @ (R * phase[None, :]).T`, but `a_u_j_dense`
    (shape (N_unique, n_radial_modes)) is typically under 1% nonzero, so we
    never build it.

    Uses the deterministic cumulative-sum trick from precompute_bin_blocks
    instead of a scatter-add, and processes k_batch modes at a time (see
    `blocks`), so the largest thing ever held in memory is
    (n_rows, k_batch), not (n_rows, n_k_modes).

    Inputs
    ------
    aj_sorted, parent_j_sorted : per-k-mode amplitude and parent radial
        mode, both sorted and zero-padded by precompute_bin_blocks
    blocks   : (block_start, bin_lo, bin_hi, bin_id), from
               precompute_bin_blocks(...).as_arrays()
    phase_c  : (n_radial_modes,) complex, e^{-i E_j t}
    R        : (n_rows, n_radial_modes) - e.g. R_j_r_fixed (n_rows = n_radii)
               or R_j_at_particles (n_rows = n_particles)
    N_unique : number of distinct (l, m) bins (static)
    k_batch  : modes per block (static, must equal blocks.k_block)
    aj_phase : optional precomputed fold_phase_into_aj(...) result, for
               callers that call this once per r-chunk with the same
               aj/phase but different R

    Returns
    -------
    (N_unique, n_rows) - each bin's summed contribution.
    """
    if not isinstance(blocks, (tuple, list)) or len(blocks) != 4:
        raise TypeError(
            "sparse_a_u_j_matmul now takes a bin-block plan, not `lm_idx`. "
            "Pass `precompute_bin_blocks(lm_idx, N_unique, k_batch).as_arrays()` "
            "and permute `aj`/`parent_j` by its `.perm` (see `sort_modes_for_sphht`).")

    n_rows = R.shape[0]
    cdtype = jnp.result_type(aj_sorted.dtype, phase_c.dtype, R.dtype)

    if aj_phase is None:
        aj_phase = fold_phase_into_aj(aj_sorted, parent_j_sorted, phase_c, R.dtype)

    zero_column = jnp.zeros((n_rows, 1), cdtype)

    def sum_one_block(bin_totals, block):
        block_start, bin_lo, bin_hi, bin_row = block

        amplitudes_in_block = jax.lax.dynamic_slice(aj_phase, (block_start,), (k_batch,))
        parents_in_block = jax.lax.dynamic_slice(parent_j_sorted, (block_start,), (k_batch,))

        # contribution[:, k] = amplitude[k] * R[:, parent_j[k]]  (phase already folded in)
        contribution = amplitudes_in_block[None, :] * R[:, parents_in_block]   # (n_rows, k_batch)

        # A running total along the block: each bin's sum is then just the
        # difference between where it ends and where it starts, so there's
        # a fixed reduction order - no atomics, so no run-to-run drift.
        running_total = jnp.cumsum(contribution, axis=1)
        running_total = jnp.concatenate([zero_column, running_total], axis=1)      # (n_rows, k_batch + 1)
        bin_sums_in_block = running_total[:, bin_hi] - running_total[:, bin_lo]     # (n_rows, bins_per_block)

        # Every real bin is written by exactly one block (blocks never
        # split a bin), so this add only ever collides on the dead row.
        return bin_totals.at[bin_row, :].add(bin_sums_in_block.T), None

    dead_row_count = 1   # unused (padding) bins scatter into this extra row and get discarded
    initial_totals = jnp.zeros((N_unique + dead_row_count, n_rows), cdtype)
    final_totals, _ = jax.lax.scan(sum_one_block, initial_totals, blocks)
    return final_totals[:N_unique]


def _density_from_bin_sums(bin_sums_row_major, lm_pairs, total_mass, L_out, cdtype):
    """
    Turns per-bin sums (one row per radius or per particle) into density
    on the McEwen-Wiaux angular grid: scatter the bins into an (l, m)
    grid, inverse spherical-harmonic-transform to real space, then square.
    """
    n_rows = bin_sums_row_major.shape[0]
    flm = jnp.zeros((n_rows, L_out, 2 * L_out - 1), dtype=cdtype)
    flm = flm.at[:, lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(bin_sums_row_major)

    psi = jax.vmap(
        lambda f: s2fft.inverse(f, L_out, sampling='mw', method='jax')
    )(flm)
    return total_mass * (jnp.abs(psi) ** 2)


def _density_then_forward_sht(bin_sums_row_major, lm_pairs, total_mass, L_out, cdtype):
    """_density_from_bin_sums, then transformed back to (l, m) space -
    used everywhere the final answer needs to be rho_lm rather than
    rho(theta, phi)."""
    rho = _density_from_bin_sums(bin_sums_row_major, lm_pairs, total_mass, L_out, cdtype)
    return jax.vmap(
        lambda r: s2fft.forward(r, L_out, sampling='mw', method='jax')
    )(rho).astype(cdtype)


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch", "r_chunk"))
def build_sphht_rho_rtp_jit(R_j_r_fixed, phase_c, aj, parent_j, blocks,
                              lm_pairs, total_mass, L_out, N_unique, k_batch, r_chunk):
    """
    Builds the density rho(r, theta, phi) on the angular grid, at every
    radius, streamed r_chunk radii at a time.

    Streaming matters because holding the whole (n_radii, L_out,
    2*L_out - 1) result in memory at once - on top of s2fft's own working
    set - can exceed the GPU's memory at high L_out. Streaming caps the
    peak at (r_chunk, L_out, 2*L_out - 1).
    """
    n_radii = R_j_r_fixed.shape[0]
    n_radial_modes = R_j_r_fixed.shape[1]
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_r_fixed.dtype)
    real_dtype = R_j_r_fixed.dtype

    n_chunks = (n_radii + r_chunk - 1) // r_chunk
    n_radii_padded = n_chunks * r_chunk
    pad_amount = n_radii_padded - n_radii

    if pad_amount > 0:
        R_padded = jnp.concatenate(
            [R_j_r_fixed, jnp.zeros((pad_amount, n_radial_modes), dtype=real_dtype)], axis=0)
    else:
        R_padded = R_j_r_fixed

    R_chunks = R_padded.reshape(n_chunks, r_chunk, n_radial_modes)

    # aj * phase doesn't depend on R, so fold it in once rather than once per chunk.
    aj_phase = fold_phase_into_aj(aj, parent_j, phase_c, real_dtype)

    def one_chunk(R_chunk):
        bin_sums = sparse_a_u_j_matmul(aj, parent_j, blocks, phase_c, R_chunk,
                                        N_unique, k_batch, aj_phase=aj_phase)   # (N_unique, r_chunk)
        return _density_from_bin_sums(bin_sums.T, lm_pairs, total_mass, L_out, cdtype)   # (r_chunk, n_theta, n_phi)

    rho_chunks = jax.lax.map(one_chunk, R_chunks)              # (n_chunks, r_chunk, n_theta, n_phi)
    n_theta, n_phi = L_out, 2 * L_out - 1
    rho_padded = rho_chunks.reshape(n_radii_padded, n_theta, n_phi)
    return rho_padded[:n_radii]


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch", "r_chunk", "out_sharding"))
def build_sphht_rho_lms_jit(R_j_r_fixed, phase_c, aj, parent_j, blocks,
                              lm_pairs, total_mass,
                              ramp_c, rho_static_r_l00,
                              L_out, N_unique, k_batch, r_chunk,
                              out_sharding=None):
    """
    Builds rho_lm(r): the density, as a spherical-harmonic expansion, at
    every radius - streamed r_chunk radii at a time (see
    build_sphht_rho_rtp_jit), and blended per chunk against the static
    background according to the ramp fraction:

        rho_lm = (1 - ramp_c) * rho_static_lm + ramp_c * rho_full_lm

    If out_sharding is given (splitting the output's L axis across
    devices), each chunk is instead computed sharded along the *radial*
    axis: s2fft isn't itself distributed across L, but every radius's
    transform is independent of every other, so splitting a chunk's
    radii across devices needs no communication. The chunk's result is
    then re-sharded from radius-sharded to L-sharded before being written
    into the output accumulator, so no device ever holds more than its
    own L-slice of the full output.
    """
    n_radii = R_j_r_fixed.shape[0]
    n_radial_modes = R_j_r_fixed.shape[1]
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_r_fixed.dtype)
    real_dtype = R_j_r_fixed.dtype

    n_chunks = (n_radii + r_chunk - 1) // r_chunk
    n_radii_padded = n_chunks * r_chunk
    pad_amount = n_radii_padded - n_radii

    if pad_amount > 0:
        R_padded = jnp.concatenate(
            [R_j_r_fixed, jnp.zeros((pad_amount, n_radial_modes), dtype=real_dtype)], axis=0)
        rho_static_padded = jnp.concatenate(
            [rho_static_r_l00, jnp.zeros((pad_amount,), dtype=rho_static_r_l00.dtype)], axis=0)
    else:
        R_padded = R_j_r_fixed
        rho_static_padded = rho_static_r_l00

    # aj * phase doesn't depend on R, so fold it in once rather than once per chunk.
    aj_phase = fold_phase_into_aj(aj, parent_j, phase_c, real_dtype)

    def density_then_forward_sht(bin_sums_r_major):
        return _density_then_forward_sht(bin_sums_r_major, lm_pairs, total_mass, L_out, cdtype)

    def rho_lm_for_chunk(R_chunk):
        bin_sums = sparse_a_u_j_matmul(aj, parent_j, blocks, phase_c, R_chunk,
                                        N_unique, k_batch, aj_phase=aj_phase)   # (N_unique, r_chunk)
        bin_sums_r_major = bin_sums.T                                          # (r_chunk, N_unique)

        if out_sharding is None:
            return density_then_forward_sht(bin_sums_r_major)

        mesh = out_sharding.mesh
        bin_sums_r_major = jax.lax.with_sharding_constraint(
            bin_sums_r_major, NamedSharding(mesh, P('x', None)))

        rho_lm_chunk = shard_map(
            density_then_forward_sht, mesh=mesh,
            in_specs=P('x', None),
            out_specs=P('x', None, None),
            check_rep=False,
        )(bin_sums_r_major)

        return jax.lax.with_sharding_constraint(rho_lm_chunk, out_sharding)

    out_shape = (n_radii_padded, L_out, 2 * L_out - 1)
    rho_lm_accumulator = jnp.zeros(out_shape, dtype=cdtype)
    if out_sharding is not None:
        rho_lm_accumulator = jax.lax.with_sharding_constraint(rho_lm_accumulator, out_sharding)

    one_minus_ramp = jnp.asarray(1.0, dtype=ramp_c.dtype) - ramp_c

    # A plain jax.lax.scan + dynamic_update_slice, not jax.lax.map: map
    # would stack every chunk's result into one (n_chunks, r_chunk, L_out,
    # 2*L_out - 1) array replicated on every device - as large as the
    # whole output - which would defeat the point of sharding it.
    def write_one_chunk(rho_lm_accumulator, chunk_index):
        R_chunk = jax.lax.dynamic_slice_in_dim(R_padded, chunk_index * r_chunk, r_chunk, axis=0)
        rho_lm_chunk = rho_lm_for_chunk(R_chunk)

        # Blend against the static background. Only the (l=0, m=0)
        # coefficient of the static density is nonzero (it's spherically
        # symmetric), so only that one slice needs adding.
        rho_lm_chunk = ramp_c * rho_lm_chunk
        static_chunk = jax.lax.dynamic_slice_in_dim(rho_static_padded, chunk_index * r_chunk, r_chunk, axis=0)
        rho_lm_chunk = rho_lm_chunk.at[:, 0, L_out - 1].add(
            (one_minus_ramp * static_chunk).astype(cdtype))

        rho_lm_accumulator = jax.lax.dynamic_update_slice(
            rho_lm_accumulator, rho_lm_chunk, (chunk_index * r_chunk, 0, 0))
        if out_sharding is not None:
            rho_lm_accumulator = jax.lax.with_sharding_constraint(rho_lm_accumulator, out_sharding)
        return rho_lm_accumulator, None

    rho_lm_accumulator, _ = jax.lax.scan(write_one_chunk, rho_lm_accumulator, jnp.arange(n_chunks))
    return rho_lm_accumulator[:n_radii]


def normalised_legendre_table(n_max, x):
    """
    Normalised associated Legendre functions, as an
    (n_max+1, n_max+1, len(x)) table indexed [order m, degree l, point].

    Same recurrences, in the same order, as
    `jax._src.scipy.special._gen_associated_legendre(n_max, x, True)` -
    values are bit-identical - with one thing removed: that function
    materialises its per-iteration coefficients as two DENSE
    (n_max+1, n_max+1, n_max+1) arrays, `d0_mask_3d` / `d1_mask_3d`, and
    reads row i out of them at iteration i. Each is 8 (n_max+1)^3 bytes,
    5.7 GB apiece at n_max = 891, and neither carries any information
    beyond an (n_max+1, n_max+1) coefficient matrix restricted to the
    plane i + j - k = 0.

    That is what killed the m22 = 100, L_out_frac = 0.185 run: the fused
    executable reported "temp 11.80 GiB", which is exactly 2 * 8 * 892^3,
    and the module carrying it could no longer be loaded onto the device -
    "Failed to load in-memory CUBIN ... CUDA_ERROR_OUT_OF_MEMORY" - even
    with ~20 GiB free under the BFC limit, because a module load draws on
    driver memory outside the pool.

    Selecting that plane inside the loop instead costs one (n_max+1)^2
    mask per iteration - 6.4 MB at n_max = 891, against the
    (n_max+1, n_max+1, n_points) working set the recurrence already
    carries - so the table scales as n_max^2, not n_max^3.

    Only the is_normalized=True branch is reproduced; it is the only one
    compute_Ylm_and_dtheta_jit ever asked for.
    """
    n_points = x.shape[0]
    table_width = n_max + 1

    p = jnp.zeros((table_width, table_width, n_points), dtype=x.dtype)

    a_idx = jnp.arange(1, n_max + 1, dtype=x.dtype)
    b_idx = jnp.arange(n_max, dtype=x.dtype)
    initial_value = 0.5 / jnp.sqrt(jnp.pi)          # p(0, 0)
    f_a = jnp.cumprod(-1 * jnp.sqrt(1.0 + 0.5 / a_idx))
    f_b = jnp.sqrt(2.0 * b_idx + 3.0)

    p = p.at[(0, 0)].set(initial_value)

    # Diagonal entries p(l, l).
    y = jnp.cumprod(
        jnp.broadcast_to(jnp.sqrt(1.0 - x * x), (n_max, n_points)), axis=0)
    # jnp.einsum, not the equivalent-looking broadcast multiply: einsum
    # lowers these through dot_general, which rounds differently from an
    # elementwise multiply, and keeping them makes this table bit-identical
    # to the JAX original rather than merely equal to ~1e-16.
    p_diag = initial_value * jnp.einsum('i,ij->ij', f_a, y)
    diag_indices = jnp.diag_indices(table_width)
    p = p.at[(diag_indices[0][1:], diag_indices[1][1:])].set(p_diag)

    # First off-diagonal, from the diagonal.
    p_offdiag = jnp.einsum('ij,ij->ij',
                           jnp.einsum('i,j->ij', f_b, x),
                           p[jnp.diag_indices(n_max)])
    offdiag_indices = (diag_indices[0][:n_max], diag_indices[1][:n_max] + 1)
    p = p.at[offdiag_indices].set(p_offdiag)

    # Two-term recurrence coefficients, (order, degree). Only the strict
    # upper triangles are meaningful - the expressions divide by
    # l^2 - m^2, which is zero on the diagonal - so everything else stays
    # at exactly zero, as in the JAX original.
    m_mat, l_mat = jnp.meshgrid(
        jnp.arange(table_width, dtype=x.dtype),
        jnp.arange(table_width, dtype=x.dtype),
        indexing='ij')
    c0 = l_mat * l_mat
    c1 = m_mat * m_mat
    c2 = 2.0 * l_mat
    c3 = (l_mat - 1.0) * (l_mat - 1.0)
    d0 = jnp.sqrt((4.0 * c0 - 1.0) / (c0 - c1))
    d1 = jnp.sqrt(((c2 + 1.0) * (c3 - c1)) / ((c2 - 3.0) * (c0 - c1)))

    d_zeros = jnp.zeros((table_width, table_width), dtype=x.dtype)
    d0_indices = jnp.triu_indices(table_width, 1)
    d1_indices = jnp.triu_indices(table_width, 2)
    d0_mask = d_zeros.at[d0_indices].set(d0[d0_indices])
    d1_mask = d_zeros.at[d1_indices].set(d1[d1_indices])

    # The plane the 3D masks encoded: at iteration i, only the entries with
    # degree = order + i contribute. Built per iteration from these two
    # (table_width, 1) / (1, table_width) index vectors.
    order_index = jnp.arange(table_width)[:, None]
    degree_index = jnp.arange(table_width)[None, :]

    p = p.astype(jnp.result_type(p.dtype, x.dtype, d0_mask.dtype))

    def body_fun(i, p_val):
        on_plane = (order_index + i - degree_index) == 0
        coeff_0 = jnp.where(on_plane, d0_mask, 0.0)
        coeff_1 = jnp.where(on_plane, d1_mask, 0.0)

        h = (jnp.einsum('ij,ijk->ijk',
                        coeff_0,
                        jnp.einsum('ijk,k->ijk',
                                   jnp.roll(p_val, shift=1, axis=1), x))
             - jnp.einsum('ij,ijk->ijk', coeff_1, jnp.roll(p_val, shift=2, axis=1)))
        return p_val + h

    if n_max > 1:
        p = jax.lax.fori_loop(2, n_max + 1, body_fun, p)

    return p


@functools.partial(jax.jit, static_argnames=("n_max",))
def compute_Ylm_and_dtheta_jit(lm_pairs, theta_arr, phi_arr, n_max):
    """
    Spherical harmonics Y_lm and their theta-derivative, evaluated at a
    set of particle positions, fully on the GPU.

    Replaces a previous scipy.special.sph_harm_y call, which ran
    single-threaded on the CPU and needed a GPU<->host round trip every
    timestep.

    Why not jax.scipy.special.sph_harm_y directly: it requires (l, m,
    theta, phi) to all share one flat shape, so evaluating our (n_modes,)
    (l, m) at (n_particles,) (theta, phi) forces theta to broadcast out to
    (n_modes, n_particles) first. It then builds an internal Legendre
    table of shape (n_max+1, n_max+1, n_modes) - at L_max=481 that's
    hundreds of GB, almost all of it duplicated work, since every column
    of that broadcast theta is identical for a given particle.

    Fix: build the Legendre table ourselves at the true (n_particles,)
    shape - (n_max+1, n_max+1, n_particles) - then gather out exactly the
    (l, m, particle) triples needed. A single JVP with respect to theta
    gives the theta-derivative in the same pass, with no second
    evaluation.

    The table comes from normalised_legendre_table above, not from JAX's
    _gen_associated_legendre: same recurrence, same values to the bit, but
    without the two dense (n_max+1)^3 coefficient masks the JAX one builds
    on the way, which are 5.7 GB *each* at n_max = 891. See that function.

    Inputs
    ------
    lm_pairs           : (n_modes, 2) int - [l, m] for each output mode
    theta_arr, phi_arr : (n_particles,) float - particle colatitude / azimuth
    n_max              : largest l in lm_pairs[:, 0] (static)

    Returns
    -------
    Y, dY_dtheta: (n_modes, n_particles) complex
    """
    l_values = lm_pairs[:, 0]
    m_values = lm_pairs[:, 1]
    abs_m_values = jnp.abs(m_values)

    # e^{i |m| phi}: depends only on phi, so compute it once, outside the
    # theta-derivative below.
    angle = abs_m_values[:, None] * phi_arr[None, :]
    azimuthal_phase = jnp.cos(angle) + 1j * jnp.sin(angle)
    negative_m_sign = ((-1.0) ** abs_m_values)[:, None]
    is_negative_m = (m_values < 0)[:, None]

    def Y_at_theta(theta):
        legendre_table = normalised_legendre_table(n_max, jnp.cos(theta))        # (n_max+1, n_max+1, n_particles)
        legendre_values = legendre_table[abs_m_values, l_values, :]              # (n_modes, n_particles)

        Y_for_positive_m = legendre_values * azimuthal_phase
        return jnp.where(
            is_negative_m,
            negative_m_sign * jnp.conjugate(Y_for_positive_m),
            Y_for_positive_m,
        )

    # JVP with tangent 1 in theta gives (Y, dY/dtheta) in a single pass.
    Y, dY_dtheta = jax.jvp(Y_at_theta, (theta_arr,), (jnp.ones_like(theta_arr),))
    return Y, dY_dtheta


@functools.partial(jax.jit, static_argnames=("L_out", "N_unique", "k_batch"))
def compute_rho_lm_at_particles_sphht_jit(R_j_at_parts, phase_c, aj, parent_j,
                                            blocks, lm_pairs, total_mass,
                                            L_out, N_unique, k_batch):
    """
    rho_lm evaluated at a set of particle radii: the same calculation as
    build_sphht_rho_lms_jit, just with R_j_at_parts (one row per particle)
    in place of R_j_r_fixed (one row per background radius). No
    streaming or sharding here - there are far fewer particles than
    background radii.
    """
    cdtype = jnp.result_type(aj.dtype, phase_c.dtype, R_j_at_parts.dtype)

    bin_sums = sparse_a_u_j_matmul(aj, parent_j, blocks, phase_c, R_j_at_parts,
                                    N_unique, k_batch)          # (N_unique, n_particles)
    bin_sums_particle_major = bin_sums.T                        # (n_particles, N_unique)

    return _density_then_forward_sht(bin_sums_particle_major, lm_pairs, total_mass, L_out, cdtype)
