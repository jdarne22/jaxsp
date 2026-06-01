
from collections import defaultdict
import functools

import numpy as np
import wigners
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
print(jax.devices())

jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp


from jaxsp.constants import hbar


@functools.partial(jax.jit, static_argnums=(5,))
def _accum_batch(F, i_b, j_b, G_b, Lf_b, N_flat):
    weighted = (F[:, i_b] * jnp.conj(F[:, j_b]) * G_b[None, :]).T   # (B, Nr)
    return jax.ops.segment_sum(weighted, Lf_b, num_segments=N_flat)   # (N_flat, Nr)


def _process_l1l2(args):
    l1, l2, L_max_out = args
    parts = []
    L_min = abs(l1 - l2)
    L_max = min(l1 + l2, L_max_out - 1)
    for L in range(L_min, L_max + 1):
        if (l1 + l2 + L) % 2 != 0:
            continue
        w0 = wigners.wigner_3j(l1, l2, L, 0, 0, 0)
        if w0 == 0.0:
            continue
        prefac0 = np.sqrt((2*l1+1)*(2*l2+1)*(2*L+1) / (4*np.pi)) * w0

        arr = wigners.clebsch_gordan_array(l1, l2, L)  # (2l1+1, 2l2+1, 2L+1)
        m1g, m2g = np.meshgrid(np.arange(-l1, l1+1), np.arange(-l2, l2+1), indexing='ij')
        Mg = m1g - m2g
        valid = np.abs(Mg) <= L
        m1v, m2v, Mv = m1g[valid], m2g[valid], Mg[valid]

        neg_m2_idx = (-m2v) + l2
        M_idx_L    = Mv + L
        cg_vals    = arr[m1v + l1, neg_m2_idx, M_idx_L]

        phase = ((-1.0) ** (l1 - l2 + Mv + m2v + Mv)) / np.sqrt(2*L+1)
        G_vals = prefac0 * phase * cg_vals

        nz = np.abs(G_vals) > 1e-14
        if not np.any(nz):
            continue

        parts.append((l1, m1v[nz], l2, m2v[nz], G_vals[nz],
                      L * (2*L_max_out - 1) + (Mv[nz] + L_max_out - 1)))
    return parts


def precompute_gaunt_table(lm_l, lm_m, L_max_out, cache_dir=".", n_workers=64):
    lm_l = list(map(int, lm_l))
    lm_m = list(map(int, lm_m))
    unique_lm = sorted(set(zip(lm_l, lm_m)))
    lm_to_idx = {lm: i for i, lm in enumerate(unique_lm)}
    L_max = max(l for l, m in unique_lm)

    # --- disk cache ---
    cache_path = Path(cache_dir) / f"gaunt_lmax{L_max}_Lout{L_max_out}.npz"
    if cache_path.exists():
        print(f"Loading Gaunt table from {cache_path} ...")
        data = np.load(cache_path)
        return (jnp.array(data["i"]), jnp.array(data["j"]), jnp.array(data["G"]),
                jnp.array(data["Lf"]), unique_lm)


    # --- build ---
    print(f"Building Gaunt table ({len(unique_lm)} unique (l,m), L_max_out={L_max_out}) ...")
    tasks = [(l1, l2, L_max_out) for l1 in range(L_max+1) for l2 in range(L_max+1)]

    all_i, all_j, all_G, all_Lf = [], [], [], []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for parts in pool.map(_process_l1l2, tasks, chunksize=4):
            for l1, m1v, l2, m2v, G_vals, Lf_vals in parts:
                all_i.append([lm_to_idx[(l1, int(m))] for m in m1v])
                all_j.append([lm_to_idx[(l2, int(m))] for m in m2v])
                all_G.append(G_vals)
                all_Lf.append(Lf_vals)

    ai = np.concatenate(all_i).astype(np.int32)
    aj = np.concatenate(all_j).astype(np.int32)
    aG = np.concatenate(all_G)
    aLf = np.concatenate(all_Lf).astype(np.int32)

    np.savez(cache_path, i=ai, j=aj, G=aG, Lf=aLf)
    print(f"  {len(ai):,} non-zero entries — saved to {cache_path}")
    return jnp.array(ai), jnp.array(aj), jnp.array(aG), jnp.array(aLf), unique_lm

def make_scatter_matrix(lm_l, lm_m, unique_lm):
    """
    Returns a (N_unique, Nmodes) float32 matrix S where S[i, k] = 1
    if unique_lm[i] == (lm_l[k], lm_m[k]), else 0.
    F = S @ aR.T  gives F_{l,m}(r) as a (N_unique, Nr) matrix via one matmul.
    """
    lm_l = list(map(int, lm_l))
    lm_m = list(map(int, lm_m))
    lm_to_idx = {lm: i for i, lm in enumerate(unique_lm)}
    N_unique = len(unique_lm)
    Nmodes = len(lm_l)
    S = np.zeros((N_unique, Nmodes), dtype=np.float64)
    for k, (ell, m) in enumerate(zip(lm_l, lm_m)):
        S[lm_to_idx[(ell, m)], k] = 1.0
    return jnp.array(S)



def make_lm_idx_sorted_per_mode(lm_l_per_mode, lm_m_per_mode, unique_lm):
    """
    Mode k -> index into the sorted unique_lm list.
    unique_lm comes from precompute_gaunt_table() (sorted(set(...))).
    """
    lm_l = list(map(int, lm_l_per_mode))
    lm_m = list(map(int, lm_m_per_mode))
    lm_to_idx = {lm: i for i, lm in enumerate(unique_lm)}
    return jnp.array([lm_to_idx[(l, m)] for l, m in zip(lm_l, lm_m)], dtype=jnp.int32)


def compute_rho_lm_gaunt_F(F, total_mass, L_max_out, all_i, all_j, all_G, all_Lf, batch_size):
    """
    Reduce a precomputed F[r, u] = sum_j coeff_uj * R_j(r) into rho_lm(r) via
    the Gaunt coefficients.

    F's leading axis is a batch axis: Nr for background grid evaluation,
    or N_particles for the per-particle batched call from _compute_radial_batch.

    The Gaunt arrays (all_i, all_j, all_G, all_Lf) are passed here as explicit
    arguments — NOT captured via a closure / self — so that when this function
    is called from inside a jit'd parent, XLA treats them as dynamic input
    buffers rather than compile-time constants. Baking them in as constants
    would require an extra ~6 GB copy of all_G alongside the live array, and
    that's what the "Failed to allocate N bytes for new constant" OOM was.

    The accumulation over N_nz is split into full batches (lax.scan with
    lax.dynamic_slice reading batch_size windows at a traced offset) plus a
    final tail. No padding, no extra allocation.

    Parameters
    ----------
    F                         : complex (Nr_or_Nparticles, N_unique)
    total_mass                : float
    L_max_out                 : int (static)
    all_i, all_j, all_G, all_Lf : the 4 array fields of gaunt_table
    batch_size                : int (static)  Gaunt entries per scan step

    Returns
    -------
    rho_lm : complex (Nr_or_Nparticles, L_max_out, 2*L_max_out-1)
    """
    Nr, _N_unique = F.shape
    N_flat = L_max_out * (2 * L_max_out - 1)

    N_nz     = int(all_i.shape[0])
    n_batches = N_nz // batch_size   # number of full batches
    n_tail    = N_nz % batch_size    # leftover entries

    # Accumulate full batches via lax.scan. The loop variable `start` indexes
    # into the Gaunt arrays with lax.dynamic_slice (traced start, static size).
    # This reads one batch_size window per step without any extra allocation.
    def body(rho_flat, start):
        i_b  = jax.lax.dynamic_slice(all_i,  (start,), (batch_size,))
        j_b  = jax.lax.dynamic_slice(all_j,  (start,), (batch_size,))
        G_b  = jax.lax.dynamic_slice(all_G,  (start,), (batch_size,))
        Lf_b = jax.lax.dynamic_slice(all_Lf, (start,), (batch_size,))
        weighted = (F[:, i_b] * jnp.conj(F[:, j_b]) * G_b[None, :]).T   # (batch_size, Nr)
        contrib  = jax.ops.segment_sum(weighted, Lf_b, num_segments=N_flat)  # (N_flat, Nr)
        return rho_flat + contrib, None

    rho_flat_init = jnp.zeros((N_flat, Nr), dtype=jnp.complex128)
    starts = jnp.arange(n_batches, dtype=jnp.int32) * batch_size   # (n_batches,)
    rho_flat, _ = jax.lax.scan(body, rho_flat_init, starts)

    # Tail: fewer than batch_size entries left after the last full batch.
    # Handled with a plain Python if (n_tail is a compile-time int) and static
    # JAX slice — no extra allocation beyond the small tail arrays.
    if n_tail > 0:
        tail_start = n_batches * batch_size
        i_b  = all_i[tail_start:]
        j_b  = all_j[tail_start:]
        G_b  = all_G[tail_start:]
        Lf_b = all_Lf[tail_start:]
        weighted = (F[:, i_b] * jnp.conj(F[:, j_b]) * G_b[None, :]).T
        rho_flat = rho_flat + jax.ops.segment_sum(weighted, Lf_b, num_segments=N_flat)

    return total_mass * rho_flat.T.reshape(Nr, L_max_out, 2 * L_max_out - 1)


def compute_rho_lm_gaunt(aj_modes, R_j_r_phased, parent_j, lm_idx_sorted_per_mode,
                          total_mass, L_max_out, gaunt_table, batch_size):
    """
    Backwards-compatible drop-in for the original compute_rho_lm_gaunt.
    Scatters aj into coeff_uj, matmuls with R_j_r_phased, then delegates the
    Gaunt accumulation to compute_rho_lm_gaunt_F (scan-based).

    Parameters
    ----------
    aj_modes               : complex (Nmodes,)    per-mode a_k
    R_j_r_phased           : complex (Nr, Nj)     R_j(r) * exp(-i E_j t/hbar)
    parent_j               : int    (Nmodes,)     mode k -> radial eigenstate j
    lm_idx_sorted_per_mode : int    (Nmodes,)     mode k -> index in unique_lm
    total_mass             : float
    L_max_out              : int
    gaunt_table            : tuple  from precompute_gaunt_table()
    batch_size             : int    Gaunt entries per scan step

    Returns
    -------
    rho_lm : JAX complex (Nr, L_max_out, 2*L_max_out-1)
    """
    _Nr, Nj = R_j_r_phased.shape
    all_i, all_j, all_G, all_Lf, unique_lm = gaunt_table
    N_unique = len(unique_lm)

    coeff_uj = jnp.zeros((N_unique, Nj), dtype=aj_modes.dtype)
    coeff_uj = coeff_uj.at[lm_idx_sorted_per_mode, parent_j].add(aj_modes)
    F = (coeff_uj @ R_j_r_phased.T).T   # (Nr, N_unique)

    return compute_rho_lm_gaunt_F(F, total_mass, L_max_out,
                                   all_i, all_j, all_G, all_Lf, batch_size)
