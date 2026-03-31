
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



def compute_rho_lm_gaunt(aj_modes, R_modes, lm_l, lm_m, total_mass, scatter_matrix,
                          L_max_out=None, gaunt_table=None, batch_size=500_000):
    """
    Compute rho_LM(r) using Gaunt coefficients — no grid, no SHT.

    Speedups vs naive version
    -------------------------
    1. Modes sharing (l,m) combined into F_{l,m}(r) first, reducing the
       double sum from O(N_modes^2) to O(N_unique^2).
    2. Gaunt table precomputed once and passed in via gaunt_table so it can
       be reused across many time steps at no extra cost.
    3. Accumulation is a single JIT-compiled segment_sum on GPU per batch —
       ~10 Python iterations instead of ~2209.

    Parameters
    ----------
    aj_modes   : complex (Nmodes,)    per-mode a_k; rand_phase included
    R_modes    : complex (Nr, Nmodes) R_{j(k)}(r) * exp(-i E_k t/hbar)
    lm_l, lm_m : int array-like (Nmodes,)
    total_mass : float
    L_max_out  : int, optional  default 2*l_max+1 captures all density harmonics
    gaunt_table : tuple, optional  from precompute_gaunt_table(); computed here if None
    batch_size  : int  Gaunt entries per GPU batch — tune to available VRAM
                       (batch_size=100_000 ≈ 1.6 GB complex128 for Nr=1000)

    Returns
    -------
    rho_lm : JAX complex array (Nr, L_max_out, 2*L_max_out-1)
        rho_lm[r, L, L_max_out-1+M] = rho_LM(r)
        Same indexing convention as s2fft.forward (mw sampling).
    """
    Nr, Nmodes = R_modes.shape

    if L_max_out is None:
        lm_l = list(map(int, lm_l))
        L_max_out = 2 * max(lm_l) + 1
    N_flat = L_max_out * (2 * L_max_out - 1)

    # aR on GPU directly — no CPU transfer
    aR = R_modes * aj_modes[None, :]   # (Nr, Nmodes), stays on GPU

    # F_{l,m}(r) via matmul instead of Python loop
    F_jax = (scatter_matrix @ aR.T).T  # (Nr, N_unique)

    # Unpack Gaunt table
    all_i, all_j, all_G, all_Lf, unique_lm = gaunt_table
    N_nz = len(all_i)

    # JIT-compiled batch accumulation on GPU
    rho_flat = jnp.zeros((N_flat, Nr), dtype=complex)
    for start in range(0, N_nz, batch_size):
        end = min(start + batch_size, N_nz)
        rho_flat = rho_flat + _accum_batch(
            F_jax, all_i[start:end], all_j[start:end],
            all_G[start:end], all_Lf[start:end], N_flat,
        )

    return total_mass * rho_flat.T.reshape(Nr, L_max_out, 2 * L_max_out - 1)
