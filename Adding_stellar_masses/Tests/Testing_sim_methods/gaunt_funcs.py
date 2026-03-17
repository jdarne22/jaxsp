
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


def compute_rho_lm_gaunt(aj_modes, R_modes, lm_l, lm_m, total_mass, time_step, dt, eigen_energies, parent_j,
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
    lm_l = list(map(int, lm_l))
    lm_m = list(map(int, lm_m))
    l_max = max(lm_l)

    if L_max_out is None:
        L_max_out = 2 * l_max + 1
    N_flat = L_max_out * (2 * L_max_out - 1)

    # ── 1. aR[r,k] = a_k * R_k(r)  (time phase already baked into R_modes) ───
    # Keep on CPU so the accumulation loop below is cheap numpy ops
    aR = np.array(R_modes, dtype=complex) * np.array(aj_modes, dtype=complex)[None, :] * np.exp(-1j * time_step * dt * np.array(eigen_energies[parent_j]) / hbar.value)[None, :]

    # ── 2. F_{l,m}(r) = sum_{k:(l_k,m_k)=(l,m)} a_k R_k(r) ──────────────────
    # CPU numpy accumulation avoids N_unique un-JIT'd GPU ops
    lm_to_F = defaultdict(lambda: np.zeros(Nr, dtype=complex))
    for k, (ell, m) in enumerate(zip(lm_l, lm_m)):
        lm_to_F[(ell, m)] += aR[:, k]

    # ── 3. Gaunt table (build once, reuse across time steps) ──────────────────
    if gaunt_table is None:
        gaunt_table = precompute_gaunt_table(lm_l, lm_m, L_max_out)

    all_i, all_j, all_G, all_Lf, unique_lm = gaunt_table
    N_nz = len(all_i)

    # F_jax[r, i] = F_{unique_lm[i]}(r)  on GPU — single CPU→GPU transfer
    F_jax = jnp.array(np.stack([lm_to_F[lm] for lm in unique_lm], axis=1))  # (Nr, N_unique)

    # ── 4. JIT-compiled batch accumulation on GPU ──────────────────────────────
    # For each entry e:  weighted[e, r] = G_e * F_{l1,m1}(r) * conj(F_{l2,m2}(r))
    # segment_sum groups by flat (L,M) index → (N_flat, Nr)
    rho_flat = jnp.zeros((N_flat, Nr), dtype=complex)
    n_batches = (N_nz + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, N_nz, batch_size)):
        end = min(start + batch_size, N_nz)
        rho_flat = rho_flat + _accum_batch(
            F_jax, all_i[start:end], all_j[start:end],
            all_G[start:end], all_Lf[start:end], N_flat,
        )
        #print(f"\r  batch {b+1}/{n_batches}", end="", flush=True)
    #print()

    # ── 5. (N_flat, Nr).T → (Nr, L_max_out, 2*L_max_out-1) ───────────────────
    return total_mass * rho_flat.T.reshape(Nr, L_max_out, 2 * L_max_out - 1)
