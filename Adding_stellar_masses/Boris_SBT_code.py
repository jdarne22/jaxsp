import numpy as np
from scipy.special import spherical_jn, jv, jvp
from scipy.optimize import brentq

def continuousSphericalBesselTransform(f_r, r_finegrid, k_nodes, ell):
    kr = r_finegrid[:, None] * k_nodes[None, :]
    j_ell_k_r = spherical_jn(ell, kr, derivative=False)
    r2 = r_finegrid[:, None]**2.
    flk = np.trapz(f_r[:, None] * j_ell_k_r * r2, x=r_finegrid, axis=0)
    return np.sqrt(2/np.pi) * flk


def constructGridAndNorms(R, N, ell):
    # TODO: add other boundary conditions
    roots = rootsSphericalBesselFunctions(ell+1, N)
    qln = roots[ell, :]
    kln_nodes = qln / R
    cln_norms = np.sqrt(2*np.pi) / R**3 / spherical_jn(ell+1, qln)**2.
    return kln_nodes, cln_norms


def forwardDiscreteSphericalBesselTransform(f_r, r_grid, ell,
                                            kln_nodes, cln_norms):
    R = r_grid[-1]
    f_l_kln = continuousSphericalBesselTransform(f_r, r_grid, kln_nodes, ell)
    return f_l_kln


def inverseDiscreteSphericalBesselTransform(f_l_kln, klns, clns, r_grid, ell):
    kr = r_grid[:, None] * klns[None, :]
    j_ell_k_r = spherical_jn(ell, kr, derivative=False)
    f_r = np.sum(f_l_kln[None, :] * clns[None, :] * j_ell_k_r, axis=1)
    return f_r


def rootsSphericalBesselFunctions(L, N):
    ir = np.arange(N+1)
    ranges = (ir + 1/2.)*np.pi - 1./(ir + 1/2.)/np.pi
    roots = np.zeros((L, N))
    for ell in range(L):
        def Jn(r):
            return spherical_jn(ell, r)
        for j in range(N):
            roots[ell, j] = brentq(Jn, ranges[j], ranges[j+1])
        ranges[:-1] = roots[ell, :]
        ranges[-1] = ranges[-2] + ranges[-2] - ranges[-3]
    return roots


def fastTransformMatrix(ell1, ell2, N, qlns=None):
    # TODO: implement in cython
    L = np.max([ell1, ell2]) + 1
    if qlns is None:
        qlns = rootsSphericalBesselFunctions(L, N)
    mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mat[i, j] = np.sqrt(2*np.pi) /\
                spherical_jn(ell1+1, qlns[ell1, j])**2.0\
                * spherical_jn(ell1, qlns[ell2, i] * qlns[ell1, j] /
                               qlns[ell1, -1])
    return mat


def orderCouplingMatrix(ell1, ell2, N, Nr, qlns=None):
    # TODO: implement in cython
    L = np.max([ell1, ell2]) + 1
    if qlns is None:
        qlns = rootsSphericalBesselFunctions(L, N)
    qln1 = qlns[ell1, :]
    qln2 = qlns[ell2, :]
    mat = np.zeros((N, N))
    grid = np.linspace(0, 1, Nr)
    for i in range(N):
        for j in range(N):
            f_r = spherical_jn(ell1, grid*qln1[i]) *\
                spherical_jn(ell2, grid*qln2[j])
            val = np.trapz(grid**2 * f_r, x=grid)
            mat[i, j] = val
        mat[i, :] *= 2 / spherical_jn(ell1+1, qln1)**2.
    return mat


#-----------------------------------------------------------------------------------------
#JAX VERSION


import jax
import jax.numpy as jnp
from jax import lax


def trapz_axis1(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Batched trapezoid over axis=1.
    y: (B, Nr, K)
    x: (Nr,)
    -> (B, K)
    """
    dx = (x[1:] - x[:-1])[None, :, None]  # (1, Nr-1, 1)
    return jnp.sum(0.5 * (y[:, :-1, :] + y[:, 1:, :]) * dx, axis=1)


def spherical_jn_dynamic(ell: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute spherical Bessel j_ell(x) with ell as a *runtime* value (no per-ell compilation).

    ell: scalar int32 (can be traced)
    x:   (...), returns (...)

    Uses:
      j0(x) = sin(x)/x, j0(0)=1
      j1(x) = sin(x)/x^2 - cos(x)/x, j1(0)=0
      j_{l+1}(x) = (2l+1)/x * j_l(x) - j_{l-1}(x)
    via lax.while_loop up to ell.
    """
    ell = ell.astype(jnp.int32)

    x0 = (x == 0)
    x_safe = jnp.where(x0, 1e-30, x)

    j0 = jnp.sin(x_safe) / x_safe
    j0 = jnp.where(x0, 1.0, j0)

    # If ell == 0
    def case0(_):
        return j0

    # Otherwise compute j1 then loop if needed
    def case_ge1(_):
        j1 = jnp.sin(x_safe) / (x_safe**2) - jnp.cos(x_safe) / x_safe
        j1 = jnp.where(x0, 0.0, j1)

        # If ell == 1
        def case1(_):
            return j1

        # If ell >= 2, while_loop recurrence
        def case_ge2(_):
            # state: (l, j_{l-1}, j_l) starting at l=1 with (j0, j1)
            def cond_fun(state):
                l, _, _ = state
                return l < ell

            def body_fun(state):
                l, j_lm1, j_l = state
                j_lp1 = (2 * l + 1) / x_safe * j_l - j_lm1
                j_lp1 = jnp.where(x0, 0.0, j_lp1)
                return (l + 1, j_l, j_lp1)

            l0 = jnp.int32(1)
            state0 = (l0, j0, j1)
            l_final, j_lm1_final, j_l_final = lax.while_loop(cond_fun, body_fun, state0)
            return j_l_final

        return lax.cond(ell == 1, case1, case_ge2, operand=None)

    return lax.cond(ell == 0, case0, case_ge1, operand=None)


@jax.jit
def forward_dsbt_chunk(
    f_r_batch: jnp.ndarray,   # (B, Nr)
    ell_batch: jnp.ndarray,   # (B,)
    r: jnp.ndarray,           # (Nr,)
    k_nodes_batch: jnp.ndarray,  # (B, K)
) -> jnp.ndarray:
    """
    Forward DSBT for a chunk of (ell,m) rows in one compiled kernel.

    Returns: (B, K)
    """
    # kr: (B, Nr, K)
    kr = r[None, :, None] * k_nodes_batch[:, None, :]

    # Compute j_ell(kr) per row (B) with ell as runtime value.
    # vmap here is fine; crucially it DOES NOT create per-ell compilations.
    j_ell = jax.vmap(spherical_jn_dynamic, in_axes=(0, 0))(ell_batch, kr)  # (B, Nr, K)

    integrand = f_r_batch[:, :, None] * j_ell * (r[None, :, None] ** 2)  # (B, Nr, K)
    flk = trapz_axis1(integrand, r)  # (B, K)

    return jnp.sqrt(2.0 / jnp.pi) * flk


def forward_dsbt_all(
    rho_lm_r: jnp.ndarray,       # (Nr, Lmax+1, 2L-1)
    r: jnp.ndarray,              # (Nr,)
    lm_l,                        # (Nlm,) ints
    lm_m,                        # (Nlm,) ints
    L: int,
    kln_nodes_by_ell: jnp.ndarray,  # (Lmax+1, K)
    chunk_size: int = 256,
) -> jnp.ndarray:
    """
    Full forward transform for all (ell,m).

    - Single compiled kernel (forward_dsbt_chunk)
    - Chunking controls memory: intermediate arrays scale ~ chunk_size * Nr * K

    Returns: (Nlm, K)
    """
    ell_arr = jnp.asarray(lm_l, dtype=jnp.int32)         # (Nlm,)
    m_arr   = jnp.asarray(lm_m, dtype=jnp.int32)         # (Nlm,)
    Nlm = int(ell_arr.shape[0])
    K = int(kln_nodes_by_ell.shape[1])

    out = jnp.zeros((Nlm, K), dtype=rho_lm_r.dtype)

    # Process in chunks on host, calling one jitted kernel repeatedly.
    for start in range(0, Nlm, chunk_size):
        stop = min(start + chunk_size, Nlm)
        idx = jnp.arange(start, stop, dtype=jnp.int32)

        ell_b = ell_arr[idx]                       # (B,)
        m_b   = m_arr[idx]                         # (B,)
        # Gather f(r) rows: rho[:, ell, m+L-1] -> (Nr, B) -> transpose -> (B, Nr)
        f_b = rho_lm_r[:, ell_b, m_b + (L - 1)].T

        # Gather k nodes per row from per-ell table: (B, K)
        k_b = kln_nodes_by_ell[ell_b, :]

        res = forward_dsbt_chunk(f_b, ell_b, r, k_b)   # (B, K)
        out = out.at[idx].set(res)

    return out


def forward_dsbt_function(
    f_r: jnp.ndarray,             # (Nr,) — function evaluated on r grid
    r: jnp.ndarray,               # (Nr,)
    ell: int,
    kln_nodes: jnp.ndarray,       # (K,) — k-nodes for this ell
) -> jnp.ndarray:
    """
    Forward DSBT for a gridded f(r) at a single order ell.

    Same computation as forward_dsbt_all but takes a 1D array directly
    rather than a pre-computed rho_lm_r array, with no loop over (ell, m).

    Returns: (K,)
    """
    f_r = jnp.asarray(f_r)                         # (Nr,)
    f_b = f_r[None, :]                              # (1, Nr)
    ell_b = jnp.array([ell], dtype=jnp.int32)       # (1,)
    k_b = kln_nodes[None, :]                        # (1, K)
    res = forward_dsbt_chunk(f_b, ell_b, r, k_b)   # (1, K)
    return res[0]                                   # (K,)