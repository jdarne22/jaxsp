
import jax
jax.config.update("jax_enable_x64", True)

import functools
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
import s2fft
from s2fft.recursions.price_mcewen import generate_precomputes_jax
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

# The associated-Legendre table is built directly at the shape we actually
# need - see compute_Ylm_and_dtheta_jit for why - by normalised_legendre_table
# below, which is jax._src.scipy.special._gen_associated_legendre's recurrence
# without its (n_max+1)^3 coefficient masks. See that function for what those
# cost.

# Vocabulary used throughout this file:
#   radial mode j : one radial eigenfunction R_j(r), with a fixed angular
#            momentum l_j and energy E_j.
#   k-mode : one basis function - a radial mode j paired with an m in
#            -l_j .. l_j. So radial mode j contributes 2*l_j + 1 k-modes, each
#            carrying its own random-phase amplitude.


def build_lm_pairs(l_per_radial_mode):
    """
    Every distinct (l, m) the density expansion needs a coefficient for,
    ordered by l ascending and, within an l, by m from -l to l.

    Only *which* l values occur matters, not how many radial modes share each
    one, so this is built from the distinct l values directly - never from the
    (vastly longer) list of k-modes.
    """
    l_values = np.unique(np.asarray(l_per_radial_mode))
    return np.concatenate([
        np.stack([np.full(2 * l + 1, l), np.arange(-l, l + 1)], axis=1)
        for l in l_values
    ]).astype(np.int32)


class RadialModeGroups(NamedTuple):
    """
    Where each l's radial modes and amplitudes sit in the flat arrays, so the
    density can be evaluated one l at a time with no gathers at all.

    Two facts about the library's layout make this work:

      - radial modes arrive sorted by l, so all the modes sharing an l are one
        contiguous slice of the radial-mode axis;
      - radial mode j's 2*l_j + 1 amplitudes are stored consecutively, with m
        running -l_j .. l_j.

    Together those mean one l's amplitudes are a single contiguous run of the
    amplitude array, and reshaping that run to (modes_at_l, 2l+1) recovers
    them indexed by (radial mode, m + l).

    Every field is a tuple of plain Python ints so they work as static slice
    bounds inside jit.
    """
    l_values: tuple
    first_radial_mode: tuple
    modes_at_l: tuple
    first_amplitude: tuple


def group_radial_modes_by_l(l_per_radial_mode):
    """Build the RadialModeGroups table. Runs once at setup, pure numpy."""
    l_per_radial_mode = np.asarray(l_per_radial_mode)

    if np.any(np.diff(l_per_radial_mode) < 0):
        raise ValueError(
            "radial modes must be sorted by l - psi_lm_at_rows slices each l's "
            "modes out as a contiguous range, which assumes that ordering.")

    k_modes_per_radial_mode = 2 * l_per_radial_mode.astype(np.int64) + 1
    first_amplitude_per_radial_mode = np.concatenate(
        [[0], np.cumsum(k_modes_per_radial_mode)[:-1]])

    l_values = np.unique(l_per_radial_mode)
    first = np.searchsorted(l_per_radial_mode, l_values, side='left')
    last = np.searchsorted(l_per_radial_mode, l_values, side='right')

    return RadialModeGroups(
        l_values=tuple(int(l) for l in l_values),
        first_radial_mode=tuple(int(i) for i in first),
        modes_at_l=tuple(int(n) for n in last - first),
        first_amplitude=tuple(int(first_amplitude_per_radial_mode[i]) for i in first),
    )


def psi_lm_at_rows(amplitudes, phase_per_radial_mode, R_at_rows, groups):
    """
    The wavefunction's (l, m) coefficients at every row of R_at_rows - rows
    being background radii, or particle radii:

        psi_lm[row, (l, m)] = sum over radial modes j with l_j = l of
                              amplitude[j, m] * phase[j] * R_at_rows[row, j]

    One dense matmul per l. Because each l's amplitudes are a contiguous,
    reshapeable slice and its radial modes are a contiguous column slice of
    R_at_rows (see RadialModeGroups), there is no gather, no sort and no
    scatter anywhere - just (n_rows x modes_at_l) @ (modes_at_l x 2l+1).

    Columns come out in exactly build_lm_pairs' order.

    On determinism: the sum over radial modes has to be reproducible run to
    run, because the simulation is chaotic and a 1e-7 difference grows to O(1)
    by late times. A matmul reduces in an order fixed at compile time, so this
    is reproducible for the same executable - which is what the old
    sort-then-cumsum scheme was protecting against (a scatter-add would have
    used GPU atomics, whose order varies between runs).

    On precision: HIGHEST is not optional. JAX's default float32 matmul
    precision lets XLA run the product on tensor cores in TF32, which carries
    10 mantissa bits instead of 24 - a ~1e-3 relative error, against the ~1e-7
    of the explicit summation this replaced. That is far too coarse for a
    chaotic system, and it would have been invisible in any short test.
    """
    psi_lm_per_l = []

    for l, first_mode, n_modes, first_amp in zip(*groups):
        m_width = 2 * l + 1
        modes = slice(first_mode, first_mode + n_modes)

        amplitudes_at_l = amplitudes[first_amp:first_amp + n_modes * m_width]
        amplitudes_at_l = amplitudes_at_l.reshape(n_modes, m_width)

        # Fold the phase into the amplitudes rather than into R. Either is
        # correct, but this one leaves R real, which is what lets the matmul
        # below split into two real products instead of the four a
        # complex-times-complex product needs.
        amplitudes_at_l = amplitudes_at_l * phase_per_radial_mode[modes, None]

        R_at_l = R_at_rows[:, modes]
        exact = jax.lax.Precision.HIGHEST
        psi_lm_per_l.append(jax.lax.complex(
            jnp.matmul(R_at_l, amplitudes_at_l.real, precision=exact),
            jnp.matmul(R_at_l, amplitudes_at_l.imag, precision=exact)))

    return jnp.concatenate(psi_lm_per_l, axis=1)


class SHTPrecomputes(NamedTuple):
    """Wigner-d recursion coefficients for the two transforms of the round trip.

    Left to itself s2fft rebuilds these *inside every transform call* - see the
    `if precomps is None` in s2fft.transforms.otf_recursions - and they depend
    only on (L, spin, sampling), all fixed for a whole run. Building them once
    and handing them in is bit-identical; it just skips the recursion.

    ~49 MB for both at L_out = 602.
    """
    inverse: list
    forward: list


def build_sht_precomputes(L_out):
    """Build the SHT recursion coefficients once, at setup."""
    return SHTPrecomputes(
        inverse=generate_precomputes_jax(L_out, 0, 'mw', None, forward=False),
        forward=generate_precomputes_jax(L_out, 0, 'mw', None, forward=True),
    )


def _density_from_psi_lm(psi_lm_rows, lm_pairs, total_mass, L_out, cdtype, sht):
    """
    Turns psi_lm (one row per radius or per particle) into density
    on the McEwen-Wiaux angular grid: scatter the coefficients into an
    (l, m) grid, inverse spherical-harmonic-transform to real space, then
    square.
    """
    n_rows = psi_lm_rows.shape[0]
    flm = jnp.zeros((n_rows, L_out, 2 * L_out - 1), dtype=cdtype)
    flm = flm.at[:, lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(psi_lm_rows)

    # reality stays False here: psi is genuinely complex.
    psi = jax.vmap(
        lambda f: s2fft.inverse(f, L_out, sampling='mw', method='jax',
                                precomps=sht.inverse)
    )(flm)
    return total_mass * (jnp.abs(psi) ** 2)


def _density_then_forward_sht(psi_lm_rows, lm_pairs, total_mass, L_out, cdtype, sht):
    """_density_from_psi_lm, then transformed back to (l, m) space -
    used everywhere the final answer needs to be rho_lm rather than
    rho(theta, phi).

    reality=True because rho is |psi|^2, which is real by construction. s2fft
    then does the longitudinal FFT as an rfft and runs the latitudinal
    recursion over m >= 0 only, filling the m < 0 coefficients back in from
    Hermitian symmetry itself - so the result is still the full (L, 2L-1)
    array, to round-off.
    """
    rho = _density_from_psi_lm(psi_lm_rows, lm_pairs, total_mass, L_out, cdtype, sht)
    return jax.vmap(
        lambda r: s2fft.forward(r, L_out, sampling='mw', method='jax',
                                reality=True, precomps=sht.forward)
    )(rho).astype(cdtype)


@functools.partial(jax.jit, static_argnames=("L_out", "groups", "r_chunk", "out_sharding", "sph_sym"))
def build_sphht_rho_lms_jit(R_j_r_fixed, phase_c, amplitudes,
                              lm_pairs, total_mass,
                              ramp_c, rho_static_r_l00, sht,
                              groups, L_out, r_chunk,
                              out_sharding=None, sph_sym=False):
    """
    Builds rho_lm(r): the density, as a spherical-harmonic expansion, at
    every radius, blended against the static background by the ramp fraction:

        rho_lm = (1 - ramp_c) * rho_static_lm + ramp_c * rho_full_lm

    Two stages. psi_lm at every radius comes first, in one shot - it is only
    (n_radii, n_lm_pairs) and psi_lm_at_rows is matmul-bound, so there is
    nothing to gain by splitting it. The s2fft round trip that turns psi_lm
    into rho_lm is then streamed r_chunk radii at a time, because *that* is
    what needs the memory: the result alone is (n_radii, L_out, 2*L_out - 1).

    sph_sym keeps only the (l=0, m=0) coefficient of the result, i.e.
    replaces rho by its angular average at each radius.

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
    cdtype = jnp.result_type(amplitudes.dtype, phase_c.dtype, R_j_r_fixed.dtype)

    psi_lm = psi_lm_at_rows(amplitudes, phase_c, R_j_r_fixed, groups)  # (n_radii, n_lm_pairs)

    n_chunks = (n_radii + r_chunk - 1) // r_chunk
    n_radii_padded = n_chunks * r_chunk
    pad_amount = n_radii_padded - n_radii

    # Zero-padded rows give zero psi, hence zero rho, so the padding needs no
    # masking later - it is simply sliced off at the end.
    if pad_amount > 0:
        psi_lm_padded = jnp.concatenate(
            [psi_lm, jnp.zeros((pad_amount, psi_lm.shape[1]), dtype=psi_lm.dtype)], axis=0)
        rho_static_padded = jnp.concatenate(
            [rho_static_r_l00, jnp.zeros((pad_amount,), dtype=rho_static_r_l00.dtype)], axis=0)
    else:
        psi_lm_padded = psi_lm
        rho_static_padded = rho_static_r_l00

    def density_then_forward_sht(psi_lm_rows):
        return _density_then_forward_sht(psi_lm_rows, lm_pairs, total_mass, L_out, cdtype, sht)

    def rho_lm_for_chunk(psi_lm_rows):
        if out_sharding is None:
            return density_then_forward_sht(psi_lm_rows)

        mesh = out_sharding.mesh
        psi_lm_rows = jax.lax.with_sharding_constraint(
            psi_lm_rows, NamedSharding(mesh, P('x', None)))

        rho_lm_chunk = shard_map(
            density_then_forward_sht, mesh=mesh,
            in_specs=P('x', None),
            out_specs=P('x', None, None),
            check_rep=False,
        )(psi_lm_rows)

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
        psi_lm_chunk = jax.lax.dynamic_slice_in_dim(
            psi_lm_padded, chunk_index * r_chunk, r_chunk, axis=0)
        rho_lm_chunk = rho_lm_for_chunk(psi_lm_chunk)

        if sph_sym:
            # Keep the monopole, drop every other coefficient. It has to be
            # rho's (l, m) coefficients that are zeroed, not psi's: every
            # psi_lm feeds the monopole of |psi|^2, so zeroing psi_lm for
            # l > 0 would change rho_00 as well as removing the anisotropy.
            rho_lm_chunk = jnp.zeros_like(rho_lm_chunk).at[:, 0, L_out - 1].set(
                rho_lm_chunk[:, 0, L_out - 1])

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


@functools.partial(jax.jit, static_argnames=("L_out", "groups", "p_chunk", "sph_sym"))
def compute_rho_lm_at_particles_sphht_jit(R_j_at_parts, phase_c, amplitudes,
                                            lm_pairs, total_mass, sht,
                                            groups, L_out,
                                            p_chunk=None, sph_sym=False):
    """
    rho_lm evaluated at a set of particle radii: the same calculation as
    build_sphht_rho_lms_jit, just with R_j_at_parts (one row per particle)
    in place of R_j_r_fixed (one row per background radius).

    Same two stages, for the same reason. psi_lm for every particle at once,
    then the s2fft round trip streamed p_chunk particles at a time - that
    round trip costs about 0.2 GB of working set per row at L_out = 892 (and
    in complex128; s2fft hardcodes that dtype, so compute_dtype does not
    shrink it), so all 1000 particles in one shot would need ~205 GB.

    Every row is independent all the way through, so the chunking is exact:
    each particle gets identical values whatever p_chunk is.

    sph_sym masks the result exactly as build_sphht_rho_lms_jit does, so
    the particle rows stay consistent with the background grid they get
    spliced into.
    """
    cdtype = jnp.result_type(amplitudes.dtype, phase_c.dtype, R_j_at_parts.dtype)

    psi_lm = psi_lm_at_rows(amplitudes, phase_c, R_j_at_parts, groups)

    n_particles = psi_lm.shape[0]
    p_chunk = n_particles if not p_chunk else max(1, min(int(p_chunk), n_particles))
    n_chunks = (n_particles + p_chunk - 1) // p_chunk
    n_particles_padded = n_chunks * p_chunk
    pad_amount = n_particles_padded - n_particles

    # Zero-padded rows give zero rho, so they cost one chunk slot and need no
    # masking afterwards - they are just sliced off at the end.
    if pad_amount > 0:
        psi_lm = jnp.concatenate(
            [psi_lm, jnp.zeros((pad_amount, psi_lm.shape[1]), dtype=psi_lm.dtype)], axis=0)

    def rho_lm_for_chunk(psi_lm_rows):
        rho_lm_chunk = _density_then_forward_sht(
            psi_lm_rows, lm_pairs, total_mass, L_out, cdtype, sht)

        if sph_sym:
            rho_lm_chunk = jnp.zeros_like(rho_lm_chunk).at[:, 0, L_out - 1].set(
                rho_lm_chunk[:, 0, L_out - 1])

        return rho_lm_chunk

    if n_chunks == 1:
        return rho_lm_for_chunk(psi_lm)

    # Default batch_size (None) makes this a scan - one chunk's working set
    # live at a time, which is the whole point.
    rho_lm_chunks = jax.lax.map(
        rho_lm_for_chunk, psi_lm.reshape(n_chunks, p_chunk, psi_lm.shape[1]))

    return rho_lm_chunks.reshape(
        n_particles_padded, L_out, 2 * L_out - 1)[:n_particles]
