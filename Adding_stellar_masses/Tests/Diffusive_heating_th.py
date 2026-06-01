import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pickle

from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
import numpy as np
import sys
sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxsp as jsp
from jaxsp.constants import GN, hbar


def compute_theoretical_trajectory(m22, R0_kpc, total_time_Gyr=10.0,
                                   C_heat=1.0, b_heat=1/60, r_max_enclosing_frac=0.99):
    """
    Integrate the Eq 21-only (quasi-particle / outer-halo) heating prediction
    for a given fuzzy DM mass and starting radius. Returns (t [Gyr], R [kpc], info).
    """
    u = jsp.set_schroedinger_units(m22)

    # ------------------------------------------------------------------
    # Load (or build + cache) the eigenstate library and wavefunction.
    # These depend only on m22 and r_max_enclosing_frac, not on R0, so
    # without this cache every (m22, R0) pair rebuilds the same objects
    # from scratch. Same disk-cache technique as
    # Testing_sim_methods/Analytic_t_dep_sim_Ylm_skip_mem_saver.py.
    # ------------------------------------------------------------------
    # The eigenstate cache is co-located with the builder script
    # (Testing_sim_methods/Analytic_t_dep_sim_Ylm_skip_mem_saver.py), not this one.
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "Testing_sim_methods", "precomputed_wf")
    os.makedirs(cache_dir, exist_ok=True)

    cache_suffix = f"m22_{float(m22):.6g}_rbins_{int(1000)}"
    r_j_r_fname = os.path.join(cache_dir, f"precomputed_R_j_r_{cache_suffix}.npz")
    pkl_fname   = os.path.join(cache_dir, f"precomputed_objs_{cache_suffix}.pkl")

    cache_params = {
        'm22': float(m22),
        'r_min': float(20),
        'r_max_enclosing_frac': float(r_max_enclosing_frac),
        'no_radius_bins': int(1000),
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

    eigenstate_lib = None
    if os.path.isfile(r_j_r_fname) and os.path.isfile(pkl_fname):
        data = np.load(r_j_r_fname)
        if _cache_valid(data, cache_params):
            print(f"Loaded cached wavefunction objects from {pkl_fname}.")
            with open(pkl_fname, 'rb') as f:
                objs = pickle.load(f)
            eigenstate_lib      = objs['eigenstate_lib']
            wavefunction_params = objs['wavefunction_params']
            rmax_lib            = float(data['rmax'])  # builder pkl doesn't store rmax_lib
        else:
            print(f"Cached {r_j_r_fname} stale (parameter mismatch); recomputing.")


    rmin_wf = 20 * u.from_pc


    if eigenstate_lib is None:
        cNFWtides_params = jnp.array([
            357964808.148399 * u.from_Msun,
            25.690207,
            0.407461,
            0.012670 * u.from_Kpc,
            1.857991 * u.from_Kpc,
            3.729259,
        ])
        density_params = jsp.init_core_NFW_tides_params_from_sample(cNFWtides_params)

        rmin_pot = 0.1 * u.from_pc
        rmax_pot = jsp.enclosing_radius(0.999, density_params)
        potential_params = jsp.init_potential_params(density_params, rmin_pot, rmax_pot, 512)

        rmax_lib = jsp.enclosing_radius(r_max_enclosing_frac, density_params)
        eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin_pot, rmax_lib, 1, 10, 1024)

        wavefunction_params = jsp.init_wavefunction_params(
            eigenstate_lib, density_params, rmin_wf, rmax_lib, 1e-7
        )

        with open(pkl_fname, 'wb') as f:
            pickle.dump({
                'eigenstate_lib': eigenstate_lib,
                'wavefunction_params': wavefunction_params,
                'rmax_lib': float(rmax_lib),
                'cache_params': cache_params,
            }, f)
        print(f"Cached wavefunction objects to {pkl_fname}.")

    r_grid = np.logspace(np.log10(rmin_wf), np.log10(rmax_lib), 1000)

    rho_psi_vmap = jax.vmap(jsp.rho_psi, in_axes=(0, None, None))

    rho_ULDM = np.array(rho_psi_vmap(r_grid, wavefunction_params, eigenstate_lib))

    M_ULDM = 4 * np.pi * cumulative_trapezoid(r_grid ** 2 * rho_ULDM, r_grid, initial=0)

    rho_ULDM_interp = interp1d(r_grid, rho_ULDM, kind='cubic', fill_value='extrapolate')

    r_c_tol = 0.95 * rho_ULDM[0]
    r_c_idx = np.where(rho_ULDM < r_c_tol)[0][0]
    r_c = r_grid[r_c_idx]
    print(f"Core radius r_c = {r_c * u.to_Kpc:.3f} kpc (where rho drops to {r_c_tol:.2e} g/cm^3))")


    M_ULDM_interp = interp1d(r_grid, M_ULDM, kind='cubic', fill_value='extrapolate')

    hbar_c = u.from_hbar
    m_a_c = m22 * u.from_m22
    G_c = GN.value * u.from_cm ** 3 / (u.from_g * u .from_s ** 2)
    

    # ANDREWS THEORY

    R_grid = np.logspace(np.log10(r_grid[0]), np.log10(r_grid[-1]), 4000)
    M_E = np.asarray(M_ULDM_interp(R_grid))
    rho_E = np.asarray(rho_ULDM_interp(R_grid))

    r_min_i, r_max_i = float(r_grid[0]), float(r_grid[-1])

    dEdR_arr = G_c * M_E / (2 * R_grid ** 2) + 2 * np.pi * G_c * rho_E * R_grid

    dEdR_interp = interp1d(R_grid, dEdR_arr, kind='cubic', fill_value='extrapolate')



    def dR_dt(t, y):
        R_val = y[0]
        M = float(M_ULDM_interp(R_val))
        sigma = np.sqrt(G_c * M / R_val)
        local_regime = R_val / r_c

        if local_regime > 5:
            rho = float(rho_ULDM_interp(R_val))
            dE = (C_heat * G_c ** 2 * np.pi ** 3 * hbar_c ** 3 * rho ** 2
                  / (m_a_c ** 3 * sigma ** 4))
            return [dE / float(dEdR_interp(R_val))]

        elif local_regime < 1:
            return [b_heat / 2 * sigma]

        else:
            dE = b_heat * G_c * M / R_val ** 2 * sigma
            return [dE / float(dEdR_interp(R_val))]
    


    def hit_boundary(t, y):
        return r_max_i - y[0]
    hit_boundary.terminal = True
    hit_boundary.direction = -1
    


    R0 = R0_kpc * u.from_Kpc
    # solve_ivp(func, t_span, y0, method='RK45', rtol=1e-8, 
    # atol=1e-12, max_step=0.01 * u.from_Gyr, events=hit_boundary)
    # for dy / dt = f(t, y), t in the interval [t0, tf], with initial value y0.
    sol = solve_ivp(
        dR_dt, (0, total_time_Gyr * u.from_Gyr), [R0],
        method='RK45', rtol=1e-8, atol=1e-12,
        max_step=float(0.01 * u.from_Gyr),
        events=hit_boundary, dense_output=True,
    )

    t_arr = np.linspace(0, sol.t[-1], 500)

    # if dense_output=True, then sol.sol is a function that evaluates the 
    # solution at any point in the interval [t0, tf]
    R_arr = sol.sol(t_arr)[0] 

    sigma0 = np.sqrt(G_c * float(M_ULDM_interp(R0)) / R0)
    lam_dB0 = hbar_c / (m_a_c * sigma0)
    info = {
        'sigma_kms': sigma0 * u.to_kms,
        'lambda_dB_kpc': lam_dB0 * u.to_Kpc,
        'r_c_kpc': r_c * u.to_Kpc,
        'R_over_r_c': R0 / r_c,
        'tau_dB_Myr': (lam_dB0 / sigma0) * u.to_Gyr * 1000,
        'R_final_kpc': R_arr[-1] * u.to_Kpc,
        'delta_R_over_R': (R_arr[-1] - R0) / R0,
        'u': u,
    }
    return t_arr, R_arr, info


m22_list = [1, 2, 5, 10]
R0_list_kpc = [0.19, 0.3, 0.5, 0.7, 1.0]

results = {}
for m22 in m22_list:
    for R0 in R0_list_kpc:
        print(f'm22={m22:>3}, R0={R0:.1f} kpc ...', end=' ', flush=True)
        t_arr, R_arr, info = compute_theoretical_trajectory(m22, R0)
        results[(m22, R0)] = (t_arr, R_arr, info)
        print(f"sigma={info['sigma_kms']:5.1f} km/s, "
              f"lam_dB={info['lambda_dB_kpc']:6.3f} kpc, "
              f"R/r_c={info['R_over_r_c']:5.1f}, "
              f"tau_dB={info['tau_dB_Myr']:6.2f} Myr, "
              f"dR/R={info['delta_R_over_R']*100:+6.1f}%")
    

import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(m22_list), len(R0_list_kpc),
                          figsize=(4 * len(R0_list_kpc), 2.8 * len(m22_list)),
                          sharex=True)
for i, m22 in enumerate(m22_list):
    for j, R0 in enumerate(R0_list_kpc):
        ax = axes[i, j]
        t_arr, R_arr, info = results[(m22, R0)]
        u = info['u']
        ax.plot(t_arr * u.to_Gyr, R_arr * u.to_Kpc, lw=2)
        ax.axhline(R0, color='red', linestyle='--', alpha=0.5, label=f'R0={R0} kpc')
        # highlight if in granular regime AND visibly heating
        granular = info['R_over_r_c'] >= 5
        visible = abs(info['delta_R_over_R']) >= 0.2
        if granular and visible:
            for spine in ax.spines.values():
                spine.set_edgecolor('green'); spine.set_linewidth(2.5)
        ax.set_title(
            f"m22={m22}, R0={R0} kpc\n"
            f"R/r_c={info['R_over_r_c']:.1f}, dR/R={info['delta_R_over_R']*100:+.1f}%",
            fontsize=9,
        )
        if i == len(m22_list) - 1:
            ax.set_xlabel('t [Gyr]')
        if j == 0:
            ax.set_ylabel('R [kpc]')

plt.suptitle('Eq 21-only heating prediction (green = granular AND visible)', y=1.00)
plt.tight_layout()
plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/th_heating_w_soliton_plot.png', dpi=300)

# summary table
print('\nSummary (sorted by delta_R/R, descending):')
print(f"{'m22':>4} {'R0 kpc':>7} {'R/r_c':>7} {'sigma':>7} {'tau_dB Myr':>11} {'dR/R %':>8}")
rows = sorted(results.items(), key=lambda kv: -kv[1][2]['delta_R_over_R'])
for (m22, R0), (_, _, info) in rows:
    print(f"{m22:>4} {R0:>7.1f} {info['R_over_r_c']:>7.1f} "
          f"{info['sigma_kms']:>7.1f} {info['tau_dB_Myr']:>11.2f} {info['delta_R_over_R']*100:>+8.1f}")