import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"



from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.interpolate import interp1d
import numpy as np
import sys
sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxsp as jsp
from jaxsp.constants import GN


def compute_theoretical_trajectory(m22_val, R0_kpc, total_time_Gyr=10.0,
                                   C_heat=1.0, r_max_enclosing_frac=0.99):
    """
    Integrate the Eq 21-only (quasi-particle / outer-halo) heating prediction
    for a given fuzzy DM mass and starting radius. Returns (t [Gyr], R [kpc], diag).
    """
    u_loc = jsp.set_schroedinger_units(m22_val)

    cNFWtides_params = jnp.array([
        357964808.148399 * u_loc.from_Msun,
        25.690207,
        0.407461,
        0.012670 * u_loc.from_Kpc,
        1.857991 * u_loc.from_Kpc,
        3.729259,
    ])
    density_params = jsp.init_core_NFW_tides_params_from_sample(cNFWtides_params)

    rmin_pot = 0.1 * u_loc.from_pc
    rmax_pot = jsp.enclosing_radius(0.999, density_params)
    potential_params = jsp.init_potential_params(density_params, rmin_pot, rmax_pot, 512)

    rmax_lib = jsp.enclosing_radius(r_max_enclosing_frac, density_params)
    eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin_pot, rmax_lib, 1, 10, 1024)

    rmin_wf = 20 * u_loc.from_pc
    wavefunction_params = jsp.init_wavefunction_params(
        eigenstate_lib, density_params, rmin_wf, rmax_lib, 1e-7
    )

    r_grid = np.logspace(np.log10(rmin_wf), np.log10(rmax_lib), 1000)
    rho_psi_vmap = jax.vmap(jsp.rho_psi, in_axes=(0, None, None))
    rho_r = np.array(rho_psi_vmap(r_grid, wavefunction_params, eigenstate_lib))
    M_r = 4 * np.pi * cumulative_trapezoid(r_grid ** 2 * rho_r, r_grid, initial=0)

    rho_of_R = interp1d(r_grid, rho_r, kind='cubic', fill_value='extrapolate')
    M_of_R = interp1d(r_grid, M_r, kind='cubic', fill_value='extrapolate')

    hbar_c = u_loc.from_hbar
    m_a_c = m22_val * u_loc.from_m22
    G_c = GN.value * u_loc.from_cm ** 3 / (u_loc.from_g * u_loc.from_s ** 2)

    R_E = np.logspace(np.log10(r_grid[0]), np.log10(r_grid[-1]), 4000)
    M_E = np.asarray(M_of_R(R_E))
    rho_E = np.asarray(rho_of_R(R_E))
    U_arr = cumulative_trapezoid(G_c * M_E / R_E ** 2, R_E, initial=0.0)
    T_arr = 0.5 * G_c * M_E / R_E
    E_arr = U_arr + T_arr
    dEdR_arr = G_c * M_E / (2 * R_E ** 2) + 2 * np.pi * G_c * rho_E * R_E
    if not np.all(np.diff(E_arr) > 0):
        raise RuntimeError(f'E(R) non-monotonic for m22={m22_val}')
    dEdR_of_R = interp1d(R_E, dEdR_arr, kind='cubic', fill_value='extrapolate')

    def dE_dt_eq21(R_val):
        M = float(M_of_R(R_val))
        rho = float(rho_of_R(R_val))
        if M <= 0 or rho <= 0:
            return 0.0
        sigma = np.sqrt(G_c * M / R_val)
        return (C_heat * G_c ** 2 * np.pi ** 3 * hbar_c ** 3 * rho ** 2
                / (m_a_c ** 3 * sigma ** 4))

    r_min_i, r_max_i = float(r_grid[0]), float(r_grid[-1])

    def dR_dt(t, y):
        R_val = float(np.clip(y[0], r_min_i, r_max_i))
        if float(M_of_R(R_val)) <= 0:
            return [0.0]
        return [dE_dt_eq21(R_val) / float(dEdR_of_R(R_val))]

    def hit_boundary(t, y):
        return r_max_i - y[0]
    hit_boundary.terminal = True
    hit_boundary.direction = -1

    R0 = R0_kpc * u_loc.from_Kpc
    sol = solve_ivp(
        dR_dt, (0, total_time_Gyr * u_loc.from_Gyr), [R0],
        method='RK45', rtol=1e-8, atol=1e-12,
        max_step=float(0.01 * u_loc.from_Gyr),
        events=hit_boundary, dense_output=True,
    )

    t_arr = np.linspace(0, sol.t[-1], 500)
    R_arr = sol.sol(t_arr)[0]

    sigma0 = np.sqrt(G_c * float(M_of_R(R0)) / R0)
    lam_dB0 = hbar_c / (m_a_c * sigma0)
    diag = {
        'sigma_kms': sigma0 * u_loc.to_kms,
        'lambda_dB_kpc': lam_dB0 * u_loc.to_Kpc,
        'R_over_lambda_dB': R0 / lam_dB0,
        'tau_dB_Myr': (lam_dB0 / sigma0) * u_loc.to_Gyr * 1000,
        'rho0_Msun_kpc3': float(rho_of_R(R0)) * u_loc.to_Msun / u_loc.to_Kpc ** 3,
        'R_final_kpc': R_arr[-1] * u_loc.to_Kpc,
        'delta_R_over_R': (R_arr[-1] - R0) / R0,
    }
    return t_arr * u_loc.to_Gyr, R_arr * u_loc.to_Kpc, diag


m22_list = [1, 2, 3, 10]
R0_list_kpc = [0.19, 1.0, 2.0]

results = {}
for m22_v in m22_list:
    for R0_v in R0_list_kpc:
        print(f'm22={m22_v:>3}, R0={R0_v:.1f} kpc ...', end=' ', flush=True)
        t_arr, R_arr, diag = compute_theoretical_trajectory(m22_v, R0_v)
        results[(m22_v, R0_v)] = (t_arr, R_arr, diag)
        print(f"sigma={diag['sigma_kms']:5.1f} km/s, "
              f"lam_dB={diag['lambda_dB_kpc']:6.3f} kpc, "
              f"R/lam={diag['R_over_lambda_dB']:5.1f}, "
              f"tau_dB={diag['tau_dB_Myr']:6.2f} Myr, "
              f"dR/R={diag['delta_R_over_R']*100:+6.1f}%")
    

import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(m22_list), len(R0_list_kpc),
                          figsize=(4 * len(R0_list_kpc), 2.8 * len(m22_list)),
                          sharex=True)
for i, m22_v in enumerate(m22_list):
    for j, R0_v in enumerate(R0_list_kpc):
        ax = axes[i, j]
        t_arr, R_arr, diag = results[(m22_v, R0_v)]
        ax.plot(t_arr, R_arr, lw=2)
        ax.axhline(R0_v, color='red', linestyle='--', alpha=0.5, label=f'R0={R0_v} kpc')
        # highlight if in granular regime AND visibly heating
        granular = diag['R_over_lambda_dB'] >= 5
        visible = abs(diag['delta_R_over_R']) >= 0.2
        if granular and visible:
            for spine in ax.spines.values():
                spine.set_edgecolor('green'); spine.set_linewidth(2.5)
        ax.set_title(
            f"m22={m22_v}, R0={R0_v} kpc\n"
            f"R/lam_dB={diag['R_over_lambda_dB']:.1f}, dR/R={diag['delta_R_over_R']*100:+.1f}%",
            fontsize=9,
        )
        if i == len(m22_list) - 1:
            ax.set_xlabel('t [Gyr]')
        if j == 0:
            ax.set_ylabel('R [kpc]')

plt.suptitle('Eq 21-only heating prediction (green = granular AND visible)', y=1.00)
plt.tight_layout()
plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/theoretical_heating_plot.png', dpi=300)

# summary table
print('\nSummary (sorted by delta_R/R, descending):')
print(f"{'m22':>4} {'R0 kpc':>7} {'R/lam':>7} {'sigma':>7} {'tau_dB Myr':>11} {'dR/R %':>8}")
rows = sorted(results.items(), key=lambda kv: -kv[1][2]['delta_R_over_R'])
for (m22_v, R0_v), (_, _, d) in rows:
    print(f"{m22_v:>4} {R0_v:>7.1f} {d['R_over_lambda_dB']:>7.1f} "
          f"{d['sigma_kms']:>7.1f} {d['tau_dB_Myr']:>11.2f} {d['delta_R_over_R']*100:>+8.1f}")