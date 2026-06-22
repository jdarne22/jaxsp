
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import sys

sys.path.insert(0, "/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses")


import jax
#jax.config.update("jax_compilation_cache_dir", "/home/joshua/.jax_cache")
#jax.config.update("jax_persistent_cache_min_compile_time_secs", 60)
#jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_enable_x64", True)


#import Analytic_t_dep_sim as ATD

#import A_nl_altered_sims as A_nl

import Analytic_t_dep_sim_mem_saver as ATDS_MS

#import Analytic_t_dep_sim_mem_saver_a_nl as ATDS_MS_anl

import numpy as np
import jax.numpy as jnp
import jaxsp as jsp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd

from collections import defaultdict


import importlib
#importlib.reload(ATD)
#importlib.reload(A_nl)
importlib.reload(ATDS_MS)
#importlib.reload(ATDS_MS_anl)




SphHT = True
integrator = 'leapfrog'
plot = False
dt_override = 1
ramp_time = 0

l_band_size = 128
use_multi_gpu = True
sparse_k_batch=16384
r_chunk_size = 32
compute_dtype = jnp.complex64

L_out_frac = 0.5

m22 = 30
R0 = [1]

for R0 in R0:

    sim = ATDS_MS.StellarSimTDep(m22 = m22, r_half = R0, r_half_width = 0.05, no_of_particles = 100, no_time_steps = 1000, total_evolve_time = 10, r_min = 20, 
                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, SphHT = SphHT, integrator = integrator, 
                                plot = plot, dt_override=dt_override, ramp_time=ramp_time, l_band_size=l_band_size, use_multi_gpu = use_multi_gpu,
                                    sparse_k_batch=sparse_k_batch, r_chunk_size=r_chunk_size, compute_dtype=compute_dtype, L_out_frac=L_out_frac)

    sim.run_simulation(checkpoint_dir=f'/rds/general/user/jd925/ephemeral/Checkpoints/checkpoints_m22_{m22}_r0_{R0}_Lout_{L_out_frac}')


    positions_all = np.array([p.positions_xyz for p in sim.particles])  # (N_particles, N_steps+1, 3)
    r_all         = np.array([p.r_values      for p in sim.particles])  # (N_particles, N_steps+1)
    all_vels_cart = np.array([[np.array(v) for v in p.velocities_cart] for p in sim.particles])
    kinetic_energy_all = np.array([p.kinetic_energy for p in sim.particles]) # (N_particles, N_steps+1)
    potential_energy_all = np.array([p.potential_energy for p in sim.particles]) # (N_particles, N_steps+1)
    ang_mom_all = np.array([p.ang_mom for p in sim.particles]) # (N_particles, N_steps+1, 3)

    time_step = sim.time_step

    average_r = np.mean(r_all, axis=0)  # Average over particles

    no_time_steps = sim.no_time_steps

    x = np.arange(no_time_steps + 1)  # Time steps array


    plt.plot(x * sim.dt * sim.u.to_Gyr, average_r * sim.u.to_Kpc, label='Average Particle Radius')
    for particle in range(r_all.shape[0]):
        plt.plot(x * sim.dt * sim.u.to_Gyr, r_all[particle] * sim.u.to_Kpc, alpha = 0.2, color='gray')
    plt.axhline(sim.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')

    plt.xlabel('Time [Gyr]')
    plt.ylabel('Average Stellar Radius [Kpc]')
    plt.title('Average Stellar Radius over Time')

    # Timescale diagnostics
    v0 = np.sqrt(2 * kinetic_energy_all[:, 0])
    mean_T_orb = float(np.mean(2 * np.pi * r_all[:, 0] / v0) * sim.u.to_Gyr)

    lambda_db_kpc = 19.15 / (sim.m22 * v0 * sim.u.to_kms)
    T_c = lambda_db_kpc / (v0 * sim.u.to_Kpc) * sim.u.to_Gyr
    plt.axhline(np.mean(lambda_db_kpc), color='g', linestyle='--', label='$\\lambda_{{\\rm db}}$')

    E = np.array(sim.eigen_energies)
    freq_diff = np.abs(E[:, None] - E[None, :])
    T_beat = (2 * np.pi / freq_diff) * sim.u.to_Gyr
    min_T_beat = np.min(T_beat[np.isfinite(T_beat)])
    max_T_beat = np.max(T_beat[np.isfinite(T_beat)])

    dt_Gyr = sim.dt * sim.u.to_Gyr


    info = (
        f"$T_{{\\rm orb}}$ (mean) = {mean_T_orb:.3f} Gyr\n"
        f"$T_{{\\rm c}}$ = {float(np.mean(T_c)):.3f} Gyr\n"
        f"Beat time band: [{min_T_beat:.3f}, {max_T_beat:.3f}] Gyr\n"
        f"$\\Delta t$ = {dt_Gyr:.4f} Gyr"
    )
    plt.text(1.02, 0.98, info, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='paleturquoise', alpha=0.6))

    plt.legend(bbox_to_anchor=(1, 0.7))

    plt.savefig(f'/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/Plots/T_dep/t_dep_lf_m{m22}_R{R0}.png', dpi=300, bbox_inches='tight')
    plt.close()


    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for particle in range(ang_mom_all.shape[0]):
        ax[0].plot(x * sim.dt * sim.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
    ax[0].set_xlabel('Time [Gyr]')
    ax[0].set_ylabel('Total Energy [J]')
    ax[0].set_title('Total Energy over Time')




    for particle in range(ang_mom_all.shape[0]):
        ax[1].plot(x * sim.dt * sim.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
    ax[1].set_xlabel('Time [Gyr]')
    ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
    ax[1].set_title('Total angular momentum over Time')


    plt.tight_layout()
    plt.savefig(f'/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/Plots/T_dep/t_dep_lf_eng_amom_m{m22}_R{R0}.png', dpi=300, bbox_inches='tight')
    plt.close()

    data_dict = defaultdict(list)

    data_dict['time_Gyr'] = x * sim.dt * sim.u.to_Gyr
    for particle in range(r_all.shape[0]):
        data_dict[f'particle_{particle}_kinetic_energy_J'] = kinetic_energy_all[particle]
        data_dict[f'particle_{particle}_potential_energy_J'] = potential_energy_all[particle]
        data_dict[f'particle_{particle}_ang_mom_kg_m2_s'] = ang_mom_all[particle]
        data_dict[f'particle_{particle}_x_kpc'] = positions_all[particle][:, 0] * sim.u.to_Kpc
        data_dict[f'particle_{particle}_y_kpc'] = positions_all[particle][:, 1] * sim.u.to_Kpc
        data_dict[f'particle_{particle}_z_kpc'] = positions_all[particle][:, 2] * sim.u.to_Kpc
        data_dict[f'particle_{particle}_v_x_kms'] = all_vels_cart[particle][:, 0] * sim.u.to_kms
        data_dict[f'particle_{particle}_v_y_kms'] = all_vels_cart[particle][:, 1] * sim.u.to_kms
        data_dict[f'particle_{particle}_v_z_kms'] = all_vels_cart[particle][:, 2] * sim.u.to_kms

    df = pd.DataFrame(data_dict)

    df.to_csv(f'/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/Plots/T_dep/t_dep_lf_data_m{m22}_R{R0}.csv', index=False)




# SphHT = True
# integrator = 'leapfrog'
# plot = False
# dt_override = 1
# ramp_time = 0


# l_band_size = 128
# use_multi_gpu = True
# sparse_k_batch=16384
# r_chunk_size = 32
# compute_dtype = jnp.complex64


# L_sims = []

# m22 = 10


# for L_out_frac in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:

#   print(f"Running sim with L_out_frac = {L_out_frac}...")

#   sim = ATDS_MS.StellarSimTDep(m22 = 10, r_half = 1, r_half_width = 0.05, no_of_particles = 5, no_time_steps = 1000, total_evolve_time = 10, r_min = 20, 
#                                 r_max_enclosing_frac = 0.99, no_radius_bins = 1000, SphHT = SphHT, integrator = integrator, 
#                                 plot = plot, dt_override=dt_override, ramp_time=ramp_time, l_band_size=l_band_size, use_multi_gpu = use_multi_gpu,
#                                   sparse_k_batch=sparse_k_batch, r_chunk_size=r_chunk_size, compute_dtype=compute_dtype, L_out_frac=L_out_frac)


#   sim.run_simulation(checkpoint_dir=f'checkpoints_m22_{m22}_Lout_{L_out_frac}')

  
#   L_sims.append(sim)


# positions_all = []
# r_all = []
# all_vels_cart = []
# kinetic_energy_all = []
# potential_energy_all = []
# ang_mom_all = []

# for sim in L_sims:

#     positions_all.append(np.array([p.positions_xyz for p in sim.particles]))  # (N_particles, N_steps+1, 3)
#     r_all.append(np.array([p.r_values      for p in sim.particles]))  # (N_particles, N_steps+1)
#     all_vels_cart.append(np.array([[np.array(v) for v in p.velocities_cart] for p in sim.particles]))
#     kinetic_energy_all.append(np.array([p.kinetic_energy for p in sim.particles])) # (N_particles, N_steps+1)
#     potential_energy_all.append(np.array([p.potential_energy for p in sim.particles])) # (N_particles, N_steps+1)
#     ang_mom_all.append(np.array([p.ang_mom for p in sim.particles])) # (N_particles, N_steps+1, 3)


# no_time_steps = L_sims[0].no_time_steps


# x = np.arange(no_time_steps + 1)  # Time steps array

# for i, sim in enumerate(L_sims):

#     R_half = np.mean(r_all[i], axis=0)  # (N_steps+1,) 

#     plt.plot(x * sim.dt * sim.u.to_Gyr, R_half * sim.u.to_Kpc, label=f'Half-Mass Radius (L_out_frac={sim.L_out_frac})', alpha=0.7)
#     plt.xlabel('Time [Gyr]')
#     plt.ylabel('Average Stellar Radius [Kpc]')
#     plt.title('Average Stellar Radius over Time')

#     # Timescale diagnostics
#     v0 = np.linalg.norm(all_vels_cart[i][:, 0, :], axis=1)
#     mean_T_orb = float(np.mean(2 * np.pi * r_all[i][:, 0] / v0) * sim.u.to_Gyr)

#     lambda_db_kpc = 19.15 / (sim.m22 * v0 * sim.u.to_kms)
#     T_c = lambda_db_kpc / (v0 * sim.u.to_Kpc) * sim.u.to_Gyr

#     #plt.axhline(np.mean(lambda_db_kpc), color='g', linestyle='--', label='$\\lambda_{{\\rm db}}$')


#     E = np.array(sim.eigen_energies)
#     freq_diff = np.abs(E[:, None] - E[None, :])
#     T_beat = (2 * np.pi / freq_diff) * sim.u.to_Gyr
#     min_T_beat = np.min(T_beat[np.isfinite(T_beat)])
#     max_T_beat = np.max(T_beat[np.isfinite(T_beat)])

#     dt_Gyr = sim.dt * sim.u.to_Gyr

#     info = (
#         f"$T_{{\\rm orb}}$ (mean) = {mean_T_orb:.3f} Gyr\n"
#         f"$T_{{\\rm c}}$ = {float(np.mean(T_c)):.3f} Gyr\n"
#         f"Beat time band: [{min_T_beat:.3f}, {max_T_beat:.3f}] Gyr\n"
#         f"$\\Delta t$ = {dt_Gyr:.4f} Gyr"
#     )

#     plt.text(1.02, 0.98 - i * 0.2, info, transform=plt.gca().transAxes,
#             verticalalignment='top', fontsize=9,
#             bbox=dict(boxstyle='round', facecolor='paleturquoise', alpha=0.6))


    
# plt.axhline(L_sims[0].r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
    
# plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Plots/Reducing_L/L_reduc_m_22_{m22}.png', dpi=300, bbox_inches='tight')
# plt.close()


# R_half_truth = np.mean(r_all[0], axis=0)  # L_out_frac = 1
# errors = []
# for i, sim in enumerate(L_sims):
#     R_half = np.mean(r_all[i], axis=0)
#     error = np.abs(R_half - R_half_truth) / R_half_truth
#     errors.append(error)


# plt.figure(figsize=(10, 6))
# for i, sim in enumerate(L_sims):
#     plt.plot(x * sim.dt * sim.u.to_Gyr, errors[i], label=f'L_out_frac={sim.L_out_frac}', alpha=0.7)
# plt.xlabel('Time [Gyr]')
# plt.ylabel('Relative Error in Half-Mass Radius')
# plt.title('Relative Error in Half-Mass Radius over Time')
# plt.yscale('log')
# plt.legend()
# plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Plots/Reducing_L/L_reduc_error_m_22_{m22}.png', dpi=300, bbox_inches='tight')
# plt.close()
