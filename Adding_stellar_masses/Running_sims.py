
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import sys

sys.path.insert(0, "/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses")


import jax
jax.config.update("jax_compilation_cache_dir", "/home/joshua/.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 60)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_enable_x64", True)


import Analytic_t_dep_sim as ATD

import A_nl_altered_sims as A_nl

import Analytic_t_dep_sim_mem_saver as ATDS_MS

import Analytic_t_dep_sim_mem_saver_a_nl as ATDS_MS_anl

import numpy as np
import jax.numpy as jnp
import jaxsp as jsp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import pandas as pd

from collections import defaultdict



import importlib
importlib.reload(ATD)
importlib.reload(A_nl)
importlib.reload(ATDS_MS)
importlib.reload(ATDS_MS_anl)



#----------------------------------------------------------------------------------

# animate = False
# static = True
# frozen = False
# SphHT = False
# plot = False
# integrator = 'ias15'

# static_test_ias15 = ATG.StellarSimTDep(m22 = 1, r_half = 0.19, no_of_particles = 15, no_time_steps = 100, total_evolve_time = 5, r_min = 20, 
#                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, static = static, frozen = frozen, SphHT = SphHT, integrator = integrator, plot = plot, animate=animate, animate_every=10)


# static_test_ias15.run_simulation()


# positions_all = np.array([p.positions_xyz for p in static_test_ias15.particles])  # (N_particles, N_steps+1, 3)
# r_all         = np.array([p.r_values      for p in static_test_ias15.particles])  # (N_particles, N_steps+1)
# v_disp_all    = np.array([p.stellar_v_disp for p in static_test_ias15.particles]) # (N_particles, N_steps+1)
# kinetic_energy_all = np.array([p.kinetic_energy for p in static_test_ias15.particles]) # (N_particles, N_steps+1)
# potential_energy_all = np.array([p.potential_energy for p in static_test_ias15.particles]) # (N_particles, N_steps+1)
# ang_mom_all = np.array([p.ang_mom for p in static_test_ias15.particles]) # (N_particles, N_steps+1, 3)

# time_step2 = static_test_ias15.time_step
# stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

# average_r2 = np.mean(r_all, axis=0)  # Average over particles


# x = np.linspace(0, time_step2, len(stellar_v_disp2))

# plt.plot(x * static_test_ias15.dt * static_test_ias15.u.to_Gyr, average_r2 * static_test_ias15.u.to_Kpc, label='Average Particle Radius')
# for particle in range(r_all.shape[0]):
#     plt.plot(x * static_test_ias15.dt * static_test_ias15.u.to_Gyr, r_all[particle] * static_test_ias15.u.to_Kpc, alpha = 0.2, color='gray')
# plt.axhline(static_test_ias15.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
# plt.xlabel('Time [Gyr]')
# plt.ylabel('Average Stellar Radius [Kpc]')
# plt.title('Average Stellar Radius over Time')
# plt.legend()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/static_ias15.png', dpi=300)
# plt.close()


# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# for particle in range(ang_mom_all.shape[0]):
#     ax[0].plot(x * static_test_ias15.dt * static_test_ias15.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
# ax[0].set_xlabel('Time [Gyr]')
# ax[0].set_ylabel('Total Energy [J]')
# ax[0].set_title('Total Energy over Time')




# for particle in range(ang_mom_all.shape[0]):
#     ax[1].plot(x * static_test_ias15.dt * static_test_ias15.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
# ax[1].set_xlabel('Time [Gyr]')
# ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
# ax[1].set_title('Total angular momentum over Time')


# plt.tight_layout()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/static_ias15_energy_angmom.png', dpi=300)
# plt.close()



# #----------------------------------------------------------------------------------



# animate = False
# static = True
# frozen = False
# SphHT = False
# plot = False
# integrator = 'leapfrog'

# static_test_leapfrog = ATG.StellarSimTDep(m22 = 1, r_half = 0.19, no_of_particles = 15, no_time_steps = 1000, total_evolve_time = 5, r_min = 20, 
#                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, static = static, frozen = frozen, SphHT = SphHT, integrator = integrator, plot = plot, animate=animate, animate_every=10)

# static_test_leapfrog.run_simulation()


# positions_all = np.array([p.positions_xyz for p in static_test_leapfrog.particles])  # (N_particles, N_steps+1, 3)
# r_all         = np.array([p.r_values      for p in static_test_leapfrog.particles])  # (N_particles, N_steps+1)
# v_disp_all    = np.array([p.stellar_v_disp for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1)
# kinetic_energy_all = np.array([p.kinetic_energy for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1)
# potential_energy_all = np.array([p.potential_energy for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1)
# ang_mom_all = np.array([p.ang_mom for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1, 3)

# time_step2 = static_test_leapfrog.time_step
# stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

# average_r2 = np.mean(r_all, axis=0)  # Average over particles


# x = np.linspace(0, time_step2, len(stellar_v_disp2))

# plt.plot(x * static_test_leapfrog.dt * static_test_leapfrog.u.to_Gyr, average_r2 * static_test_leapfrog.u.to_Kpc, label='Average Particle Radius')
# for particle in range(r_all.shape[0]):
#     plt.plot(x * static_test_leapfrog.dt * static_test_leapfrog.u.to_Gyr, r_all[particle] * static_test_leapfrog.u.to_Kpc, alpha = 0.2, color='gray')
# plt.axhline(static_test_leapfrog.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
# plt.xlabel('Time [Gyr]')
# plt.ylabel('Average Stellar Radius [Kpc]')
# plt.title('Average Stellar Radius over Time')
# plt.legend()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/static_leapfrog.png', dpi=300)
# plt.close()


# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# for particle in range(ang_mom_all.shape[0]):
#     ax[0].plot(x * static_test_leapfrog.dt * static_test_leapfrog.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
# ax[0].set_xlabel('Time [Gyr]')
# ax[0].set_ylabel('Total Energy [J]')
# ax[0].set_title('Total Energy over Time')




# for particle in range(ang_mom_all.shape[0]):
#     ax[1].plot(x * static_test_leapfrog.dt * static_test_leapfrog.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
# ax[1].set_xlabel('Time [Gyr]')
# ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
# ax[1].set_title('Total angular momentum over Time')

# plt.tight_layout()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/static_leapfrog_energy_angmom.png', dpi=300)
# plt.close()


# #----------------------------------------------------------------------------------


# animate = False
# static = False
# frozen = True
# SphHT = False
# plot = False
# integrator = 'ias15'

# frozen_test_ias15 = ATG.StellarSimTDep(m22 = 1, r_half = 0.19, no_of_particles = 15, no_time_steps = 100, total_evolve_time = 5, r_min = 20, 
#                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, static = static, frozen = frozen, SphHT = SphHT, integrator = integrator, plot = plot, animate=animate, animate_every=10)


# frozen_test_ias15.run_simulation()

# positions_all = np.array([p.positions_xyz for p in frozen_test_ias15.particles])  # (N_particles, N_steps+1, 3)
# r_all         = np.array([p.r_values      for p in frozen_test_ias15.particles])  # (N_particles, N_steps+1)
# v_disp_all    = np.array([p.stellar_v_disp for p in frozen_test_ias15.particles]) # (N_particles, N_steps+1)
# kinetic_energy_all = np.array([p.kinetic_energy for p in frozen_test_ias15.particles]) # (N_particles, N_steps+1)
# potential_energy_all = np.array([p.potential_energy for p in frozen_test_ias15.particles]) # (N_particles, N_steps+1)
# ang_mom_all = np.array([p.ang_mom for p in frozen_test_ias15.particles]) # (N_particles, N_steps+1, 3)

# time_step2 = frozen_test_ias15.time_step
# stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

# average_r2 = np.mean(r_all, axis=0)  # Average over particles


# x = np.linspace(0, time_step2, len(stellar_v_disp2))

# plt.plot(x * frozen_test_ias15.dt * frozen_test_ias15.u.to_Gyr, average_r2 * frozen_test_ias15.u.to_Kpc, label='Average Particle Radius')
# for particle in range(r_all.shape[0]):
#     plt.plot(x * frozen_test_ias15.dt * frozen_test_ias15.u.to_Gyr, r_all[particle] * frozen_test_ias15.u.to_Kpc, alpha = 0.2, color='gray')
# plt.axhline(frozen_test_ias15.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
# plt.xlabel('Time [Gyr]')
# plt.ylabel('Average Stellar Radius [Kpc]')
# plt.title('Average Stellar Radius over Time')
# plt.legend()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/frozen_ias15.png', dpi=300)
# plt.close()


# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# for particle in range(ang_mom_all.shape[0]):
#     ax[0].plot(x * frozen_test_ias15.dt * frozen_test_ias15.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
# ax[0].set_xlabel('Time [Gyr]')
# ax[0].set_ylabel('Total Energy [J]')
# ax[0].set_title('Total Energy over Time')




# for particle in range(ang_mom_all.shape[0]):
#     ax[1].plot(x * frozen_test_ias15.dt * frozen_test_ias15.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
# ax[1].set_xlabel('Time [Gyr]')
# ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
# ax[1].set_title('Total angular momentum over Time')


# plt.tight_layout()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/frozen_ias15_energy_angmom.png', dpi=300)
# plt.close()


# #----------------------------------------------------------------------------------


# animate = False
# static = False
# frozen = True
# SphHT = False
# plot = False
# integrator = 'leapfrog'

# frozen_test_leapfrog = ATG.StellarSimTDep(m22 = 1, r_half = 0.19, no_of_particles = 15, no_time_steps = 1000, total_evolve_time = 5, r_min = 20, 
#                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, static = static, frozen = frozen, SphHT = SphHT, integrator = integrator, plot = plot, animate=animate, animate_every=10)

# frozen_test_leapfrog.run_simulation()


# positions_all = np.array([p.positions_xyz for p in frozen_test_leapfrog.particles])  # (N_particles, N_steps+1, 3)
# r_all         = np.array([p.r_values      for p in frozen_test_leapfrog.particles])  # (N_particles, N_steps+1)
# v_disp_all    = np.array([p.stellar_v_disp for p in frozen_test_leapfrog.particles]) # (N_particles, N_steps+1)
# kinetic_energy_all = np.array([p.kinetic_energy for p in frozen_test_leapfrog.particles]) # (N_particles, N_steps+1)
# potential_energy_all = np.array([p.potential_energy for p in frozen_test_leapfrog.particles]) # (N_particles, N_steps+1)
# ang_mom_all = np.array([p.ang_mom for p in frozen_test_leapfrog.particles]) # (N_particles, N_steps+1, 3)

# time_step2 = frozen_test_leapfrog.time_step
# stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

# average_r2 = np.mean(r_all, axis=0)  # Average over particles


# x = np.linspace(0, time_step2, len(stellar_v_disp2))

# plt.plot(x * frozen_test_leapfrog.dt * frozen_test_leapfrog.u.to_Gyr, average_r2 * frozen_test_leapfrog.u.to_Kpc, label='Average Particle Radius')
# for particle in range(r_all.shape[0]):
#     plt.plot(x * frozen_test_leapfrog.dt * frozen_test_leapfrog.u.to_Gyr, r_all[particle] * frozen_test_leapfrog.u.to_Kpc, alpha = 0.2, color='gray')
# plt.axhline(frozen_test_leapfrog.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
# plt.xlabel('Time [Gyr]')
# plt.ylabel('Average Stellar Radius [Kpc]')
# plt.title('Average Stellar Radius over Time')
# plt.legend()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/frozen_leapfrog.png', dpi=300)
# plt.close()


# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# for particle in range(ang_mom_all.shape[0]):
#     ax[0].plot(x * frozen_test_leapfrog.dt * frozen_test_leapfrog.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
# ax[0].set_xlabel('Time [Gyr]')
# ax[0].set_ylabel('Total Energy [J]')
# ax[0].set_title('Total Energy over Time')




# for particle in range(ang_mom_all.shape[0]):
#     ax[1].plot(x * frozen_test_leapfrog.dt * frozen_test_leapfrog.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
# ax[1].set_xlabel('Time [Gyr]')
# ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
# ax[1].set_title('Total angular momentum over Time')


# plt.tight_layout()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/frozen_leapfrog_energy_angmom.png', dpi=300)
# plt.close()


#----------------------------------------------------------------------------------
# ENTERING T DEP SIMS

# m22_list = [2]
# R0_list_kpc = [0.19, 1]


# SphHT = True
# integrator = 'leapfrog'
# plot = False
# dt_override = 10
# ramp_time = 0.5


# for m22 in m22_list:
#     for R0 in R0_list_kpc:

#         print('Running t dep sim for m22 =', m22, 'and R0 =', R0, 'kpc ...', flush=True)

#         t_dep_leapfrog = ATD.StellarSimTDep(m22 = m22, r_half = R0, no_of_particles = 100, no_time_steps = 1000, total_evolve_time = 10, r_min = 1, 
#                                     r_max_enclosing_frac = 0.99, no_radius_bins = 2000, SphHT = SphHT, integrator = integrator, plot = plot, dt_override=dt_override, ramp_time=ramp_time)
        

#         t_dep_leapfrog.run_simulation()


#         positions_all = np.array([p.positions_xyz for p in t_dep_leapfrog.particles])  # (N_particles, N_steps+1, 3)
#         r_all         = np.array([p.r_values      for p in t_dep_leapfrog.particles])  # (N_particles, N_steps+1)
#         v_disp_all    = np.array([p.stellar_v_disp for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1)
#         kinetic_energy_all = np.array([p.kinetic_energy for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1)
#         potential_energy_all = np.array([p.potential_energy for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1)
#         ang_mom_all = np.array([p.ang_mom for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1, 3)

#         time_step = t_dep_leapfrog.time_step
#         stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

#         average_r = np.mean(r_all, axis=0)  # Average over particles

#         no_time_steps = t_dep_leapfrog.no_time_steps
        
#         x = np.arange(no_time_steps + 1)  # Time steps array


#         plt.plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, average_r * t_dep_leapfrog.u.to_Kpc, label='Average Particle Radius')
#         for particle in range(r_all.shape[0]):
#             plt.plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, r_all[particle] * t_dep_leapfrog.u.to_Kpc, alpha = 0.2, color='gray')
#         plt.axhline(t_dep_leapfrog.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')

#         plt.xlabel('Time [Gyr]')
#         plt.ylabel('Average Stellar Radius [Kpc]')
#         plt.title('Average Stellar Radius over Time')

#         # Timescale diagnostics
#         v0 = np.sqrt(2 * kinetic_energy_all[:, 0])
#         mean_T_orb = float(np.mean(2 * np.pi * r_all[:, 0] / v0) * t_dep_leapfrog.u.to_Gyr)

#         lambda_db_kpc = 19.15 / (t_dep_leapfrog.m22 * v0 * t_dep_leapfrog.u.to_kms)
#         T_c = lambda_db_kpc / (v0 * t_dep_leapfrog.u.to_Kpc) * t_dep_leapfrog.u.to_Gyr
#         plt.axhline(np.mean(lambda_db_kpc), color='g', linestyle='--', label='$\\lambda_{{\\rm db}}$')

#         E = np.array(t_dep_leapfrog.eigen_energies)
#         freq_diff = np.abs(E[:, None] - E[None, :])
#         T_beat = (2 * np.pi / freq_diff) * t_dep_leapfrog.u.to_Gyr
#         min_T_beat = np.min(T_beat[np.isfinite(T_beat)])
#         max_T_beat = np.max(T_beat[np.isfinite(T_beat)])

#         dt_Gyr = t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr


#         info = (
#             f"$T_{{\\rm orb}}$ (mean) = {mean_T_orb:.3f} Gyr\n"
#             f"$T_{{\\rm c}}$ = {float(np.mean(T_c)):.3f} Gyr\n"
#             f"Beat time band: [{min_T_beat:.3f}, {max_T_beat:.3f}] Gyr\n"
#             f"$\\Delta t$ = {dt_Gyr:.4f} Gyr"
#         )
#         plt.text(1.02, 0.98, info, transform=plt.gca().transAxes,
#                 verticalalignment='top', fontsize=9,
#                 bbox=dict(boxstyle='round', facecolor='paleturquoise', alpha=0.6))

#         plt.legend(bbox_to_anchor=(1, 0.7))

#         plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/T_dep/DIFF/t_dep_lf_m{m22}_R{R0}.png', dpi=300, bbox_inches='tight')
#         plt.close()


#         fig, ax = plt.subplots(1, 2, figsize=(12, 5))

#         for particle in range(ang_mom_all.shape[0]):
#             ax[0].plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
#         ax[0].set_xlabel('Time [Gyr]')
#         ax[0].set_ylabel('Total Energy [J]')
#         ax[0].set_title('Total Energy over Time')




#         for particle in range(ang_mom_all.shape[0]):
#             ax[1].plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
#         ax[1].set_xlabel('Time [Gyr]')
#         ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
#         ax[1].set_title('Total angular momentum over Time')


#         plt.tight_layout()
#         plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/T_dep/DIFF/t_dep_lf_eng_amom_m{m22}_R{R0}.png', dpi=300, bbox_inches='tight')
#         plt.close()




SphHT = True
integrator = 'leapfrog'
plot = False
dt_override = 2
ramp_time = 1

l_band_size = 128
use_multi_gpu = True
sparse_k_batch=16384
r_chunk_size = 32
compute_dtype = jnp.complex64

L_out_frac = 1

m22 = 10
R0 = [1]

for R0 in R0:

    sim = ATDS_MS.StellarSimTDep(m22 = m22, r_half = R0, r_half_width = 0.05, no_of_particles = 5, no_time_steps = 1000, total_evolve_time = 10, r_min = 20, 
                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, SphHT = SphHT, integrator = integrator, 
                                plot = plot, dt_override=dt_override, ramp_time=ramp_time, l_band_size=l_band_size, use_multi_gpu = use_multi_gpu,
                                    sparse_k_batch=sparse_k_batch, r_chunk_size=r_chunk_size, compute_dtype=compute_dtype, L_out_frac=L_out_frac)

    sim.run_simulation()


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

    plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Plots/T_dep/t_dep_lf_m{m22}_R{R0}.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Plots/T_dep/t_dep_lf_eng_amom_m{m22}_R{R0}.png', dpi=300, bbox_inches='tight')
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

    df.to_csv(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Plots/T_dep/t_dep_lf_data_m{m22}_R{R0}.csv', index=False)





# #----------------------------------------------------------------------------------

# animate = False
# SphHT = False
# integrator = 'ias15'

# t_dep_ias15 = ATD.StellarSimTDep(m22 = 1, r_half = 0.19, no_of_particles = 15, no_time_steps = 1000, total_evolve_time = 10, r_min = 20, 
#                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, SphHT = SphHT, integrator = integrator, animate=animate, animate_every=10,)


# t_dep_ias15.run_simulation()

# positions_all = np.array([p.positions_xyz for p in t_dep_ias15.particles])  # (N_particles, N_steps+1, 3)
# r_all         = np.array([p.r_values      for p in t_dep_ias15.particles])  # (N_particles, N_steps+1)
# v_disp_all    = np.array([p.stellar_v_disp for p in t_dep_ias15.particles]) # (N_particles, N_steps+1)
# kinetic_energy_all = np.array([p.kinetic_energy for p in t_dep_ias15.particles]) # (N_particles, N_steps+1)
# potential_energy_all = np.array([p.potential_energy for p in t_dep_ias15.particles]) # (N_particles, N_steps+1)
# ang_mom_all = np.array([p.ang_mom for p in t_dep_ias15.particles]) # (N_particles, N_steps+1, 3)
# time_step2 = t_dep_ias15.time_step
# stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

# average_r2 = np.mean(r_all, axis=0)  # Average over particles


# x = np.linspace(0, time_step2, len(stellar_v_disp2))

# plt.plot(x * t_dep_ias15.dt * t_dep_ias15.u.to_Gyr, average_r2 * t_dep_ias15.u.to_Kpc, label='Average Particle Radius')
# for particle in range(r_all.shape[0]):
#     plt.plot(x * t_dep_ias15.dt * t_dep_ias15.u.to_Gyr, r_all[particle] * t_dep_ias15.u.to_Kpc, alpha = 0.2, color='gray')
# plt.axhline(t_dep_ias15.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
# plt.xlabel('Time [Gyr]')
# plt.ylabel('Average Stellar Radius [Kpc]')
# plt.title('Average Stellar Radius over Time')
# plt.legend()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/t_dep_ias15.png', dpi=300)
# plt.close()


# fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# for particle in range(ang_mom_all.shape[0]):
#     ax[0].plot(x * t_dep_ias15.dt * t_dep_ias15.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
# ax[0].set_xlabel('Time [Gyr]')
# ax[0].set_ylabel('Total Energy [J]')
# ax[0].set_title('Total Energy over Time')




# for particle in range(ang_mom_all.shape[0]):
#     ax[1].plot(x * t_dep_ias15.dt * t_dep_ias15.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
# ax[1].set_xlabel('Time [Gyr]')
# ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
# ax[1].set_title('Total angular momentum over Time')


# plt.tight_layout()
# plt.savefig('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/t_dep_ias15_energy_angmom.png', dpi=300)
# plt.close()

#----------------------------------------------------------------------------------------
#ENTERING CHANGING A_NL SIMS


# a_nl_boosting_factors = np.logspace(8, 17, 10)

# print("a_nl boosting factors:", a_nl_boosting_factors)

# animate = False
# SphHT = False
# integrator = 'leapfrog'
# a_nl_range = 'orbital'
# plot = False

# for boost_factor in a_nl_boosting_factors:

#     a_nl_sim = A_nl.StellarSimTDep(m22 = 1, r_half = 0.19, no_of_particles = 10, no_time_steps = 2000, total_evolve_time = 10, r_min = 20, 
#                                    r_max_enclosing_frac = 0.99, no_radius_bins = 1000, SphHT = SphHT, integrator = integrator, a_nl_range= a_nl_range, boost_factor=boost_factor, plot = plot, animate=animate, animate_every=10)


#     a_nl_sim.run_simulation()

#     positions_all = np.array([p.positions_xyz for p in a_nl_sim.particles])  # (N_particles, N_steps+1, 3)
#     r_all         = np.array([p.r_values      for p in a_nl_sim.particles])  # (N_particles, N_steps+1)
#     v_disp_all    = np.array([p.stellar_v_disp for p in a_nl_sim.particles]) # (N_particles, N_steps+1)
#     kinetic_energy_all = np.array([p.kinetic_energy for p in a_nl_sim.particles]) # (N_particles, N_steps+1)
#     potential_energy_all = np.array([p.potential_energy for p in a_nl_sim.particles]) # (N_particles, N_steps+1)
#     ang_mom_all = np.array([p.ang_mom for p in a_nl_sim.particles]) # (N_particles, N_steps+1, 3)
#     time_step2 = a_nl_sim.time_step
#     stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

#     average_r2 = np.mean(r_all, axis=0)  # Average over particles


#     x = np.linspace(0, time_step2, len(stellar_v_disp2))

#     plt.plot(x * a_nl_sim.dt * a_nl_sim.u.to_Gyr, average_r2 * a_nl_sim.u.to_Kpc, label='Average Particle Radius')
#     for particle in range(r_all.shape[0]):
#         plt.plot(x * a_nl_sim.dt * a_nl_sim.u.to_Gyr, r_all[particle] * a_nl_sim.u.to_Kpc, alpha = 0.2, color='gray')
#     plt.axhline(a_nl_sim.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
#     plt.xlabel('Time [Gyr]')
#     plt.ylabel('Average Stellar Radius [Kpc]')
#     plt.title('Average Stellar Radius over Time with boost factor: ' + str(boost_factor))
#     plt.legend()
#     plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/a_nl_boosting/t_dep_a_nl_orbital_{boost_factor}.png', dpi=300)
#     plt.close()


#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))

#     for particle in range(ang_mom_all.shape[0]):
#         ax[0].plot(x * a_nl_sim.dt * a_nl_sim.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
#     ax[0].set_xlabel('Time [Gyr]')
#     ax[0].set_ylabel('Total Energy [J]')
#     ax[0].set_title('Total Energy over Time')




#     for particle in range(ang_mom_all.shape[0]):
#         ax[1].plot(x * a_nl_sim.dt * a_nl_sim.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
#     ax[1].set_xlabel('Time [Gyr]')
#     ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
#     ax[1].set_title('Total angular momentum over Time with boost factor: ' + str(boost_factor))


#     plt.tight_layout()
#     plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/a_nl_boosting/t_dep_a_nl_orbital_energy_angmom_{boost_factor}.png', dpi=300)
#     plt.close()

