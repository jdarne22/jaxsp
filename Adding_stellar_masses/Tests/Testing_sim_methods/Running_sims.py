import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"   # or: "cuda_async"

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import sys

sys.path.insert(0, "/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses")



import analytic_test_gaunt as ATG

import Analytic_t_dep_sim as ATD

import A_nl_altered_sims as A_nl

import Analytic_t_dep_sim_CC_speed as ATD_CC

import Analytic_test_CC_speed as ATCCS



import jaxsp as jsp

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



import importlib
importlib.reload(ATG)
importlib.reload(ATD)
importlib.reload(A_nl)
importlib.reload(ATD_CC)
importlib.reload(ATCCS)



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

m22_list = [2, 3]
R0_list_kpc = [0.19, 1, 2]


animate = False
SphHT = True
integrator = 'leapfrog'
plot = False
frozen = True
static = False
dt_override = True


for m22 in m22_list:
    for R0 in R0_list_kpc:

        print('Running frozen ULDM for m22 =', m22, 'and R0 =', R0, 'kpc ...', flush=True)

        t_dep_leapfrog = ATCCS.StellarSimTDep(m22 = m22, r_half = R0, no_of_particles = 10, no_time_steps = 1000, total_evolve_time = 10, r_min = 20, 
                                    r_max_enclosing_frac = 0.99, no_radius_bins = 1000, SphHT = SphHT, integrator = integrator, plot = plot, frozen = frozen,
                                    static = static, animate=animate, animate_every=10, dt_override=dt_override)
        

        t_dep_leapfrog.run_simulation()


        positions_all = np.array([p.positions_xyz for p in t_dep_leapfrog.particles])  # (N_particles, N_steps+1, 3)
        r_all         = np.array([p.r_values      for p in t_dep_leapfrog.particles])  # (N_particles, N_steps+1)
        v_disp_all    = np.array([p.stellar_v_disp for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1)
        kinetic_energy_all = np.array([p.kinetic_energy for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1)
        potential_energy_all = np.array([p.potential_energy for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1)
        ang_mom_all = np.array([p.ang_mom for p in t_dep_leapfrog.particles]) # (N_particles, N_steps+1, 3)

        time_step2 = t_dep_leapfrog.time_step
        stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

        average_r2 = np.mean(r_all, axis=0)  # Average over particles


        x = np.linspace(0, time_step2, len(stellar_v_disp2))

        plt.plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, average_r2 * t_dep_leapfrog.u.to_Kpc, label='Average Particle Radius')
        for particle in range(r_all.shape[0]):
            plt.plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, r_all[particle] * t_dep_leapfrog.u.to_Kpc, alpha = 0.2, color='gray')
        plt.axhline(t_dep_leapfrog.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
        plt.xlabel('Time [Gyr]')
        plt.ylabel('Average Stellar Radius [Kpc]')
        plt.title('Average Stellar Radius over Time')

        # Timescale diagnostics
        v0 = np.sqrt(2 * kinetic_energy_all[:, 0])
        mean_T_orb = float(np.mean(2 * np.pi * r_all[:, 0] / v0) * t_dep_leapfrog.u.to_Gyr)

        lambda_db_kpc = 19.15 / (t_dep_leapfrog.m22 * v0 * t_dep_leapfrog.u.to_kms)
        T_c = lambda_db_kpc / (v0 * t_dep_leapfrog.u.to_Kpc) * t_dep_leapfrog.u.to_Gyr


        E = np.array(t_dep_leapfrog.eigen_energies)
        freq_diff = np.abs(E[:, None] - E[None, :])
        T_beat = (2 * np.pi / freq_diff) * t_dep_leapfrog.u.to_Gyr
        min_T_beat = np.min(T_beat[np.isfinite(T_beat)])
        max_T_beat = np.max(T_beat[np.isfinite(T_beat)])

        dt_Gyr = t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr


        info = (
            f"$T_{{\\rm orb}}$ (mean) = {mean_T_orb:.3f} Gyr\n"
            f"$T_{{\\rm c}}$ = {float(T_c[0]):.3f} Gyr\n"
            f"Beat time band: [{min_T_beat:.3f}, {max_T_beat:.3f}] Gyr\n"
            f"$\\Delta t$ = {dt_Gyr:.4f} Gyr"
        )
        plt.text(0.02, 0.98, info, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='paleturquoise', alpha=0.6))

        plt.legend(loc = 'lower right')

        plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/Frozen/frozen_lf_m{m22}_R{R0}.png', dpi=300)
        plt.close()


        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        for particle in range(ang_mom_all.shape[0]):
            ax[0].plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, (kinetic_energy_all[particle] + potential_energy_all[particle]), label='Total Energy')
        ax[0].set_xlabel('Time [Gyr]')
        ax[0].set_ylabel('Total Energy [J]')
        ax[0].set_title('Total Energy over Time')




        for particle in range(ang_mom_all.shape[0]):
            ax[1].plot(x * t_dep_leapfrog.dt * t_dep_leapfrog.u.to_Gyr, ang_mom_all[particle], label='Total angular momentum')
        ax[1].set_xlabel('Time [Gyr]')
        ax[1].set_ylabel('Total angular momentum [kg m^2/s]')
        ax[1].set_title('Total angular momentum over Time')


        plt.tight_layout()
        plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/Frozen/frozen_lf_eng_amom_m{m22}_R{R0}.png', dpi=300)
        plt.close()


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

