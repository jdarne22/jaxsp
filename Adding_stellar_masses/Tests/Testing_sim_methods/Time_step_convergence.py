import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import PhD_year_1.jaxsp.Adding_stellar_masses.Analytic_t_dep_sim as ATD

import Analytic_test as AT

import time

import jaxsp as jsp

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import importlib
importlib.reload(ATD)
importlib.reload(AT)

#--------------------------------------------------------------------------------------------------------------


N_t = np.linspace(200, 2000, 11)

# N_t = [2500, 3000, 4000]

# N_t = [2500]

print(N_t)


for N in N_t:

    animate = False
    static = True
    frozen = False
    SphHT = False
    plot = False
    integrator = 'leapfrog'

    static_test_leapfrog = AT.StellarSimTDep(m22 = 1, r_half = 0.19, no_of_particles = 20, no_time_steps = N, total_evolve_time = 10, r_min = 20, 
                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, static = static, frozen = frozen, SphHT = SphHT, integrator = integrator,
                                plot = plot, animate=animate, animate_every=10)
    
    start = time.time()

    static_test_leapfrog.run_simulation()

    end = time.time()
    print(f"Time taken for leapfrog using {N} time steps: {end - start:.2f} seconds")


    positions_all = np.array([p.positions_xyz for p in static_test_leapfrog.particles])  # (N_particles, N_steps+1, 3)
    r_all         = np.array([p.r_values      for p in static_test_leapfrog.particles])  # (N_particles, N_steps+1)
    v_disp_all    = np.array([p.stellar_v_disp for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1)
    kinetic_energy_all = np.array([p.kinetic_energy for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1)
    potential_energy_all = np.array([p.potential_energy for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1)
    ang_mom_all = np.array([p.ang_mom for p in static_test_leapfrog.particles]) # (N_particles, N_steps+1, 3)

    time_step2 = static_test_leapfrog.time_step
    stellar_v_disp2 = np.mean(v_disp_all, axis=0)  # Average over particles

    average_r2 = np.mean(r_all, axis=0)  # Average over particles


    x = np.linspace(0, time_step2, len(stellar_v_disp2))

    plt.plot(x * static_test_leapfrog.dt * static_test_leapfrog.u.to_Gyr, average_r2 * static_test_leapfrog.u.to_Kpc, label='Average Particle Radius')
    for particle in range(r_all.shape[0]):
        plt.plot(x * static_test_leapfrog.dt * static_test_leapfrog.u.to_Gyr, r_all[particle] * static_test_leapfrog.u.to_Kpc, alpha = 0.2, color='gray')
    plt.axhline(static_test_leapfrog.r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
    plt.xlabel('Time [Gyr]')
    plt.ylabel('Average Stellar Radius [Kpc]')
    plt.title('Average Stellar Radius over Time, sim took {:.2f} seconds'.format(end - start))
    plt.legend()
    plt.savefig(f'/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses/Tests/Testing_sim_methods/Plots/T_convergence/leapfrog_{N}_timesteps.png', dpi=300)
    plt.close()

