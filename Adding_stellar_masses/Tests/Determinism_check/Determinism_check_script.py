import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import Analytic_t_dep_sim_mem_saver as ATDS_MS

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp

import importlib
importlib.reload(ATDS_MS)


SphHT = True
integrator = 'leapfrog'
plot = False
dt_override = 1
ramp_time = 0

l_band_size = 250
use_multi_gpu = True
sparse_k_batch=16384
r_chunk_size = 1000

compute_dtype = jnp.complex64

L_out_frac = 0.5

Runs = []

for i in range(3):
  print('Run number:', i+1)
  sim = ATDS_MS.StellarSimTDep(m22 = 5, r_half = 0.19, r_half_width = 0.05, no_of_particles = 5, no_time_steps = 1000, total_evolve_time = 10, r_min = 20, 
                                r_max_enclosing_frac = 0.99, no_radius_bins = 1000, SphHT = SphHT, integrator = integrator, 
                                plot = plot, dt_override=dt_override, ramp_time=ramp_time, l_band_size=l_band_size, use_multi_gpu = use_multi_gpu,
                                  sparse_k_batch=sparse_k_batch, r_chunk_size=r_chunk_size, compute_dtype=compute_dtype, L_out_frac=L_out_frac)


  sim.run_simulation()
  Runs.append(sim)

positions_all = []
r_all = []
all_vels_cart = []
kinetic_energy_all = []
potential_energy_all = []
ang_mom_all = []

for sim in Runs:

    positions_all.append(np.array([p.positions_xyz for p in sim.particles]))  # (N_particles, N_steps+1, 3)
    r_all.append(np.array([p.r_values      for p in sim.particles]))  # (N_particles, N_steps+1)
    all_vels_cart.append(np.array([[np.array(v) for v in p.velocities_cart] for p in sim.particles]))
    kinetic_energy_all.append(np.array([p.kinetic_energy for p in sim.particles])) # (N_particles, N_steps+1)
    potential_energy_all.append(np.array([p.potential_energy for p in sim.particles])) # (N_particles, N_steps+1)
    ang_mom_all.append(np.array([p.ang_mom for p in sim.particles])) # (N_particles, N_steps+1, 3)

np.savez('Determinism_Check.npz', positions_all=positions_all, r_all=r_all, all_vels_cart=all_vels_cart, kinetic_energy_all=kinetic_energy_all, potential_energy_all=potential_energy_all, ang_mom_all=ang_mom_all)

no_time_steps = Runs[0].no_time_steps


x = np.arange(no_time_steps + 1)  # Time steps array


for i, sim in enumerate(Runs):

    R_half = np.mean(r_all[i], axis=0)  # (N_steps+1,) 

    plt.plot(x * sim.dt * sim.u.to_Gyr, R_half * sim.u.to_Kpc, label=f'Half-Mass Radius (L_out_frac={sim.L_out_frac})', alpha=0.7)
    #for part in range(len(sim.particles)):
    #    plt.plot(x * sim.dt * sim.u.to_Gyr, r_all[i][part] * sim.u.to_Kpc, alpha=0.3, color='gray')
    plt.xlabel('Time [Gyr]')
    plt.ylabel('Average Stellar Radius [Kpc]')
    plt.title('Average Stellar Radius over Time using GPU deterministic')

    # Timescale diagnostics
    v0 = np.linalg.norm(all_vels_cart[i][:, 0, :], axis=1)
    mean_T_orb = float(np.mean(2 * np.pi * r_all[i][:, 0] / v0) * sim.u.to_Gyr)

    lambda_db_kpc = 19.15 / (sim.m22 * v0 * sim.u.to_kms)
    T_c = lambda_db_kpc / (v0 * sim.u.to_Kpc) * sim.u.to_Gyr

    #plt.axhline(np.mean(lambda_db_kpc), color='g', linestyle='--', label='$\\lambda_{{\\rm db}}$')


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

    plt.text(1.02, 0.98 - i * 0.2, info, transform=plt.gca().transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='paleturquoise', alpha=0.6))


    
plt.axhline(Runs[0].r_half, color='r', linestyle='--', label='Initial Particle Position (r_half)')
    
plt.legend(bbox_to_anchor=(1.6, 0.7))
#plt.yscale('log')
plt.savefig('Determinism_Check.png', bbox_inches='tight')


R_half_truth = np.mean(r_all[0], axis=0)  # L_out_frac = 1
errors = []
for i, sim in enumerate(Runs):
    R_half = np.mean(r_all[i], axis=0)
    error = np.abs(R_half - R_half_truth) / R_half_truth
    errors.append(error)


plt.figure(figsize=(10, 6))
for i, sim in enumerate(Runs):
    plt.plot(x * sim.dt * sim.u.to_Gyr, errors[i], label=f'L_out_frac={sim.L_out_frac}', alpha=0.7)
plt.xlabel('Time [Gyr]')
plt.ylabel('Relative Error in Half-Mass Radius')
plt.title('Relative Error in Half-Mass Radius over Time using GPU deterministic')
plt.yscale('log')
plt.legend()
plt.savefig('Determinism_Check_Error.png', bbox_inches='tight')