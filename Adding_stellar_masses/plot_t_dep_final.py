"""
Usage: python plot_t_dep_final.py --m22 <m22> --r0 <r0> --Lout <Lout>

Load the final checkpoint from checkpoints_m22_{m22}_{r0}_{Lout}/ and produce
T_dep style plots, saved to Plots/T_dep/.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses")
import jax
jax.config.update("jax_enable_x64", True)
import jaxsp as jsp

parser = argparse.ArgumentParser()
parser.add_argument("--m22",  type=float, required=True)
parser.add_argument("--r0",   type=float, required=True)
parser.add_argument("--Lout", type=float, required=True)
args = parser.parse_args()

WORKDIR   = "/rds/general/user/jd925/ephemeral/Checkpoints/"
M22       = args.m22
R0        = args.r0
LOUT      = args.Lout
m22_str   = f"{M22:g}"
r0_str_in = f"{R0:g}"
lout_str  = f"{LOUT:g}"
CKPT_DIR  = os.path.join(WORKDIR, f"checkpoints_m22_{m22_str}_r0_{r0_str_in}_Lout_{lout_str}")
PLOT_DIR = "/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/Plots/T_dep"
os.makedirs(PLOT_DIR, exist_ok=True)

u = jsp.set_schroedinger_units(M22)
to_Kpc = float(u.to_Kpc)
to_kms = float(u.to_kms)
to_Gyr = float(u.to_Gyr)
print(f"m22={M22}, r0={R0}, Lout={LOUT} (checkpoints: {CKPT_DIR})")
print(f"Unit conversions: 1 code length = {to_Kpc:.4f} kpc, 1 code time = {to_Gyr:.4f} Gyr")

# Load final checkpoint
ckpt_path = os.path.join(CKPT_DIR, "checkpoint_final.pkl")
print(f"\nLoading {ckpt_path}")
with open(ckpt_path, "rb") as f:
    state = pickle.load(f)

n_steps = state["sim_time_step"]
n_ramp  = state.get("n_ramp_steps", 0)
dt_Gyr  = (state["sim_t"] / n_steps) * to_Gyr
print(f"Steps: {n_steps}, ramp steps: {n_ramp}, dt = {dt_Gyr:.5f} Gyr")

particles = state["particle_states"]

r_all  = np.array([p["r_values"] for p in particles])                              # (N_part, N_steps+1)
v_cart = np.array([[np.asarray(v) for v in p["velocities_cart"]] for p in particles])  # (N_part, N_steps+1, 3)
ke_all = np.array([p["kinetic_energy"]   for p in particles])                       # (N_part, N_steps+1)
pe_all = np.array([p["potential_energy"] for p in particles])                       # (N_part, N_steps+1)
am_all = np.array([p["ang_mom"]          for p in particles])                       # (N_part, N_steps+1[, 3])

N_part, N_t = r_all.shape
print(f"Particles: {N_part}, time points: {N_t}")

# Infer r_half from initial particle radii
r_half_kpc  = float(np.mean(r_all[:, 0])) * to_Kpc
r_half_str  = f"{r_half_kpc:.2f}"
print(f"Inferred r_half = {r_half_kpc:.4f} kpc")

avg_r = np.mean(r_all, axis=0)         # (N_steps+1,)
x     = np.arange(N_t) * dt_Gyr       # time axis in Gyr

# Timescale diagnostics from initial conditions (consistent with Running_sims.py)
v0 = np.sqrt(2 * ke_all[:, 0])                        # (N_part,) initial speeds, code units
mean_T_orb    = float(np.mean(2 * np.pi * r_all[:, 0] / v0)) * to_Gyr
lambda_db_kpc = 19.15 / (M22 * v0 * to_kms)           # de Broglie wavelength, kpc
T_c           = float(np.mean(lambda_db_kpc / (v0 * to_Kpc))) * to_Gyr

print(f"T_orb = {mean_T_orb:.3f} Gyr, T_c = {T_c:.3f} Gyr")


# --- Plot 1: Average stellar radius vs time ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, avg_r * to_Kpc, label="Average Particle Radius", zorder=3)
for i in range(N_part):
    ax.plot(x, r_all[i] * to_Kpc, alpha=0.2, color="gray")
ax.axhline(r_half_kpc, color="r", linestyle="--",
           label=f"Initial $r_{{1/2}}$ = {r_half_kpc:.3f} kpc")
ax.axhline(float(np.mean(lambda_db_kpc)), color="g", linestyle="--",
           label=r"$\lambda_{\rm db}$")

info = (
    f"$T_{{\\rm orb}}$ (mean) = {mean_T_orb:.3f} Gyr\n"
    f"$T_c$ = {T_c:.3f} Gyr\n"
    f"$\\Delta t$ = {dt_Gyr:.4f} Gyr"
)
ax.text(1.02, 0.98, info, transform=ax.transAxes,
        verticalalignment="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="paleturquoise", alpha=0.6))

ax.set_xlabel("Time [Gyr]")
ax.set_ylabel("Average Stellar Radius [kpc]")
ax.set_title(f"Average Stellar Radius vs Time  (m22={M22}, r0={R0}, Lout={LOUT}, $r_{{1/2}}$={r_half_kpc:.3f} kpc)")
ax.legend(bbox_to_anchor=(1, 0.7))

out1 = os.path.join(PLOT_DIR, f"t_dep_lf_m{m22_str}_R0_{r_half_str}_Lout{lout_str}.png")
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out1}")


# --- Plot 2: Energy and angular momentum ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i in range(N_part):
    axes[0].plot(x, ke_all[i] + pe_all[i], alpha=0.7, label=f"Particle {i}")
axes[0].set_xlabel("Time [Gyr]")
axes[0].set_ylabel("Total Energy")
axes[0].set_title("Total Energy vs Time")

for i in range(N_part):
    axes[1].plot(x, am_all[i], alpha=0.7)
axes[1].set_xlabel("Time [Gyr]")
axes[1].set_ylabel("Angular Momentum")
axes[1].set_title("Angular Momentum vs Time")

plt.tight_layout()
out2 = os.path.join(PLOT_DIR, f"t_dep_lf_eng_amom_m{m22_str}_R0_{r_half_str}_Lout{lout_str}.png")
fig.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")
