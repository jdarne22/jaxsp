"""
Usage: python plot_L_reduc.py --m22 <m22> --r0 <r0_kpc>

Finds all checkpoint dirs matching checkpoints_m22_{m22}_r0_{r0}_Lout_*,
loads the latest checkpoint from each, and saves both plots to
Plots/Reducing_L/.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
import glob
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses")
import jax
jax.config.update("jax_enable_x64", True)
import jaxsp as jsp

WORKDIR = "/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses"
PLOT_DIR = os.path.join(WORKDIR, "Plots", "Reducing_L")
os.makedirs(PLOT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--m22", type=float, required=True)
parser.add_argument("--r0",  type=float, required=True, help="r_half in kpc")
args = parser.parse_args()

m22_val   = args.m22
r0_kpc    = args.r0
m22_str   = f"{m22_val:g}"   # "10" not "10.0", "0.5" not "0.5"
r0_str    = f"{r0_kpc:g}"

u = jsp.set_schroedinger_units(m22_val)
to_Kpc = float(u.to_Kpc)
to_kms = float(u.to_kms)
to_Gyr = float(u.to_Gyr)
print(f"m22={m22_val}, r0={r0_kpc} kpc")
print(f"Unit conversions: 1 code length = {to_Kpc:.4f} kpc, 1 code time = {to_Gyr:.4f} Gyr")


def load_latest_checkpoint(checkpoint_dir):
    files = sorted(
        [f for f in os.listdir(checkpoint_dir)
         if f.startswith("checkpoint_step_") and f.endswith(".pkl")],
        key=lambda f: int(f[len("checkpoint_step_"):-len(".pkl")])
    )
    if not files:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    path = os.path.join(checkpoint_dir, files[-1])
    step = int(files[-1][len("checkpoint_step_"):-len(".pkl")])
    print(f"  Loading {path} (step {step})")
    with open(path, "rb") as f:
        state = pickle.load(f)
    return state, step


# Auto-detect all L_out_frac directories for this m22 / r0 combination
pattern = os.path.join(WORKDIR, "Checkpoints", f"checkpoints_m22_{m22_str}_r0_{r0_str}_Lout_*")
dirs = sorted(glob.glob(pattern),
              key=lambda d: float(d.split("_Lout_")[-1]),
              reverse=True)   # descending so L=1 is first (truth)

if not dirs:
    sys.exit(f"No checkpoint directories found matching: {pattern}")

L_out_fracs = [float(d.split("_Lout_")[-1]) for d in dirs]
print(f"Found L_out_fracs: {L_out_fracs}")

R_halves    = []
r0_per_part = []   # per-particle initial radius (code units), for T_orb
v0_all      = []
dt_Gyr      = None

for ckpt_dir, L_frac in zip(dirs, L_out_fracs):
    print(f"\nL_out_frac = {L_frac}")
    state, n_steps = load_latest_checkpoint(ckpt_dir)

    # Recover dt from checkpoint (sim_t is total elapsed code time)
    if dt_Gyr is None:
        dt_Gyr = (state["sim_t"] / state["sim_time_step"]) * to_Gyr
        print(f"  dt = {dt_Gyr:.5f} Gyr  (recovered from checkpoint)")

    particles = state["particle_states"]
    r_vals = np.array([p["r_values"] for p in particles])       # (N_part, N_steps+1)
    v_cart = np.array([[np.asarray(v) for v in p["velocities_cart"]]
                       for p in particles])                      # (N_part, N_steps+1, 3)

    R_halves.append(np.mean(r_vals, axis=0))
    r0_per_part.append(r_vals[:, 0])                            # (N_part,) initial radii
    v0_all.append(np.linalg.norm(v_cart[:, 0, :], axis=1))     # (N_part,) initial speeds
    print(f"  Particles: {r_vals.shape[0]}, steps: {r_vals.shape[1]-1}")


# --- Plot 1: Average stellar radius vs time ---
fig, ax = plt.subplots(figsize=(10, 6))

for i, L_frac in enumerate(L_out_fracs):
    x = np.arange(len(R_halves[i])) * dt_Gyr
    ax.plot(x, R_halves[i] * to_Kpc, label=f"L_out_frac={L_frac}", alpha=0.7)

    v0_code    = v0_all[i]
    mean_T_orb = float(np.mean(2 * np.pi * r0_per_part[i] / v0_code)) * to_Gyr
    lambda_db_kpc = 19.15 / (m22_val * v0_code * to_kms)
    T_c = float(np.mean(lambda_db_kpc / (v0_code * to_Kpc) * to_Gyr))

    info = (
        f"L_out_frac={L_frac}\n"
        f"$T_{{\\rm orb}}$ = {mean_T_orb:.3f} Gyr\n"
        f"$T_c$ = {T_c:.3f} Gyr\n"
        f"$\\Delta t$ = {dt_Gyr:.4f} Gyr"
    )
    ax.text(1.02, 0.98 - i * 0.17, info, transform=ax.transAxes,
            verticalalignment="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="paleturquoise", alpha=0.6))

ax.axhline(r0_kpc, color="r", linestyle="--", label=f"Initial r_half = {r0_kpc} kpc")
ax.set_xlabel("Time [Gyr]")
ax.set_ylabel("Average Stellar Radius [kpc]")
ax.set_title(f"Average Stellar Radius vs Time  (m22={m22_val}, r0={r0_kpc} kpc)")
ax.legend(loc="upper left")
out1 = os.path.join(PLOT_DIR, f"L_reduc_all_m22_{m22_str}_r0_{r0_str}.png")
fig.savefig(out1, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out1}")


# --- Plot 2: Relative error vs time (truth = largest L_out_frac) ---
R_truth = R_halves[5]

fig, ax = plt.subplots(figsize=(10, 6))
for i, L_frac in enumerate(L_out_fracs):
    n = min(len(R_halves[i]), len(R_truth))
    x = np.arange(n) * dt_Gyr
    err = np.abs(R_halves[i][:n] - R_truth[:n]) / R_truth[:n]
    ax.plot(x, err, label=f"L_out_frac={L_frac}", alpha=0.7)

ax.set_xlabel("Time [Gyr]")
ax.set_ylabel("Relative Error in Half-Mass Radius")
ax.set_title(f"Relative Error in Half-Mass Radius  (m22={m22_val}, r0={r0_kpc} kpc)")
ax.set_yscale("log")
ax.legend()
out2 = os.path.join(PLOT_DIR, f"L_reduc_all_error_m22_{m22_str}_r0_{r0_str}.png")
fig.savefig(out2, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out2}")
