
import functools

import os
import pickle
from time import time

import sys

sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses')

import Stellar_sim_funcs as SSF

import importlib

import jaxsp as jsp

import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

from jaxsp.constants import GN, hbar

import rebound

import s2fft

from scipy.special import sph_harm_y

from collections import defaultdict

import gaunt_funcs_CC_speed as gf

importlib.reload(SSF)
importlib.reload(gf)


import matplotlib.pyplot as plt


def precompute_lm_pairs_Ylms(l):

    '''Precompute (l, m) pair for spherical harmonics'''

    lm_l = []      # list of l for each mode k
    lm_m = []      # list of m for each mode k
    parent_j = []  # which radial eigenstate j this (l,m) mode comes from
    lm_pairs_dict = defaultdict(int)

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            lm_l.append(ell)
            lm_m.append(m)
            parent_j.append(j_idx)
            lm_pairs_dict[(ell, m)] += 1

    lm_pairs_list = list(lm_pairs_dict.keys())       # unique (l,m) tuples, insertion order
    lm_pairs = jnp.array(lm_pairs_list)              # (N_unique_lm, 2)

    # Map each mode k -> index into lm_pairs_list (its unique (l,m) slot).
    lm_to_idx = {pair: i for i, pair in enumerate(lm_pairs_list)}
    lm_idx_per_mode = jnp.array(
        [lm_to_idx[(ell, m)] for ell, m in zip(lm_l, lm_m)], dtype=jnp.int32)


    '''Precompute Y_lm's for wavefunction reconstruction'''

    L = int(max(l)) + 1

    L_max_out = 2 * L - 1

    n_theta = L_max_out
    n_phi = 2 * L_max_out - 1

    i = np.arange(n_theta)
    theta_np = (np.pi * (2 * i + 1)) / (2 * L_max_out - 1)
    j = np.arange(n_phi)
    phi_np = (2 * np.pi * j) / (2 * L_max_out - 1)

    Theta, Phi = np.meshgrid(theta_np, phi_np, indexing="ij")  # both (n_theta, n_phi), numpy

    Y_lm_np = np.empty((len(lm_pairs_list), n_theta, n_phi), dtype=np.complex64)
    for u, (ell, m) in enumerate(lm_pairs_list):
        Y_lm_np[u] = sph_harm_y(ell, m, Theta, Phi).astype(np.complex64, copy=False)

    Y_lm = jnp.asarray(Y_lm_np)  # single host->device transfer, complex64

    return (jnp.array(parent_j), Y_lm, lm_pairs,
            jnp.array(lm_l), jnp.array(lm_m),
            jnp.asarray(theta_np), jnp.asarray(phi_np),
            lm_idx_per_mode)

@functools.partial(jax.jit, static_argnums=(5,))
def _compute_all_phi(rho_lm_updated, r_updated, output_lm_pairs, mask_int, mask_ext, L_max_out, G, particle_r):
    """
    Vmapped computation of dphi_dr and phi_lm for all (l,m) pairs.
    All args are JAX arrays
    """
    Nr = rho_lm_updated.shape[0] - 1
    dr     = jnp.diff(r_updated)
    dr_rev = jnp.diff(r_updated[::-1])

    def compute_phi_for_lm(lm_pair):
        l_val = lm_pair[0]
        m_val = lm_pair[1]
        prefix = -4.0 * jnp.pi * G / (2 * l_val + 1)
        m_ind  = m_val + L_max_out - 1

        f_at_lm = rho_lm_updated[:, l_val, m_ind]

        integrand_ext = r_updated ** (1 - l_val) * f_at_lm
        integrand_int = r_updated ** (l_val + 2) * f_at_lm

        avg_int = 0.5 * (integrand_int[1:] + integrand_int[:-1])
        integral_int = jnp.sum(jnp.where(mask_int, avg_int * dr, 0.0 + 0.0j))

        integrand_ext_rev = integrand_ext[::-1]
        avg_ext = 0.5 * (integrand_ext_rev[1:] + integrand_ext_rev[:-1])
        integral_ext = -jnp.sum(jnp.where(mask_ext, avg_ext * dr_rev, 0.0 + 0.0j))

        dphi_lm_dr = prefix * (l_val * particle_r ** (l_val - 1) * integral_ext
                            - (l_val + 1) * particle_r ** (-l_val - 2) * integral_int)
        phi_lm  = prefix * (particle_r ** l_val * integral_ext
                            + particle_r ** (-l_val - 1) * integral_int)
        return dphi_lm_dr, phi_lm

    return jax.vmap(compute_phi_for_lm)(output_lm_pairs)


def _sample_q_plummer(seed):
    """Rejection-sample q ∈ (0, 1) from the Plummer isotropic-DF speed
    distribution g(q) = q² (1 − q²)^(7/2), following Eberhardt, Gosenca, Hui
    arXiv:2510.17079 Appendix A.

    g attains a maximum of ≈ 0.0918 on (0, 1), so the bound 0.1 used in the
    von Neumann acceptance test g(X4) > 0.1 X5 is valid.

    Uses jax.random for reproducibility but executes a host-side Python loop —
    only called at IC time so the overhead doesn't matter.
    """
    key = jax.random.PRNGKey(seed)
    for _ in range(1000):
        key, k1, k2 = jax.random.split(key, 3)
        X4 = float(jax.random.uniform(k1, (), jnp.float64))
        X5 = float(jax.random.uniform(k2, (), jnp.float64))
        if X4**2 * (1.0 - X4**2)**3.5 > 0.1 * X5:
            return X4
    raise RuntimeError("Plummer DF rejection sampler failed after 1000 trials")




class Simulation_Particle:
    """
    Stores the state (position + velocity) and history for a single stellar particle.
    """

    def __init__(self, particle_id, init_pos_cart, init_vel_cart, u):

        self.id = particle_id
        self.u = u

        # Current Cartesian state
        self.r_pos = jnp.array(init_pos_cart)   # (3,)
        self.v     = jnp.array(init_vel_cart)    # (3,)

        # Convert to spherical for initial record
        self.r_pos_sph = SSF.Cartesian_to_sph(
            self.r_pos[0], self.r_pos[1], self.r_pos[2]
        )
        self.v_sph = SSF.Cartesian_to_sph_vel(
            self.r_pos[0], self.r_pos[1], self.r_pos[2],
            self.v[0],     self.v[1],     self.v[2]
        )

        # History buffers (same structure as original StellarSimTDep)
        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]
        self.stellar_v_disp = [0]
        self.r_values       = [float(self.r_pos_sph[0])]
        self.average_r      = [float(self.r_pos_sph[0])]
        self.positions_xyz  = [[float(self.r_pos[0]),
                                 float(self.r_pos[1]),
                                 float(self.r_pos[2])]]

        self.potential_energy = []
        self.kinetic_energy = [1/2 * jnp.sum(self.v**2)]
        self.ang_mom = [jnp.linalg.norm(jnp.cross(self.r_pos, self.v))]

        self.time_step = 0

    def Change_to_new_vel(self, v_corrected):

        self.v = jnp.array(v_corrected)
        self.v_sph = SSF.Cartesian_to_sph_vel(
            self.r_pos[0], self.r_pos[1], self.r_pos[2],
            v_corrected[0],     v_corrected[1],     v_corrected[2]
        )
        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]

        self.kinetic_energy = [1/2 * jnp.sum(self.v**2)]
        self.ang_mom = [jnp.linalg.norm(jnp.cross(self.r_pos, self.v))]


    def update_state(self, new_pos_cart, new_vel_cart):
        """
        Called after each rebound integration step to update this particle's
        Cartesian and spherical state and append to history arrays.

        """
        self.r_pos = jnp.array(new_pos_cart)
        self.v     = jnp.array(new_vel_cart)

        self.r_pos_sph = SSF.Cartesian_to_sph(
            self.r_pos[0], self.r_pos[1], self.r_pos[2]
        )
        self.v_sph = SSF.Cartesian_to_sph_vel(
            self.r_pos[0], self.r_pos[1], self.r_pos[2],
            self.v[0],     self.v[1],     self.v[2]
        )

        self.velocities.append(self.v_sph)
        self.velocities_cart.append(self.v)
        velocities_arr = jnp.array(self.velocities)
        new_vel_disp = (
            jnp.std(velocities_arr[:, 0])**2
            + jnp.std(velocities_arr[:, 1])**2
            + jnp.std(velocities_arr[:, 2])**2
        ) ** 0.5
        self.stellar_v_disp.append(new_vel_disp)

        R = float(self.r_pos_sph[0])
        self.r_values.append(R)
        self.positions_xyz.append([float(self.r_pos[0]),
                                    float(self.r_pos[1]),
                                    float(self.r_pos[2])])

        self.kinetic_energy.append(1/2 * jnp.sum(self.v**2))

        self.ang_mom.append(jnp.linalg.norm(jnp.cross(self.r_pos, self.v)))

        self.time_step += 1


#--------------------------------------------------------------------------------------------------------------------


class StellarSimTDep:

    '''
    Diffusive variant: stellar tracers are halo-centered (cluster COM = halo
    centre), positions and velocities drawn from the Plummer isotropic
    distribution function exactly as in Eberhardt, Gosenca & Hui
    arXiv:2510.17079 Appendix A.

    Initial conditions:
        - Position: invert the Plummer M(r)/M_tot CDF (eq A.1) — same r_orbit
          inversion as CC_speed but the Plummer scale is now r_half itself
          (no brentq matching).
        - Speed: rejection-sample q = v / v_e from g(q) = q²(1−q²)^(7/2)
          (eq A.9), v_e set by the Plummer's own potential (eq A.8) using
          cluster_M_Msun and r_half.
        - Direction: isotropic via eqs A.10–A.12.
        - Virial rescaling (eqs A.13–A.14): all velocities multiplied by
          √(|W|/(2K)) where W and K are computed in the actual halo
          potential (the stars will evolve in this potential, not in the
          Plummer self-potential). This corresponds to the paper's "settle
          in external NFW" step but applied as a one-shot rescaling.

    Evolution:
        - 1 Gyr linear ramp of the fluctuating density on top of the
          spherically-averaged static base ρ_base, ported from the
          TIDAL_HEAT variant. ρ_base is mode-diagonal (only the (l,m)=(0,0)
          coefficient is non-zero) and time-independent, giving a clean
          symmetric potential at t = 0. Over the first n_ramp_steps the
          density linearly transitions to the full time-dependent ρ_total.

    Geometry: cluster centred on the halo origin (no orbital offset).
    '''

    def __init__(self, m22, r_half, no_of_particles, no_time_steps, total_evolve_time, r_min, r_max_enclosing_frac, no_radius_bins, SphHT, integrator, plot, dt_override,
                 cluster_M_Msun=7.0e5):

        self.stellar_v_disp = []
        self.average_r = []
        self.time_step = 0
        self.SphHT = SphHT
        self.integrator = integrator
        self.plot = plot
        self.dt_override = dt_override

        # Cluster total mass — sets the Plummer escape velocity v_e at IC time
        # via eq A.8. The stars are zero-mass test particles during the
        # integration; cluster_M_Msun only enters the initial speed scaling.
        self.cluster_M_Msun = float(cluster_M_Msun)


        self.m22 = m22
        self.u = jsp.set_schroedinger_units(self.m22)

        self.no_of_particles = no_of_particles

        # List of Simulation_Particle instances — populated in initialising_simulation()
        self.particles = []

        self.r_half = r_half
        self.no_time_steps = no_time_steps
        self.total_evolve_time = total_evolve_time
        self.dt = (self.total_evolve_time * self.u.from_Gyr) / self.no_time_steps

        self.r_min = r_min
        self.r_max_enclosing_frac = r_max_enclosing_frac

        self.no_radius_bins = no_radius_bins

        self.G = GN.value * (self.u.from_cm**3) / (self.u.from_g * self.u.from_s**2)

        # Precomputed quantities shared across all IAS15 sub-steps within a macro timestep.

        self.current_phase = None   # exp(-i E_j * t / hbar), shape (Nj,)
        self.R_j_r_phased = None    # R_j_r_fixed * current_phase,  shape (Nr, Nj)
        self.eigen_energies = None  # stored from eigenstate_lib
        self.lm_pairs_np = None     # numpy copy of lm_pairs – avoids GPU to CPU


    def initialising_simulation(self):

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_wf")
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"m22_{float(self.m22):.6g}_rbins_{int(self.no_radius_bins)}"
        r_j_r_fname = os.path.join(cache_dir, f"precomputed_R_j_r_{cache_suffix}.npz")
        pkl_fname   = os.path.join(cache_dir, f"precomputed_objs_{cache_suffix}.pkl")
        y_lm_fname  = os.path.join(cache_dir, f"precomputed_Y_lm_{cache_suffix}.npz")

        cache_params = {
            'm22': float(self.m22),
            'r_min': float(self.r_min),
            'r_max_enclosing_frac': float(self.r_max_enclosing_frac),
            'no_radius_bins': int(self.no_radius_bins),
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

        R_j_r = None
        if os.path.isfile(r_j_r_fname) and os.path.isfile(pkl_fname):
            data = np.load(r_j_r_fname)
            if _cache_valid(data, cache_params):
                print(f"Loading precomputed R_j_r from {r_j_r_fname}...")
                R_j_r = data['R_j_r']
                rmin = data['rmin'].item()
                rmax = data['rmax'].item()
                with open(pkl_fname, 'rb') as f:
                    objs = pickle.load(f)
                eigenstate_lib    = objs['eigenstate_lib']
                wavefunction_params = objs['wavefunction_params']
                eval_library = jax.vmap(jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0, None))
            else:
                print(f"Cached {r_j_r_fname} stale (parameter mismatch); recomputing.")
        if R_j_r is None:

            cNFWtides_params = jnp.array([
            357964808.148399 * self.u.from_Msun,
            25.690207,
            0.407461,
            0.012670 * self.u.from_Kpc,
            1.857991 * self.u.from_Kpc,
            3.729259
            ])

            density_params = jsp.init_core_NFW_tides_params_from_sample(cNFWtides_params)

            N = 512
            rmin = .1 * self.u.from_pc
            rmax = jsp.enclosing_radius(0.999, density_params)
            potential_params = jsp.init_potential_params(density_params, rmin, rmax, N)

            eval_library = jax.vmap(jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0,None))

            N = 1024
            a = 1
            b = 10

            rmax = jsp.enclosing_radius(self.r_max_enclosing_frac, density_params)
            eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin, rmax, a, b, N)


            rmin = self.r_min * self.u.from_pc

            tol = 1e-7
            wavefunction_params = jsp.init_wavefunction_params(eigenstate_lib, density_params, rmin, rmax, tol)


            r = jnp.logspace(jnp.log10(rmin), jnp.log10(rmax), self.no_radius_bins)
            R_j_r = eval_library(r, eigenstate_lib.radial_eigenmode_params)  # (Nr, Nj)

            np.savez(r_j_r_fname, R_j_r=np.array(R_j_r), rmin=rmin, rmax=rmax, **cache_params)
            with open(pkl_fname, 'wb') as f:
                pickle.dump({'eigenstate_lib': eigenstate_lib, 'wavefunction_params': wavefunction_params}, f)


        l = eigenstate_lib.radial_eigenmode_params.l
        self.l = l

        print('l max from jaxsp:', max(l))
        L = int(max(l) + 1)

        self.L = L

        self.rmin = rmin
        self.rmax = rmax


        r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)
        self.r = r

        total_mass = wavefunction_params.total_mass
        self.total_mass = total_mass
        aj_2 = wavefunction_params.aj_2        # shape (Nj,)
        self.aj_2_per_j = aj_2                 # used by _compute_rho_base_lms

        self.eigen_energies = eigenstate_lib.radial_eigenmode_params.E

        self.R_j_r_fixed = R_j_r

        phase = jnp.exp(-1j * self.eigen_energies * 0 * self.dt / 1)
        R_j_r_phased = self.R_j_r_fixed * phase[None, :]

        Y_lm = None
        if os.path.isfile(y_lm_fname):
            data = np.load(y_lm_fname)
            if _cache_valid(data, cache_params):
                print(f"Loading precomputed Y_lm and lm_pairs from {y_lm_fname}...")
                parent_j = data['parent_j']
                Y_lm = data['Y_lm']
                lm_pairs = data['lm_pairs']
                lm_l_per_mode = data['lm_l_per_mode']
                lm_m_per_mode = data['lm_m_per_mode']
                theta = data['theta']
                phi = data['phi']
                lm_idx_per_mode = data['lm_idx_per_mode']
            else:
                print(f"Cached {y_lm_fname} stale (parameter mismatch); recomputing.")
        if Y_lm is None:
            (parent_j, Y_lm, lm_pairs, lm_l_per_mode, lm_m_per_mode,
            theta, phi, lm_idx_per_mode) = precompute_lm_pairs_Ylms(l)

            np.savez(y_lm_fname,
                    parent_j=parent_j, Y_lm=Y_lm, lm_pairs=lm_pairs,
                    lm_l_per_mode=lm_l_per_mode, lm_m_per_mode=lm_m_per_mode,
                    theta=theta, phi=phi, lm_idx_per_mode=lm_idx_per_mode,
                    **cache_params)



        Nmodes = len(parent_j)
        rand_phase_per_mode = jax.random.uniform(jax.random.PRNGKey(42), shape=(Nmodes,), minval=0.0, maxval=2 * jnp.pi)
        aj = jnp.sqrt(aj_2[parent_j]) * jnp.exp(1j * rand_phase_per_mode)  # shape (Nmodes,)


        self.parent_j = parent_j
        self.lm_l = lm_pairs[:, 0]        # unique pairs — used for Gaunt table
        self.lm_m = lm_pairs[:, 1]
        self.lm_l_per_mode = lm_l_per_mode  # one per mode — used for scatter matrix
        self.lm_m_per_mode = lm_m_per_mode
        self.lm_idx_per_mode = lm_idx_per_mode  # mode k -> unique-lm slot in Y_lm
        self.theta = theta
        self.phi = phi


        rho_rtp = self.construct_rho_rtp(R_j_r_phased, aj, self.parent_j, Y_lm, self.lm_idx_per_mode)  # (Nr, n_theta, n_phi)


        if self.plot:

            plotting_theta = int(len(theta) / 2)

            plotting_phi = 0

            rho_r = rho_rtp[:, plotting_theta, plotting_phi]

            plt.plot(self.r * self.u.to_Kpc, rho_r * self.u.to_Msun / (self.u.to_Kpc)**3)

            plt.xlabel('r (kpc)')
            plt.ylabel(r'$\rho$ [$M_\odot / kpc^3$]')
            plt.title(f'Density profile with m22 = {self.m22} at $theta = \pi/2, \phi=0$')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid()
            plt.show()



        #------------------------------------------------------------------
        # SIMULATION

        sim_step = rebound.Simulation()


        if self.integrator == 'ias15':

            sim_step.integrator = "ias15"
            sim_step.force_is_velocity_dependent = False
            sim_step.ri_ias15.min_dt = self.dt
            sim_step.ri_ias15.epsilon = 1e-9

        elif self.integrator == 'leapfrog':

            sim_step.integrator = "leapfrog"
            sim_step.dt = self.dt

        # Plummer initial conditions per Eberhardt, Gosenca, Hui arXiv:2510.17079
        # Appendix A. Cluster centred on the halo origin (no orbital offset);
        # r_half IS the Plummer scale R, M_cluster sets the speed normalisation
        # via the cluster's own escape velocity.
        a_plummer = self.r_half * self.u.from_Kpc
        M_cluster = self.cluster_M_Msun * self.u.from_Msun

        # CDF inversion bounds for the position sampler. F_max capped so
        # samples cannot land outside the halo radial grid (where the
        # eigenmode evaluation is undefined).
        def _F(r, a):
            return r**3 / (r**2 + a**2)**1.5
        F_min = 1e-6
        F_max = min(0.999, float(_F(rmax, a_plummer)))

        print(f"Plummer cluster: scale R = r_half = {a_plummer / self.u.from_Kpc:.4f} kpc, "
              f"M_c = {self.cluster_M_Msun:.2e} Msun, "
              f"F sampling bounds = [{F_min:.2e}, {F_max:.4f}]")

        init_vels = []
        self.particles = []
        for i in range(self.no_of_particles):

            # Position: invert M(r)/M_tot to draw radius (eq A.1), then place
            # uniformly on the spherical shell (eqs A.2–A.4 are equivalent
            # to three Gaussians normalised to unit length).
            u_i = jax.random.uniform(jax.random.PRNGKey(i), shape=(), dtype=jnp.float64, minval=F_min, maxval=F_max)
            r_orbit = a_plummer * (u_i**(-2.0/3.0) - 1.0)**(-0.5)


            X1 = jax.random.normal(jax.random.PRNGKey(i+1000), shape=(), dtype=jnp.float64)
            X2 = jax.random.normal(jax.random.PRNGKey(i+2000), shape=(), dtype=jnp.float64)
            X3 = jax.random.normal(jax.random.PRNGKey(i+3000), shape=(), dtype=jnp.float64)

            mag = jnp.sqrt(X1**2 + X2**2 + X3**2)

            r_i = r_orbit * jnp.array([X1, X2, X3]) / mag

            # Speed: rejection-sample q = v/v_e from g(q) = q²(1−q²)^(7/2)
            # (eq A.9), v_e from eq A.8 using the Plummer's own potential.
            a_dim = r_orbit / a_plummer
            v_e = jnp.sqrt(2.0 * self.G * M_cluster / a_plummer) * (1.0 + a_dim**2)**(-0.25)
            q = _sample_q_plummer(seed=i + 8000)
            v_mag = v_e * q

            # Direction: isotropic via eqs A.10–A.12.
            key_dir = jax.random.PRNGKey(i + 9000)
            k1, k2 = jax.random.split(key_dir)
            X6 = jax.random.uniform(k1, (), jnp.float64)
            X7 = jax.random.uniform(k2, (), jnp.float64)
            vz = v_mag * (1.0 - 2.0 * X6)
            v_perp = jnp.sqrt(jnp.maximum(v_mag**2 - vz**2, 0.0))
            vx = v_perp * jnp.cos(2.0 * jnp.pi * X7)
            vy = v_perp * jnp.sin(2.0 * jnp.pi * X7)
            v_i = jnp.array([vx, vy, vz])

            init_pos = r_i
            init_vel = v_i

            init_vels.append(v_mag)

            print(f"Particle {i}: r = {float(r_orbit) / self.u.from_Kpc:.4f} kpc, "
                  f"q = {float(q):.3f}, |v| = {float(v_mag) * self.u.to_kms:.3f} km/s")

            particle = Simulation_Particle(i, init_pos, init_vel, self.u)
            self.particles.append(particle)

            sim_step.add(
                m=0.0,
                x=float(init_pos[0]), y=float(init_pos[1]), z=float(init_pos[2]),
                vx=float(init_vel[0]), vy=float(init_vel[1]), vz=float(init_vel[2])
            )

        ps_step = sim_step.particles

        autodiff_data = {'eval_library': eval_library, 'eigenstate_lib': eigenstate_lib}
        self.autodiff_data = autodiff_data


        self._force_call_count = 0

        def additional_forces_step(_reb_sim):
            """
            IAS15 calls this multiple times per timestep at different positions.
            All particle accelerations are computed in a single batched JAX call
            (vmap over the radial integrals + one vectorised scipy angular call),
            then written back to each rebound particle.
            """
            # Collect all current positions into a single JAX array (N, 3)
            positions_sph = jnp.array([
                SSF.Cartesian_to_sph(ps_step[i].x, ps_step[i].y, ps_step[i].z)
                for i in range(self.no_of_particles)
            ])

            self._force_call_count += 1


            # Single batched acceleration computation — parallel over all particles
            a_r_all, a_theta_all, a_phi_all = self.construct_acc_batch(
                positions_sph,
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib']
            )

            # Write accelerations back to rebound particles
            for i in range(self.no_of_particles):
                pos_sph = positions_sph[i]
                ax, ay, az = SSF.acceleration_spherical_to_cartesian(
                    a_r_all[i], a_theta_all[i], a_phi_all[i], pos_sph[1], pos_sph[2]
                )
                ps_step[i].ax += float(ax)
                ps_step[i].ay += float(ay)
                ps_step[i].az += float(az)

        sim_step.additional_forces = additional_forces_step

        self.sim_step = sim_step
        self.ps_step = ps_step

        radii = jnp.array([jnp.linalg.norm(p.r_pos) for p in self.particles])
        print(f"empirical median r = {jnp.median(radii) / self.u.from_Kpc:.4f} kpc, "
              f"min = {radii.min() / self.u.from_Kpc:.4f}, "
              f"max = {radii.max() / self.u.from_Kpc:.4f}")

        if self.dt_override is not None:

            mean_init_vel = jnp.mean(jnp.array(init_vels), axis=0)

            r_orbit_mean = self.r_half * self.u.from_Kpc
            orbital_P = 2 * jnp.pi * r_orbit_mean / mean_init_vel

            lambda_db_kpc = 19.15 / (self.m22 * mean_init_vel * self.u.to_kms)
            T_c = lambda_db_kpc / (mean_init_vel * self.u.to_Kpc)

            new_dt_orb = orbital_P / self.dt_override

            new_dt_c = T_c / self.dt_override

            new_dt = min(new_dt_orb, new_dt_c)

            self.sim_step.dt = float(new_dt)

            self.dt = new_dt

            self.no_time_steps = int(self.total_evolve_time * self.u.from_Gyr / new_dt)

            print('New dt [Gyr]:', self.dt * self.u.to_Gyr, 'No of time steps:', self.no_time_steps)


        return aj, Y_lm


    def _compute_rho_base_lms(self):
        """
        Spherically-symmetric, mode-diagonal density rho_base(r) and its (l,m)
        decomposition. Splitting psi = Σ_k a_k R_{j(k)} Y_{l_k m_k} e^{-iE_j t/ℏ}
        and keeping only k=k' terms collapses the angular sum via
        Σ_m |Y_lm|² = (2l+1)/(4π) into a purely radial, time-independent
        profile. Only the (l=0, m=0) coefficient is non-zero in the (l,m) basis,
        with f_00 = sqrt(4π) · f(r) since Y_00 = 1/sqrt(4π).
        """
        weights = self.aj_2_per_j * (2 * self.l + 1) / (4 * jnp.pi)              # (Nj,)
        rho_base_r = self.total_mass * (self.R_j_r_fixed ** 2) @ weights          # (Nr,)

        L_max_out = 2 * self.L - 1
        Nr = self.R_j_r_fixed.shape[0]
        rho_base_lms = jnp.zeros(
            (Nr, L_max_out, 2 * L_max_out - 1), dtype=jnp.complex128
        )
        rho_base_lms = rho_base_lms.at[:, 0, L_max_out - 1].set(
            rho_base_r.astype(jnp.complex128) * jnp.sqrt(4 * jnp.pi)
        )

        return rho_base_r, rho_base_lms


    def construct_rho_rtp(self, R_j_r_phased, aj, parent_j, Y_lm, lm_idx_per_mode):

        # Exact algebraic regrouping of psi = sum_k aj[k] * R[:, parent_j[k]] *
        # Y_lm[lm_idx[k]]. Because each (j, l, m) is unique, we can scatter aj
        # into a small (N_unique_lm, Nj) table and contract with the (Nr, Nj)
        # radial basis directly — skipping the (Nr, Nmodes) R_modes gather that
        # previously dominated memory.
        Nj = R_j_r_phased.shape[1]
        coeff_uj = jnp.zeros((Y_lm.shape[0], Nj), dtype=aj.dtype)
        coeff_uj = coeff_uj.at[lm_idx_per_mode, parent_j].add(aj)       # (N_unique_lm, Nj)
        S_ur = coeff_uj @ R_j_r_phased.T                                # (N_unique_lm, Nr)
        full_psi_rtp = jnp.einsum('ur,utp->rtp', S_ur, Y_lm)            # (Nr, n_theta, n_phi)

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2
        rho_rtp = self.total_mass * psi_abs2

        return rho_rtp


    def construct_rho_lms(self, aj, parent_j, R_j_r_phased):

        # Gaunt kernel now takes (Nr, Nj) R_j_r_phased directly and does the
        # mode -> (l,m) collapse internally via scatter-add, avoiding the
        # (Nr, Nmodes) R_modes intermediate.
        rho_lm_gaunt = gf.compute_rho_lm_gaunt(
            aj, R_j_r_phased, parent_j, self.lm_idx_sorted_per_mode,
            self.total_mass,
            L_max_out=self.L_max_out,
            gaunt_table=self.gaunt_table,
            batch_size = 100_000
        )

        return rho_lm_gaunt


    def forward_s2fft(self, rho_rtp):

        '''Forward s2fft of 3d density to get rho_lm(r)
        '''

        #Parallel forward SHT over all radii
        def forward_sht_single_r(rho_at_r):
            return s2fft.forward(rho_at_r, self.L_max_out, sampling='mw', method='jax')

        flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)


        return flm_r



    def _compute_rho_lm_at_particles_gaunt(self, R_j_phased_all, coeff_uj,
                                            all_i, all_j, all_G, all_Lf):
        """Batched Gaunt path: compute rho_lm at every particle's radius in
        ONE call instead of once per particle inside a vmap.

        coeff_uj and the Gaunt arrays are passed as explicit arguments (not
        read from self) so that when this is called from inside the jit'd
        _compute_radial_batch, XLA treats them as dynamic input buffers and
        does NOT bake them into the compiled module's constant pool. Baking
        all_G in as a constant would need another ~6 GB alongside the live
        array and caused the "Failed to allocate new constant" OOM.
        """
        F_all = (coeff_uj @ R_j_phased_all.T).T   # (N_particles, N_unique)
        return gf.compute_rho_lm_gaunt_F(
            F_all, self.total_mass, self.L_max_out,
            all_i, all_j, all_G, all_Lf,
            batch_size=100_000,
        )

    def _compute_rho_lm_at_particles_sphht(self, R_j_phased_all, Q_j_tp):
        """SphHT path: rho_lm per particle still depends on a per-particle
        einsum (psi), so we keep a vmap'd helper here.

        Q_j_tp is passed in (rather than read from self) so XLA treats it as
        a dynamic input buffer instead of a compile-time constant. At
        L_max_out=143 with thousands of radial modes Q_j_tp is ~1.7 GB c128;
        baking it as a constant requires a second copy of the same size,
        which caused a "Failed to allocate new constant" OOM during XLA
        compilation."""
        L_max_out = self.L_max_out
        total_mass = self.total_mass

        def single(R_j_phased):
            psi_at_r = jnp.einsum('j,jtp->tp', R_j_phased, Q_j_tp)
            rho_at_r = total_mass * jnp.abs(psi_at_r) ** 2
            return s2fft.forward(rho_at_r, L_max_out, sampling='mw', method='jax')

        return jax.vmap(single)(R_j_phased_all)   # (N_particles, L_max_out, 2*L_max_out-1)

    def _construct_acc_radial(self, r_pos_sph, rho_lm_at_particle,
                               rho_lms_below, rho_lms_above):
        """
        Per-particle insertion into the background radial grid + call to
        _compute_all_phi. Safe to vmap. rho_lm_at_particle is supplied by
        the caller (computed once for all particles in _compute_radial_batch),
        so this body no longer contains the scatter / Gaunt loop that was
        causing the compile-time blow-up.

        rho_lms_below and rho_lms_above are passed in (rather than read from
        self) so XLA treats them as dynamic input buffers and does not bake
        them into the compiled module's constant pool. Each is c128 of shape
        (Nr+1, L_max_out, 2*L_max_out-1) — at L_max_out=143, Nr=1000 that's
        ~650 MB, and constant-baking required a second copy → OOM. Threading
        them as args also fixes a correctness issue: they are reassigned each
        macro timestep, but as closure-captured constants the JIT cache would
        keep using the values from the first compile.
        """
        particle_r = r_pos_sph[0]

        insert_idx = jnp.searchsorted(self.r, particle_r)

        r_updated = jnp.where(
            self.all_idx < insert_idx,
            self.r_below,
            jnp.where(self.all_idx == insert_idx, particle_r, self.r_above)
        )

        rho_lm_updated = jnp.where(
            self.all_idx[:, None, None] < insert_idx,
            rho_lms_below,
            jnp.where(
                self.all_idx[:, None, None] == insert_idx,
                rho_lm_at_particle[None, :, :],
                rho_lms_above
            )
        )

        mask_int = jnp.arange(self.Nr) < insert_idx
        mask_ext = jnp.arange(self.Nr) < (self.Nr - insert_idx)

        dphi_lm_dr_at_r, phi_lm_at_r = _compute_all_phi(
            rho_lm_updated, r_updated, self.output_lm_pairs,
            mask_int, mask_ext, int(self.L_max_out), self.G, particle_r
        )

        return dphi_lm_dr_at_r, phi_lm_at_r  # (Nmodes,), (Nmodes,)

    def _compute_radial_batch(self, positions_sph, current_phase, radial_eigenmode_params,
                               coeff_uj, all_i, all_j, all_G, all_Lf,
                               rho_lms_below, rho_lms_above, Q_j_tp):
        """JIT-compilable: radial basis evaluation + batched rho_lm + vmap
        over the per-particle insertion step.

        current_phase and radial_eigenmode_params are passed explicitly so JAX
        traces them as dynamic values (they change between macro timesteps).
        eval_library is accessed via self._eval_library, captured as a static
        closure constant since it never changes after setup.

        coeff_uj + the 4 Gaunt arrays + rho_lms_below/above are ALSO explicit
        arguments (instead of being read from self) so XLA treats them as
        dynamic input buffers rather than compile-time constants. all_G alone
        is ~6 GB; rho_lms_below/above are each ~650 MB at L_max_out=143.
        Baking any of them as a constant requires an extra copy of the same
        size, which caused the "Failed to allocate new constant" OOM. The
        rho_lms arrays also change every macro timestep — passing them as
        args ensures the JIT cache uses fresh values rather than stale ones
        baked at first compile.
        """
        particle_rs = positions_sph[:, 0]                                      # (N_particles,)
        R_j_at_particles = self._eval_library(particle_rs, radial_eigenmode_params)
        # R_j_at_particles : (N_particles, Nj)

        R_j_phased_all = R_j_at_particles * current_phase[None, :]             # (N_particles, Nj)

        # Compute rho_lm at each particle's radius OUTSIDE the vmap so the
        # expensive coeff_uj matmul + Gaunt reduction happen once per call,
        # not once per particle.
        if self.SphHT:
            rho_lm_at_particles = self._compute_rho_lm_at_particles_sphht(R_j_phased_all, Q_j_tp)
        else:
            rho_lm_at_particles = self._compute_rho_lm_at_particles_gaunt(
                R_j_phased_all, coeff_uj, all_i, all_j, all_G, all_Lf,
            )
        # rho_lm_at_particles : (N_particles, L_max_out, 2*L_max_out-1)

        # Sequential per-particle evaluation via lax.map. vmap here materialises
        # the per-particle rho_lm_updated tensor (Nr+1, L_max_out, 2*L_max_out-1)
        # ≈ 650 MB in c128 for ALL particles at once — ~10 GB for 15 particles,
        # blowing past device memory once the 15 GB Gaunt table is also resident.
        # lax.map runs one iteration at a time and reuses that per-iter buffer.
        dphi_dr_all, phi_lm_all = jax.lax.map(
            lambda inp: self._construct_acc_radial(
                inp[0], inp[1], rho_lms_below, rho_lms_above
            ),
            (positions_sph, rho_lm_at_particles),
        )
        # dphi_dr_all : (N_particles, Nmodes)
        # phi_lm_all  : (N_particles, Nmodes)

        return dphi_dr_all, phi_lm_all

    @staticmethod
    def _combine_acc(dphi_lm_dr_all, phi_lm_all, Ylm_all, dY_dtheta, dY_dphi, particle_r, particle_theta):
        """JIT-compilable: contract radial outputs with angular terms to get accelerations."""
        dphi_lm_dr_T = dphi_lm_dr_all.T   # (Nmodes, N_particles)
        phi_lm_T  = phi_lm_all.T   # (Nmodes, N_particles)

        a_r     = jnp.sum(-dphi_lm_dr_T * Ylm_all, axis=0).real                                                      # (N_particles,)
        a_theta = jnp.sum(-phi_lm_T  * dY_dtheta / particle_r[None, :], axis=0).real                             # (N_particles,)
        a_phi   = jnp.sum(-phi_lm_T  * dY_dphi   / (particle_r[None, :] * jnp.sin(particle_theta[None, :])), axis=0).real  # (N_particles,)

        return a_r, a_theta, a_phi

    def construct_acc_batch(self, positions_sph, eval_library, eigenstate_lib, poten = False):

        # Lazy JIT compilation on first call — compiles once, reused for all ~110
        # IAS15 sub-steps per macro timestep and across all subsequent timesteps.
        if not hasattr(self, '_compute_radial_batch_jit'):
            self._eval_library = eval_library
            self._compute_radial_batch_jit = jax.jit(self._compute_radial_batch)
            self._combine_acc_jit = jax.jit(StellarSimTDep._combine_acc)

        # JIT-compiled radial part. coeff_uj + the 4 Gaunt arrays are passed
        # as explicit args (rather than captured via self) so XLA treats them
        # as dynamic input buffers, not compile-time constants. For SphHT the
        # static branch inside _compute_radial_batch doesn't use them; tiny
        # placeholder arrays are stored on self in that case.
        dphi_lm_dr_at_r, phi_lm_at_r = self._compute_radial_batch_jit(
            positions_sph,
            self.current_phase,
            eigenstate_lib.radial_eigenmode_params,
            self.coeff_uj,
            self._jit_all_i,
            self._jit_all_j,
            self._jit_all_G,
            self._jit_all_Lf,
            self.rho_lms_below,
            self.rho_lms_above,
            self.Q_j_tp,
        )


        # Scipy angular part: cannot be JIT-compiled
        thetas = np.array(positions_sph[:, 1])   # (N_particles,)
        phis   = np.array(positions_sph[:, 2])   # (N_particles,)

        Ylm_all, dY_all = sph_harm_y(
            self.lm_pairs_np[:, 0, None],   # (Nmodes, 1) — broadcast over particles
            self.lm_pairs_np[:, 1, None],
            thetas[None, :],                # (1, N_particles)
            phis[None, :],
            diff_n=1
        )
        # Ylm_all : (Nmodes, N_particles),  dY_all : (Nmodes, N_particles, 1)
        Ylm_all   = jnp.array(Ylm_all)
        dY_dtheta = jnp.array(dY_all[:, :, 0])

        m_vals  = self.output_lm_pairs[:, 1, None]  # (Nmodes, 1)
        dY_dphi = 1j * m_vals * Ylm_all             # (Nmodes, N_particles)

        # JIT-compiled final contraction

        if poten:
            accs = self._combine_acc_jit(
                dphi_lm_dr_at_r, phi_lm_at_r, Ylm_all, dY_dtheta, dY_dphi,
                positions_sph[:, 0], positions_sph[:, 1],
            )
            return *accs, phi_lm_at_r, Ylm_all



        return self._combine_acc_jit(
            dphi_lm_dr_at_r, phi_lm_at_r, Ylm_all, dY_dtheta, dY_dphi,
            positions_sph[:, 0], positions_sph[:, 1],
        )


    def time_step_particle(self):
        """
        Synchronise all Simulation_Particle states into rebound, integrate
        one macro timestep, then read back and update each particle instance.
        """
        # Write current state of every particle into the rebound simulation
        for i, particle in enumerate(self.particles):
            p = self.ps_step[i]
            p.x,  p.y,  p.z  = float(particle.r_pos[0]), float(particle.r_pos[1]), float(particle.r_pos[2])
            p.vx, p.vy, p.vz = float(particle.v[0]),     float(particle.v[1]),     float(particle.v[2])

        self._force_call_count = 0   # reset counter for this macro step
        target_time = self.sim_step.t + self.dt
        self.sim_step.integrate(target_time)
        print(f"  Force calls this timestep: {self._force_call_count}")

        # Read back and update each Simulation_Particle
        for i, particle in enumerate(self.particles):
            p = self.sim_step.particles[i]
            particle.update_state(
                [p.x,  p.y,  p.z],
                [p.vx, p.vy, p.vz]
            )
            #print(f"  Particle {i}: r = {float(particle.r_pos_sph[0]) * self.u.to_Kpc:.4f} kpc")


    def run_simulation(self):

        start = time()
        aj, Y_lm = self.initialising_simulation()
        end = time()
        self.aj = aj


        Nr = len(self.r)        # number of background radial bins
        all_idx = jnp.arange(Nr + 1)   # indices 0 .. Nr
        self.Nr = Nr
        self.all_idx = all_idx


        r_below = jnp.concatenate([self.r, self.r[-1:]])    # (Nr+1,)
        r_above = jnp.concatenate([self.r[:1], self.r])     # (Nr+1,)
        self.r_below = r_below
        self.r_above = r_above


        L_max_out = 2 * self.L - 1  # captures all density harmonics up to l1+l2 <= 2*(L-1)

        self.L_max_out = L_max_out



        if self.SphHT == False:

            # Precompute Gaunt table ONCE — reuse this across all time steps
            gaunt_table = gf.precompute_gaunt_table(self.lm_l, self.lm_m, L_max_out)
            self.gaunt_table = gaunt_table

            _, _, _, _, unique_lm = gaunt_table
            # Mode -> index in the sorted unique_lm list (matches gaunt_table's
            # i/j indices). Replaces the old (N_unique, Nmodes) scatter_matrix:
            # with ~126k modes that one-hot matrix was itself ~5 GB in float64.
            self.lm_idx_sorted_per_mode = gf.make_lm_idx_sorted_per_mode(
                self.lm_l_per_mode, self.lm_m_per_mode, unique_lm)

            # Pre-scatter aj into the (N_unique, Nj) coefficient table ONCE,
            # here at setup. aj, parent_j and lm_idx_sorted_per_mode are fixed
            # for the whole run, so doing this inside the jitted
            # _compute_radial_batch just made XLA try to constant-fold a
            # ~214 MB c128 array on every call — stored once on self, the
            # matmul downstream sees it as a single traced constant.
            Nj = len(self.eigen_energies)
            N_unique = len(unique_lm)
            coeff_uj_init = jnp.zeros((N_unique, Nj), dtype=self.aj.dtype)
            self.coeff_uj = coeff_uj_init.at[self.lm_idx_sorted_per_mode, self.parent_j].add(self.aj)

            # Bind the 4 Gaunt arrays to explicit names used by the jit call
            # site. Passing them as args (not closure-captured via self inside
            # the jit trace) prevents XLA from baking them into the compiled
            # module's constant pool — all_G alone is ~6 GB and that
            # duplication caused the most recent OOM.
            self._jit_all_i, self._jit_all_j, self._jit_all_G, self._jit_all_Lf, _ = gaunt_table

            # Gaunt path doesn't use Q_j_tp, but the jit signature includes it.
            # Tiny placeholder — the static `if self.SphHT` branch eliminates
            # any actual use.
            self.Q_j_tp = jnp.zeros((1, 1, 1), dtype=self.aj.dtype)

        else:

            Nj = len(self.eigen_energies)          # Nj is number of distinct (n,l) modes
            # Y_lm is (N_unique_lm, n_theta, n_phi). Avoid materialising the
            # (Nmodes, n_theta, n_phi) per-mode array — instead scatter aj into
            # a small (N_unique_lm, Nj) coefficient table and contract with
            # Y_lm. Each (j, l, m) mode is unique, so each (u, j) slot receives
            # at most one aj contribution.
            coeff_uj = jnp.zeros((Y_lm.shape[0], Nj), dtype=self.aj.dtype)
            coeff_uj = coeff_uj.at[self.lm_idx_per_mode, self.parent_j].add(self.aj)
            Q_j_tp = jnp.einsum('uj,utp->jtp', coeff_uj, Y_lm)                       # (Nj, n_theta, n_phi)
            self.Q_j_tp = Q_j_tp

            # SphHT path doesn't need Gaunt arrays, but the jit'd
            # _compute_radial_batch has a fixed signature including them.
            # Provide tiny placeholder arrays — the static `if self.SphHT`
            # branch inside the jit body eliminates any actual use of them.
            self.coeff_uj    = jnp.zeros((1, 1), dtype=self.aj.dtype)
            self._jit_all_i  = jnp.zeros(1, dtype=jnp.int32)
            self._jit_all_j  = jnp.zeros(1, dtype=jnp.int32)
            self._jit_all_G  = jnp.zeros(1, dtype=jnp.float64)
            self._jit_all_Lf = jnp.zeros(1, dtype=jnp.int32)



        # Pre-convert lm_pairs to numpy once so scipy sph_harm_y receives a
        # plain numpy array and avoids a GPU to CPU device transfer every sub-step.
        out_lm = [(L, M) for L in range(L_max_out) for M in range(-L, L+1)]
        output_lm_pairs = jnp.array(out_lm)

        self.output_lm_pairs = output_lm_pairs
        self.lm_pairs_np = np.array(output_lm_pairs)


        # Mode-diagonal piece — used as the constant background during the ramp
        # phase and as the t=0 density for the initial virial-rescale calc so
        # particles see the symmetric (l=0) potential at IC.
        _, self.rho_base_lms = self._compute_rho_base_lms()

        # Linear ramp: the fluctuating piece is scaled from 0 → 1 over the
        # first 1 Gyr so particles are not abruptly exposed to the full
        # fluctuation spectrum. Ported from the TIDAL_HEAT variant.
        ramp_time = 1.0 * self.u.from_Gyr
        self.n_ramp_steps = int(jnp.ceil(ramp_time / self.dt).item())
        self.no_time_steps = self.n_ramp_steps + self.no_time_steps
        print(f"Ramp phase: {self.n_ramp_steps} steps "
              f"({self.n_ramp_steps * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Main phase: {self.no_time_steps - self.n_ramp_steps} steps "
              f"({(self.no_time_steps - self.n_ramp_steps) * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Total: {self.no_time_steps} steps")

        self.current_phase = jnp.exp(-1j * self.eigen_energies * 0 * self.dt)
        self.rho_lms = self.rho_base_lms

        rho_lm_below = jnp.concatenate([self.rho_lms, self.rho_lms[-1:]], axis=0)    # (Nr+1, L, 2L-1)
        rho_lm_above = jnp.concatenate([self.rho_lms[:1], self.rho_lms], axis=0)     # (Nr+1, L, 2L-1)
        self.rho_lms_below = rho_lm_below
        self.rho_lms_above = rho_lm_above


        ##########

        if self.plot == True:

            def inverse_s2fft(single_rho_lm_r):
                return s2fft.inverse(single_rho_lm_r, self.L_max_out, sampling='mw', method='jax')

            rho_rtp = jax.vmap(inverse_s2fft)(self.rho_lms)  # (Nr, L, 2*L-1)


            import yt
            from yt.visualization.volume_rendering.api import (
                Scene,
                Camera,
                TransferFunctionHelper,
                create_volume_source
            )

            def rho_rtp_to_cart(rho_rtp, r, theta, phi, Ncart=None):
                import numpy as np
                from scipy.interpolate import RegularGridInterpolator

                if Ncart is None:
                    Ncart = len(r)

                r = np.asarray(r)
                theta = np.asarray(theta)
                phi = np.asarray(phi)
                rho_rtp = np.asarray(rho_rtp)

                r_max = r[-1]
                r_min = r[0]

                # Interpolate in log-r space since r is logspaced
                log_r = np.log10(r)
                interp = RegularGridInterpolator(
                    (log_r, theta, phi), rho_rtp,
                    bounds_error=False, fill_value=0.0
                )

                # Cartesian grid
                x = np.linspace(-r_max, r_max, Ncart)
                y = np.linspace(-r_max, r_max, Ncart)
                z = np.linspace(-r_max, r_max, Ncart)
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

                # Cartesian -> spherical
                R = np.sqrt(X**2 + Y**2 + Z**2)
                Theta = np.arccos(np.clip(Z / np.clip(R, 1e-30, None), -1, 1))
                Phi = np.arctan2(Y, X) % (2 * np.pi)

                # Interpolate in log-r space
                log_R = np.log10(np.clip(R, r_min, None))
                pts = np.stack([log_R.ravel(), Theta.ravel(), Phi.ravel()], axis=-1)
                rho_xyz = interp(pts).reshape(X.shape)

                return rho_xyz, x, y, z

            rho_xyz, x, y, z = rho_rtp_to_cart(rho_rtp, self.r, self.theta, self.phi)


            ds = yt.load_uniform_grid(
            dict(density=np.asarray(rho_xyz) * float(self.u.to_Msun)/float(self.u.to_Kpc)**3),
            [1000,1000,1000],
            bbox=np.array([[-self.rmax, self.rmax], [-self.rmax, self.rmax], [-self.rmax, self.rmax]]) * float(self.u.to_Kpc),
            length_unit="kpc",
            mass_unit="Msun"
            )

            ds_section = ds.sphere(ds.domain_center,((self.rmax * self.u.to_Kpc).item(),"kpc"))
            sc = yt.create_scene(ds_section, ("stream", "density"), "perspective")
            source = sc.get_source()
            source.set_log(True)
            bounds=(1e-2, 3e5)

            tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=False)

            def quadramp(vals, minval, maxval):
                return ((vals - vals.min()) / (vals.max() - vals.min()))**0.5

            tf.map_to_colormap(
                np.log10(bounds[0]), np.log10(bounds[1]),
                colormap="gist_stern",
                scale_func=quadramp
            )


            tf.add_layers(8,
                        colormap="gist_stern",
                        alpha=np.geomspace(1, 6, 8))

            source.tfh.tf = tf
            source.tfh.bounds = bounds

            camera = sc.camera
            camera.position = [1.,0,0]
            camera.resolution = (900,900)
            camera.zoom(1.)

            camera.switch_orientation()
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            # Render the scene to an image array
            im = sc.render()

            # Plot with matplotlib so we can add a colorbar
            fig, ax = plt.subplots(1, 1, figsize=(9, 9))
            ax.imshow(im[:, :, :3] / im[:, :, :3].max(), origin="lower")
            ax.set_axis_off()

            # Add colorbar matching your transfer function bounds
            norm = mcolors.LogNorm(vmin=bounds[0], vmax=bounds[1])
            sm = plt.cm.ScalarMappable(cmap="gist_stern", norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r"Density [M$_\odot$ / kpc$^3$]")

            plt.tight_layout()
            plt.show()

        #################


        # Initial radial accelerations and potential energies in the static
        # symmetric base potential (rho_lms = rho_base at this point).
        r_pos_sphs = jnp.array([particle.r_pos_sph for particle in self.particles])

        a_r, _, _, phi_lm_at_r, Y_lm_all = self.construct_acc_batch(
            r_pos_sphs,
            self.autodiff_data['eval_library'],
            self.autodiff_data['eigenstate_lib'],
            poten = True
        )

        phi_at_parts = jnp.sum(phi_lm_at_r * Y_lm_all.T, axis=1)  # (N_particles,)

        # Virial rescaling per Eberhardt, Gosenca, Hui arXiv:2510.17079
        # eqs A.13–A.14, applied with the actual halo Φ rather than the
        # Plummer self-potential (the sampled DF is a good first guess but
        # the cluster sits in the real halo — this rescaling enforces
        # 2K + W = 0 in the right potential before the integrator starts).
        # a_r = -∂Φ/∂r ⇒ |W| = Σ m_i r_i ∂Φ/∂r = Σ r_i |a_r_i|  (m_i=1).
        # 2K = Σ |v_i|² for unit masses.
        v_sq_per_part = jnp.array([jnp.sum(p.v**2) for p in self.particles])
        W_abs = jnp.sum(r_pos_sphs[:, 0] * jnp.abs(a_r))
        K2    = jnp.sum(v_sq_per_part)
        virial_scale = jnp.sqrt(W_abs / K2)
        print(f"Virial rescale: |W| = {float(W_abs):.4e}, 2K = {float(K2):.4e}, "
              f"sqrt(|W|/2K) = {float(virial_scale):.4f}")

        for i, particle in enumerate(self.particles):
            v_new = particle.v * virial_scale
            p = self.ps_step[i]
            p.vx, p.vy, p.vz = float(v_new[0]), float(v_new[1]), float(v_new[2])
            particle.potential_energy.append(phi_at_parts[i].real)
            particle.Change_to_new_vel(v_new)



        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")

            phase = jnp.exp(-1j * self.eigen_energies * self.time_step * self.dt / 1)
            R_j_r_phased = self.R_j_r_fixed * phase[None, :]
            self.current_phase = phase  # shape (Nj,)

            if self.SphHT == True:

                rho_rtp = self.construct_rho_rtp(R_j_r_phased, self.aj, self.parent_j, Y_lm, self.lm_idx_per_mode)  # (Nr, n_theta, n_phi)
                rho_total_lms = self.forward_s2fft(rho_rtp)  # (Nr, L, 2*L-1)


            else:
                # 1. Construct total psi and rho on background grid
                rho_total_lms = self.construct_rho_lms(self.aj, self.parent_j, R_j_r_phased)


            # Linear ramp on the fluctuating piece during the first Gyr; after
            # that the particles see the full time-dependent density.
            if self.time_step < self.n_ramp_steps:
                ramp = (self.time_step + 1) / self.n_ramp_steps
                self.rho_lms = self.rho_base_lms + ramp * (rho_total_lms - self.rho_base_lms)
            else:
                self.rho_lms = rho_total_lms


            rho_lm_below = jnp.concatenate([self.rho_lms, self.rho_lms[-1:]], axis=0)    # (Nr+1, L, 2L-1)
            rho_lm_above = jnp.concatenate([self.rho_lms[:1], self.rho_lms], axis=0)     # (Nr+1, L, 2L-1)
            self.rho_lms_below = rho_lm_below
            self.rho_lms_above = rho_lm_above


            # Time step all particles (IAS15 calls additional_forces_step ~8× internally,
            # which loops over every particle each call)
            start = time()
            self.time_step_particle()
            end = time()
            print(f"Time stepping all particles completed in {end - start:.2f} seconds")

            self.current_phase = jnp.exp(-1j * self.eigen_energies * (self.time_step + 1) * self.dt / 1)
            # Compute phi at updated particle positions for potential energy tracking
            r_pos_sphs_new = jnp.array([p.r_pos_sph for p in self.particles])
            _, _, _, phi_lm_new, Ylm_new = self.construct_acc_batch(
                r_pos_sphs_new,
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib'],
                poten=True
            )
            phi_at_parts = jnp.sum(phi_lm_new * Ylm_new.T, axis=1).real  # (N_particles,)
            for i, particle in enumerate(self.particles):
                particle.potential_energy.append(float(phi_at_parts[i]))


            self.time_step += 1
