
import functools
import os
import pickle
from time import time
import matplotlib.pyplot as plt
import rebound
import s2fft
from scipy.special import sph_harm_y
from collections import defaultdict

import sys
sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses')

import jaxsp as jsp

import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jaxsp.constants import GN, hbar

import gaunt_funcs_CC_speed as gf
import Stellar_sim_funcs as SSF

import importlib
importlib.reload(SSF)
importlib.reload(gf)

# k mode is defined as a unique nlm pair
# j mode is defined as a unique nl pair (multiple j modes can have the same l, but different n)
# u is index of unique lm pair

def precompute_lm_pairs(l):
    '''Precompute (l, m) bookkeeping for spherical harmonics.
    '''

    l_for_kmode = []   
    m_for_kmode = []      
    ind_for_jmode_over_all_k = []  
    lm_pairs_dict = defaultdict(int)

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            l_for_kmode.append(ell)
            m_for_kmode.append(m)
            ind_for_jmode_over_all_k.append(j_idx)
            lm_pairs_dict[(ell, m)] += 1

    lm_pairs_list = list(lm_pairs_dict.keys())      
    lm_pairs = jnp.array(lm_pairs_list)  

    lm_pair_to_idx = {pair: i for i, pair in enumerate(lm_pairs_list)} # {(l, m): idx in unique lm pairs}

    lm_pairs_idx_for_kmode = jnp.array(
        [lm_pair_to_idx[(ell, m)] for ell, m in zip(l_for_kmode, m_for_kmode)], dtype=jnp.int32) # way of going from each lm for k to its index in lm pairs

    L = int(max(l)) + 1
    L_max_out = 2 * L - 1
    n_theta = L_max_out
    n_phi = 2 * L_max_out - 1

    i = np.arange(n_theta)
    theta_np = (np.pi * (2 * i + 1)) / (2 * L_max_out - 1)
    j = np.arange(n_phi)
    phi_np = (2 * np.pi * j) / (2 * L_max_out - 1)

    return (jnp.array(ind_for_jmode_over_all_k), lm_pairs,
            jnp.array(l_for_kmode), jnp.array(m_for_kmode),
            jnp.asarray(theta_np), jnp.asarray(phi_np),
            lm_pairs_idx_for_kmode)



@functools.partial(jax.jit, static_argnums=(5,))
def compute_phi_lm_and_deriv(rho_lm_updated, r_updated, output_lm_pairs, mask_int, mask_ext, L_max_out, G, particle_r):

    """
    Vmapped computation of dphi_dr and phi_lm for all (l,m) pairs.
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

        dphi_lm_dr = prefix * (l_val * particle_r ** (l_val - 1) * integral_ext - (l_val + 1) * particle_r ** (-l_val - 2) * integral_int)

        phi_lm  = prefix * (particle_r ** l_val * integral_ext + particle_r ** (-l_val - 1) * integral_int)

        return dphi_lm_dr, phi_lm

    return jax.vmap(compute_phi_for_lm)(output_lm_pairs)



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
        self.r_pos_sph = SSF.Cartesian_to_sph(self.r_pos[0], self.r_pos[1], self.r_pos[2])
        self.v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2],self.v[0], self.v[1], self.v[2])

        # History buffers (same structure as original StellarSimTDep)
        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]
        self.stellar_v_disp = [0]
        self.r_values       = [float(self.r_pos_sph[0])]
        self.average_r      = [float(self.r_pos_sph[0])]
        self.positions_xyz  = [[float(self.r_pos[0]), float(self.r_pos[1]), float(self.r_pos[2])]]

        self.potential_energy = []
        self.kinetic_energy = [1/2 * jnp.sum(self.v**2)]
        self.ang_mom = [jnp.linalg.norm(jnp.cross(self.r_pos, self.v))]

        self.time_step = 0
        
    def Change_to_new_vel(self, v_corrected):

        self.v = jnp.array(v_corrected)
        self.v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2], v_corrected[0], v_corrected[1], v_corrected[2])
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

        self.r_pos_sph = SSF.Cartesian_to_sph(self.r_pos[0], self.r_pos[1], self.r_pos[2])
        self.v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2], self.v[0], self.v[1], self.v[2])

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
    Stellar simulation which controls how everything is done and calls the particle
    class to update particle states.
    '''

    def __init__(self, m22, r_half, r_half_width, no_of_particles, no_time_steps, total_evolve_time, r_min, r_max_enclosing_frac, 
                 no_radius_bins, SphHT, integrator, plot, dt_override, ramp_time):

        self.stellar_v_disp = []
        self.average_r = []
        self.time_step = 0
        self.SphHT = SphHT
        self.integrator = integrator
        self.plot = plot
        self.dt_override = dt_override
        self.ramp_time = ramp_time

        self.m22 = m22
        self.u = jsp.set_schroedinger_units(self.m22)

        self.no_of_particles = no_of_particles

        # List of Simulation_Particle instances — populated in initialising_simulation()
        self.particles = []

        self.r_half = r_half
        self.r_half_width = r_half_width
        self.no_time_steps = no_time_steps
        self.total_evolve_time = total_evolve_time
        self.dt = (self.total_evolve_time * self.u.from_Gyr) / self.no_time_steps

        self.r_min = r_min
        self.r_max_enclosing_frac = r_max_enclosing_frac

        self.no_radius_bins = no_radius_bins

        self.G = GN.value * (self.u.from_cm**3) / (self.u.from_g * self.u.from_s**2)


        self.current_phase = None
        self.R_j_r_phased = None   
        self.eigen_energies = None  
        self.lm_pairs_np = None     


    def initialising_simulation(self):

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_wf")
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"m22_{float(self.m22):.6g}_rbins_{int(self.no_radius_bins)}"
        r_j_r_fname = os.path.join(cache_dir, f"precomputed_R_j_r_{cache_suffix}.npz")
        pkl_fname   = os.path.join(cache_dir, f"precomputed_objs_{cache_suffix}.pkl")

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

        self.L_max_out = 2 * L - 1

        self.rmin = rmin
        self.rmax = rmax
        

        r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)
        self.r = r

        total_mass = wavefunction_params.total_mass
        self.total_mass = total_mass
        aj_2 = wavefunction_params.aj_2 
            
        self.eigen_energies = eigenstate_lib.radial_eigenmode_params.E

        self.R_j_r_fixed = R_j_r

        phase = jnp.exp(-1j * self.eigen_energies * 0 * self.dt / 1)
        R_j_r_phased = self.R_j_r_fixed * phase[None, :]

        (parent_j, lm_pairs, lm_l_per_mode, lm_m_per_mode, theta, phi, lm_idx_per_mode) = precompute_lm_pairs(l)

        Nmodes = len(parent_j)
        rand_phase_per_mode = jax.random.uniform(jax.random.PRNGKey(42), shape=(Nmodes,), minval=0.0, maxval=2 * jnp.pi)
        aj = jnp.sqrt(aj_2[parent_j]) * jnp.exp(1j * rand_phase_per_mode) 


        self.parent_j = parent_j
        self.lm_l = lm_pairs[:, 0]
        self.lm_m = lm_pairs[:, 1]
        self.lm_l_per_mode = lm_l_per_mode 
        self.lm_m_per_mode = lm_m_per_mode
        self.lm_idx_per_mode = lm_idx_per_mode  
        self.lm_pairs_jax = jnp.asarray(lm_pairs, dtype=jnp.int32)
        self.theta = theta
        self.phi = phi
        
        # Constructing initial conditions based on Andrew paper

        rho_rtp = self.construct_rho_rtp(R_j_r_phased, aj, self.parent_j, self.lm_pairs_jax, self.lm_idx_per_mode,)

        # M_enc_tot = SSF.Enclosed_mass_3d(self.r, self.theta, self.phi, rho_rtp, self.rmax)

        # print(f"Total enclosed mass at rmax: {M_enc_tot:.3e}")
        # print(f"Total mass from wavefunction: {total_mass:.3e}")

        # multiply_factor = total_mass / M_enc_tot

        # print(f"Scaling density and mass by factor {multiply_factor} to match total mass")

        # self.total_mass *= multiply_factor

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

        sim = rebound.Simulation()


        if self.integrator == 'ias15':
        
            sim.integrator = "ias15"
            sim.force_is_velocity_dependent = False
            sim.integrator.ri_ias15.epsilon = 1e-5
        
        elif self.integrator == 'leapfrog':

            sim.integrator = "leapfrog"
            sim.dt = self.dt

        init_vels = []
        r_orbit_mean = self.r_half * self.u.from_Kpc
        r_orbit_min = r_orbit_mean - self.r_half_width/2 * self.u.from_Kpc
        r_orbit_max = r_orbit_mean + self.r_half_width/2 * self.u.from_Kpc


        self.particles = []
        for i in range(self.no_of_particles):

            r_orbit = jax.random.uniform(jax.random.PRNGKey(i), shape=(), minval=r_orbit_min, maxval=r_orbit_max)


            X1 = jax.random.normal(jax.random.PRNGKey(i+1000), shape=(), dtype=jnp.float64)
            X2 = jax.random.normal(jax.random.PRNGKey(i+2000), shape=(), dtype=jnp.float64)
            X3 = jax.random.normal(jax.random.PRNGKey(i+3000), shape=(), dtype=jnp.float64)

            mag = jnp.sqrt(X1**2 + X2**2 + X3**2)

            r_i = r_orbit * jnp.array([X1, X2, X3]) / mag

            r_i_unit = r_i / r_orbit

            #avoid degeneracy near z-axis
            ref = jnp.where(jnp.abs(r_i_unit[2]) < 0.9, 
                            jnp.array([0., 0., 1.]), 
                            jnp.array([1., 0., 0.]))
            o_i_unit = jnp.cross(r_i_unit, ref)
            o_i_unit = o_i_unit / jnp.linalg.norm(o_i_unit)


            t_i_unit = jnp.cross(r_i_unit, o_i_unit)

            b_i_unit = jnp.cross(t_i_unit, r_i_unit)

            rand_theta = jax.random.uniform(jax.random.PRNGKey(i+4000), shape=(), minval=0.0, maxval=2 * jnp.pi,)

            v_i_unit = t_i_unit * jnp.sin(rand_theta) + b_i_unit * jnp.cos(rand_theta)

            # Compute circular velocity from spherically-averaged enclosed mass
            M_enc_at_r = SSF.Enclosed_mass_3d(self.r, self.theta, self.phi, rho_rtp, float(r_orbit))
            v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / r_orbit)

            init_pos = r_i
            init_vel = v_circ_mag * v_i_unit

            init_vels.append(v_circ_mag)

            print(f"Particle {i}: v_circ = {v_circ_mag * self.u.to_kms:.3f} km/s")

            particle = Simulation_Particle(i, init_pos, init_vel, self.u)
            self.particles.append(particle) # Adding instances of particles to simulation class

            sim.add(
                m=0.0,
                x=float(init_pos[0]), y=float(init_pos[1]), z=float(init_pos[2]),
                vx=float(init_vel[0]), vy=float(init_vel[1]), vz=float(init_vel[2])
            )

        sim_particles = sim.particles

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
                SSF.Cartesian_to_sph(sim_particles[i].x, sim_particles[i].y, sim_particles[i].z)
                for i in range(self.no_of_particles)
            ])

            self._force_call_count += 1
            

            # Single batched acceleration computation — parallel over all particles
            a_r_all, a_theta_all, a_phi_all = self.construct_acc_master_func(
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
                sim_particles[i].ax += float(ax)
                sim_particles[i].ay += float(ay)
                sim_particles[i].az += float(az)

        sim.additional_forces = additional_forces_step

        self.sim = sim
        self.sim_particles = sim_particles

        r_orbits = jnp.array([p.r_values[0] for p in self.particles])

        r_orbit_mean = jnp.mean(r_orbits)

        print(f"Mean r: {r_orbit_mean * self.u.to_Kpc:.3f} kpc")

        if self.dt_override is not None:

            orbital_P = 2 * jnp.pi * r_orbits / jnp.array(init_vels)

            min_orbital_P = jnp.min(orbital_P)

            mean_init_vel = jnp.mean(jnp.array(init_vels))

            lambda_db_kpc = 19.15 / (self.m22 * mean_init_vel * self.u.to_kms)
            T_c = lambda_db_kpc / (mean_init_vel * self.u.to_Kpc) 

            print(f"Min T_orb: {min_orbital_P * self.u.to_Myr:.3f} Myr")
            print(f"T_c: {T_c * self.u.to_Myr:.3f} Myr")

            new_dt_orb = min_orbital_P / self.dt_override

            new_dt_c = T_c / self.dt_override

            new_dt = min(new_dt_orb, new_dt_c)

            self.sim.dt = float(new_dt)

            self.dt = new_dt

            self.no_time_steps = int(self.total_evolve_time * self.u.from_Gyr / new_dt)

            print(f"dt: {self.dt * self.u.to_Gyr:.3f} Gyr")
            print(f"Number of time steps: {self.no_time_steps}")

        return aj


    def ramp_frac_for_step(self, time_step):

        """Scalar in [0, 1] giving the fraction of the off-diagonal (j != j') 
        cross-terms switched on.
        Linear from ~0 to 1 over n_ramp_steps
        """

        if time_step < self.n_ramp_steps:
            return (time_step + 1) / self.n_ramp_steps
        else:
            return 1.0

    def compute_diagonal_rho_expansion(self):

        """Time-averaged (diagonal) density. This only multiplies terms with same nlm.

        Because |a_{j,m}|² is m-independent (isotropic random phases, line 401),
        the addition theorem Σ_m |Y_{lm}(θ,φ)|² = (2l+1)/4π collapses the
        angular sum to a constant, giving a spherically symmetric static density:

            ρ_static(r) = total_mass · Σ_j  weight_j · |R_j(r)|²
            weight_j     = |a_j|² · (2l_j + 1) / (4π)
        """

        Nj = self.R_j_r_fixed.shape[1]

        # Recover |a|² per j-mode: all k-modes sharing the same j carry the same value.
        aj_sq_k = jnp.abs(self.aj) ** 2                                       # (Nk,)
        aj_sq_j = jnp.zeros(Nj, dtype=jnp.float64).at[self.parent_j].set(aj_sq_k)  # (Nj,)

        weight_j = aj_sq_j * (2.0 * self.l.astype(jnp.float64) + 1.0) / (4.0 * jnp.pi)
        self.weight_j = weight_j                                               # (Nj,) — reused by static rho_lm calls

        R_sq = (jnp.abs(self.R_j_r_fixed) ** 2).astype(jnp.float64)
        rho_static_r = self.total_mass * (R_sq @ weight_j)                    # (Nr,)


        Nr    = rho_static_r.shape[0]
        L_out = self.L_max_out
        rho_static_lms = jnp.zeros((Nr, L_out, 2 * L_out - 1), dtype=jnp.complex128)

        # Since rho_static only depends on r (anglular part collapses), only the 0,0 mode will survive when we project onto Y_lm basis
        # Y00 = 1/sqrt(4pi) so we need to multiply by sqrt(4pi) to get the correct coefficient for the 0,0 mode as the integral will be 4pi.
        return rho_static_lms.at[:, 0, L_out - 1].set(rho_static_r * jnp.sqrt(4.0 * jnp.pi))

    def compute_rho_lm_at_particles_diagonal_only(self, R_j_at_particles):
        """Time-averaged (diagonal-only) rho_lm at each particle's exact r.

        Because ρ_static is spherically symmetric, only (l=0, m=0) is nonzero:
            ρ_lm(r_p) = ρ_static(r_p) · sqrt(4π) · δ_{l0} δ_{m0}
        No SHT needed — just a dot product against weight_j.
        """
        R_sq_all = (jnp.abs(R_j_at_particles) ** 2).astype(jnp.float64)  # (N_p, Nj)
        rho_r_p  = self.total_mass * (R_sq_all @ self.weight_j)           # (N_p,)
        N_p      = rho_r_p.shape[0]
        L_out    = self.L_max_out
        out = jnp.zeros((N_p, L_out, 2 * L_out - 1), dtype=jnp.complex128)
        return out.at[:, 0, L_out - 1].set(rho_r_p * jnp.sqrt(4.0 * jnp.pi))

    def construct_rho_lms_gaunt(self, aj, parent_j, R_j_r_phased):

        '''
        Construct rho_lms using the Gaunt kernel.
        '''

        rho_lm_gaunt = gf.compute_rho_lm_gaunt(
            aj, R_j_r_phased, parent_j, self.lm_idx_sorted_per_mode,
            self.total_mass,
            L_max_out=self.L_max_out,
            gaunt_table=self.gaunt_table,
            batch_size = 100_000
        )

        return rho_lm_gaunt


    def compute_rho_lms_s2fft(self, rho_rtp):

        '''Forward s2fft of 3d density to get rho_lm(r)
        '''

        #Parallel forward SHT over all radii
        def forward_sht_single_r(rho_at_r):
            return s2fft.forward(rho_at_r, self.L_max_out, sampling='mw', method='jax')

        flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)


        return flm_r

    def Build_rho_lms_for_timestep(self, time_step):
        """Build rho_lms at `time_step`, applying the two-phase schedule.

        Phase 1 (ramp, time_step < n_ramp_steps):
            rho = rho_static + ramp_frac * (rho_full(t) - rho_static).
            The off-diagonal (j != j') cross-terms — i.e. the time-dependent
            piece — are linearly switched on from 0 to 1 over `ramp_time`.

        Phase 2 (main, time_step >= n_ramp_steps):
            rho = rho_full(t). Full instantaneous ULDM density, all terms.
        """


        phase = jnp.exp(-1j * self.eigen_energies * time_step * self.dt / 1)
        R_j_r_phased = self.R_j_r_fixed * phase[None, :]
        self.current_phase = phase

        if self.SphHT:
            rho_rtp = self.construct_rho_rtp(
                R_j_r_phased, self.aj, self.parent_j,
                self.lm_pairs_jax, self.lm_idx_per_mode,
            )
            rho_lms_full = self.compute_rho_lms_s2fft(rho_rtp)
        else:
            rho_lms_full = self.construct_rho_lms_gaunt(self.aj, self.parent_j, R_j_r_phased)


        ramp_frac = self.ramp_frac_for_step(time_step)
        self.current_ramp_frac = jnp.float64(ramp_frac)

        return self.rho_static_lms + ramp_frac * (rho_lms_full - self.rho_static_lms)


    def construct_rho_rtp(self, R_j_r_phased, aj, parent_j, lm_pairs, lm_idx_per_mode):

        '''
        Construct rho_rtp without using Y_lms.
        Instead its just an inverse SHT to get psi on the dense grid, then square and forward SHT back to get rho_lm.
        '''

        Nj = R_j_r_phased.shape[1]
        Nr = R_j_r_phased.shape[0]
        L_out = self.L_max_out

        a_u_j = jnp.zeros((lm_pairs.shape[0], Nj), dtype=aj.dtype)
        a_u_j = a_u_j.at[lm_idx_per_mode, parent_j].add(aj)     
        S_ur = a_u_j @ R_j_r_phased.T            # (N_unique_lm, Nr)             


        flm_r = jnp.zeros((Nr, L_out, 2 * L_out - 1), dtype=aj.dtype)
        flm_r = flm_r.at[:, lm_pairs[:, 0], (L_out - 1) + lm_pairs[:, 1]].set(S_ur.T)

        psi_rtp = jax.vmap(
            lambda f: s2fft.inverse(f, L_out, sampling='mw', method='jax')
        )(flm_r)                                                        # (Nr, T, P)

        rho_rtp = self.total_mass * (jnp.abs(psi_rtp) ** 2)
        return rho_rtp




    def compute_rho_lm_at_particles_gaunt(self, R_j_phased_at_parts, a_u_j, all_i, all_j, all_G, all_Lf):
        
        """Batched Gaunt path: compute rho_lm at every particle's radius in
        ONE call instead of once per particle inside a vmap.
        """

        F_all = (a_u_j @ R_j_phased_at_parts.T).T   # (N_particles, N_unique)
        return gf.compute_rho_lm_gaunt_F(
            F_all, self.total_mass, self.L_max_out,
            all_i, all_j, all_G, all_Lf,
            batch_size=100_000,
        )

    def compute_rho_lm_at_particles_sphht(self, R_j_phased_at_parts, a_u_j_sphht):

        """SphHT path: rho_lm per particle via s2fft.inverse
        """

        L_max_out = self.L_max_out
        total_mass = self.total_mass
        lm_pairs = self.lm_pairs_jax

        def single(R_j_phased_at_parts):
            S_u = a_u_j_sphht @ R_j_phased_at_parts                                # (N_unique_lm,)
            flm = jnp.zeros((L_max_out, 2 * L_max_out - 1), dtype=S_u.dtype)
            flm = flm.at[lm_pairs[:, 0], (L_max_out - 1) + lm_pairs[:, 1]].set(S_u)
            psi_at_r = s2fft.inverse(flm, L_max_out, sampling='mw', method='jax')
            rho_at_r = total_mass * jnp.abs(psi_at_r) ** 2
            return s2fft.forward(rho_at_r, L_max_out, sampling='mw', method='jax')

        return jax.vmap(single)(R_j_phased_at_parts)   # (N_particles, L_max_out, 2*L_max_out-1)

    def insert_particle_rholm_and_get_philm(self, r_pos_sph, rho_lm_at_particle, rho_lms_below, rho_lms_above):

        """
        Per-particle insertion into the background radial grid + call to
        _compute_all_phi. Safe to vmap. rho_lm_at_particle is supplied by
        the caller
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

        dphi_lm_dr_at_r, phi_lm_at_r = compute_phi_lm_and_deriv(
            rho_lm_updated, r_updated, self.output_lm_pairs,
            mask_int, mask_ext, int(self.L_max_out), self.G, particle_r
        )

        # if self.plot and not isinstance(particle_r, jax.core.Tracer):

        #     if self.current_ramp_frac == 1.0:
        #         for l in range(3):
        #             plt.plot(r_updated * self.u.to_Kpc, rho_lm_updated[:, l, self.L_max_out - 1] * self.u.to_Msun / (self.u.to_Kpc)**3, label=f'l={l}')
        #         plt.xlabel('r (kpc)')
        #         plt.ylabel(r'$\rho_{lm}$ [$M_\odot / kpc^3$]')
        #         plt.title(f'Updated rho_lm with inserted particle at r={particle_r * self.u.to_Kpc:.3f} kpc')
        #         plt.xscale('log')
        #         plt.yscale('log')
        #         plt.grid()
        #         plt.legend()
        #         plt.show()

        #         def inverse_s2fft(rho_lm):
        #             return s2fft.inverse(rho_lm, int(self.L_max_out), sampling='mw', method='jax')
                
        #         rho_rtp = jax.vmap(inverse_s2fft)(rho_lm_updated)  # (Nr, n_theta, n_phi)

        #         plotting_theta = int(self.theta.shape[0] / 2)
        #         plotting_phi = 0

        #         rho_r = rho_rtp[:, plotting_theta, plotting_phi]
        #         plt.plot(r_updated * self.u.to_Kpc, rho_r * self.u.to_Msun / (self.u.to_Kpc)**3, label='Updated rho(r)')
        #         plt.xlabel('r (kpc)')
        #         plt.ylabel(r'$\rho$ [$M_\odot / kpc^3$]')
        #         plt.title(f'Updated rho(r) with inserted particle at r={particle_r * self.u.to_Kpc:.3f} kpc')
        #         plt.xscale('log')
        #         plt.yscale('log')
        #         plt.grid()
        #         plt.legend()
        #         plt.show()


        return dphi_lm_dr_at_r, phi_lm_at_r  # (Nmodes,), (Nmodes,)

    def calc_rho_lm_at_parts_and_call_insert(self, positions_sph, current_phase, radial_eigenmode_params, a_u_j, all_i, all_j,
                              all_G, all_Lf, rho_lms_below, rho_lms_above, a_u_j_sphht, ramp_frac):
        
        """JIT-compilable: radial basis evaluation + batched rho_lm + vmap
        over the per-particle insertion step.
        """

        particle_rs = positions_sph[:, 0]                         
        R_j_at_particles = self._eval_library(particle_rs, radial_eigenmode_params)
        # R_j_at_particles : (N_particles, Nj)

        R_j_phased_at_parts = R_j_at_particles * current_phase[None, :]             # (N_particles, Nj)

        # Compute rho_lm at each particle's radius OUTSIDE the vmap so the
        # expensive a_u_j matmul + Gaunt reduction happen once per call,
        # not once per particle.

        if self.SphHT:
            rho_lm_at_particles_full = self.compute_rho_lm_at_particles_sphht(R_j_phased_at_parts, a_u_j_sphht)

        else:
            rho_lm_at_particles_full = self.compute_rho_lm_at_particles_gaunt(
                R_j_phased_at_parts, a_u_j, all_i, all_j, all_G, all_Lf,
            )

        # Time-averaged (diagonal-only) rho_lm at the same radii. Used as
        # the baseline for the ramp.
        rho_lm_at_particles_static = self.compute_rho_lm_at_particles_diagonal_only(R_j_at_particles)


        rho_lm_at_particles = (rho_lm_at_particles_static + 
                               ramp_frac * (rho_lm_at_particles_full - rho_lm_at_particles_static))
        # rho_lm_at_particles : (N_particles, L_max_out, 2*L_max_out-1)


        dphi_lm_dr_at_parts, phi_lm_at_parts = jax.lax.map(
            lambda inp: self.insert_particle_rholm_and_get_philm(
                inp[0], inp[1], rho_lms_below, rho_lms_above),
            (positions_sph, rho_lm_at_particles),
        )
        # dphi_dr_all : (N_particles, Nmodes)
        # phi_lm_all  : (N_particles, Nmodes)

        return dphi_lm_dr_at_parts, phi_lm_at_parts

    @staticmethod
    def combine_acc(dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi, particle_r, particle_theta):

        """
        JIT-compilable: contract radial outputs with angular terms to get accelerations.
        """

        dphi_lm_dr_T = dphi_lm_dr_at_parts.T   # (Nmodes, N_particles)
        phi_lm_T  = phi_lm_at_parts.T   # (Nmodes, N_particles)

        a_r     = jnp.sum(-dphi_lm_dr_T * Ylm_all, axis=0).real                                                      # (N_particles,)
        a_theta = jnp.sum(-phi_lm_T  * dY_dtheta / particle_r[None, :], axis=0).real                             # (N_particles,)
        a_phi   = jnp.sum(-phi_lm_T  * dY_dphi   / (particle_r[None, :] * jnp.sin(particle_theta[None, :])), axis=0).real  # (N_particles,)

        return a_r, a_theta, a_phi

    def construct_acc_master_func(self, positions_sph, eval_library, eigenstate_lib, poten = False):

        '''
        Constructing acc vectors for all particles master function.
        With JIT-compilable radial eval + rho_lm + insertion, followed by a non-JIT angular contraction to get accs.
        '''

        if not hasattr(self, 'calc_rho_lm_at_parts_and_call_insert_jit'):
            self._eval_library = eval_library
            self.calc_rho_lm_at_parts_and_call_insert_jit = jax.jit(self.calc_rho_lm_at_parts_and_call_insert)
            self.combine_acc_jit = jax.jit(StellarSimTDep.combine_acc)


        dphi_lm_dr_at_parts, phi_lm_at_parts = self.calc_rho_lm_at_parts_and_call_insert_jit(
            positions_sph,
            self.current_phase,
            eigenstate_lib.radial_eigenmode_params,
            self.a_u_j,
            self._jit_all_i,
            self._jit_all_j,
            self._jit_all_G,
            self._jit_all_Lf,
            self.rho_lms_below,
            self.rho_lms_above,
            self.a_u_j_sphht,
            self.current_ramp_frac,
        )

        # if self.plot:
        #     # Eagerly recompute rho_lm for particle 0 so matplotlib gets concrete
        #     # arrays. _construct_acc_radial is called directly here (outside JIT /
        #     # lax.map), so all arrays are concrete and plt.plot works normally.
        #     _p0_r = positions_sph[0:1, 0]
        #     _R_j_p0 = self._eval_library(_p0_r, eigenstate_lib.radial_eigenmode_params)
        #     _R_j_phased_p0 = _R_j_p0 * self.current_phase[None, :]
        #     if self.SphHT:
        #         _rho_full_p0 = self._compute_rho_lm_at_particles_sphht(_R_j_phased_p0, self.a_u_j_sphht)
        #     else:
        #         _rho_full_p0 = self._compute_rho_lm_at_particles_gaunt(
        #             _R_j_phased_p0, self.a_u_j, self._jit_all_i, self._jit_all_j,
        #             self._jit_all_G, self._jit_all_Lf,
        #         )
        #     _rho_static_p0 = self._compute_rho_lm_at_particles_static(_R_j_p0, self.M_j_t)
        #     _rho_p0 = (_rho_static_p0[0]
        #                + self.current_ramp_frac * (_rho_full_p0[0] - _rho_static_p0[0]))
        #     self._construct_acc_radial(positions_sph[0], _rho_p0,
        #                                self.rho_lms_below, self.rho_lms_above)


        thetas = np.array(positions_sph[:, 1])   # (N_particles,)
        phis   = np.array(positions_sph[:, 2])   # (N_particles,)

        Ylm_all, dY_all = sph_harm_y(
            self.lm_pairs_np[:, 0, None],   # (Nmodes, 1) — broadcast over particles
            self.lm_pairs_np[:, 1, None],
            thetas[None, :],                # (1, N_particles)
            phis[None, :],
            diff_n=1
        )

        Ylm_all   = jnp.array(Ylm_all)
        dY_dtheta = jnp.array(dY_all[:, :, 0])

        m_vals  = self.output_lm_pairs[:, 1, None]  # (Nmodes, 1)
        dY_dphi = 1j * m_vals * Ylm_all             # (Nmodes, N_particles)


        if poten:
            accs = self.combine_acc_jit(
                dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
                positions_sph[:, 0], positions_sph[:, 1],
            )
            return *accs, phi_lm_at_parts, Ylm_all

        

        return self.combine_acc_jit(
            dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
            positions_sph[:, 0], positions_sph[:, 1],
        )


    def time_step_particle(self):

        """
        Synchronise all Simulation_Particle states into rebound, integrate
        one macro timestep, then read back and update each particle instance.
        """

        # Write current state of every particle into the rebound simulation
        for i, particle in enumerate(self.particles):
            p = self.sim_particles[i]
            p.x,  p.y,  p.z  = float(particle.r_pos[0]), float(particle.r_pos[1]), float(particle.r_pos[2])
            p.vx, p.vy, p.vz = float(particle.v[0]),     float(particle.v[1]),     float(particle.v[2])

        self._force_call_count = 0   # reset counter for this macro step
        target_time = self.sim.t + self.dt
        self.sim.integrate(target_time)
        print(f"  Force calls this timestep: {self._force_call_count}")

        # Read back and update each Simulation_Particle
        for i, particle in enumerate(self.particles):
            p = self.sim_particles[i]
            particle.update_state(
                [p.x,  p.y,  p.z],
                [p.vx, p.vy, p.vz]
            )
            #print(f"  Particle {i}: r = {float(particle.r_pos_sph[0]) * self.u.to_Kpc:.4f} kpc")


    def run_simulation(self):

        start = time()
        aj = self.initialising_simulation()
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

            self.lm_idx_sorted_per_mode = gf.make_lm_idx_sorted_per_mode(
                self.lm_l_per_mode, self.lm_m_per_mode, unique_lm)


            Nj = len(self.eigen_energies)
            N_unique = len(unique_lm)
            a_u_j_init = jnp.zeros((N_unique, Nj), dtype=self.aj.dtype)
            self.a_u_j = a_u_j_init.at[self.lm_idx_sorted_per_mode, self.parent_j].add(self.aj)


            self._jit_all_i, self._jit_all_j, self._jit_all_G, self._jit_all_Lf, _ = gaunt_table

            # Gaunt path doesn't use a_u_j_sphht, but the jit signature
            # includes it. Tiny placeholder — the static `if self.SphHT`
            # branch eliminates any actual use.
            self.a_u_j_sphht = jnp.zeros((1, 1), dtype=self.aj.dtype)

        else:

            Nj = len(self.eigen_energies)          # Nj is number of distinct (n,l) modes


            N_unique = self.lm_pairs_jax.shape[0]

            a_u_j_sphht = jnp.zeros((N_unique, Nj), dtype=self.aj.dtype)
            a_u_j_sphht = a_u_j_sphht.at[self.lm_idx_per_mode, self.parent_j].add(self.aj)
            self.a_u_j_sphht = a_u_j_sphht

            # SphHT path doesn't need Gaunt arrays, but the jit'd
            # _compute_radial_batch has a fixed signature including them.
            # Provide tiny placeholder arrays 
            self.a_u_j    = jnp.zeros((1, 1), dtype=self.aj.dtype)
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

        # Ramp phase: linearly switch on the off-diagonal cross-terms — i.e.
        # the time-dependent piece — over `ramp_time` so particles are not
        # abruptly exposed to the full fluctuating spectrum. After the ramp
        # the system evolves for `total_evolve_time` in the full ULDM
        # potential.
        ramp_time = self.ramp_time * self.u.from_Gyr
        self.n_ramp_steps = int(jnp.ceil(ramp_time / self.dt).item())

        # Update no_time_steps to include ramp steps on top of the original
        # main-phase steps.
        self.no_time_steps = self.n_ramp_steps + self.no_time_steps
        print(f"Ramp phase: {self.n_ramp_steps} steps "
              f"({self.n_ramp_steps * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Main phase: {self.no_time_steps - self.n_ramp_steps} steps "
              f"({(self.no_time_steps - self.n_ramp_steps) * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Total: {self.no_time_steps} steps")


        # One-shot precomputation of the time-averaged (diagonal-only) density.
        # Also sets self.weight_j as a side-effect.
        self.rho_static_lms = self.compute_diagonal_rho_expansion()

        #Build_rho_lms_for_timestep(timestep) - needs rho_static_lms computed and in self
        self.rho_lms = self.Build_rho_lms_for_timestep(0)

        rho_lm_below = jnp.concatenate([self.rho_lms, self.rho_lms[-1:]], axis=0)    # (Nr+1, L, 2L-1)
        rho_lm_above = jnp.concatenate([self.rho_lms[:1], self.rho_lms], axis=0)     # (Nr+1, L, 2L-1)
        self.rho_lms_below = rho_lm_below
        self.rho_lms_above = rho_lm_above


        ###################################

        # if self.plot == True:

        #     def inverse_s2fft(single_rho_lm_r):
        #         return s2fft.inverse(single_rho_lm_r, self.L_max_out, sampling='mw', method='jax')

        #     rho_rtp = jax.vmap(inverse_s2fft)(self.rho_lms)  # (Nr, L, 2*L-1)


        #     import yt
        #     from yt.visualization.volume_rendering.api import (
        #         Scene, 
        #         Camera, 
        #         TransferFunctionHelper, 
        #         create_volume_source
        #     )

        #     def rho_rtp_to_cart(rho_rtp, r, theta, phi, Ncart=None):
        #         import numpy as np
        #         from scipy.interpolate import RegularGridInterpolator

        #         if Ncart is None:
        #             Ncart = len(r)

        #         r = np.asarray(r)
        #         theta = np.asarray(theta)
        #         phi = np.asarray(phi)
        #         rho_rtp = np.asarray(rho_rtp)

        #         r_max = r[-1]
        #         r_min = r[0]

        #         # Interpolate in log-r space since r is logspaced
        #         log_r = np.log10(r)
        #         interp = RegularGridInterpolator(
        #             (log_r, theta, phi), rho_rtp,
        #             bounds_error=False, fill_value=0.0
        #         )

        #         # Cartesian grid
        #         x = np.linspace(-r_max, r_max, Ncart)
        #         y = np.linspace(-r_max, r_max, Ncart)
        #         z = np.linspace(-r_max, r_max, Ncart)
        #         X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        #         # Cartesian -> spherical
        #         R = np.sqrt(X**2 + Y**2 + Z**2)
        #         Theta = np.arccos(np.clip(Z / np.clip(R, 1e-30, None), -1, 1))
        #         Phi = np.arctan2(Y, X) % (2 * np.pi)

        #         # Interpolate in log-r space
        #         log_R = np.log10(np.clip(R, r_min, None))
        #         pts = np.stack([log_R.ravel(), Theta.ravel(), Phi.ravel()], axis=-1)
        #         rho_xyz = interp(pts).reshape(X.shape)

        #         return rho_xyz, x, y, z

        #     rho_xyz, x, y, z = rho_rtp_to_cart(rho_rtp, self.r, self.theta, self.phi)


        #     ds = yt.load_uniform_grid(
        #     dict(density=np.asarray(rho_xyz) * float(self.u.to_Msun)/float(self.u.to_Kpc)**3),
        #     [1000,1000,1000],
        #     bbox=np.array([[-self.rmax, self.rmax], [-self.rmax, self.rmax], [-self.rmax, self.rmax]]) * float(self.u.to_Kpc),
        #     length_unit="kpc",
        #     mass_unit="Msun"
        #     )

        #     ds_section = ds.sphere(ds.domain_center,((self.rmax * self.u.to_Kpc).item(),"kpc"))
        #     sc = yt.create_scene(ds_section, ("stream", "density"), "perspective")
        #     source = sc.get_source()
        #     source.set_log(True)
        #     bounds=(1e-2, 3e5)
                
        #     tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=False)

        #     def quadramp(vals, minval, maxval):
        #         return ((vals - vals.min()) / (vals.max() - vals.min()))**0.5

        #     tf.map_to_colormap(
        #         np.log10(bounds[0]), np.log10(bounds[1]), 
        #         colormap="gist_stern", 
        #         scale_func=quadramp
        #     )

            
        #     tf.add_layers(8,
        #                 colormap="gist_stern", 
        #                 alpha=np.geomspace(1, 6, 8))

        #     source.tfh.tf = tf
        #     source.tfh.bounds = bounds

        #     camera = sc.camera
        #     camera.position = [1.,0,0]
        #     camera.resolution = (900,900)
        #     camera.zoom(1.)

        #     camera.switch_orientation()
        #     import matplotlib.pyplot as plt
        #     import matplotlib.colors as mcolors

        #     # Render the scene to an image array
        #     im = sc.render()

        #     # Plot with matplotlib so we can add a colorbar
        #     fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        #     ax.imshow(im[:, :, :3] / im[:, :, :3].max(), origin="lower")
        #     ax.set_axis_off()

        #     # Add colorbar matching your transfer function bounds
        #     norm = mcolors.LogNorm(vmin=bounds[0], vmax=bounds[1])
        #     sm = plt.cm.ScalarMappable(cmap="gist_stern", norm=norm)
        #     sm.set_array([])
        #     cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        #     cbar.set_label(r"Density [M$_\odot$ / kpc$^3$]")

        #     plt.tight_layout()
        #     plt.show()

        ######################################


        # Compute initial potential energy for each particle
                
        r_pos_sphs = jnp.array([particle.r_pos_sph for particle in self.particles])  # (N_particles, 3)

        a_r, _, _, phi_lm_at_r, Y_lm_all = self.construct_acc_master_func(
            r_pos_sphs,
            self.autodiff_data['eval_library'],
            self.autodiff_data['eigenstate_lib'],
            poten = True
        )

        #v_circ_true = jnp.sqrt(jnp.abs(a_r) * r_pos_sphs[:, 0])

        phi_at_parts = jnp.sum(phi_lm_at_r * Y_lm_all.T, axis=1)  # (N_particles,)

        for i, particle in enumerate(self.particles):

            #v_old = particle.v
            #v_dir = v_old / jnp.linalg.norm(v_old)
            #v_new = v_dir * v_circ_true[i]
            #p = self.ps_step[i]
            #p.vx, p.vy, p.vz = float(v_new[0]), float(v_new[1]), float(v_new[2])
            particle.potential_energy.append(phi_at_parts[i].real)

            #particle.Change_to_new_vel(v_new)

        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")


            self.rho_lms = self.Build_rho_lms_for_timestep(self.time_step)

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
            _, _, _, phi_lm_new, Ylm_new = self.construct_acc_master_func(
                r_pos_sphs_new,
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib'],
                poten=True
            )

            phi_at_parts = jnp.sum(phi_lm_new * Ylm_new.T, axis=1).real  # (N_particles,)
            for i, particle in enumerate(self.particles):
                particle.potential_energy.append(float(phi_at_parts[i]))


            self.time_step += 1


    def run_simulation_profiled(self, time_output='profile_time.prof',
                                memory_output='profile_memory.txt', top_n=30):
        """Run `run_simulation` under cProfile (time) and tracemalloc (memory).

        cProfile output goes to `time_output` (binary). Inspect with e.g.
            python -m pstats profile_time.prof
            # or interactively: snakeviz profile_time.prof
        A console summary of the top `top_n` functions is printed for both
        cumulative-time and self-time sorts.

        tracemalloc takes a single snapshot at the end of the run and writes
        the top `top_n` allocation sites (by total size) to `memory_output`.
        Peak/current allocated bytes are also reported.

        Caveat for JAX: most heavy work here is JIT'd and dispatched
        asynchronously, so cProfile timings reflect Python-side overhead +
        time blocking on async results. For pure kernel timing, use
        `jax.profiler` or insert `jax.block_until_ready(...)` at key points.
        """
        import cProfile
        import pstats
        import tracemalloc

        tracemalloc.start()
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            self.run_simulation()
        finally:
            profiler.disable()
            snapshot = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        profiler.dump_stats(time_output)

        stats = pstats.Stats(profiler).strip_dirs()
        print(f"\n===== cProfile: top {top_n} by cumulative time =====")
        stats.sort_stats('cumulative').print_stats(top_n)
        print(f"\n===== cProfile: top {top_n} by self (tottime) =====")
        stats.sort_stats('tottime').print_stats(top_n)

        top_stats = snapshot.statistics('lineno')
        header = (f"tracemalloc: peak = {peak / 1e6:.2f} MB, "
                  f"current = {current / 1e6:.2f} MB")
        print(f"\n===== {header} =====")
        print(f"Top {top_n} allocation sites (by size):")
        with open(memory_output, 'w') as f:
            f.write(header + "\n\n")
            f.write(f"Top {top_n} allocation sites (by size):\n")
            for stat in top_stats[:top_n]:
                line = str(stat)
                print(line)
                f.write(line + "\n")

        print(f"\nFull cProfile data written to: {time_output}")
        print(f"Memory summary written to:     {memory_output}")
