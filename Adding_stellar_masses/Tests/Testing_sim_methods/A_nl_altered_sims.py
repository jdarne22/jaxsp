
import functools

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

import gaunt_funcs as gf

importlib.reload(SSF)
importlib.reload(gf)


'''For Animations and plotting'''
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib import cm
from matplotlib.colors import Normalize


def precompute_lm_pairs_Ylms(l):

    '''Precompute (l, m) pair for spherical harmonics'''

    lm_l = []      # list of l for each mode k
    lm_m = []      # list of m for each mode k
    parent_j = []  # which radial eigenstate j this (l,m) mode comes from
    lm_pairs = defaultdict(int)

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            lm_l.append(ell)
            lm_m.append(m)
            parent_j.append(j_idx)
            lm_pairs[(ell, m)] += 1

    lm_pairs = list(lm_pairs.keys()) # list of ((l,m), count) pairs

    lm_pairs = jnp.array(lm_pairs)  # shape (Nmodes, 2)


    '''Precompute Y_lm's for wavefunction reconstruction'''

    # McEwen-Wiaux-style equiangular grid

    L = max(l)+1

    L_max_out = 2 * L - 1

    n_theta = L_max_out
    n_phi = 2 * L_max_out - 1

    # Generate theta values
    i = jnp.arange(n_theta)
    theta = (jnp.pi * (2 * i + 1)) / (2 * L_max_out - 1)
    # Generate phi values
    j = jnp.arange(n_phi)
    phi = (2 * jnp.pi * j) / (2 * L_max_out - 1)


    Theta, Phi = jnp.meshgrid(theta, phi, indexing="ij")  # both (n_theta, n_phi)

    Y_list = []
    for ell, m in zip(lm_l, lm_m):
        Y_lm_mode = sph_harm_y(ell, m, Theta, Phi)  # (n_theta, n_phi), complex
        Y_list.append(Y_lm_mode)

    Y_lm = jnp.stack(Y_list, axis=0)  # (Nmodes, n_theta, n_phi), complex


    return jnp.array(parent_j), Y_lm, lm_pairs, jnp.array(lm_l), jnp.array(lm_m), theta, phi

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



#--------------------------------------------------------------------------------------------------------------------------
"""
3D animation of axion dark matter density in the equatorial (z=0) plane.
The surface height at each (x, y) point encodes the local density.
"""
class SimulationAnimator:

    def __init__(self, L, r_grid, u):

        self.u = u
        self.L = L

        # MW-sampling angular grid
        n_theta = L
        n_phi   = 2 * L - 1
        i_arr   = np.arange(n_theta)
        theta   = (np.pi * (2 * i_arr + 1)) / (2 * L - 1)
        j_arr   = np.arange(n_phi)
        phi     = (2 * np.pi * j_arr) / (2 * L - 1)
        phi     = np.append(phi, 2 * np.pi)  # close phi ring for smooth plotting

        # Index of the theta slice closest to pi/2 (equatorial plane, z=0)
        self.theta_eq_idx = int(np.argmin(np.abs(theta - np.pi / 2)))
        print(f"Equatorial slice: theta index {self.theta_eq_idx}, "
              f"theta = {theta[self.theta_eq_idx]:.4f} rad (pi/2 = {np.pi/2:.4f})")

        # 2D polar grid in the xy-plane: r x phi → (X, Y)
        r_kpc = np.array(r_grid) * float(u.to_Kpc)
        R_grid, Phi_grid = np.meshgrid(r_kpc, phi, indexing='ij')  # (Nr, n_phi+1)
        self.X = R_grid * np.cos(Phi_grid)
        self.Y = R_grid * np.sin(Phi_grid)

        # Frame storage — particle_positions is now a list of lists (one per particle)
        self.density_frames = []
        self.particle_positions = []   # list of frames; each frame is a list of (x,y,z) per particle

    def store_frame(self, rho_rtp, particle_xyz_list):
        """
        Capture one snapshot: equatorial density slice + all particle positions.

        Parameters
        ----------
        rho_rtp : array (Nr, n_theta, n_phi)
        particle_xyz_list : list of (x, y, z) Cartesian positions, one per particle
        """
        rho_eq = np.array(np.real(rho_rtp[:, self.theta_eq_idx, :]))  # (Nr, n_phi)
        rho_eq = np.append(rho_eq, rho_eq[:, 0:1], axis=1)            # close phi: (Nr, n_phi+1)
        self.density_frames.append(rho_eq)
        frame_positions = [
            [float(xyz[0]) * float(self.u.to_Kpc),
             float(xyz[1]) * float(self.u.to_Kpc),
             float(xyz[2]) * float(self.u.to_Kpc)]
            for xyz in particle_xyz_list
        ]
        self.particle_positions.append(frame_positions)

    def create_animation(self, interval=200, save_path=None, orbit_radius_kpc=None):
        """
        Build a matplotlib FuncAnimation from the stored frames.
        X/Y axes are spatial coordinates in the equatorial plane (kpc).
        Z axis is the density value at each (x, y) point.
        All particles are shown as individual scatter markers.
        """

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        cmap = cm.inferno

        # Use a fixed global colour scale across all frames so the oscillation is visible
        all_rho = np.concatenate([f.ravel() for f in self.density_frames])
        global_norm = Normalize(vmin=all_rho.min(), vmax=all_rho.max())

        # Number of particles (inferred from first frame)
        n_particles = len(self.particle_positions[0])

        # Pre-convert positions: shape (n_frames, n_particles, 3)
        all_positions = np.array(self.particle_positions)  # (n_frames, n_particles, 3)

        # Initial density surface
        rho0 = self.density_frames[0]
        surf = [ax.plot_surface(self.X, self.Y, rho0,
                                cmap=cmap, norm=global_norm, shade=False, alpha=0.85)]

        # One scatter marker per particle
        colors = plt.cm.Set1(np.linspace(0, 1, n_particles))
        particle_dots = []
        for i in range(n_particles):
            p0 = self.particle_positions[0][i]
            dot = ax.scatter([p0[0]], [p0[1]], [0.0],
                             color=colors[i], s=80, zorder=5,
                             edgecolors='white', linewidths=0.5,
                             depthshade=False)
            particle_dots.append(dot)

        # Orbit trails per particle
        trail_lines = []
        for i in range(n_particles):
            p0 = self.particle_positions[0][i]
            line, = ax.plot([p0[0]], [p0[1]], [0.0],
                            color=colors[i], alpha=0.5, linewidth=1)
            trail_lines.append(line)

        # Axis limits
        if orbit_radius_kpc is not None:
            xy_lim = orbit_radius_kpc * 1.3
        else:
            xy_lim = np.max(np.abs(all_positions[:, :, :2])) * 1.3
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-xy_lim, xy_lim)
        ax.set_xlabel(r'$x$ [kpc]')
        ax.set_ylabel(r'$y$ [kpc]')
        ax.set_zlabel(r'$\rho$ [a.u.]')
        title = ax.set_title('Time step: 0')

        mappable = cm.ScalarMappable(norm=global_norm, cmap=cmap)
        fig.colorbar(mappable, ax=ax, shrink=0.5, label=r'$\rho$')

        def update(frame):
            surf[0].remove()
            rho = self.density_frames[frame]
            surf[0] = ax.plot_surface(self.X, self.Y, rho,
                                      cmap=cmap, norm=global_norm, shade=False, alpha=0.85)

            for i in range(n_particles):
                pos = self.particle_positions[frame][i]
                particle_dots[i]._offsets3d = ([pos[0]], [pos[1]], [0.0])
                trail_lines[i].set_data(all_positions[:frame + 1, i, 0],
                                        all_positions[:frame + 1, i, 1])
                trail_lines[i].set_3d_properties(np.zeros(frame + 1))

            title.set_text(f'Time step: {frame + 1}')
            return [surf[0]] + particle_dots + trail_lines

        import matplotlib
        matplotlib.rcParams['animation.embed_limit'] = 2**10

        ani = mpl_animation.FuncAnimation(fig, update,
                                          frames=len(self.density_frames),
                                          interval=interval, blit=False)

        if save_path:
            ani.save(save_path, writer='ffmpeg', fps=10)
            print(f"Animation saved to {save_path}")

        else:
            from IPython.display import HTML, display
            html = HTML(ani.to_jshtml())
            plt.close(fig)
            display(html)
            return html

#--------------------------------------------------------------------------------------------------------------------


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
        self.velocities     = [self.v_sph]
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
        self.velocities = [self.v_sph]

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
    Methodology:

    Initialise simulation - construct wavefunctions using jaxsp, construct eigenenergies and l values
    Make rho consistent with the total mass of LeoII, setup particle IC (position and vel dir), create persistent simulation
    which calls the force function at each microstep

    Setup - Precompute number of radial bins for insertion indices later, Pre compute Gaunt table and scatter matrix for rho_lm construction
    Precompute Q_j_tp for SphHT method, create phased wavefunctions (in this case the phase is fixed at exp(-i E_j * t / hbar)

    Get rho_lms - For s2fft call constructing rho_rtp and then call forward s2fft, for gaunt, call construct_rho_lms which does the summation over j and 
    l', m' using the precomputed Gaunt table and scatter matrix. If static, set all l > 0 modes to zero. Expands up to L_max_out = 2L - 1

    Calculate v_corrected - Use batched acceleration pipeline to get accelerations for each particles position, make these related to the magnitude 
    of the velocities |v| = sqrt(a_r*r) but keep direction same as in initialising simulation

    Main loop - Continue to run function timestep_particle until we reach the number of timesteps initially set. This function writes particle states (pos and vel) 
    to rebound sim, calls the integrator which in turn, calls the force function inside the simulation multiple times to get acc vector for each microstep and 
    advance the particles, then reads the new particle states and updates the particle class.

    Forces Function: 

    Force function is called at every microstep of each macro timestep for ias15.
    It collects all the current particle positions
    Then it runs construct_acc_batch (the main function)
    Construct_acc_batch then calls _compute_radial_batch_jit which is a jax.jit version of _compute_radial_batch.
    This then calculates the wavefunctions at the positions of each of the particles and multiplies it by the current phase (constant)
    Then it runs a vmap over _construct_acc_radial - this computes the rho_lms at each particle position depending on whether its static/frozen or SphHT/Gaunt
    It inserts r and the rho_lm's into the correct index and runs _compute_all_phi
    _compute_all_phi is a jax.jit function which vmaps over all lm pairs to calculate the integrals for dphi_lm/dr and phi_lm evaluated at the particle r positions
    Then it takes all particle thetas and phis to construct the spherical harmonics at the positions of the particles up to 2L-1
    Runs _combine_acc_jit, the jit version of _combine_acc, a static function which contracts the radial and angular parts to get the accelerations at the positions 
    of each particle

    Reason for this setup: can jax jit and jax vmap over alot of things, however, sph_harm function isnt jax compatible therefore need to have it separate
    (in main function construct_acc_batch)
    '''

    def __init__(self, m22, r_half, no_of_particles, no_time_steps, total_evolve_time, r_min, r_max_enclosing_frac, no_radius_bins, SphHT, integrator, a_nl_range, plot,
                 boost_factor, animate=False, animate_every=1):

        self.stellar_v_disp = []
        self.average_r = []
        self.time_step = 0
        self.SphHT = SphHT
        self.integrator = integrator
        self.a_nl_range = a_nl_range
        self.plot = plot
        self.boost_factor = boost_factor


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

        # Animation settings
        self.animate = animate
        self.animate_every = animate_every
        self.animator = None

        # Precomputed quantities shared across all IAS15 sub-steps within a macro timestep.

        self.current_phase = None   # exp(-i E_j * t / hbar), shape (Nj,)
        self.R_j_r_phased = None    # R_j_r_fixed * current_phase,  shape (Nr, Nj)
        self.eigen_energies = None  # stored from eigenstate_lib
        self.lm_pairs_np = None     # numpy copy of lm_pairs – avoids GPU to CPU


    def initialising_simulation(self):


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

        self.eigen_energies = eigenstate_lib.radial_eigenmode_params.E

        l = eigenstate_lib.radial_eigenmode_params.l
        n = eigenstate_lib.radial_eigenmode_params.n
        self.l = l


        print('l max from jaxsp:', max(l))
        L = int(max(l) + 1)
        self.L = L

        rmin = self.r_min * self.u.from_pc
        self.rmin = rmin
        self.rmax = rmax

        r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)
        self.r = r


        tol = 1e-7
        wavefunction_params = jsp.init_wavefunction_params(eigenstate_lib, density_params, rmin, rmax, tol)

        

        total_mass = wavefunction_params.total_mass
        self.total_mass = total_mass
        aj_2 = wavefunction_params.aj_2        # shape (Nj,)

 
        nl_labels = list(zip(l.tolist(), n.tolist()))


        T_periods = -(1 / self.eigen_energies)

        first_mode_amp = aj_2[0]

        average_part_period = 0.3 * self.u.from_Gyr

        if self.a_nl_range == 'orbital':


            sigma = average_part_period  # width of resonance
            weight = jnp.exp(-0.5 * ((T_periods - average_part_period) / sigma)**2)

            weight = jnp.where(l == 0, 0.0, weight)  # no boost for l=0 modes

            boost = 1.0 + (self.boost_factor - 1.0) * weight  # smooth boost peaking at orbital period
            aj_2_new = aj_2 * boost
            aj_2_new = aj_2_new * (jnp.sum(aj_2) / jnp.sum(aj_2_new))  # normalisation


        
        elif self.a_nl_range == 'large':

            mask = (T_periods > 15*average_part_period)

            aj_2_new = jnp.where(mask, first_mode_amp, aj_2)
        


        elif self.a_nl_range == 'small':

            mask = (T_periods > 5*average_part_period) & (T_periods < 15*average_part_period)

            aj_2_new = jnp.where(mask, first_mode_amp, aj_2)


        if self.plot: 

            plt.figure(figsize=(10, 5))
            plt.scatter(range(len(aj_2_new)), aj_2_new, s=5, label='Modified a_nl^2', color='orange', alpha=0.7)
            plt.scatter(range(len(aj_2)), aj_2, s=5, label='Original a_nl^2', color='blue', alpha=0.5)
            plt.legend()

            step = max(1, len(nl_labels) // 30)  # show ~30 labels
            tick_positions = range(0, len(nl_labels), step)
            tick_labels = [nl_labels[i] for i in tick_positions]
            plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)

            plt.xlabel('(l, n)')
            plt.ylabel('aj_2')
            plt.title('Comparing old/new aj_2 values per (l, n) pair')
            plt.grid(axis='y')
            plt.yscale('log')
            plt.tight_layout()
            plt.show()

        aj_2 = aj_2_new


        R_j_r = eval_library(self.r, eigenstate_lib.radial_eigenmode_params)  # (Nr, Nj)
        self.R_j_r_fixed = R_j_r


        phase = jnp.exp(-1j * self.eigen_energies * 0 * self.dt / 1)
        R_j_r_phased = self.R_j_r_fixed * phase[None, :]
        self.current_phase = phase  # shape (Nj,)
        self.R_j_r_phased = R_j_r_phased

        parent_j, Y_lm, lm_pairs, lm_l_per_mode, lm_m_per_mode, theta, phi = precompute_lm_pairs_Ylms(l)

        Nmodes = len(parent_j)
        rand_phase_per_mode = jax.random.uniform(jax.random.PRNGKey(42), shape=(Nmodes,), minval=0.0, maxval=2 * jnp.pi)
        aj = jnp.sqrt(aj_2[parent_j]) * jnp.exp(1j * rand_phase_per_mode)  # shape (Nmodes,)

        
        self.parent_j = parent_j
        self.lm_l = lm_pairs[:, 0]        # unique pairs — used for Gaunt table
        self.lm_m = lm_pairs[:, 1]
        self.lm_l_per_mode = lm_l_per_mode  # one per mode — used for scatter matrix
        self.lm_m_per_mode = lm_m_per_mode
        self.theta = theta
        self.phi = phi
        
        # Constructing initial conditions based on Andrew paper

        r_orbit_mean = self.r_half * self.u.from_Kpc


        rho_rtp = self.construct_rho_rtp(R_j_r_phased, aj, self.parent_j, Y_lm)  # (Nr, n_theta, n_phi)

        M_enc_tot = SSF.Enclosed_mass_3d(self.r, self.theta, self.phi, rho_rtp, self.rmax)

        print(f"Total enclosed mass at rmax: {M_enc_tot:.3e}")
        print(f"Total mass from wavefunction: {total_mass:.3e}")

        multiply_factor = total_mass / M_enc_tot

        print(f"Scaling density and mass by factor {multiply_factor} to match total mass")

        self.total_mass *= multiply_factor


        if self.plot:

            rho_rtp_old_anl = self.construct_rho_rtp(self.R_j_r_phased, jnp.sqrt(wavefunction_params.aj_2[parent_j]) * jnp.exp(1j * rand_phase_per_mode), self.parent_j, Y_lm)

            rho_rtp_new_anl = self.construct_rho_rtp(self.R_j_r_phased, aj, self.parent_j, Y_lm)

            plotting_theta = int(len(theta) / 2)

            plotting_phi = 0

            rho_r_new_anl = rho_rtp_new_anl[:, plotting_theta, plotting_phi]

            rho_r_old_anl = rho_rtp_old_anl[:, plotting_theta, plotting_phi]

            plt.plot(self.r * self.u.to_Kpc, rho_r_old_anl * self.u.to_Msun / (self.u.to_Kpc)**3, label='Old a_nl', alpha = 0.7)
            plt.plot(self.r * self.u.to_Kpc, rho_r_new_anl * self.u.to_Msun / (self.u.to_Kpc)**3, label='New a_nl', alpha = 0.7)
            plt.legend()
            plt.xlabel('r (kpc)')
            plt.ylabel(r'$\rho$ [$M_\odot / kpc^3$]')
            plt.title(r'Density profile with new/old a_nl at $\theta = \pi/2, \phi=0$')
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


        self.particles = []
        for i in range(self.no_of_particles):

            r_orbit = jnp.abs(jax.random.normal(jax.random.PRNGKey(i), shape=(), dtype=jnp.float64) * 0.1 * r_orbit_mean + r_orbit_mean)

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
            rho_rtp = self.construct_rho_rtp(R_j_r_phased, aj, self.parent_j, Y_lm)
            M_enc_at_r = SSF.Enclosed_mass_3d(self.r, self.theta, self.phi, rho_rtp, float(r_orbit))
            v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / r_orbit)

            init_pos = r_i
            init_vel = v_circ_mag * v_i_unit

            print(f"Particle {i}: v_circ = {jnp.linalg.norm(init_vel) * self.u.to_kms:.3f} km/s")

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

        return aj, Y_lm


    def construct_rho_rtp(self, R_j_r_phased, aj, parent_j, Y_lm):

        R_modes = R_j_r_phased[:, parent_j]  # (Nr, Nmodes)
        aj_modes = aj  # (Nmodes,) — already per-mode with independent phases

        full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm)

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2
        rho_rtp = self.total_mass * psi_abs2

        return rho_rtp


    def construct_rho_lms(self, aj, parent_j, R_j_r_phased):

        # Use precomputed R_j_r_phased (set once per macro timestep in run_simulation)
        # to avoid recomputing exp(-i E t / hbar) on every call.
        R_modes = R_j_r_phased[:, parent_j]  # (Nr, Nmodes)
        aj_modes = aj  # (Nmodes,) — already per-mode with independent phases


        rho_lm_gaunt = gf.compute_rho_lm_gaunt(
        aj_modes, R_modes, self.lm_l, self.lm_m, self.total_mass,
        L_max_out=self.L_max_out,
        gaunt_table=self.gaunt_table,
        scatter_matrix=self.scatter_matrix,
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



    def _construct_acc_radial(self, r_pos_sph, R_j_at_particle_phased):
        """
        Pure-JAX portion of the acceleration computation.  R_j has already been
        evaluated (via eval_library) and phased before this is called, so this
        method is safe to vmap over a batch of particle positions.

        """
        particle_r = r_pos_sph[0]

        if self.SphHT == False:
            # construct_rho_lms expects shape (Nr, Nj); wrap the single-radius row
            R_j_row = R_j_at_particle_phased[None, :]                     # (1, Nj)
            rho_lm_at_particle = self.construct_rho_lms(self.aj, self.parent_j, R_j_row)[0]


        else:
            
            psi_at_r  = jnp.einsum('j,jtp->tp', R_j_at_particle_phased, self.Q_j_tp)
            rho_at_r  = self.total_mass * jnp.abs(psi_at_r) ** 2
            rho_lm_at_particle = s2fft.forward(rho_at_r, self.L_max_out, sampling='mw', method='jax')


        insert_idx = jnp.searchsorted(self.r, particle_r)

        r_updated = jnp.where(
            self.all_idx < insert_idx,
            self.r_below,
            jnp.where(self.all_idx == insert_idx, particle_r, self.r_above)
        )

        rho_lm_updated = jnp.where(
            self.all_idx[:, None, None] < insert_idx,
            self.rho_lms_below,
            jnp.where(
                self.all_idx[:, None, None] == insert_idx,
                rho_lm_at_particle[None, :, :],
                self.rho_lms_above
            )
        )

        mask_int = jnp.arange(self.Nr) < insert_idx
        mask_ext = jnp.arange(self.Nr) < (self.Nr - insert_idx)

        dphi_lm_dr_at_r, phi_lm_at_r = _compute_all_phi(
            rho_lm_updated, r_updated, self.output_lm_pairs,
            mask_int, mask_ext, int(self.L_max_out), self.G, particle_r
        )

        return dphi_lm_dr_at_r, phi_lm_at_r  # (Nmodes,), (Nmodes,)

    def _compute_radial_batch(self, positions_sph, current_phase, radial_eigenmode_params):
        """JIT-compilable: radial basis evaluation + vmap over _construct_acc_radial.

        current_phase and radial_eigenmode_params are passed explicitly so JAX
        traces them as dynamic values (they change between macro timesteps).
        eval_library is accessed via self._eval_library, captured as a static
        closure constant since it never changes after setup.
        """
        particle_rs = positions_sph[:, 0]                                      # (N_particles,)
        R_j_at_particles = self._eval_library(particle_rs, radial_eigenmode_params)
        # R_j_at_particles : (N_particles, Nj)

        R_j_phased_all = R_j_at_particles * current_phase[None, :]             # (N_particles, Nj)

        dphi_dr_all, phi_lm_all = jax.vmap(
            lambda pos, R_j_phased: self._construct_acc_radial(pos, R_j_phased)
        )(positions_sph, R_j_phased_all)
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

        # JIT-compiled radial part 
        dphi_lm_dr_at_r, phi_lm_at_r = self._compute_radial_batch_jit(
            positions_sph,
            self.current_phase,
            eigenstate_lib.radial_eigenmode_params,
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

        #print(f"Precomputation of (l,m) pairs and Y_lm grid completed in {end - start:.2f} seconds")

        L_max_out = 2 * self.L - 1  # captures all density harmonics up to l1+l2 <= 2*(L-1)

        self.L_max_out = L_max_out



        if self.SphHT == False:

            # Precompute Gaunt table ONCE — reuse this across all time steps
            gaunt_table = gf.precompute_gaunt_table(self.lm_l, self.lm_m, L_max_out)
            self.gaunt_table = gaunt_table

            _, _, _, _, unique_lm = gaunt_table
            scatter_matrix = gf.make_scatter_matrix(self.lm_l_per_mode, self.lm_m_per_mode, unique_lm)
            self.scatter_matrix = scatter_matrix

        else:

            Nj = len(self.eigen_energies)          # Nj is number of distinct (n,l) modes
            # aj is now per-mode (Nmodes,) — multiply into Y_lm before summing over m
            aj_Y_lm = self.aj[:, None, None] * Y_lm                                  # (Nmodes, n_theta, n_phi)
            Q_j_tp = jax.ops.segment_sum(aj_Y_lm, self.parent_j, num_segments=Nj)   # (Nj, n_theta, n_phi)
            self.Q_j_tp = Q_j_tp



        # Pre-convert lm_pairs to numpy once so scipy sph_harm_y receives a
        # plain numpy array and avoids a GPU to CPU device transfer every sub-step.
        out_lm = [(L, M) for L in range(L_max_out) for M in range(-L, L+1)]
        output_lm_pairs = jnp.array(out_lm)

        self.output_lm_pairs = output_lm_pairs
        self.lm_pairs_np = np.array(output_lm_pairs)



        # Set up animator
        if self.animate:
            self.animator = SimulationAnimator(self.L_max_out, self.r, self.u)
            print(f"Animation enabled: equatorial (z=0) density slice")



        if self.SphHT == True:

            rho_rtp = self.construct_rho_rtp(self.R_j_r_phased, self.aj, self.parent_j, Y_lm)  # (Nr, n_theta, n_phi)
            rho_lm = self.forward_s2fft(rho_rtp)  # (Nr, L, 2*L-1)
            self.rho_lms = rho_lm


        else:
            # 1. Construct total psi and rho on background grid
            rho_lms = self.construct_rho_lms(self.aj, self.parent_j, self.R_j_r_phased)
            self.rho_lms = rho_lms


        rho_lm_below = jnp.concatenate([self.rho_lms, self.rho_lms[-1:]], axis=0)    # (Nr+1, L, 2L-1)
        rho_lm_above = jnp.concatenate([self.rho_lms[:1], self.rho_lms], axis=0)     # (Nr+1, L, 2L-1)
        self.rho_lms_below = rho_lm_below
        self.rho_lms_above = rho_lm_above


        # Compute initial potential energy for each particle
                
        r_pos_sphs = jnp.array([particle.r_pos_sph for particle in self.particles])  # (N_particles, 3)

        a_r, _, _, phi_lm_at_r, Y_lm_all = self.construct_acc_batch(
            r_pos_sphs,
            self.autodiff_data['eval_library'],
            self.autodiff_data['eigenstate_lib'],
            poten = True
        )

        v_circ_true = jnp.sqrt(jnp.abs(a_r) * r_pos_sphs[:, 0])

        phi_at_parts = jnp.sum(phi_lm_at_r * Y_lm_all.T, axis=1)  # (N_particles,)

        for i, particle in enumerate(self.particles):

            v_old = particle.v
            v_dir = v_old / jnp.linalg.norm(v_old)
            v_new = v_dir * v_circ_true[i]
            p = self.ps_step[i]
            p.vx, p.vy, p.vz = float(v_new[0]), float(v_new[1]), float(v_new[2])
            particle.potential_energy.append(phi_at_parts[i].real)

            particle.Change_to_new_vel(v_new)

        

        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")

            # Capture animation frame (density at this instant + all particle positions)
            if self.animate and self.time_step % self.animate_every == 0:
                def inverse_sht_single_r(rho_lm_r):
                    return s2fft.inverse(rho_lm_r, self.L_max_out, sampling='mw', method='jax')
                rho_rtp = jax.vmap(inverse_sht_single_r)(self.rho_lms)  # (Nr, L, 2*L-1)
                all_positions = [p.r_pos for p in self.particles]
                self.animator.store_frame(rho_rtp, all_positions)


            phase = jnp.exp(-1j * self.eigen_energies * self.time_step * self.dt / 1)
            R_j_r_phased = self.R_j_r_fixed * phase[None, :]
            self.current_phase = phase  # shape (Nj,)
            self.R_j_r_phased = R_j_r_phased

            if self.SphHT == True:

                rho_rtp = self.construct_rho_rtp(self.R_j_r_phased, self.aj, self.parent_j, Y_lm)  # (Nr, n_theta, n_phi)
                rho_lm = self.forward_s2fft(rho_rtp)  # (Nr, L, 2*L-1)
                self.rho_lms = rho_lm


            else:
                # 1. Construct total psi and rho on background grid
                rho_lms = self.construct_rho_lms(self.aj, self.parent_j, self.R_j_r_phased)
                self.rho_lms = rho_lms


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




