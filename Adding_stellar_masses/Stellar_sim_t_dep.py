'''CODE FOR RUNNING TIME DEPENDENT STELLAR SIMULATIONS WITH JAXSP'''


from time import time
import Stellar_sim_funcs as SSF

import jaxsp as jsp

import jax
print(jax.devices())
print(jax.default_backend())
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

from jaxsp.constants import GN, hbar

import rebound

import s2fft

from scipy.special import sph_harm

from math import factorial

from collections import defaultdict


'''For Animations and plotting'''
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation
from matplotlib import cm
from matplotlib.colors import Normalize



def compute_plm_table(L, x):
    """
    Compute associated Legendre polynomials P_lm(x)
    All operations are pure JAX, so jax.grad can differentiate
    """
    P = jnp.zeros((L, L))
    P = P.at[0, 0].set(1.0)

    sin_theta = jnp.sqrt(jnp.maximum(1.0 - x**2, 0.0))

    # Diagonal: P_m^m = -(2m-1) * sin(theta) * P_{m-1}^{m-1}
    for m in range(1, L):
        P = P.at[m, m].set(-(2*m - 1) * sin_theta * P[m-1, m-1])

    # First superdiagonal: P_{m+1}^m = (2m+1) * x * P_m^m
    for m in range(0, L - 1):
        P = P.at[m+1, m].set((2*m + 1) * x * P[m, m])

    # Fill remaining entries via standard recurrence
    # (l-m) P_l^m = x(2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m
    for m in range(0, L):
        for l_val in range(m + 2, L):
            P = P.at[l_val, m].set(
                ((2*l_val - 1) * x * P[l_val-1, m] - (l_val + m - 1) * P[l_val-2, m]) / (l_val - m)
            )
    return P



def build_grad_Phi(L, lm_pairs, norms):
    """
    Function that computes dPhi/dr, dPhi/dtheta, dPhi/dphi
    at a single point using jax.grad

    The potential is evaluated as:
        Phi(r, theta, phi) = sum_{l,m} phi_lm(r) * Y_l^m(theta, phi)
    where phi_lm(r) is obtained by Lagrange quadratic interpolation through 3 radial points,
    and Y_l^m is computed via the associated Legendre recurrence + exp(i*m*phi).
    """
    lm_l = lm_pairs[:, 0].astype(int)
    lm_m = lm_pairs[:, 1].astype(int)
    abs_m_arr = jnp.abs(lm_m)
    lm_m_float = lm_m.astype(float)

    # Sign correction for negative m: Y_l^{-|m|} = (-1)^|m| * conj(Y_l^{|m|})
    # P_l^{|m|} from compute_plm_table includes the CS phase (-1)^|m|, but for
    # negative m the correct Y_lm requires an extra (-1)^|m| factor to undo it.
    neg_m_sign = jnp.where(lm_m < 0, (-1.0)**abs_m_arr, 1.0)

    def Phi_at_point(r, theta, phi, phi_lm_real, phi_lm_imag, r_vals):
        """
        Evaluate the gravitational potential at a single point (r, theta, phi).
        """
        # --- Radial: Lagrange quadratic interpolation ---
        r0, r1, r2 = r_vals[0], r_vals[1], r_vals[2]
        w0 = (r - r1) * (r - r2) / ((r0 - r1) * (r0 - r2))
        w1 = (r - r0) * (r - r2) / ((r1 - r0) * (r1 - r2))
        w2 = (r - r0) * (r - r1) / ((r2 - r0) * (r2 - r1))

        coeff_real = phi_lm_real[:, 0]*w0 + phi_lm_real[:, 1]*w1 + phi_lm_real[:, 2]*w2
        coeff_imag = phi_lm_imag[:, 0]*w0 + phi_lm_imag[:, 1]*w1 + phi_lm_imag[:, 2]*w2

        # --- Angular: evaluate Y_lm(theta, phi) ---
        # Y_l^m = N_{l,|m|} * P_l^{|m|}(cos theta) * exp(i*m*phi)
        # For m < 0: Y_l^{-|m|} = (-1)^|m| * conj(Y_l^{|m|})
        # Split into real/imag to keep everything real for jax.grad
        x = jnp.cos(theta)
        P = compute_plm_table(L, x)
        P_vals = P[lm_l, abs_m_arr]

        Y_real = neg_m_sign * norms * P_vals * jnp.cos(lm_m_float * phi)
        Y_imag = neg_m_sign * norms * P_vals * jnp.sin(lm_m_float * phi)

        # Phi = Re[sum(coeff * Y)] = sum(coeff_real * Y_real - coeff_imag * Y_imag)
        return jnp.dot(coeff_real, Y_real) - jnp.dot(coeff_imag, Y_imag)


    # Differentiate w.r.t. (r, theta, phi) only; SH coefficients and r_vals are parameters
    grad_Phi = jax.jit(jax.grad(Phi_at_point, argnums=(0, 1, 2)))

    return grad_Phi

def precompute_lm_pairs_Ylm_norms(l):

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

    """
    Precompute normalization constants for Y_l^m.
    N_{l,|m|} = sqrt((2l+1)/(4pi) * (l-|m|)! / (l+|m|)!)
    """
    norms = []
    for l_val, m_val in lm_pairs:
        l_val, m_val = int(l_val), int(m_val)
        abs_m = abs(m_val)
        norms.append(
            np.sqrt((2*l_val + 1) / (4 * np.pi) * factorial(l_val - abs_m) / factorial(l_val + abs_m))
        )

    '''Precompute Y_lm's for wavefunction reconstruction'''

    # McEwen-Wiaux-style equiangular grid

    L = max(l)+1

    n_theta = L
    n_phi = 2 * L - 1

    # Generate theta values
    i = jnp.arange(n_theta)
    theta = (jnp.pi * (2 * i + 1)) / (2 * L - 1)

    # Generate phi values
    j = jnp.arange(n_phi)
    phi = (2 * jnp.pi * j) / (2 * L - 1)


    Theta, Phi = jnp.meshgrid(theta, phi, indexing="ij")  # both (n_theta, n_phi)

    Y_list = []
    for ell, m in zip(lm_l, lm_m):
        Y_lm_mode = sph_harm(m, ell, Phi, Theta)  # (n_theta, n_phi), complex
        Y_list.append(Y_lm_mode)

    Y_lm = jnp.stack(Y_list, axis=0)  # (Nmodes, n_theta, n_phi), complex


    return jnp.array(parent_j), Y_lm, lm_pairs, jnp.array(norms)


class SimulationAnimator:
    """
    3D animation of axion dark matter density granules and stellar orbit.
    """

    def __init__(self, L, r_grid, r_pos, u):

        self.u = u
        self.L = L

        # MW-sampling angular grid (same as used for the SHT)
        n_theta = L
        n_phi = 2 * L - 1
        i_arr = np.arange(n_theta)
        self.theta = (np.pi * (2 * i_arr + 1)) / (2 * L - 1)
        j_arr = np.arange(n_phi)
        self.phi = (2 * np.pi * j_arr) / (2 * L - 1)
        self.phi = np.append(self.phi, 2 * np.pi)  # Add phi=2pi point for smooth plotting

        # Closest radial bin to the requested shell radius
        self.r_shell_idx = int(np.argmin(np.abs(np.array(r_grid) - r_pos)))
        self.r_shell = float(r_grid[self.r_shell_idx])

        # Cartesian mesh for the spherical surface (in kpc for display)
        Theta, Phi = np.meshgrid(self.theta, self.phi, indexing='ij')
        R_kpc = self.r_shell * float(u.to_Kpc)
        self.X = R_kpc * np.sin(Theta) * np.cos(Phi)
        self.Y = R_kpc * np.sin(Theta) * np.sin(Phi)
        self.Z = R_kpc * np.cos(Theta)
        self.R_kpc = R_kpc

        # Frame storage
        self.density_frames = []
        self.particle_positions = []

    def store_frame(self, rho_rtp, particle_xyz):
        """Capture one snapshot of the density shell and particle position."""
        rho_shell = np.array(np.real(rho_rtp[self.r_shell_idx, :, :]))
        self.density_frames.append(rho_shell)
        self.particle_positions.append([
            float(particle_xyz[0]) * float(self.u.to_Kpc),
            float(particle_xyz[1]) * float(self.u.to_Kpc),
            float(particle_xyz[2]) * float(self.u.to_Kpc),
        ])

    def create_animation(self, interval=200, save_path=None, orbit_radius_kpc=None):
        """
        Build a matplotlib FuncAnimation from the stored frames.
        """

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        cmap = cm.inferno

        # Per-frame normalisation so angular (theta, phi) structure is visible
        def frame_colors(frame_idx):
            rho = self.density_frames[frame_idx]
            frame_norm = Normalize(vmin=rho.min(), vmax=rho.max())
            return cmap(frame_norm(rho))

        # Initial density surface
        colors0 = frame_colors(0)
        surf = [ax.plot_surface(self.X, self.Y, self.Z,
                                facecolors=colors0, shade=False, alpha=0.6)]

        # Particle marker
        p0 = self.particle_positions[0]
        particle_dot = ax.scatter([p0[0]], [p0[1]], [p0[2]],
                                  color='green', s=80, zorder=5,
                                  edgecolors='white', linewidths=0.5,
                                  depthshade=False)

        # Orbit trail
        trail_line, = ax.plot([p0[0]], [p0[1]], [p0[2]],
                              color='green', alpha=0.4, linewidth=1)

        # Axis limits
        if orbit_radius_kpc is not None:
            lim = orbit_radius_kpc * 1.8
        else:
            all_pos = np.array(self.particle_positions)
            lim = np.max(np.abs(all_pos)) * 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel(r'$x$ [kpc]')
        ax.set_ylabel(r'$y$ [kpc]')
        ax.set_zlabel(r'$z$ [kpc]')
        title = ax.set_title('Time step: 0')

        # Colorbar (uses first frame range as reference)
        rho0 = self.density_frames[0]
        mappable = cm.ScalarMappable(norm=Normalize(vmin=rho0.min(), vmax=rho0.max()), cmap=cmap)
        fig.colorbar(mappable, ax=ax, shrink=0.5, label=r'$\rho$')

        def update(frame):
            # Remove old surface and redraw with per-frame normalised colours
            surf[0].remove()
            colors = frame_colors(frame)
            surf[0] = ax.plot_surface(self.X, self.Y, self.Z,
                                      facecolors=colors, shade=False, alpha=0.6)

            # Update particle position
            pos = self.particle_positions[frame]
            particle_dot._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

            # Update orbit trail up to current frame
            trail = np.array(self.particle_positions[:frame + 1])
            trail_line.set_data(trail[:, 0], trail[:, 1])
            trail_line.set_3d_properties(trail[:, 2])

            title.set_text(f'Time step: {frame + 1}')
            return surf[0], particle_dot, trail_line

        import matplotlib
        matplotlib.rcParams['animation.embed_limit'] = 2**128

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


class StellarSimTDep:

    def __init__(self, m22, r_half, no_of_particles, no_time_steps, total_evolve_time, r_min, r_max_enclosing_frac, no_radius_bins,
                 animate=False):

        # Instance variables - reset for each new simulation
        self.velocities = []
        self.stellar_v_disp = []
        self.average_r = []
        self.r_values = []
        self.positions_xyz = []
        self.time_step = 0

        self.m22 = m22
        self.u = jsp.set_schroedinger_units(self.m22)

        self.no_of_particles = no_of_particles

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
        self.animator = None


    def first_time_step(self):

        '''Complete the first timestep using the static potential provided by the Jaxsp tutorial code.
        This is the wavefunction reconstructed potential.

        Set up the Hanno Reins rebound simulation for later time stepping and orbit integration
        '''

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

        rmax = jsp.enclosing_radius(0.99, density_params)
        eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin, rmax, a, b, N)

        n = eigenstate_lib.radial_eigenmode_params.n
        l = eigenstate_lib.radial_eigenmode_params.l
        eigen_energies = eigenstate_lib.radial_eigenmode_params.E

        print('l max from jaxsp:', max(l))
        L = int(max(l) + 1)
        self.L = L

        rmin = 20 * self.u.from_pc
        self.rmin = rmin
        self.rmax = rmax

        r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)
        self.r = r

        R_j_r = eval_library(r, eigenstate_lib.radial_eigenmode_params)

        rho_psi = jax.vmap(jsp.rho_psi, in_axes=(0,None,None))

        tol = 1e-7
        wavefunction_params = jsp.init_wavefunction_params(eigenstate_lib, density_params, rmin, rmax, tol)

        total_mass = wavefunction_params.total_mass
        aj_2 = wavefunction_params.aj_2        # shape (Nj,)
        rand_phase = jax.random.uniform(jax.random.PRNGKey(0), shape=aj_2.shape, minval=0.0, maxval=2 * jnp.pi,)
        aj = jnp.sqrt(aj_2) * jnp.exp(1j * rand_phase)  # shape (Nj,)

        rho_psi_vals = rho_psi(r, wavefunction_params, eigenstate_lib)

        Phi_psi = SSF.Obtain_pot(self.rmin, self.rmax, rho_psi_vals, r)

        r_orbit = self.r_half * self.u.from_Kpc

        init_pos = jnp.array([r_orbit, 0, 0]) #Starting as position (r_vir, 0, 0) = (x, y, z)

        acc_mag = SSF.Find_acc_mag_from_Phi(r, Phi_psi, r_orbit)

        acc_vec = acc_mag * (-init_pos / jnp.linalg.norm(init_pos))
        init_vel = jnp.sqrt(acc_mag * r_orbit) * jnp.array([0, 1, 0])  # Circular orbit velocity (x, y, z)

        init_vel_sph = SSF.Cartesian_to_sph_vel(init_pos[0], init_pos[1], init_pos[2], init_vel[0], init_vel[1], init_vel[2])

        self.velocities.append(init_vel_sph)
        self.stellar_v_disp.append(0)
        self.average_r.append(r_orbit)
        self.r_values.append(r_orbit)
        self.positions_xyz.append([float(init_pos[0]), float(init_pos[1]), float(init_pos[2])])

        sim = rebound.Simulation()
        sim.integrator = "ias15"
        sim.add(m=0.0, x=init_pos[0], y=init_pos[1], z=init_pos[2],
                vx=init_vel[0], vy=init_vel[1], vz=init_vel[2])

        ps = sim.particles

        def additional_forces_init(_reb_sim):
            p = ps[0]
            p.ax += acc_vec[0]
            p.ay += acc_vec[1]
            p.az += acc_vec[2]

        sim.additional_forces = additional_forces_init
        sim.integrate(self.dt)

        p = sim.particles[0]
        r_pos = jnp.array([p.x, p.y, p.z])
        v = jnp.array([p.vx, p.vy, p.vz])
        self.r_pos = r_pos
        self.v = v

        r_pos_sph = SSF.Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
        v_sph = SSF.Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

        self.update_summary_stats(v_sph, r_pos_sph)

        self.r_pos_sph = r_pos_sph
        self.v_sph = v_sph

        #Create persistent rebound simulation for time stepping
        sim_step = rebound.Simulation()
        sim_step.integrator = "ias15"
        sim_step.add(m=0.0, x=r_pos[0], y=r_pos[1], z=r_pos[2], vx=v[0], vy=v[1], vz=v[2])
        ps_step = sim_step.particles

        autodiff_data = {'phi_lm_real': None, 'phi_lm_imag': None, 'r_vals': None}
        self.autodiff_data = autodiff_data


        def additional_forces_step(_reb_sim):
            """
            Rebound callback: compute acceleration at the sub-step position using jax.grad.
            IAS15 calls this multiple times per timestep at different positions.
            """
            p = ps_step[0]
            pos_sph = SSF.Cartesian_to_sph(p.x, p.y, p.z)

            # Evaluate exact gradient via autodiff
            dPhi_dr, dPhi_dtheta, dPhi_dphi = self._grad_Phi(
                pos_sph[0], pos_sph[1], pos_sph[2],
                autodiff_data['phi_lm_real'],
                autodiff_data['phi_lm_imag'],
                autodiff_data['r_vals']
            )

            # Convert gradient to spherical acceleration components: a = -grad(Phi)
            a_r = -float(dPhi_dr)
            a_theta = -float(dPhi_dtheta) / float(pos_sph[0])
            a_phi = -float(dPhi_dphi) / (float(pos_sph[0]) * float(jnp.sin(pos_sph[1])))

            ax, ay, az = SSF.acceleration_spherical_to_cartesian(a_r, a_theta, a_phi, pos_sph[1], pos_sph[2])
            p.ax += float(ax)
            p.ay += float(ay)
            p.az += float(az)

        sim_step.additional_forces = additional_forces_step

        self.sim_step = sim_step
        self.ps_step = ps_step

        return R_j_r, l, total_mass, eigen_energies, aj


    def update_summary_stats(self, vel, r_vec):

        '''Plotted stats from simulation updated after each timestep
        '''

        self.velocities.append(vel)

        self.velocities = jnp.array(self.velocities)

        new_vel_disp = (jnp.std(self.velocities[:, 0])**2 + jnp.std(self.velocities[:, 1])**2 + jnp.std(self.velocities[:, 2])**2)**0.5
        self.stellar_v_disp.append(new_vel_disp)

        self.velocities = self.velocities.tolist()

        self.r_values.append(r_vec[0])

        new_avg_r = jnp.mean(jnp.array(self.r_values))
        self.average_r.append(new_avg_r)

        self.positions_xyz.append([float(self.r_pos[0]), float(self.r_pos[1]), float(self.r_pos[2])])

        self.time_step += 1



    def construct_total_psi(self, R_j_r, eigen_energies, aj, total_mass, Y_lm, parent_j):

        '''Wavefunction constructed density profile in 3d as a function of r, theta and phi
        '''

        # Update wavefunction potential
        radial_eigen_function_time_stepped = R_j_r * jnp.exp(-1j * self.time_step * self.dt * eigen_energies / hbar.value)


        R_modes = radial_eigen_function_time_stepped[:, parent_j]
        aj_modes = aj[parent_j]

        full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm)

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2
        rho_rtp = total_mass * psi_abs2

        return rho_rtp

    def forward_s2fft(self, rho_rtp):

        '''Forward s2fft of 3d density to get rho_lm(r)
        '''

        #Parallel forward SHT over all radii
        def forward_sht_single_r(rho_at_r):
            return s2fft.forward(rho_at_r, self.L, sampling='mw', method='jax')

        flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)

        return flm_r


    def integrate_rho_lm(self, r, flm_r, lm_pairs):

        '''Integration of the rho_lm(r)'s to get phi_lm(r) at the particle position r_orbit, using the formula from Binney & Tremaine
        Evaluate integration at r index closest to particles r position and then +- an index aswell
        '''

        r_orbit = self.r_pos_sph[0]

        r_finder = jnp.abs(r - r_orbit)
        r_index = jnp.argmin(r_finder)
        r_vals = r[r_index - 1 : r_index + 2]  # three r values around r_orbit where we are evaluating integrals at

        r_vals = jnp.array(r_vals)

        def compute_phi_for_lm(lm_pair):
            """Compute phi_lm for a single (l, m) pair at r = r_orbit +- small delta"""

            l_val = lm_pair[0]
            m_val = lm_pair[1]

            prefix = -4.0 * jnp.pi * self.G / (2 * l_val + 1)
            m_ind = m_val + self.L - 1

            f_at_lm = flm_r[:, l_val, m_ind]  # (Nr). This is rho_lm(r)

            # Integrands - compute over ALL r
            integrand_ext = r**(1 - l_val) * f_at_lm
            integrand_int = r**(l_val + 2) * f_at_lm

            dr = jnp.diff(r)

            # Internal integral (from 0 to r) - FULL computation
            avg_int = 0.5 * (integrand_int[1:] + integrand_int[:-1])

            avg_int = avg_int[:r_index + 2]
            dr = dr[:r_index + 2]

            integral_int = jnp.concatenate([
                jnp.array([0.0 + 0.0j]),
                jnp.cumsum(avg_int * dr)
            ])

            integral_int = integral_int[r_index - 1:r_index + 2]


            # External integral (from r to r_max) - FULL computation
            integrand_ext_rev = integrand_ext[::-1]
            r_rev = r[::-1]
            dr_rev = jnp.diff(r_rev)
            avg_ext = 0.5 * (integrand_ext_rev[1:] + integrand_ext_rev[:-1])

            avg_ext = avg_ext[:len(r) - r_index + 1]
            dr_rev = dr_rev[:len(r) - r_index + 1]

            integral_ext_rev = jnp.concatenate([
                jnp.array([0.0 + 0.0j]),
                jnp.cumsum(avg_ext * dr_rev)
            ])
            integral_ext = -integral_ext_rev[::-1][:3]


            phi_spec_r = prefix * (r_vals**(-(l_val + 1)) * integral_int + r_vals**l_val * integral_ext)

            return phi_spec_r

        #Parallel computation over all (l, m) pairs

        all_phi_lm = jax.vmap(compute_phi_for_lm)(lm_pairs)  # shape (N_pairs, 3)


        return all_phi_lm, r_vals


    def time_step_particle(self):

        # Update particle state and integrate
        p = self.ps_step[0]
        p.x, p.y, p.z = self.r_pos[0], self.r_pos[1], self.r_pos[2]
        p.vx, p.vy, p.vz = self.v[0], self.v[1], self.v[2]

        target_time = self.sim_step.t + self.dt
        self.sim_step.integrate(target_time)

        # Read back state
        p = self.sim_step.particles[0]
        self.r_pos = jnp.array([p.x, p.y, p.z])
        self.v = jnp.array([p.vx, p.vy, p.vz])

        r_pos_sph = SSF.Cartesian_to_sph(self.r_pos[0], self.r_pos[1], self.r_pos[2])
        v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2], self.v[0], self.v[1], self.v[2])

        self.update_summary_stats(v_sph, r_pos_sph)
        self.r_pos_sph = r_pos_sph
        self.v_sph = v_sph


    def run_simulation(self):

        start = time()
        R_j_r, l, total_mass, eigen_energies, aj = self.first_time_step()
        end = time()
        #print(f"First time step completed in {end - start:.2f} seconds")

        start = time()
        parent_j, Y_lm, lm_pairs, ylm_norms = precompute_lm_pairs_Ylm_norms(l)
        end = time()
        #print(f"Precomputation of (l,m) pairs and Y_lm grid completed in {end - start:.2f} seconds")

        start = time()
        self._grad_Phi = build_grad_Phi(self.L, lm_pairs, ylm_norms)
        end = time()
        #print(f"Building grad_Phi completed in {end - start:.2f} seconds")

        # Set up animator
        if self.animate:
            r_shell = self.r_values[0]  # Start with the initial radius of the particle
            self.animator = SimulationAnimator(self.L, self.r, r_shell, self.u)
            print(f"Animation enabled: density shell at r = {self.animator.r_shell * self.u.to_Kpc:.4f} kpc")

        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")

            # 1. Construct total psi and rho
            start = time()
            rho_rtp = self.construct_total_psi(R_j_r, eigen_energies, aj, total_mass, Y_lm, parent_j)
            end = time()
            #print(f"Constructing total psi and rho completed in {end - start:.2f} seconds")

            # 1b. Capture animation frame (density at this instant + particle position)
            if self.animate:
                self.animator.store_frame(rho_rtp, self.r_pos)

            # 2. Forward S2FFT to get rho_lm(r)
            start = time()
            flm_r = self.forward_s2fft(rho_rtp)
            end = time()
            #print(f"Forward S2FFT completed in {end - start:.2f} seconds")

            # 3. Integrate to get phi_lm at r values around star position
            start = time()
            all_phi_lm, r_vals = self.integrate_rho_lm(self.r, flm_r, lm_pairs)
            end = time()
            #print(f"Integrating rho_lm to get phi_lm completed in {end - start:.2f} seconds")

            # 4. Update autodiff data with the new SH coefficients
            self.autodiff_data['phi_lm_real'] = all_phi_lm.real
            self.autodiff_data['phi_lm_imag'] = all_phi_lm.imag
            self.autodiff_data['r_vals'] = r_vals

            # 5. Time step particle — IAS15 calls the callback which uses jax.grad
            start = time()
            self.time_step_particle()
            end = time()
            #print(f"Time stepping particle completed in {end - start:.2f} seconds")
