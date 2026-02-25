'''CODE FOR RUNNING TIME DEPENDENT STELLAR SIMULATIONS WITH JAXSP
Method for finding acceleration vector:
Add particle position to r_bins
Re evaluate R_jk
Get rho(r, theta, phi)
S2FFT to rho_lm's
Integrate to get r derivative of phi_lm's at particle position
Put together with analytical solution for derivative of Y_lm's to get dPhi/dr, dPhi/dtheta, dPhi/dphi at particle position
'''


from time import time
import Stellar_sim_funcs as SSF

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
        Y_lm_mode = sph_harm_y(ell, m, Theta, Phi)  # (n_theta, n_phi), complex
        Y_list.append(Y_lm_mode)

    Y_lm = jnp.stack(Y_list, axis=0)  # (Nmodes, n_theta, n_phi), complex


    return jnp.array(parent_j), Y_lm, lm_pairs


#--------------------------------------------------------------------------------------------------------------------------
"""
3D animation of axion dark matter density granules and stellar orbit.
"""
class SimulationAnimator:

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

#--------------------------------------------------------------------------------------------------------------------



class StellarSimTDep:

    def __init__(self, m22, r_half, no_of_particles, no_time_steps, total_evolve_time, r_min, r_max_enclosing_frac, no_radius_bins,
                 animate=False):


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

        # Precomputed quantities shared across all IAS15 sub-steps within a macro timestep.

        self.current_phase = None   # exp(-i E_j * t / hbar), shape (Nj,)
        self.R_j_r_phased = None    # R_j_r_fixed * current_phase,  shape (Nr, Nj)
        self.eigen_energies = None  # stored from eigenstate_lib 
        self.lm_pairs_np = None     # numpy copy of lm_pairs – avoids GPU to CPU 


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

        rmax = jsp.enclosing_radius(self.r_max_enclosing_frac, density_params)
        eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin, rmax, a, b, N)

        l = eigenstate_lib.radial_eigenmode_params.l

        print('l max from jaxsp:', max(l))
        L = int(max(l) + 1)
        self.L = L

        rmin = self.r_min * self.u.from_pc
        self.rmin = rmin
        self.rmax = rmax

        r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)
        self.r = r

        rho_psi = jax.vmap(jsp.rho_psi, in_axes=(0,None,None))

        tol = 1e-7
        wavefunction_params = jsp.init_wavefunction_params(eigenstate_lib, density_params, rmin, rmax, tol)

        total_mass = wavefunction_params.total_mass
        self.total_mass = total_mass
        aj_2 = wavefunction_params.aj_2        # shape (Nj,)
        rand_phase = jax.random.uniform(jax.random.PRNGKey(0), shape=aj_2.shape, minval=0.0, maxval=2 * jnp.pi,)
        aj = jnp.sqrt(aj_2) * jnp.exp(1j * rand_phase)  # shape (Nj,)
        eigen_energies = eigenstate_lib.radial_eigenmode_params.E  # shape (Nj,)

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
        # Gravitational force is velocity-independent: IAS15 can skip redundant
        # force evaluations at sub-steps where only the velocity differs.
        sim_step.force_is_velocity_dependent = False
        sim_step.add(m=0.0, x=r_pos[0], y=r_pos[1], z=r_pos[2], vx=v[0], vy=v[1], vz=v[2])
        ps_step = sim_step.particles

        autodiff_data = {'lm_pairs': None, 'eval_library': None, 'eigenstate_lib': None, 'flm_r': None}
        self.autodiff_data = autodiff_data


        def additional_forces_step(_reb_sim):
            """
            IAS15 calls this multiple times per timestep at different positions.
            """
            p = ps_step[0]
            pos_sph = SSF.Cartesian_to_sph(p.x, p.y, p.z)

            a_r, a_theta, a_phi = self.construct_acc(
                pos_sph,
                self.autodiff_data['lm_pairs'],
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib'],
                self.autodiff_data['flm_r']
            )

            ax, ay, az = SSF.acceleration_spherical_to_cartesian(a_r, a_theta, a_phi, pos_sph[1], pos_sph[2])
            p.ax += float(ax)
            p.ay += float(ay)
            p.az += float(az)

        sim_step.additional_forces = additional_forces_step

        self.sim_step = sim_step
        self.ps_step = ps_step

        return eval_library, eigenstate_lib, aj, l, eigen_energies


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



    def construct_total_psi_background(self, aj, Y_lm, parent_j, eigen_energies):

        # Use precomputed R_j_r_phased (set once per macro timestep in run_simulation)
        # to avoid recomputing exp(-i E t / hbar) on every call.
        R_modes = self.R_j_r_phased[:, parent_j]  # (Nr, Nmodes)
        aj_modes = aj[parent_j]

        full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm)

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2
        rho_rtp = self.total_mass * psi_abs2

        return rho_rtp


    def forward_s2fft(self, rho_rtp):

        '''Forward s2fft of 3d density to get rho_lm(r)
        '''

        #Parallel forward SHT over all radii
        def forward_sht_single_r(rho_at_r):
            return s2fft.forward(rho_at_r, self.L, sampling='mw', method='jax')

        flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)

        return flm_r


    def construct_acc(self, r_pos_sph, lm_pairs, eval_library, eigenstate_lib, flm_r):

        '''Construct the acceleration vector at the particle position.
        '''

        particle_r     = r_pos_sph[0]
        particle_theta = r_pos_sph[1]
        particle_phi   = r_pos_sph[2]

        Nr = len(self.r)        # number of background radial bins
        all_idx = jnp.arange(Nr + 1)   # indices 0 .. Nr 



        R_j_at_particle = eval_library(jnp.array([particle_r]), eigenstate_lib.radial_eigenmode_params)[0] * self.current_phase   # New bin phased radial wavefunction

        insert_idx = jnp.searchsorted(self.r, particle_r)  # index where particle_r would be inserted to keep r sorted

        r_below = jnp.concatenate([self.r, self.r[-1:]])    # (Nr+1,)
        r_above = jnp.concatenate([self.r[:1], self.r])     # (Nr+1,)

        r_updated = jnp.where(                  # jnp.where(cond, x, y) returns x if cond is True, else y
            all_idx < insert_idx,
            r_below,
            jnp.where(all_idx == insert_idx, particle_r, r_above)
        )   # (Nr+1,)


        psi_at_r   = jnp.einsum('j,jtp->tp', R_j_at_particle, self.Q_j_tp)  # (n_theta, n_phi) Q_j_tp is a_j * Y_lm
        rho_at_r   = self.total_mass * jnp.abs(psi_at_r) ** 2               # (n_theta, n_phi)


        rho_lm_at_r = s2fft.forward(rho_at_r, self.L, sampling='mw', method='jax')  # (L, 2L-1)

        # Insert the particle's rho_lm into the flm_r array at the correct radial index, shifting other entries as needed.

        flm_below = jnp.concatenate([flm_r, flm_r[-1:]], axis=0)    # (Nr+1, L, 2L-1)
        flm_above = jnp.concatenate([flm_r[:1], flm_r], axis=0)     # (Nr+1, L, 2L-1)

        flm_r_updated = jnp.where(
            all_idx[:, None, None] < insert_idx,
            flm_below,
            jnp.where(
                all_idx[:, None, None] == insert_idx,
                rho_lm_at_r[None, :, :],
                flm_above
            )
        )   # (Nr+1, L, 2L-1)


        r_index = insert_idx   # particle sits at exactly this index in r_updated

        dr     = jnp.diff(r_updated)          
        dr_rev = jnp.diff(r_updated[::-1])    

        # Masks reused by every (l,m) pair in the vmap below.
        mask_int = jnp.arange(Nr) < r_index               # True for indices 0 .. r_index-1
        mask_ext = jnp.arange(Nr) < (Nr - r_index)   # True for indices 0 .. Nr-r_index-1 in the reversed array, i.e. r_index+1 .. Nr in the forward array

        def compute_phi_for_lm(lm_pair):
            l_val = lm_pair[0]
            m_val = lm_pair[1]

            prefix = -4.0 * jnp.pi * self.G / (2 * l_val + 1)
            m_ind  = m_val + self.L - 1

            f_at_lm = flm_r_updated[:, l_val, m_ind]

            integrand_ext = r_updated ** (1 - l_val) * f_at_lm
            integrand_int = r_updated ** (l_val + 2) * f_at_lm

            avg_int = 0.5 * (integrand_int[1:] + integrand_int[:-1])   # (Nr,)

            # Internal integral at r_index: masked sum over first r_index intervals.
            integral_int = jnp.sum(jnp.where(mask_int, avg_int * dr, 0.0 + 0.0j))

            # External integral at r_index: masked sum over intervals from r_max down.
            integrand_ext_rev = integrand_ext[::-1]
            avg_ext = 0.5 * (integrand_ext_rev[1:] + integrand_ext_rev[:-1])   # (Nr,)
            integral_ext = -jnp.sum(jnp.where(mask_ext, avg_ext * dr_rev, 0.0 + 0.0j))

            r_val = particle_r

            dphi_dr = prefix * (l_val * r_val ** (l_val - 1) * integral_ext - (l_val + 1) * r_val ** (-l_val - 2) * integral_int)

            phi_lm = prefix * (r_val ** l_val * integral_ext + r_val ** (-l_val - 1) * integral_int)

            return dphi_dr, phi_lm


        # vmap over all (l,m) pairs to get straight to dphi_lm_dr and phi_lm both evaluated at the r value of the particle
        dphi_dr_at_r, phi_lm_at_r = jax.vmap(compute_phi_for_lm)(lm_pairs)


        # Generate spherical harmonics and their derivatives at theta and phi of particle
        Ylm_particle, dY_arr = sph_harm_y(
            self.lm_pairs_np[:, 0], self.lm_pairs_np[:, 1],
            float(particle_theta), float(particle_phi),
            diff_n=1
        )
        Ylm_particle = jnp.array(Ylm_particle)
        dY_dtheta    = jnp.array(dY_arr[:, 0])

        dY_dphi = 1j * lm_pairs[:, 1] * Ylm_particle

        # Sum over all (l,m) modes to get scalar acceleration components
        a_r     = jnp.sum(-dphi_dr_at_r * Ylm_particle).real
        a_theta = jnp.sum(-phi_lm_at_r * dY_dtheta / particle_r).real
        a_phi   = jnp.sum(-phi_lm_at_r * dY_dphi / (particle_r * jnp.sin(particle_theta))).real

        return a_r, a_theta, a_phi

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
        eval_library, eigenstate_lib, aj, l, eigen_energies = self.first_time_step()
        end = time()
        #print(f"First time step completed in {end - start:.2f} seconds")

        # Store eigen_energies so construct_acc can access them via self without
        # an extra argument through the rebound callback chain.
        self.eigen_energies = eigenstate_lib.radial_eigenmode_params.E

        start = time()
        parent_j, Y_lm, lm_pairs = precompute_lm_pairs_Ylms(l)
        end = time()
        #print(f"Precomputation of (l,m) pairs and Y_lm grid completed in {end - start:.2f} seconds")

        # Pre-convert lm_pairs to numpy once so scipy sph_harm_y receives a
        # plain numpy array and avoids a GPU to CPU device transfer every sub-step.
        self.lm_pairs_np = np.array(lm_pairs)

        self.autodiff_data['lm_pairs'] = lm_pairs
        self.autodiff_data['eval_library'] = eval_library
        self.autodiff_data['eigenstate_lib'] = eigenstate_lib


        # Precompute Q_j_tp = aj[j] * sum Y_lm(theta, phi)
        # Precompute R_j on the fixed r grid: only one new r point (particle position) evaluated per step
        start = time()
        Nj = len(eigenstate_lib.radial_eigenmode_params.E)          # Nj is number of distinct (n,l) modes 
        Y_lm_per_j = jax.ops.segment_sum(Y_lm, parent_j, num_segments=Nj)  # (Nj, n_theta, n_phi)
        Q_j_tp = aj[:, None, None] * Y_lm_per_j                             # (Nj, n_theta, n_phi)
        R_j_r_fixed = eval_library(self.r, eigenstate_lib.radial_eigenmode_params)  # (Nr, Nj)
        self.Q_j_tp = Q_j_tp
        self.R_j_r_fixed = R_j_r_fixed
        end = time()
        print(f"Precomputation of Q_j_tp and R_j_r_fixed completed in {end - start:.2f} seconds")


        # Set up animator
        if self.animate:
            r_shell = self.r_values[0]  # Start with the initial radius of the particle
            self.animator = SimulationAnimator(self.L, self.r, r_shell, self.u)
            print(f"Animation enabled: density shell at r = {self.animator.r_shell * self.u.to_Kpc:.4f} kpc")

        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")

            # Precompute phase and phased R_j once per macro timestep.
            self.current_phase = jnp.exp(-1j * self.time_step * self.dt * self.eigen_energies / hbar.value)
            self.R_j_r_phased = self.R_j_r_fixed * self.current_phase   # (Nr, Nj)

            # 1. Construct total psi and rho on background grid
            start = time()
            rho_rtp = self.construct_total_psi_background(aj, Y_lm, parent_j, eigen_energies)
            end = time()
            print(f"Constructing total psi and rho completed in {end - start:.2f} seconds")

            # 1b. Capture animation frame (density at this instant + particle position)
            if self.animate:
                self.animator.store_frame(rho_rtp, self.r_pos)

            # 2. Forward S2FFT to get rho_lm(r) on background grid
            start = time()
            flm_r = self.forward_s2fft(rho_rtp)
            end = time()
            print(f"Forward S2FFT completed in {end - start:.2f} seconds")


            # 3. Update autodiff data
            self.autodiff_data['flm_r'] = flm_r

            # 4. Time step particle (IAS15 calls construct_acc ~8 times internally)
            start = time()
            self.time_step_particle()
            end = time()
            print(f"Time stepping particle completed in {end - start:.2f} seconds")
