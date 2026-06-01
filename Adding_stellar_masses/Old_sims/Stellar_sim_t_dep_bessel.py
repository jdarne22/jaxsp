'''CODE FOR RUNNING TIME DEPENDENT STELLAR SIMULATIONS WITH JAXSP'''


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

from scipy.special import spherical_jn, jv

from collections import defaultdict

import Boris_SBT_code as BSBT



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


    '''Precompute Y_lm's on McEwen-Wiaux equiangular grid'''

    L = max(l) + 1
    n_theta = L
    n_phi   = 2 * L - 1

    i_arr = jnp.arange(n_theta)
    theta  = (jnp.pi * (2 * i_arr + 1)) / (2 * L - 1)
    j_arr = jnp.arange(n_phi)
    phi   = (2 * jnp.pi * j_arr) / (2 * L - 1)

    Theta, Phi = jnp.meshgrid(theta, phi, indexing="ij")  # both (n_theta, n_phi)

    Y_list = []
    for ell, m in zip(lm_l, lm_m):
        Y_lm_mode = sph_harm_y(ell, m, Theta, Phi)  # (n_theta, n_phi), complex
        Y_list.append(Y_lm_mode)

    Y_lm = jnp.stack(Y_list, axis=0)  # (Nmodes, n_theta, n_phi), complex

    return jnp.array(parent_j), Y_lm, lm_pairs


#---------------------------------------------------------------------------------------------------------------


class SimulationAnimator:
    """
    3D animation of axion dark matter density granules and stellar orbit.
    """

    def __init__(self, L, r_grid, r_pos, u):

        self.u = u
        self.L = L

        # MW-sampling angular grid (same as used for the SHT)
        n_theta = L
        n_phi   = 2 * L - 1
        i_arr   = np.arange(n_theta)
        self.theta = (np.pi * (2 * i_arr + 1)) / (2 * L - 1)
        j_arr   = np.arange(n_phi)
        self.phi   = (2 * np.pi * j_arr) / (2 * L - 1)
        self.phi   = np.append(self.phi, 2 * np.pi)  # close the surface for smooth plotting

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
        rho_shell = np.array(np.real(rho_rtp[self.r_shell_idx, :, :]))  # (n_theta, n_phi)
        rho_shell = np.append(rho_shell, rho_shell[:, 0:1], axis=1)     # close phi for plotting
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


#--------------------------------------------------------------------------------------------------------


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

        rmax = jsp.enclosing_radius(self.r_max_enclosing_frac, density_params)
        eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin, rmax, a, b, N)

        n = eigenstate_lib.radial_eigenmode_params.n
        l = eigenstate_lib.radial_eigenmode_params.l
        eigen_energies = eigenstate_lib.radial_eigenmode_params.E

        print('l max from jaxsp:', max(l))
        L = int(max(l) + 1)
        self.L = L

        rmin = self.r_min * self.u.from_pc
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
        #sim_step = rebound.Simulation()
        #sim_step.integrator = "ias15"
        #sim_step.force_is_velocity_dependent = False

        #sim_step.add(m=0.0, x=r_pos[0], y=r_pos[1], z=r_pos[2], vx=v[0], vy=v[1], vz=v[2])
        #ps_step = sim_step.particles

        #def additional_forces_step(_reb_sim):
        #   """
        #    Rebound callback: compute acceleration at the sub-step position using jax.grad.
        #    IAS15 calls this multiple times per timestep at different positions.
        #    """
        #    p = ps_step[0]
        #    pos_sph = SSF.Cartesian_to_sph(p.x, p.y, p.z)

            # Evaluate exact gradient via autodiff
            # r_pos_sph, phi_nlm
        #    a_r, a_theta, a_phi = self.construct_a(
        #        pos_sph
        #    )

        #    ax, ay, az = SSF.acceleration_spherical_to_cartesian(a_r, a_theta, a_phi, pos_sph[1], pos_sph[2])
        #    p.ax += float(ax)
        #    p.ay += float(ay)
        #    p.az += float(az)

        #sim_step.additional_forces = additional_forces_step

        #sim_step.dt = self.dt / 10           # initial sub-step size


        #Create persistent rebound simulation for time stepping
        #self.ps_step = ps_step

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

        '''Wavefunction constructed density profile as a function of r
        '''

        # Update wavefunction potential
        radial_eigen_function_time_stepped = R_j_r * jnp.exp(-1j * self.time_step * self.dt * eigen_energies / hbar.value)

        R_modes = radial_eigen_function_time_stepped[:, parent_j]
        aj_modes = aj[parent_j]

        full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm)

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2
        rho_rtp = total_mass * psi_abs2  # shape (Nr, n_theta, n_phi)

        return rho_rtp


    def forward_s2fft(self, rho_rtp):

        '''Forward s2fft of 3d density (on HEALPix pixels) to get rho_lm(r).
        rho_rp shape: (Nr, NPIX)
        '''

        #Parallel forward SHT over all radii
        def forward_sht_single_r(rho_at_r):
            return s2fft.forward(rho_at_r, self.L, sampling='mw', method='jax')

        flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)

        return flm_r


    def Bessel_function_expansion_of_rho_lm(self, rho_lm_r, lm_pairs, kln_nodes_by_ell, cln_norms_by_ell):
        # rho_lm_r has shape (Nr, L, 2*L-1) from s2fft (McEwen-Wiaux convention)
        # lm_pairs has shape (N_lm, 2) with columns [l, m]
        lm_l = lm_pairs[:, 0]
        lm_m = lm_pairs[:, 1]
        L = int(lm_l.max()) + 1

        rho_lm_r_j = jnp.asarray(rho_lm_r)   # (Nr, Lmax+1, 2L-1)
        r_j = jnp.asarray(self.r)

        f_l_kln_all = BSBT.forward_dsbt_all(
            rho_lm_r=rho_lm_r_j,
            r=r_j,
            lm_l=lm_l,
            lm_m=lm_m,
            L=L,
            kln_nodes_by_ell=kln_nodes_by_ell,
            chunk_size=16,   # tune this
        )

        # Obtain phi from rho
        Phi_nlm = -4 * jnp.pi * self.G * cln_norms_by_ell[lm_l] * f_l_kln_all / kln_nodes_by_ell[lm_l]**2  # (Nlm, N_k)

        return np.array(Phi_nlm)


    def construct_a(self, r_pos_sph):

        particle_r = r_pos_sph[0]
        particle_theta = r_pos_sph[1]
        particle_phi = r_pos_sph[2]

        r = float(particle_r)
        theta = float(particle_theta)
        phi_angle = float(particle_phi)

        # Guard: clamp r to the valid Bessel domain [rmin, rmax].
        # Outside this range the expansion is non-physical (wrong sign / oscillatory).
        r_clamped = float(np.clip(r, float(self.rmin), float(self.rmax)))
        if r != r_clamped:
            print(f"WARNING: r={r * self.u.to_Kpc:.4f} kpc outside Bessel domain "
                  f"[{float(self.rmin) * self.u.to_Kpc:.4f}, {float(self.rmax) * self.u.to_Kpc:.4f}] kpc — clamping.")

        # Guard: sin_theta → 0 at the poles causes 1/(r*sin_theta) to diverge.
        sin_theta = max(abs(np.sin(theta)), 1e-10)

        # Spherical harmonics in l-sorted order (matches _phi_sorted row order)
        start = time()
        Ylm_s, dY_arr_s = sph_harm_y(
            self._lm_sorted[:, 0], self._lm_sorted[:, 1],
            theta, phi_angle, diff_n=1
        )
        dY_dtheta_s = dY_arr_s[:, 0]                           # (N_lm,)
        dY_dphi_s   = 1j * self._m_sorted * Ylm_s              # (N_lm,)
        end = time()
        #print(f"Spherical harmonics and derivatives computed in {end - start:.2f} seconds")

        '''Bessel functions'''

        start = time()

        kr      = self.kln_np * r_clamped                   
        j_by_l  = spherical_jn(self._ell_bc, kr) 
        dj_by_l = spherical_jn(self._ell_bc, kr, derivative=True) * self.kln_np  # (L, N_k)

        end = time()
        #print(f"Bessel functions computed in {end - start:.2f} seconds")

        '''Contract n-dimension per l-block'''

        # _phi_sorted is (N_lm, N_k) with rows pre-sorted by l (computed once per timestep).
        # For each l-block, (2l+1, N_k) @ (N_k,) → (2l+1,)..
        N_lm = int(self._l_offsets[-1])
        A_lm = np.empty(N_lm, dtype=complex)   # Σ_n phi * j_l
        B_lm = np.empty(N_lm, dtype=complex)   # Σ_n phi * dj_l/dr

        start = time()
        for ell in range(self.L):

            # set start and end of l block
            s, e = int(self._l_offsets[ell]), int(self._l_offsets[ell + 1])
            A_lm[s:e] = self._phi_sorted[s:e] @ j_by_l[ell]    # (2l+1, N_k) @ (N_k,) - this is rho_nlm * j_l(kr) summed over n for each (l,m)
            B_lm[s:e] = self._phi_sorted[s:e] @ dj_by_l[ell]   # (2l+1, N_k) @ (N_k,) - this is rho_nlm * dj_l(kr)/dr summed over n for each (l,m)

        end = time()
        #print(f"Contracting Bessel sums with phi_nlm completed in {end - start:.2f} seconds")

        #print(r * self.u.to_Kpc)

        a_r     = -np.real(np.dot(B_lm, Ylm_s))
        a_theta = -np.real(np.dot(A_lm / r_clamped, dY_dtheta_s))
        a_phi   = -np.real(np.dot(A_lm / (r_clamped * sin_theta), dY_dphi_s))

        return a_r, a_theta, a_phi


    def time_step_particle(self):

        # Recreate the simulation from scratch each outer step so that IAS15
        # starts with clean (zeroed) b-coefficients.
        # Stale b-coefficients from the previous (different) potential cause
        # large errors. 
        sim = rebound.Simulation()
        sim.integrator = "ias15"
        sim.force_is_velocity_dependent = False
        sim.add(m=0.0,
                x=float(self.r_pos[0]), y=float(self.r_pos[1]), z=float(self.r_pos[2]),
                vx=float(self.v[0]),    vy=float(self.v[1]),    vz=float(self.v[2]))
        ps = sim.particles
        sim.dt = self.dt / 10

        def _forces(_reb_sim):
            p = ps[0]
            pos_sph = SSF.Cartesian_to_sph(p.x, p.y, p.z)

            a_r, a_theta, a_phi = self.construct_a(pos_sph)

            ax, ay, az = SSF.acceleration_spherical_to_cartesian(
                a_r, a_theta, a_phi, pos_sph[1], pos_sph[2])
            
            p.ax += float(ax)
            p.ay += float(ay)
            p.az += float(az)

        sim.additional_forces = _forces
        sim.integrate(self.dt)

        # Read back state
        p = sim.particles[0]
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
        parent_j, Y_lm, lm_pairs = precompute_lm_pairs_Ylms(l)
        end = time()
        #print(f"Precomputation of (l,m) pairs and Y_lm grid completed in {end - start:.2f} seconds")

        self.lm_pairs_np = np.array(lm_pairs)  # numpy copy for scipy calls inside construct_a

        #Using Boris code to get k_ln nodes and cln norms for all ell up to Lmax.
        k_list = []
        cln_list = []

        #N_min = int(round(self.no_radius_bins/(2*np.log(self.rmax/self.rmin)), 0))

        #N = int(self.no_radius_bins / 5)

        #N_max = int(round(2 * self.no_radius_bins / (np.log(self.rmax) - np.log(self.rmin)), 0))

        N = int(self.no_radius_bins)

        print(N)

        for ell in range(self.L):
            k_nodes, _cln = BSBT.constructGridAndNorms(self.rmax, N, ell)
            k_list.append(np.asarray(k_nodes, dtype=np.float64)) 
            cln_list.append(np.asarray(_cln, dtype=np.float64))


        # Rearrange so shape is defined by each ell
        kln_nodes_by_ell = jnp.asarray(np.stack(k_list, axis=0))  # (Lmax+1, K)
        cln_norms_by_ell = jnp.asarray(np.stack(cln_list, axis=0))  # (Lmax+1, K)
        self.kln_nodes_by_ell = kln_nodes_by_ell
        self.cln_norms_by_ell = cln_norms_by_ell
        self.kln_np = np.array(kln_nodes_by_ell, dtype=np.float64)  # float64 for Bessel accuracy
        self.cln_np = np.array(cln_norms_by_ell, dtype=np.float64)  # float64 for Bessel accuracy


        self._ell_bc = np.arange(self.L, dtype=np.float64)[:, None]  # (L, 1) for jv broadcast

        # Sort (l,m) pairs by l so phi_nlm rows can be sliced as contiguous l-blocks.
        # This lets construct_a replace the (N_lm, N_k) gather+broadcast with cheap gemv - general matrix vector multiplication.
        _sort = np.argsort(self.lm_pairs_np[:, 0], kind='stable')
        self._lm_sort_order = _sort
        self._lm_sorted     = self.lm_pairs_np[_sort]        # (N_lm, 2) sorted by l
        self._m_sorted      = self._lm_sorted[:, 1]          # m values in sorted order
        _counts = np.bincount(self._lm_sorted[:, 0], minlength=self.L).astype(int)
        self._l_offsets = np.concatenate([[0], np.cumsum(_counts)])  # (L+1,)



        # Set up animator
        if self.animate:
            r_shell = self.r_values[0]  # Start with the initial radius of the particle
            self.animator = SimulationAnimator(self.L, self.r, r_shell, self.u)
            print(f"Animation enabled: density shell at r = {self.animator.r_shell * self.u.to_Kpc:.4f} kpc")

        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")

            # 1. Construct total psi and rho on MW grid
            start = time()
            rho_rtp = self.construct_total_psi(R_j_r, eigen_energies, aj, total_mass, Y_lm, parent_j)
            end = time()
            print(f"Constructing total psi and rho completed in {end - start:.2f} seconds")

            # 1b. Capture animation frame (density at this instant + particle position)
            if self.animate:
                self.animator.store_frame(rho_rtp, self.r_pos)

            # 2. Forward s2fft to get rho_lm(r)
            start = time()
            rho_lm_r = self.forward_s2fft(rho_rtp)
            end = time()
            print(f"Forward s2fft completed in {end - start:.2f} seconds")

            # 3. Convert rho_lm(r) to rho_nlm for autodiff
            start = time()
            phi_nlm = self.Bessel_function_expansion_of_rho_lm(rho_lm_r, lm_pairs, kln_nodes_by_ell, cln_norms_by_ell)
            end = time()
            print(f"Conversion from rho_lm(r) to rho_nlm completed in {end - start:.2f} seconds")


            # Sort phi_nlm rows by l once here — reused across all IAS15 sub-steps
            self._phi_sorted = phi_nlm[self._lm_sort_order]   # (N_lm, N_k)

            # 5. Time step particle — IAS15 calls the callback which uses jax.grad
            start = time()
            self.time_step_particle()
            end = time()
            print(f"Time stepping particle completed in {end - start:.2f} seconds")

        print("Simulation complete.")
