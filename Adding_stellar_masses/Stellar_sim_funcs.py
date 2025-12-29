import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import jaxsp as jsp

from scipy import constants as const

import matplotlib.animation as animation
from IPython.display import HTML

from collections import defaultdict
    
from jaxsp.constants import h, om, hbar, Msun, GN, c, m22

import matplotlib.pyplot as plt

import matplotlib

from scipy.interpolate import interp1d

from scipy.special import sph_harm

import s2fft

m22 = 1
u = jsp.set_schroedinger_units(m22)


def Obtain_pot(rmin, rmax, rho_psi_vals, r):

    # Obtain wavefunction potential from density profile using Poissons equation

    G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

    import numpy as np

    def enclosed_mass(r, rho):
        r = np.asarray(r)
        rho = np.asarray(rho)

        integrand = 4 * np.pi * r**2 * rho

        M_enc = np.zeros_like(r, dtype=float)

        # mass inside [0, r[0]], assume constant density = rho[0]
        #M_core = 4 * np.pi / 3 * rho[0] * r[0]**3

        #M_enc[0] = M_core

        for i in range(1, len(r)):
            #M_enc[i] = M_core + np.trapz(integrand[:i+1], r[:i+1])
            M_enc[i] = np.trapezoid(integrand[:i+1], r[:i+1])

        return M_enc


    M_enc = enclosed_mass(r, rho_psi_vals)


    #Integrand
    h = M_enc / r**2

    #Reverse arrays to integrate from r_max down to r_min
    r_rev = r[::-1]
    h_rev = h[::-1]


    I_rev = np.zeros_like(r, dtype=float)  # integral from r_max downwards
    for k in range(1, len(r)):
        dr = r_rev[k] - r_rev[k - 1]
        I_rev[k] = I_rev[k - 1] + 0.5 * (h_rev[k] + h_rev[k - 1]) * dr

    # flip back: I[i] ≈ ∫_{r_i}^{r_max} M(s)/s^2 ds
    I = -I_rev[::-1]

    # base potential with V(r_max) = 0
    V = -G * I

    return V


def Enclosed_mass(r, rho):

    r = np.asarray(r)
    rho = np.asarray(rho)

    integrand = 4 * np.pi * r**2 * rho

    M_enc = np.zeros_like(r, dtype=float)

    # mass inside [0, r[0]], assume constant density = rho[0]
    #M_core = 4 * np.pi / 3 * rho[0] * r[0]**3

    #M_enc[0] = M_core

    for i in range(1, len(r)):
        #M_enc[i] = M_core + np.trapz(integrand[:i+1], r[:i+1])
        M_enc[i] = np.trapezoid(integrand[:i+1], r[:i+1])

    return M_enc


def Cartesian_to_sph(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = 0
    phi = jnp.arctan2(y, x)
    return np.array([r, theta, phi])


def Cartesian_to_sph_vel(x, y, z, vx, vy, vz):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = 0
    phi = jnp.arctan2(y, x)

    vr = (x * vx + y * vy + z * vz) / r
    vtheta = 0
    vphi = (x * vy - y * vx) / (x**2 + y**2)**0.5

    return np.array([vr, vtheta, vphi])


def Time_step_t_indep(r_pos, v, dt, acc_mag, velocities, avg_r, i):

    acc_vector = acc_mag * (-r_pos / np.linalg.norm(r_pos))
    v = v + acc_vector * dt
    r_pos = r_pos + v * dt

    r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
    v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

    frame = i + 1

    avg_r_new = (avg_r * frame + r_pos_sph[0])/(frame + 1)

    r_mag = r_pos_sph[0]

    velocities.append(v_sph)

    velocities_arr = np.array(velocities)

    vel_disp_r = np.std(velocities_arr[:,0])

    vel_disp_theta = np.std(velocities_arr[:,1])    

    vel_disp_phi = np.std(velocities_arr[:,2])

    vel_disp = (vel_disp_r**2 + vel_disp_theta**2 + vel_disp_phi**2)**0.5

    return r_pos, v, vel_disp, avg_r_new, r_mag, velocities


def Time_step_t_indep_leapfrog(r_pos, v, dt, acc_mag, velocities, avg_r, i):

    acc_vector = acc_mag * (-r_pos / np.linalg.norm(r_pos))

    v_half = v + 0.5 * acc_vector * dt

    r_pos = r_pos + v_half * dt

    acc_vector_new = acc_mag * (-r_pos / np.linalg.norm(r_pos))

    v = v_half + 0.5 * acc_vector_new * dt

    r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
    v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

    frame = i + 1

    avg_r_new = (avg_r * frame + r_pos_sph[0])/(frame + 1)

    r_mag = r_pos_sph[0]

    velocities.append(v_sph)

    velocities_arr = np.array(velocities)

    vel_disp = (np.std(velocities_arr[:,0])**2 + np.std(velocities_arr[:,1])**2 + np.std(velocities_arr[:,2])**2)**0.5

    return r_pos, v, vel_disp, avg_r_new, r_mag, velocities


def Time_step_t_indep_Hanno_reins(r_pos, v, dt, acc_mag, velocities, avg_r, i):
    import numpy as np
    import rebound

    sim = rebound.Simulation()
    sim.integrator = "ias15"

    # Add a massless test particle
    sim.add(
        m=0.0,
        x=r_pos[0], y=r_pos[1], z=r_pos[2],
        vx=v[0],    vy=v[1],    vz=v[2]
    )

    # Take a reference to the particle array *now*
    ps = sim.particles

    def additional_forces(_reb_sim):
        # use ps[0], do NOT touch _reb_sim.particles
        p = ps[0]
        r_vec = np.array([p.x, p.y, p.z])
        r_hat = -r_vec / np.linalg.norm(r_vec)
        p.ax += acc_mag * r_hat[0]
        p.ay += acc_mag * r_hat[1]
        p.az += acc_mag * r_hat[2]

    sim.additional_forces = additional_forces

    # Integrate one step
    sim.integrate(dt)

    # Read back state
    p = sim.particles[0]
    r_pos = np.array([p.x,  p.y,  p.z])
    v     = np.array([p.vx, p.vy, p.vz])

    # Your spherical conversions & stats
    r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
    v_sph     = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2],
                                     v[0], v[1], v[2])

    frame = i + 1
    avg_r_new = (avg_r * frame + r_pos_sph[0]) / (frame + 1)

    r_mag = r_pos_sph[0]

    velocities.append(v_sph)
    velocities_arr = np.array(velocities)

    vel_disp = (np.std(velocities_arr[:, 0])**2 + np.std(velocities_arr[:, 1])**2 + np.std(velocities_arr[:, 2])**2)**0.5

    return r_pos, v, vel_disp, avg_r_new, r_mag, velocities


def Make_animation_t_indep(r_orbit, init_pos, init_vel, dt, num_steps, acc_mag):

    global r_pos, v

    
    orbit = plt.Circle((0, 0), r_orbit * u.to_Kpc, color='black', fill=False, linestyle='--', label='Star Orbit')
    fig, ax = plt.subplots() 

    ax.add_patch(orbit)
    ax.set_xlim(-0.5 , 0.5)
    ax.set_ylim(-0.5 , 0.5)

    ax.set_aspect('equal', adjustable='box')

    plt.scatter(0, 0, color='blue', label='Halo Center', marker='x')

    ax.set_xlabel(r"$x \;\;\mathrm{[kpc]}$", fontsize = 15)
    ax.set_ylabel(r"$y \;\;\mathrm{[kpc]}$", fontsize = 15)

    point, = ax.plot([init_pos[0] * u.to_Kpc], [init_pos[1] * u.to_Kpc], 'go', label='Star', color='red')

    r_pos = init_pos
    v = init_vel

    def update(frame, acc_mag=acc_mag, dt=dt):
        global r_pos, v

        acc_vector = acc_mag * (-r_pos / np.linalg.norm(r_pos))

        v = v + acc_vector * dt
        r_pos = r_pos + v * dt

        point.set_data([r_pos[0] * u.to_Kpc], [r_pos[1] * u.to_Kpc]) 
        return point,
    
    matplotlib.rcParams['animation.embed_limit'] = 2**128

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1, blit=True)

    return ani


def Time_step_t_dep(r_pos, v, dt, acc_mag, r, eigen_energies, l, R_j_r, aj, total_mass, k):

    acc_vector = acc_mag * (-r_pos / np.linalg.norm(r_pos))

    v = v + acc_vector * dt

    r_pos = r_pos + v * dt

    #Update wavefunction potential

    radial_eigen_function_time_stepped = R_j_r * jnp.exp(-1j * k * dt * eigen_energies  / hbar.value)

    Y_l0 = jnp.sqrt((2*l.squeeze()+1)/(4*jnp.pi))

    full_psi = jnp.sum(aj * Y_l0 * radial_eigen_function_time_stepped, axis=1)

    total_psi_2 = (total_mass) * jnp.abs(full_psi)**2

    rho_psi_time_stepped = total_psi_2


    return r_pos, v, rho_psi_time_stepped


def Make_animation_t_dep(r_orbit, init_pos, init_vel, dt, num_steps, acc_mag_init, r, eigen_energies, l, radial_eigen_functions_init, aj, total_mass, rmin, rmax):

    global r_pos, v, avg_r, acc_mag, radial_eigen_functions

    r_pos = init_pos

    v = init_vel

    acc_mag = acc_mag_init

    radial_eigen_functions = radial_eigen_functions_init

    #Figure
    fig, ax = plt.subplots() 

    orbit = plt.Circle((0, 0), r_orbit * u.to_Kpc, color='black', fill=False, linestyle='--', label='Star Orbit')
    ax.add_patch(orbit)

    avg_r = (r_pos[0]**2 + r_pos[1]**2)**0.5
    avg_radial_pos = plt.Circle((0, 0), avg_r * u.to_Kpc, color='orange', fill=False, linestyle='-.', label='Approximate Average Radius of Orbit')
    ax.add_patch(avg_radial_pos)

    ax.set_xlim(-0.5 , 0.5)
    ax.set_ylim(-0.5 , 0.5)

    ax.set_aspect('equal', adjustable='box')

    ax.scatter(0, 0, color='blue', label='Halo Center', marker='x')


    point, = ax.plot([r_pos[0] * u.to_Kpc], [r_pos[1] * u.to_Kpc], 'go', label='Star', color='red')


    #plt.scatter(r_pos[0] * u.to_Kpc, r_pos[1] * u.to_Kpc, color='green', label='Star Position at time t = ' + str(i+1) + 'dt')

    ax.set_xlabel(r"$x \;\;\mathrm{[kpc]}$", fontsize = 15)
    ax.set_ylabel(r"$y \;\;\mathrm{[kpc]}$", fontsize = 15)
    #plt.show()

    def update(frame, dt=dt, eigen_energies=eigen_energies, l=l):

        global r_pos, v, acc_mag, avg_r, radial_eigen_functions


        r_dt, v_dt, rho_psi_time_stepped = Time_step_t_dep(r_pos, v, dt, acc_mag, r, eigen_energies, l, radial_eigen_functions, aj, total_mass, frame+1)
        r_pos = r_dt

        #print(frame)

        v = v_dt

        r_mag = ((r_pos[0])**2 + (r_pos[1])**2)**0.5

        acc_mag = Find_acc_mag_from_rho(r, rho_psi_time_stepped, r_mag)
        
        point.set_data([r_pos[0] * u.to_Kpc], [r_pos[1] * u.to_Kpc]) 


        avg_r_new = (avg_r * frame + r_mag) / (frame + 1)
        avg_radial_pos.set_radius(avg_r_new * u.to_Kpc)
        avg_r = avg_r_new

        return point, avg_radial_pos

    matplotlib.rcParams['animation.embed_limit'] = 2**128

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=10, blit=True)

    return ani


def Simulate_time_dep(r_orbit, init_pos, dt, num_steps, r, eigen_energies, l, radial_eigen_functions, aj, total_mass, rmin, rmax, Phi_psi):

    acc_mag = Find_acc_mag_from_Phi(r, Phi_psi, r_orbit)

    init_vel = np.sqrt(acc_mag * r_orbit) * np.array([0, 1, 0]) #Circular orbit velocity

    init_pos_sph = Cartesian_to_sph(init_pos[0], init_pos[1], init_pos[2])
    init_vel_sph = Cartesian_to_sph_vel(init_pos[0], init_pos[1], init_pos[2], init_vel[0], init_vel[1], init_vel[2])

    r_pos = init_pos
    v = init_vel

    avg_r = init_pos_sph[0]

    total_mag_r = [avg_r]
    total_avg_r = [avg_r]


    def time_Step(r_pos, v, dt, acc_mag, r, eigen_energies, l, radial_eigen_functions, avg_r, i, velocities, aj, total_mass):


        r_dt, v_dt, rho_psi_time_stepped = Time_step_t_dep(r_pos, v, dt, acc_mag, r, eigen_energies, l, radial_eigen_functions, aj, total_mass, i+1)
        r_pos = r_dt

        v = v_dt

        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
        v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

        r_mag = r_pos_sph[0]

        acc_mag = Find_acc_mag_from_rho(r, rho_psi_time_stepped, r_mag)

        frame = i+1

        avg_r_new = (avg_r * frame + r_mag) / (frame + 1)

        velocities.append(v_sph)

        velocities_arr = np.array(velocities)

        vel_disp = (np.std(velocities_arr[:,0])**2 + np.std(velocities_arr[:,1])**2 + np.std(velocities_arr[:,2])**2)**0.5

        return r_pos, v, acc_mag, r_mag, avg_r_new, vel_disp

    stellar_v_disp = [0]
    velocities = [init_vel_sph]


    for i in range(num_steps - 1):
        #print(i)
        r_pos, v, acc_mag, r_mag, avg_r, vel_disp = time_Step(r_pos, v, dt, acc_mag, r, eigen_energies, l, radial_eigen_functions, avg_r, i, velocities, aj, total_mass)
        stellar_v_disp.append(vel_disp)
        total_avg_r.append(avg_r)
        total_mag_r.append(r_mag)

    return np.array(total_mag_r), np.array(total_avg_r), np.array(stellar_v_disp)


def Find_acc_mag_from_Phi(r, Phi_psi, r_orbit):
    
    grad_pot = jnp.gradient(Phi_psi, r)

    grad_pot_func = interp1d(r, grad_pot, kind='cubic', fill_value="extrapolate")

    grad_pot_at_orbit = grad_pot_func(r_orbit)

    acc_mag = grad_pot_at_orbit

    return acc_mag


def Find_acc_mag_from_rho(r, rho_psi, r_orbit):

    G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

    M_enc = Enclosed_mass(r, rho_psi)

    M_enc_func = interp1d(r, M_enc, kind='cubic', fill_value="extrapolate")

    M_enc_at_orbit = M_enc_func(r_orbit)

    acc_mag = G * M_enc_at_orbit / r_orbit**2

    return acc_mag


def Calculating_rho_from_psi_3d(r, eigenstate_lib, wavefunction_params, dt, eval_library):

    Nr = r.shape[0]

    eigen_energies = eigenstate_lib.radial_eigenmode_params.E  # shape (Nj,)
    l = eigenstate_lib.radial_eigenmode_params.l               # shape (Nj,)
    l = jnp.asarray(l, dtype=int)
    n = eigenstate_lib.radial_eigenmode_params.n               # shape (Nj,)
    #print(l)
    #print(n)
    Nj = l.shape[0]


    aj_2 = wavefunction_params.aj_2        # shape (Nj,)
    total_mass = wavefunction_params.total_mass


    rand_phase = jax.random.uniform(jax.random.PRNGKey(0), shape=aj_2.shape, minval=0.0, maxval=2 * jnp.pi,)
    aj = jnp.sqrt(aj_2) * jnp.exp(1j * rand_phase)  # shape (Nj,)


    R_j_r = eval_library(r, eigenstate_lib.radial_eigenmode_params)  # (Nr, Nj)
    radial_eigen_functions = R_j_r

    radial_eigen_function_time_stepped = (radial_eigen_functions * jnp.exp(-1j * dt * eigen_energies / hbar.value))  # (Nr, Nj)


    lm_l = []      # list of l for each mode k
    lm_m = []      # list of m for each mode k
    parent_j = []  # which radial eigenstate j this (l,m) mode comes from

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            lm_l.append(ell)
            lm_m.append(m)
            parent_j.append(j_idx)

    Nmodes = len(lm_l)
    lm_l = np.array(lm_l, dtype=int)
    lm_m = np.array(lm_m, dtype=int)
    parent_j = np.array(parent_j, dtype=int)

    aj_modes = aj[parent_j]                             # (Nmodes,)
    R_modes = radial_eigen_function_time_stepped[:, parent_j]  # (Nr, Nmodes)


    # Band-limit (you can choose something slightly larger if you want margin)
    L = int(l.max()) + 1

    # McEwen–Wiaux–style equiangular grid
    n_theta = L
    n_phi   = 2 * L - 1

    # Generate theta values
    i = np.arange(n_theta)
    theta = (np.pi * (2 * i + 1)) / (2 * L - 1)
        
    # Generate phi values
    j = np.arange(n_phi)
    phi = (2 * np.pi * j) / (2 * L - 1)           

    Theta, Phi = jnp.meshgrid(theta, phi, indexing="ij")  # both (n_theta, n_phi)


    Y_list = []
    for ell, m in zip(lm_l, lm_m):
        Y_lm_mode = sph_harm(m, ell, Phi, Theta)  # (n_theta, n_phi), complex
        Y_list.append(Y_lm_mode)

    Y_lm = jnp.stack(Y_list, axis=0)  # (Nmodes, n_theta, n_phi), complex



    # Broadcast to (Nr, Nmodes, n_theta, n_phi)
    aj_b = aj_modes[None, :, None, None]      # (1, Nmodes, 1, 1)
    R_b  = R_modes[:, :, None, None]          # (Nr, Nmodes, 1, 1)
    Y_b  = Y_lm[None, :, :, :]                # (1, Nmodes, n_theta, n_phi)

    # R_b captures the time dependence part

    full_psi_rtp = jnp.sum(aj_b * R_b * Y_b, axis=1)  # (Nr, n_theta, n_phi)

    psi_abs2 = jnp.abs(full_psi_rtp) ** 2         # (Nr, n_theta, n_phi)
    rho_rtp  = total_mass * psi_abs2              # (Nr, n_theta, n_phi)

    # Quadrature weights on the MW equiangular grid
    dtheta = 2 * jnp.pi / n_phi
    dphi   = 2 * jnp.pi / n_phi

    w_theta = jnp.sin(theta) * dtheta            # (n_theta,)
    w_phi   = jnp.ones_like(phi) * dphi          # (n_phi,)

    w = w_theta[:, None] * w_phi[None, :]  # (n_theta, n_phi)
    w = w[None, :, :]

    norm = w.sum()                      # ≈ 4π

    # Angle-averaged radial profile ρ_ψ(r)
    # Contract over (θ, φ) with weights, then normalise.
    rho_psi_time_stepped = jnp.sum(rho_rtp * w, axis=(1, 2)) / norm  # (Nr,)

    return rho_rtp, rho_psi_time_stepped, radial_eigen_functions, radial_eigen_function_time_stepped, theta, phi, dtheta, dphi


def Calculating_Phi_from_rho_in_3d(l, rho_rtp, r, dtheta, dphi, theta, phi):

    L = int(l.max()) + 1 # = 24


    flm_r = []  # list to hold the spherical harmonic coefficients at each r

    for i in range(len(r)):

        #For each r bin we take the theta and phi variation on the shell, f
        f = rho_rtp[i, :, :]  # shape (n_theta, n_phi)

        #Compute the SHT - get 24 coefficients that span the l,m space for that radius r
        flm = s2fft.forward(f, L, sampling='mw', method='jax')  # shape (l, m) with l in [0, L-1] and m in [-l, l]

        flm_r.append(flm)


    flm_r = jnp.stack(flm_r, axis=0)  # (r, l, m) but shape (r, l, 2*L-1) therefore m = 0 corresponds to position L or index L-1

    #Perform integrals

    # Remember |m| must be less than or equal to l for the rho_lm to be non zero

    l_vals = np.arange(0, L)  # l values from 0 to L-1

    total_phi = defaultdict(list)

    G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

    for l_val in l_vals:

        prefix = - 4 * np.pi * G/(2*l_val + 1)

        m_vals = np.arange(-l_val, l_val + 1)

        for m in m_vals:

            m_ind = m + L - 1 # index in the flm array corresponding to this m value

            f_at_lm = flm_r[:, l_val, m_ind]  # shape (r,) for this (l,m) value - rho_lm for a certain l,m as a function of r

            if abs(f_at_lm).all() == 0:
                print('Skipping as all zero')
                total_phi[(l_val, m)].append(np.zeros_like(r, dtype=complex))
                continue
                
            else:

                print('(l, m) = '+ '(' + str(l_val) + ', ' + str(m) + ')')

            integrand_ext = r**(1-l_val) * f_at_lm
            integrand_int = r**(l_val + 2) * f_at_lm

            integrand_ext_rev = integrand_ext[::-1]
            r_rev = r[::-1]

            integral_ext_rev = np.zeros_like(r, dtype=complex)  # integral from r_max downwards

            integral_int = np.zeros_like(r, dtype=complex)  # integral from 0 to r
            
            for k in range(1, len(r)):
                dr_rev = r_rev[k] - r_rev[k - 1]
                integral_ext_rev[k] = integral_ext_rev[k - 1] + 0.5 * (integrand_ext_rev[k] + integrand_ext_rev[k - 1]) * dr_rev

                dr = r[k] - r[k - 1]
                integral_int[k] = integral_int[k - 1] + 0.5 * (integrand_int[k] + integrand_int[k - 1]) * dr

            integral_ext = -integral_ext_rev[::-1] #integral from r to r_max. DONT FORGET TO MINUS TO FLIP THE INTEGRATION LIMITS!!!!

            total_phi_lm = prefix * (r**(-(l_val + 1)) * integral_int + r**l_val * integral_ext)
            total_phi[(l_val, m)].append(total_phi_lm)
    
    # In total_phi, for each l,m there are 1000 values corresponding to r = r_min to r_max

    # We actually want them in the first form with r as the first index and then l, m

    phi_rlm = []


    for i in range(len(r)):

        phi_lm = np.zeros_like(flm_r[0, :, :], dtype=complex)  # shape (l, m) for this r
        
        for l_val in l_vals:

            for m in range(-l.max(), l.max() + 1):

                m_ind = m + L - 1 # index in the flm array corresponding to this m value

                if abs(m) <= l_val:

                    phi_lm[l_val, m_ind] = total_phi[(l_val, m)][0][i]
                    #print('For (l, m) = '+ str(l_val) + ',' + str(m) + ' its ' + str(total_phi[(l_val, m)][0][i]))
                
                else:

                    phi_lm[l_val, m_ind] = 0.0 + 0.0j
        
        phi_rlm.append(phi_lm)
    
    phi_rlm = np.stack(phi_rlm, axis=0)  # (r, l, m) but shape (r, l, 2*L-1) therefore m = 0 corresponds to position L or index L-1


    L = int(l.max()) + 1 # = 24

    Phi_r = []

    for i in range(len(r)):

        flm = phi_rlm[i, :, :]  # shape (l, m) for this r

        #Compute the inverse SHT - get back to f

        f = s2fft.inverse(flm, L, sampling = 'mw', method='jax')  # shape (n_theta, n_phi)

        Phi_r.append(f)

    Phi_rtp = jnp.stack(Phi_r, axis=0)  # shape (r, n_theta, n_phi)

    # Quadrature weights on the MW equiangular grid
    w_theta = jnp.sin(theta) * dtheta  # (n_theta,)
    w_phi = jnp.ones_like(phi) * dphi      # (n_phi,)

    w = w_theta[:, None] * w_phi[None, :]  # (n_theta, n_phi)
    w = w[None, :, :]

    norm = w.sum()                  # ≈ 4π

    # Angle-averaged radial profile Φ(r)
    Phi_r_dt = jnp.sum(Phi_rtp * w, axis=(1, 2)) / norm  # (Nr,)

    return Phi_rtp, Phi_r_dt, total_phi


def Calculating_Phi_from_rho_in_3d_Unit_Test(l, rho_rtp, r, dtheta, dphi, theta, phi):

    L = int(l.max()) + 1 # = 24


    flm_r = []  # list to hold the spherical harmonic coefficients at each r

    for i in range(len(r)):

        #For each r bin we take the theta and phi variation on the shell, f
        f = rho_rtp[i, :, :]  # shape (n_theta, n_phi)

        #Compute the SHT - get 24 coefficients that span the l,m space for that radius r
        flm = s2fft.forward(f, L, sampling='mw')  # shape (l, m) with l in [0, L-1] and m in [-l, l]

        flm_r.append(flm)


    flm_r = jnp.stack(flm_r, axis=0)  # (r, l, m) but shape (r, l, 2*L-1) therefore m = 0 corresponds to position L or index L-1
    print(flm_r.shape)

    f00_r = flm_r[:, 0, L - 1]  # shape (r,) - the l=0, m=0 coeff at each r
    f1n1_r = flm_r[:, 1, L - 2]  # shape (r,) - the l=1, m=-1 coeff at each r
    f11_r = flm_r[:, 1, L]      # shape (r,) - the l=1, m=1 coeff at each r

    fig = plt.figure(figsize = (8,6))
    plt.plot(r , f00_r, label='f_00', alpha = 0.6)
    plt.plot(r , f1n1_r, label='f_1-1', alpha = 0.6)
    plt.plot(r , -f11_r, label='- f_11', alpha = 0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel('Spherical Harmonic Coefficients of rho')
    plt.legend()
    plt.grid()
    plt.show()

    log_f00_r = np.log10(f00_r)
    log_f11_r = np.log10(-f11_r)
    log_f1n1_r = np.log10(f1n1_r)

    slope_00, intercept_00 = np.polyfit(np.log10(r), log_f00_r, 1)

    slope_11, intercept_11 = np.polyfit(np.log10(r), log_f11_r, 1)

    slope_1n1, intercept_1n1 = np.polyfit(np.log10(r), log_f1n1_r, 1)

    print('Slope and intercept of log-log plot of f_00 vs r: ', slope_00, intercept_00)
    print('Slope and intercept of log-log plot of f_11 vs r: ', slope_11, intercept_11)
    print('Slope and intercept of log-log plot of f_1-1 vs r: ', slope_1n1, intercept_1n1)

    #Perform integrals

    # Remember |m| must be less than or equal to l for the rho_lm to be non zero

    l_vals = np.arange(0, L)  # l values from 0 to L-1

    total_phi = defaultdict(list)

    G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

    for l_val in l_vals:

        prefix = - 4 * np.pi * G/(2*l_val + 1)

        m_vals = np.arange(-l_val, l_val + 1)

        for m in m_vals:

            m_ind = m + L - 1 # index in the flm array corresponding to this m value

            f_at_lm = flm_r[:, l_val, m_ind]  # shape (r,) for this (l,m) value - rho_lm for a certain l,m as a function of r

            if abs(f_at_lm).all() == 0:
                print('Skipping as all zero')
                total_phi[(l_val, m)].append(np.zeros_like(r, dtype=complex))
                continue

            elif l_val > 1:
                print('Skipping as l > 1')
                total_phi[(l_val, m)].append(np.zeros_like(r, dtype=complex))
                continue
                
            else:

                print('(l, m) = '+ '(' + str(l_val) + ', ' + str(m) + ')')

            integrand_ext = r**(1-l_val) * f_at_lm
            integrand_int = r**(l_val + 2) * f_at_lm

            integrand_ext_rev = integrand_ext[::-1]
            r_rev = r[::-1]

            integral_ext_rev = np.zeros_like(r, dtype=complex)  # integral from r_max downwards

            integral_int = np.zeros_like(r, dtype=complex)  # integral from 0 to r
            
            for k in range(1, len(r)):
                dr_rev = r_rev[k] - r_rev[k - 1]
                integral_ext_rev[k] = integral_ext_rev[k - 1] + 0.5 * (integrand_ext_rev[k] + integrand_ext_rev[k - 1]) * dr_rev

                dr = r[k] - r[k - 1]
                integral_int[k] = integral_int[k - 1] + 0.5 * (integrand_int[k] + integrand_int[k - 1]) * dr

            integral_ext = -integral_ext_rev[::-1] #integral from r to r_max. DONT FORGET TO MINUS TO FLIP THE INTEGRATION LIMITS!!!!

            total_phi_lm = prefix * (r**(-(l_val + 1)) * integral_int + r**l_val * integral_ext)
            total_phi[(l_val, m)].append(total_phi_lm)
    
    # In total_phi, for each l,m there are 1000 values corresponding to r = r_min to r_max

    # We actually want them in the first form with r as the first index and then l, m

    phi_rlm = []


    for i in range(len(r)):

        phi_lm = np.zeros_like(flm_r[0, :, :], dtype=complex)  # shape (l, m) for this r
        
        for l_val in l_vals:

            for m in range(-l.max(), l.max() + 1):

                m_ind = m + L - 1 # index in the flm array corresponding to this m value

                if abs(m) <= l_val:

                    phi_lm[l_val, m_ind] = total_phi[(l_val, m)][0][i]
                    #print('For (l, m) = '+ str(l_val) + ',' + str(m) + ' its ' + str(total_phi[(l_val, m)][0][i]))
                
                else:

                    phi_lm[l_val, m_ind] = 0.0 + 0.0j
        
        phi_rlm.append(phi_lm)
    
    phi_rlm = np.stack(phi_rlm, axis=0)  # (r, l, m) but shape (r, l, 2*L-1) therefore m = 0 corresponds to position L or index L-1

    print(phi_rlm.shape)

    phi_00_r = phi_rlm[:, 0, L - 1]  # shape (r,) - the l=0, m=0 coeff at each r
    phi_1n1_r = phi_rlm[:, 1, L - 2]  # shape (r,) - the l=1, m=-1 coeff at each r
    phi_11_r = phi_rlm[:, 1, L]      # shape (r,) - the l=1, m=1 coeff at each r

    fig = plt.figure(figsize = (8,6))
    r_min = r[0]
    r_max = r[-1]


    phi_00_r_func = -(4*np.pi)**(3/2)*G*(r_max - 1/2 * r - 1/2 * r_min**2/r)

    phi_1n1_func = -4*np.pi*G/3 * np.sqrt(2*np.pi/3) * (r*np.log(r_max/r) + 1/3 * r - 1/3*r_min**3/r**2)

    phi_11_func = -phi_1n1_func


    plt.plot(r , phi_00_r, label='phi_00', alpha = 0.6)
    plt.plot(r , phi_1n1_r, label='phi_1-1', alpha = 0.6)
    plt.plot(r , phi_11_r, label='phi_11', alpha = 0.6)

    plt.plot(r , phi_00_r_func, '--', label='Analytic phi_00', alpha = 0.6)
    plt.plot(r , phi_1n1_func, '--', label='Analytic phi_1-1', alpha = 0.6)
    plt.plot(r , phi_11_func, '--', label='Analytic phi_11', alpha = 0.6)

    plt.xlabel('r')
    plt.ylabel('Spherical Harmonic Coefficients of Phi')
    plt.grid()
    plt.legend()
    plt.show()


    L = int(l.max()) + 1 # = 24

    Phi_r = []

    for i in range(len(r)):

        flm = phi_rlm[i, :, :]  # shape (l, m) for this r

        #Compute the inverse SHT - get back to f

        f = s2fft.inverse(flm, L, sampling = 'mw')  # shape (n_theta, n_phi)

        Phi_r.append(f)

    Phi_rtp = jnp.stack(Phi_r, axis=0)  # shape (r, n_theta, n_phi)
    print(Phi_rtp.shape)

    # Quadrature weights on the MW equiangular grid
    w_theta = jnp.sin(theta) * dtheta  # (n_theta,)
    w_phi = jnp.ones_like(phi) * dphi      # (n_phi,)

    w = w_theta[:, None] * w_phi[None, :]  # (n_theta, n_phi)
    w = w[None, :, :]

    norm = w.sum()                  # ≈ 4π

    # Angle-averaged radial profile Φ(r)
    Phi_r_dt = jnp.sum(Phi_rtp * w, axis=(1, 2)) / norm  # (Nr,)

    return Phi_rtp, Phi_r_dt, total_phi

