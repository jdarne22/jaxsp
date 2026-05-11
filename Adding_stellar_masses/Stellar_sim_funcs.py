
import functools

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

import time

from scipy.interpolate import RegularGridInterpolator

m22 = 1
u = jsp.set_schroedinger_units(m22)


def return_u():
    return u

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

    dr = np.diff(r)

    integrand = 4 * jnp.pi * r**2 * rho

    avg_int = 0.5 * (integrand[1:] + integrand[:-1])

    M_enc = jnp.cumsum(avg_int * dr)

    M_enc = jnp.insert(M_enc, 0, 0.0)

    return M_enc


def Cartesian_to_sph(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r) 
    phi = jnp.arctan2(y, x) % (2 * jnp.pi)
    return jnp.array([r, theta, phi])


def Cartesian_to_sph_vel(x, y, z, vx, vy, vz):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.arctan2(y, x)

    vr = (x * vx + y * vy + z * vz) / r
    vtheta = (z * (x * vx + y * vy) - r**2 * vz) / (r * jnp.sqrt(x**2 + y**2))
    vphi = (x * vy - y * vx) / (x**2 + y**2)**0.5


    return jnp.array([vr, vtheta, vphi])


def acceleration_spherical_to_cartesian(a_r, a_theta, a_phi, theta, phi):

    sin_t, cos_t = jnp.sin(theta), jnp.cos(theta)
    sin_p, cos_p = jnp.sin(phi), jnp.cos(phi)
    a_x = a_r * sin_t * cos_p + a_theta * cos_t * cos_p - a_phi * sin_p
    a_y = a_r * sin_t * sin_p + a_theta * cos_t * sin_p + a_phi * cos_p
    a_z = a_r * cos_t - a_theta * sin_t
    return a_x, a_y, a_z


# Numpy variants of the spherical-coordinate transforms above. Same math, but
# pure numpy so they can be called from host-side per-step bookkeeping without
# paying JAX dispatch overhead. Accept either scalars or 1-D arrays; return
# component tuples so the caller can stack/pack however it needs.

def Cartesian_to_sph_np(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return r, theta, phi


def Cartesian_to_sph_vel_np(x, y, z, vx, vy, vz):
    r = np.sqrt(x*x + y*y + z*z)
    rho_xy = np.sqrt(x*x + y*y)
    vr = (x*vx + y*vy + z*vz) / r
    vtheta = (z*(x*vx + y*vy) - r*r*vz) / (r * rho_xy)
    vphi = (x*vy - y*vx) / rho_xy
    return vr, vtheta, vphi


def acceleration_spherical_to_cartesian_np(a_r, a_theta, a_phi, theta, phi):
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi),   np.cos(phi)
    a_x = a_r * sin_t * cos_p + a_theta * cos_t * cos_p - a_phi * sin_p
    a_y = a_r * sin_t * sin_p + a_theta * cos_t * sin_p + a_phi * cos_p
    a_z = a_r * cos_t         - a_theta * sin_t
    return a_x, a_y, a_z


# ------------------------------------------------------------------
# JAX-native spherical harmonic evaluator.
#
# Drop-in replacement for the diff_n=1 path of scipy.special.sph_harm_y
# used in the time-dependent ULDM simulations. Returns Y_l^m and
# dY_l^m/dtheta at scattered (theta, phi) points for every (l, m) with
# 0 <= l < L_max_out, -l <= m <= l, in the order
#     (0, 0), (1, -1), (1, 0), (1, 1), (2, -2), ..., (L_max_out-1, L_max_out-1).
#
# Why: scipy's sph_harm_y dominates the unjitted per-step cost (~125 ms/call
# in our profile of Analytic_t_dep_sim_Ylm_skip.py). This implementation runs
# inside a single jit, so the only host<->device traffic is the (theta, phi)
# inputs and the final Ylm / dY arrays.
#
# Convention matches scipy: Condon-Shortley phase baked into P_l^m, and
#     Y_l^{-m}(theta, phi) = (-1)^m conj(Y_l^m(theta, phi)).
# Uses the fully-normalised associated Legendre recurrence (Holmes &
# Featherstone 2002 / Wieczorek & Meschede 2018). Numerically stable for
# the lmax ~ 50 regime we run at.
#
# NOTE: dY/dtheta uses the identity
#     dP̄_l^m/dtheta = [l x P̄_l^m - C_lm P̄_{l-1}^m] / sin(theta),
# which diverges at sin(theta) = 0. Particles must not sit exactly on the
# z-axis. (This is a coordinate singularity the existing scipy path shares
# via the a_phi / (r sin theta) acceleration term.)
# ------------------------------------------------------------------

def _NALF_recurrence(L_max_out, x, u):
    """Build dict {(l, m): P̄_l^m(cos theta)} for 0 <= m <= l < L_max_out.

    x = cos(theta), u = sin(theta); both JAX arrays of shape (N_particles,).
    """
    lmax = L_max_out - 1
    P = {(0, 0): jnp.full_like(x, 1.0 / (2.0 * jnp.sqrt(jnp.pi)))}
    # Diagonal: P̄_m^m = -sqrt((2m+1)/(2m)) u P̄_{m-1}^{m-1}
    for m in range(1, lmax + 1):
        coef = -jnp.sqrt((2.0 * m + 1.0) / (2.0 * m))
        P[(m, m)] = coef * u * P[(m - 1, m - 1)]
    # First above-diagonal: P̄_{m+1}^m = sqrt(2m+3) x P̄_m^m
    for m in range(lmax):
        P[(m + 1, m)] = jnp.sqrt(2.0 * m + 3.0) * x * P[(m, m)]
    # General: P̄_l^m = a_lm [x P̄_{l-1}^m - b_lm P̄_{l-2}^m] for l >= m+2
    for m in range(lmax - 1):
        for l in range(m + 2, lmax + 1):
            a_lm = jnp.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
            b_lm = jnp.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
            P[(l, m)] = a_lm * (x * P[(l - 1, m)] - b_lm * P[(l - 2, m)])
    return P


def _dNALF_dtheta(L_max_out, P, x, u):
    """Build dict {(l, m): dP̄_l^m/dtheta} from the NALF table.

    For m == l the (l, l-1) reference is zero, which the formula handles
    via C_lm vanishing — so a single branch suffices. (0, 0) is set to
    zero explicitly because the formula yields 0/u there.
    """
    lmax = L_max_out - 1
    dP = {(0, 0): jnp.zeros_like(x)}
    for l in range(1, lmax + 1):
        for m in range(l + 1):
            if m == l:
                dP[(l, m)] = (l * x * P[(l, m)]) / u
            else:
                C_lm = jnp.sqrt((l * l - m * m) * (2.0 * l + 1.0) / (2.0 * l - 1.0))
                dP[(l, m)] = (l * x * P[(l, m)] - C_lm * P[(l - 1, m)]) / u
    return dP


@functools.partial(jax.jit, static_argnums=0)
def Ylm_dY_jax(L_max_out, theta, phi, eps=1e-8):
    """Y_l^m(theta, phi) and dY_l^m/dtheta(theta, phi) for all modes up to L_max_out.

    theta, phi : 1-D JAX arrays of shape (N_particles,).
    eps        : pole clamp. theta is clipped to [eps, pi - eps] before the
                 sin(theta) division so a particle that lands exactly on the
                 z-axis (or to within float roundoff of it) returns large-but-
                 finite values instead of NaN. The underlying singularity is
                 real (spherical coords break down at the poles, and the
                 simulation's a_phi has a 1/sin(theta) factor) — this clamp
                 only prevents the silent NaN propagation.
                 Default 1e-8 caps |dY/dtheta| at roughly 1e8.
    Returns
        Y_lm        : (L_max_out**2, N_particles) complex
        dY_dtheta   : (L_max_out**2, N_particles) complex
    Modes are returned in the same order as
        [(L, M) for L in range(L_max_out) for M in range(-L, L+1)],
    matching the existing `output_lm_pairs` in the simulation code.

    L_max_out is a static argument — JIT recompiles only if it changes,
    which it shouldn't during a run. Tracing/compile time grows ~quadratically
    in L_max_out (a few seconds for L_max_out ~ 47).
    """
    theta = jnp.clip(theta, eps, jnp.pi - eps)
    x = jnp.cos(theta)
    u = jnp.sin(theta)
    P  = _NALF_recurrence(L_max_out, x, u)
    dP = _dNALF_dtheta(L_max_out, P, x, u)

    Y_list = []
    dY_list = []
    for l in range(L_max_out):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            # sign = (-1)^|m| if m < 0 else 1   (from Y_{l,-m} = (-1)^m conj(Y_{l,|m|}))
            sign = -1.0 if (m < 0 and (abs_m % 2 == 1)) else 1.0
            e_imphi = jnp.exp(1j * m * phi)
            Y_list.append(sign * P[(l, abs_m)] * e_imphi)
            dY_list.append(sign * dP[(l, abs_m)] * e_imphi)

    Y_lm      = jnp.stack(Y_list,  axis=0)
    dY_dtheta = jnp.stack(dY_list, axis=0)
    return Y_lm, dY_dtheta


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

## THIS IS WRONG SINCE TO USE UNSOLDS THEOREM WE ARE NEGLECTING ANY CROSS TERMS

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

'''
ENTERING 3D FUNCTIONS
'''

def Calculating_rho_from_psi_3d(r, eigenstate_lib, wavefunction_params, dt, eval_library):

    Nr = r.shape[0]

    eigen_energies = eigenstate_lib.radial_eigenmode_params.E  # shape (Nj,)
    l = eigenstate_lib.radial_eigenmode_params.l               # shape (Nj,)
    l = jnp.asarray(l, dtype=int)
    n = eigenstate_lib.radial_eigenmode_params.n               # shape (Nj,)
    #print(l)
    #print(n)
    Nj = l.shape[0] #j encompasses both n and l into one index


    aj_2 = wavefunction_params.aj_2        # shape (Nj,)
    total_mass = wavefunction_params.total_mass


    rand_phase = jax.random.uniform(jax.random.PRNGKey(0), shape=aj_2.shape, minval=0.0, maxval=2 * jnp.pi,)
    aj = jnp.sqrt(aj_2) * jnp.exp(1j * rand_phase)  # shape (Nj,)


    R_j_r = eval_library(r, eigenstate_lib.radial_eigenmode_params)  # (Nr, Nj)
    print(R_j_r.shape)

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

    Nmodes = len(lm_l) #Nmodes is the total number of modes for each different n, l, m
    lm_l = jnp.array(lm_l, dtype=int)
    lm_m = jnp.array(lm_m, dtype=int)
    parent_j = jnp.array(parent_j, dtype=int)

    aj_modes = aj[parent_j]                             # (Nmodes,)
    R_modes = radial_eigen_function_time_stepped[:, parent_j]  # (Nr, Nmodes)

    # Band-limit (you can choose something slightly larger if you want margin)
    L = int(l.max()) + 1

    # McEwen–Wiaux–style equiangular grid
    n_theta = L
    n_phi   = 2 * L - 1

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


    full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm) #Use this rather than np.sum as causes memory error due to limited vram in GPU


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

    return rho_rtp, rho_psi_time_stepped, radial_eigen_functions, radial_eigen_function_time_stepped, theta, phi, dtheta, dphi, Y_lm


def precompute_lm_pairs(L):
    """Precompute (l, m) pairs for spherical harmonics - call once before simulation loop."""
    lm_pairs = []
    for l_val in range(L):
        for m_val in range(-l_val, l_val + 1):
            lm_pairs.append((l_val, m_val))
    return jnp.array(lm_pairs)


def Calculating_Phi_from_rho_in_3d_optimized(l, rho_rtp, r, dtheta, dphi, theta, phi, lm_pairs, G):

    #start = time.time()

    L = int(l.max()) + 1

    #Parallel forward SHT over all radii
    def forward_sht_single_r(rho_at_r):
        return s2fft.forward(rho_at_r, L, sampling='mw', method='jax')

    flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)

    lm_pairs = jnp.array(lm_pairs)  # shape (N_pairs, 2)

    def compute_phi_for_lm(lm_pair):
        """Compute phi_lm for a single (l, m) pair"""

        l_val = lm_pair[0]
        m_val = lm_pair[1]

        prefix = -4.0 * jnp.pi * G / (2 * l_val + 1)
        m_ind = m_val + L - 1

        f_at_lm = flm_r[:, l_val, m_ind] #(Nr)

        # Integrands
        integrand_ext = r**(1 - l_val) * f_at_lm
        integrand_int = r**(l_val + 2) * f_at_lm

        # Cumulative integration
        dr = jnp.diff(r)

        # Internal integral (from 0 to r)
        avg_int = 0.5 * (integrand_int[1:] + integrand_int[:-1])
        integral_int = jnp.concatenate([
            jnp.array([0.0 + 0.0j]),
            jnp.cumsum(avg_int * dr)
        ])

        # External integral (from r to r_max) - integrate backwards
        integrand_ext_rev = integrand_ext[::-1]
        r_rev = r[::-1]
        dr_rev = jnp.diff(r_rev)
        avg_ext = 0.5 * (integrand_ext_rev[1:] + integrand_ext_rev[:-1])
        integral_ext_rev = jnp.concatenate([
            jnp.array([0.0 + 0.0j]),
            jnp.cumsum(avg_ext * dr_rev)
        ])
        integral_ext = -integral_ext_rev[::-1]

        return prefix * (r**(-(l_val + 1)) * integral_int + r**l_val * integral_ext)

    #Parallel computation over all (l, m) pairs
    all_phi_lm = jax.vmap(compute_phi_for_lm)(lm_pairs)  # shape (N_pairs, Nr)

    #Reconstruct phi_rlm array using JAX's advanced indexing
    phi_rlm = jnp.zeros((len(r), L, 2*L-1), dtype=complex)

    # Extract l and m indices for all pairs
    l_indices = lm_pairs[:, 0].astype(int)
    m_indices = (lm_pairs[:, 1] + L - 1).astype(int)

    # Use JAX's scatter operation to fill all values at once
    # Transpose all_phi_lm to shape (Nr, N_pairs) then scatter

    # Use vectorized indexing:
    phi_rlm = phi_rlm.at[:, l_indices, m_indices].set(all_phi_lm.T)

    #Parallel inverse SHT over all radii
    def inverse_sht_single_r(phi_lm_at_r):
        return s2fft.inverse(phi_lm_at_r, L, sampling='mw', method='jax')

    Phi_rtp = jax.vmap(inverse_sht_single_r)(phi_rlm)  # (Nr, n_theta, n_phi)

    #Angle-average
    w_theta = jnp.sin(theta) * dtheta
    w_phi = jnp.ones_like(phi) * dphi
    w = w_theta[:, None] * w_phi[None, :]
    norm = w.sum()

    Phi_r_dt = jnp.sum(Phi_rtp * w[None, :, :], axis=(1, 2)) / norm

    #end = time.time()
    #print("Time taken for potential calculation (s): ", end - start)

    return Phi_rtp, Phi_r_dt


def Calculating_Phi_from_rho_in_3d_optimized_at_r_orbit(l, rho_rtp, r, r_orbit, lm_pairs, G):

    start = time.time()

    L = int(l.max()) + 1

    #Parallel forward SHT over all radii
    def forward_sht_single_r(rho_at_r):
        return s2fft.forward(rho_at_r, L, sampling='mw', method='jax')

    flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)

    r_finder = jnp.abs(r - r_orbit)
    r_index = jnp.argmin(r_finder)
    r_vals = r[r_index - 10 : r_index + 11]  # three r values around r_orbit where we are evaluating integrals at

    r_vals = jnp.array(r_vals)


    def compute_phi_for_lm(lm_pair):
        """Compute phi_lm for a single (l, m) pair at r = r_orbit +- small delta"""

        l_val = lm_pair[0]
        m_val = lm_pair[1]

        prefix = -4.0 * jnp.pi * G / (2 * l_val + 1)
        m_ind = m_val + L - 1

        f_at_lm = flm_r[:, l_val, m_ind]  # (Nr). This is rho_lm(r)

        # Integrands - compute over ALL r
        integrand_ext = r**(1 - l_val) * f_at_lm
        integrand_int = r**(l_val + 2) * f_at_lm

        dr = jnp.diff(r)

        # Internal integral (from 0 to r) - FULL computation
        avg_int = 0.5 * (integrand_int[1:] + integrand_int[:-1])
        integral_int = jnp.concatenate([
            jnp.array([0.0 + 0.0j]),
            jnp.cumsum(avg_int * dr)
        ])

        # External integral (from r to r_max) - FULL computation
        integrand_ext_rev = integrand_ext[::-1]
        r_rev = r[::-1]
        dr_rev = jnp.diff(r_rev)
        avg_ext = 0.5 * (integrand_ext_rev[1:] + integrand_ext_rev[:-1])
        integral_ext_rev = jnp.concatenate([
            jnp.array([0.0 + 0.0j]),
            jnp.cumsum(avg_ext * dr_rev)
        ])
        integral_ext = -integral_ext_rev[::-1]

        # Compute phi at ALL r, then extract only the 3 values we need
        phi_all_r = prefix * (r**(-(l_val + 1)) * integral_int + r**l_val * integral_ext)

        # Extract only at r_index-10 to r_index+10
        return phi_all_r[r_index - 10 : r_index + 11]

    #Parallel computation over all (l, m) pairs
    all_phi_lm = jax.vmap(compute_phi_for_lm)(lm_pairs)  # shape (N_pairs, 3)

    #Reconstruct phi_rlm array using JAX's advanced indexing
    phi_rlm = jnp.zeros((len(r_vals), L, 2*L-1), dtype=complex)

    # Extract l and m indices for all pairs
    l_indices = lm_pairs[:, 0].astype(int)
    m_indices = (lm_pairs[:, 1] + L - 1).astype(int)

    # Use JAX's scatter operation to fill all values at once
    # Transpose all_phi_lm to shape (Nr, N_pairs) then scatter
    # Use vectorized indexing:
    phi_rlm = phi_rlm.at[:, l_indices, m_indices].set(all_phi_lm.T)

    #Parallel inverse SHT over all radii
    def inverse_sht_single_r(phi_lm_at_r):
        return s2fft.inverse(phi_lm_at_r, L, sampling='mw', method='jax')

    Phi_rtp = jax.vmap(inverse_sht_single_r)(phi_rlm)  # (3, n_theta, n_phi)

    end = time.time()

    #print("Time taken for potential calculation at r_orbit (s): ", end - start)

    return Phi_rtp, r_vals


def Calculating_Phi_from_rho_in_3d_optimized_at_r_orbit_HR(l, rho_rtp, r, r_orbit, lm_pairs, G):

    start = time.time()

    L = int(l.max()) + 1

    #Parallel forward SHT over all radii
    def forward_sht_single_r(rho_at_r):
        return s2fft.forward(rho_at_r, L, sampling='mw', method='jax')

    flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)

    end = time.time()
    print("Time taken for forward SHT (s): ", end - start)

    r_finder = jnp.abs(r - r_orbit)
    r_index = jnp.argmin(r_finder)
    r_vals = r[r_index - 1 : r_index + 2]  # three r values around r_orbit where we are evaluating integrals at

    r_vals = jnp.array(r_vals)


    def compute_phi_for_lm(lm_pair):
        """Compute phi_lm for a single (l, m) pair at r = r_orbit +- small delta"""

        l_val = lm_pair[0]
        m_val = lm_pair[1]

        prefix = -4.0 * jnp.pi * G / (2 * l_val + 1)
        m_ind = m_val + L - 1

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


        # Compute phi at ALL r, then extract only the 3 values we need
        phi_all_r = prefix * (r_vals**(-(l_val + 1)) * integral_int + r_vals**l_val * integral_ext)

        # Extract only at r_index-1 to r_index+1
        return phi_all_r

    #Parallel computation over all (l, m) pairs
    start = time.time()
    all_phi_lm = jax.vmap(compute_phi_for_lm)(lm_pairs)  # shape (N_pairs, 3)
    end = time.time()
    print("Time taken for integration at r_orbit (s): ", end - start)

    #Reconstruct phi_rlm array using JAX's advanced indexing
    phi_rlm = jnp.zeros((len(r_vals), L, 2*L-1), dtype=complex)

    # Extract l and m indices for all pairs
    l_indices = lm_pairs[:, 0].astype(int)
    m_indices = (lm_pairs[:, 1] + L - 1).astype(int)

    # Use JAX's scatter operation to fill all values at once
    # Transpose all_phi_lm to shape (Nr, N_pairs) then scatter
    # Use vectorized indexing:
    phi_rlm = phi_rlm.at[:, l_indices, m_indices].set(all_phi_lm.T)

    #Parallel inverse SHT over all radii
    start = time.time()
    def inverse_sht_single_r(phi_lm_at_r):
        return s2fft.inverse(phi_lm_at_r, L, sampling='mw', method='jax')


    Phi_rtp = jax.vmap(inverse_sht_single_r)(phi_rlm)  # (3, n_theta, n_phi)

    end = time.time()
    print("Time taken for inverse SHT at r_orbit (s): ", end - start)

    #print("Time taken for potential calculation at r_orbit (s): ", end - start)

    return Phi_rtp, r_vals


def convergence_rho_to_pot_3d(r_in):
    

    m22 = 1
    u = jsp.set_schroedinger_units(m22)

    G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

    l = np.array([23])

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

    r = np.array(r_in)

    r_min = min(r)
    r_max = max(r)

    #Function is rho(r, theta, phi) = 1/r * (1 + sin(theta) * cos(phi))

    # Construct ρ(r, θ, φ)

    rho_rtp = jnp.einsum('r,tp->rtp', 1 / r, (1 + jnp.sin(Theta) * jnp.cos(Phi)))  # (Nr, n_theta, n_phi)

    #rho_rtp = (1 / r[:, None, None]) * (1 + jnp.sin(Theta)[None, :, :] * jnp.cos(Phi)[None, :, :])  # (Nr, n_theta, n_phi)

    dtheta = 2 * jnp.pi / n_phi
    dphi   = 2 * jnp.pi / n_phi

    Phi_rtp, Phi_r_dt = Calculating_Phi_from_rho_in_3d_optimized(l, rho_rtp, r, dtheta, dphi, theta, phi)

    R = r[:, None, None]
    theta = Theta[None, :, :]
    phi = Phi[None, :, :]

    Phi_rtp_maths = -4*np.pi*G*(r_max - 1/2*R - 1/2*r_min**2/R) - 4*np.pi*G/3*(R*np.log(r_max/R) + 1/3*R - 1/3*r_min**3/R**2)*np.sin(theta)*np.cos(phi)
    
    Phi_r_maths = -4*np.pi*G*(r_max - 1/2*r - 1/2 * r_min**2 / r)

    rms_error = np.sqrt(np.mean(abs(((Phi_rtp-Phi_rtp_maths)/Phi_rtp_maths)**2)))

    return rms_error, Phi_r_dt, Phi_r_maths
    

def Acc_from_pot_3d(Phi_rtp, r, theta, phi):


    dr = jnp.gradient(r)
    dtheta = jnp.gradient(theta)
    dphi = jnp.gradient(phi)

    dPhi_dr = jnp.gradient(Phi_rtp, axis=0) / dr[:, None, None]

    dPhi_dtheta = jnp.gradient(Phi_rtp, axis=1) / dtheta[None, :, None]

    dPhi_dphi = jnp.gradient(Phi_rtp, axis=2) / dphi[None, None, :]

    r_grid = r[:, None, None]
    theta_grid = theta[None, :, None]

    acc_r = dPhi_dr
    acc_theta = dPhi_dtheta / r_grid
    acc_phi = dPhi_dphi / (r_grid * jnp.sin(theta_grid))

    acc_vec = -jnp.stack([acc_r, acc_theta, acc_phi])

    return acc_vec


def Simulate_time_dep_3d(r_orbit, init_pos, dt, num_steps, r, eigen_energies, l, R_j_r, aj, total_mass, Phi_psi, Y_lm):

    to_kpc = jnp.array([u.to_Kpc, 1, 1])

    L = int(l.max()) + 1
    G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)
    lm_pairs = precompute_lm_pairs(L)

    acc_mag = Find_acc_mag_from_Phi(r, Phi_psi, r_orbit)

    acc_vec = acc_mag * (-init_pos / np.linalg.norm(init_pos)) 

    init_vel = np.sqrt(acc_mag * r_orbit) * np.array([0, 1, 0]) #Circular orbit velocity

    init_pos_sph = Cartesian_to_sph(init_pos[0], init_pos[1], init_pos[2])
    init_vel_sph = Cartesian_to_sph_vel(init_pos[0], init_pos[1], init_pos[2], init_vel[0], init_vel[1], init_vel[2])

    
    v = init_vel + acc_vec * dt

    r_pos = init_pos + v * dt

    r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
    v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])


    L = int(l.max()) + 1

    # McEwen–Wiaux–style equiangular grid
    n_theta = L
    n_phi   = 2 * L - 1

    # Generate theta values
    i = jnp.arange(n_theta)
    theta = (jnp.pi * (2 * i + 1)) / (2 * L - 1)
        
    # Generate phi values
    j = jnp.arange(n_phi)
    phi = (2 * jnp.pi * j) / (2 * L - 1)  

    dtheta = 2 * jnp.pi / n_phi
    dphi   = 2 * jnp.pi / n_phi         

    parent_j = []

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            parent_j.append(j_idx)


    def time_Step(r_pos, v, dt, r, eigen_energies, l, velocities, aj, total_mass, R_j_r, k, Y_lm, parent_j, average_r):

        '''
        Time step R wavefunctions
        Construct total psi(r, theta, phi)
        Calculate rho_psi(r, theta, phi)
        construct total Phi(r, theta, phi) from rho_psi
        Calculate acc vector field from Phi
        Calculate acc_vec at star position
        Update position from acc_vec
        '''

        start = time.time()

        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])


        #Update wavefunction potential

        radial_eigen_function_time_stepped = R_j_r * jnp.exp(-1j * k * dt * eigen_energies  / hbar.value)

        R_modes = radial_eigen_function_time_stepped[:, parent_j]
        aj_modes = aj[parent_j]

        full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm) #Use this rather than np.sum as causes memory error due to limited vram in GPU

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2         # (Nr, n_theta, n_phi)
        rho_rtp  = total_mass * psi_abs2              # (Nr, n_theta, n_phi)

        Phi_rtp, r_vals = Calculating_Phi_from_rho_in_3d_optimized_at_r_orbit(l, rho_rtp, r, r_pos_sph[0], lm_pairs, G)

        Phi_rtp = Phi_rtp.real

        acc_vec = Acc_from_pot_3d(Phi_rtp, r_vals, theta, phi) # (3, 3, n_theta, n_phi) = (dim, Nr, n_theta, n_phi)

        # Reshape to (r, theta, phi, 3) for a single interpolator
        acc_transposed = np.moveaxis(acc_vec, 0, -1)  # shape (Nr, n_theta, n_phi, 3)

        # Handle phi periodicity: extend grid to include 2π by copying phi=0 values
        phi_extended = np.concatenate([phi, np.array([2 * np.pi])])
        acc_extended = np.concatenate([acc_transposed, acc_transposed[:, :, :1, :]], axis=2)

        interp = RegularGridInterpolator(
            (r_vals, theta, phi_extended),
            acc_extended,
            method='linear'
        )

        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])

        acc_vec = interp(list(r_pos_sph))[0]  # Returns (a_r, a_theta, a_phi)

        #print("Acceleration vector (a_r, a_theta, a_phi) at star position: ", acc_vec)

        #LEAPFROG TIMESTEP

        a_r = acc_vec[0]
        a_theta = acc_vec[1]
        a_phi = acc_vec[2]

        pos_theta = r_pos_sph[1]
        pos_phi = r_pos_sph[2]

        #Convert acceleration to cartesian coordinates
        ax, ay, az = acceleration_spherical_to_cartesian(a_r, a_theta, a_phi, pos_theta, pos_phi)
        
        acc_vec_cart = np.array([ax, ay, az])

        v_half = v + 0.5 * acc_vec_cart * dt

        r_pos = r_pos + v_half * dt

        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])

        print(r_pos_sph * to_kpc)

        acc_vector_new = interp(list(r_pos_sph))[0]

        pos_theta = r_pos_sph[1]
        pos_phi = r_pos_sph[2]

        ax, ay, az = acceleration_spherical_to_cartesian(acc_vector_new[0], acc_vector_new[1], acc_vector_new[2], pos_theta, pos_phi)

        acc_vector_new_cart = np.array([ax, ay, az])

        v = v_half + 0.5 * acc_vector_new_cart * dt

        v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

        

        average_r_new = (average_r * (len(velocities)) + r_pos_sph[0]) / (len(velocities) + 1)


        velocities.append(v_sph)

        velocities_arr = np.array(velocities)

        vel_disp = (np.std(velocities_arr[:,0])**2 + np.std(velocities_arr[:,1])**2 + np.std(velocities_arr[:,2])**2)**0.5

        end = time.time()
        print("Time for one time step (s): ", end - start)

        return r_pos, v, vel_disp, average_r_new, velocities

    
    velocities = np.array([init_vel_sph, v_sph])

    stellar_v_disp = [0, (np.std(velocities[:,0])**2 + np.std(velocities[:,1])**2 + np.std(velocities[:,2])**2)**0.5]

    velocities = velocities.tolist()

    average_r = (init_pos_sph[0] + r_pos_sph[0]) / 2
    total_avg_r = [init_pos_sph[0], average_r]
    r_values = [init_pos_sph[0], r_pos_sph[0]]
    

    for i in range(1, num_steps-1):
        print(i)
        r_pos, v, vel_disp, average_r_new, velocities = time_Step(r_pos, v, dt, r, eigen_energies, l, velocities, aj, total_mass, R_j_r, i, Y_lm, parent_j, average_r)
        stellar_v_disp.append(vel_disp)
        total_avg_r.append(average_r_new)
        average_r = average_r_new
        r_values.append((r_pos[0]**2 + r_pos[1]**2 + r_pos[2]**2)**0.5)


    return np.array(stellar_v_disp), np.array(total_avg_r), np.array(r_values), np.array(velocities)


def Simulate_time_dep_3d_Hanno_Reins(r_orbit, init_pos, dt, num_steps, r, eigen_energies, l, R_j_r, aj, total_mass, Phi_psi, Y_lm, print_every):

    import rebound
    from scipy.interpolate import interpn

    start = time.time()

    acc_mag = Find_acc_mag_from_Phi(r, Phi_psi, r_orbit)

    acc_vec = acc_mag * (-init_pos / np.linalg.norm(init_pos))

    init_vel = np.sqrt(acc_mag * r_orbit) * np.array([0, 1, 0])  # Circular orbit velocity

    init_pos_sph = Cartesian_to_sph(init_pos[0], init_pos[1], init_pos[2])
    init_vel_sph = Cartesian_to_sph_vel(init_pos[0], init_pos[1], init_pos[2], init_vel[0], init_vel[1], init_vel[2])

    L = int(l.max()) + 1
    G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)
    lm_pairs = precompute_lm_pairs(L)

    # McEwen-Wiaux-style equiangular grid
    n_theta = L
    n_phi = 2 * L - 1

    # Generate theta values
    i = jnp.arange(n_theta)
    theta = (jnp.pi * (2 * i + 1)) / (2 * L - 1)

    # Generate phi values
    j = jnp.arange(n_phi)
    phi = (2 * jnp.pi * j) / (2 * L - 1)


    parent_j = []

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            parent_j.append(j_idx)

    
    # --- First step using IAS15 ---
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
    sim.integrate(dt)

    p = sim.particles[0]
    r_pos = np.array([p.x, p.y, p.z])
    v = np.array([p.vx, p.vy, p.vz])

    r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
    v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

    velocities = np.array([init_vel_sph, v_sph])

    stellar_v_disp = [0, (np.std(velocities[:, 0])**2 + np.std(velocities[:, 1])**2 + np.std(velocities[:, 2])**2)**0.5]

    velocities = velocities.tolist()

    average_r = (init_pos_sph[0] + r_pos_sph[0]) / 2
    total_avg_r = [init_pos_sph[0], average_r]
    r_values = [init_pos_sph[0], r_pos_sph[0]]


    # --- Create persistent rebound simulation for time stepping ---
    sim_step = rebound.Simulation()
    sim_step.integrator = "ias15"
    sim_step.add(m=0.0, x=r_pos[0], y=r_pos[1], z=r_pos[2], vx=v[0], vy=v[1], vz=v[2])
    ps_step = sim_step.particles

    # Mutable container for interpolator (to be updated each step)
    interp_data = {'grid': None, 'values': None}


    def additional_forces_step(_reb_sim):
        p = ps_step[0]
        pos_sph = Cartesian_to_sph(p.x, p.y, p.z)
        # Use interpn which is faster for repeated calls
        acc_sph = interpn(interp_data['grid'], interp_data['values'], [list(pos_sph)], method='linear', bounds_error=False, fill_value=None)[0]
        ax, ay, az = acceleration_spherical_to_cartesian(
            acc_sph[0], acc_sph[1], acc_sph[2],
            pos_sph[1], pos_sph[2]
        )
        p.ax += ax
        p.ay += ay
        p.az += az

    sim_step.additional_forces = additional_forces_step

    end = time.time()
    print("Initialization time (s): ", end - start)

    for k in range(1, num_steps - 1):
        '''
        Time step R wavefunctions
        Construct total psi(r, theta, phi)
        Calculate rho_psi(r, theta, phi)
        construct total Phi(r, theta, phi) from rho_psi
        Calculate acc vector field from Phi
        Calculate acc_vec at star position
        Update position using IAS15 (Hanno Rein) integrator
        '''

        if k % print_every == 0:
            print(f"Step {k}/{num_steps}")

        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])

        to_kpc = jnp.array([u.to_Kpc, 1, 1])

        print("Star position (r, theta, phi) in kpc: ", r_pos_sph * to_kpc)

        start = time.time()

        # Update wavefunction potential
        radial_eigen_function_time_stepped = R_j_r * jnp.exp(-1j * k * dt * eigen_energies / hbar.value)

        R_modes = radial_eigen_function_time_stepped[:, parent_j]
        aj_modes = aj[parent_j]

        full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm)

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2
        rho_rtp = total_mass * psi_abs2

        end = time.time()

        print("Time for wavefunction update (s): ", end - start)

        Phi_rtp, r_vals = Calculating_Phi_from_rho_in_3d_optimized_at_r_orbit_HR(l, rho_rtp, r, r_pos_sph[0], lm_pairs, G)

        print(Phi_rtp.shape, r_vals.shape)

        Phi_rtp = Phi_rtp.real

        start = time.time()
        acc_vec_field = Acc_from_pot_3d(Phi_rtp, r_vals, theta, phi)
        end = time.time()
        print("Time for acceleration field calculation (s): ", end - start)

        start = time.time()
        # Prepare interpolator data (update values only)
        acc_transposed = np.moveaxis(np.array(acc_vec_field), 0, -1)
        acc_extended = np.concatenate([acc_transposed, acc_transposed[:, :, :1, :]], axis=2)

        phi_ext = np.concatenate([np.array(phi), np.array([2 * np.pi])])
        interp_data['grid'] = (np.array(r_vals), np.array(theta), phi_ext)
        interp_data['values'] = acc_extended

        # Update particle state and integrate
        p = ps_step[0]
        p.x, p.y, p.z = r_pos[0], r_pos[1], r_pos[2]
        p.vx, p.vy, p.vz = v[0], v[1], v[2]

        target_time = sim_step.t + dt
        sim_step.integrate(target_time)

        # Read back state
        p = sim_step.particles[0]
        r_pos = np.array([p.x, p.y, p.z])
        v = np.array([p.vx, p.vy, p.vz])
        end = time.time()
        print("Time for updating particle position and interpolating acc (s): ", end - start)


        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
        v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

        average_r_new = (average_r * (len(velocities)) + r_pos_sph[0]) / (len(velocities) + 1)

        velocities.append(v_sph)

        velocities_arr = np.array(velocities)

        vel_disp = (np.std(velocities_arr[:, 0])**2 + np.std(velocities_arr[:, 1])**2 + np.std(velocities_arr[:, 2])**2)**0.5
        

        stellar_v_disp.append(vel_disp)
        total_avg_r.append(average_r_new)
        average_r = average_r_new
        r_values.append((r_pos[0]**2 + r_pos[1]**2 + r_pos[2]**2)**0.5)

    return np.array(stellar_v_disp), np.array(total_avg_r), np.array(r_values), np.array(velocities)


def Make_animation_t_dep_3d(r_orbit, init_pos, init_vel, dt, num_steps, r, eigen_energies, l, R_j_r, aj, total_mass, Y_lm, Phi_psi):


    global r_pos, v, acc_vec, avg_r, k

    r_pos = init_pos

    v = init_vel

    acc_mag = Find_acc_mag_from_Phi(r, Phi_psi, r_orbit)

    acc_vec = acc_mag * (-init_pos / np.linalg.norm(init_pos)) #Cartesian acceleration vector

    init_vel = np.sqrt(acc_mag * r_orbit) * np.array([0, 1, 0]) #Circular orbit velocity

    r_pos = init_pos
    v = init_vel


    L = int(l.max()) + 1

    # McEwen–Wiaux–style equiangular grid
    n_theta = L
    n_phi   = 2 * L - 1

    # Generate theta values
    i = jnp.arange(n_theta)
    theta = (jnp.pi * (2 * i + 1)) / (2 * L - 1)
        
    # Generate phi values
    j = jnp.arange(n_phi)
    phi = (2 * jnp.pi * j) / (2 * L - 1)  

    dtheta = 2 * jnp.pi / n_phi
    dphi   = 2 * jnp.pi / n_phi         

    parent_j = []

    for j_idx, ell in enumerate(l.tolist()):
        for m in range(-ell, ell + 1):
            parent_j.append(j_idx)
    

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

    def update(frame, dt=dt, eigen_energies=eigen_energies, l=l, aj=aj, R_j_r = R_j_r, total_mass=total_mass, r=r, dtheta=dtheta, dphi=dphi, theta=theta, phi=phi, parent_j=parent_j, Y_lm=Y_lm):

        global r_pos, v, acc_vec, avg_r, k

        k = frame + 1

        print("Frame: ", frame)

        a_r = acc_vec[0]
        a_theta = acc_vec[1]
        a_phi = acc_vec[2]

        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
        pos_theta = r_pos_sph[1]
        pos_phi = r_pos_sph[2]

        print("Acceleration vector (a_r, a_theta, a_phi) at star position: ", acc_vec)


        #Convert acceleration to cartesian coordinates
        ax, ay, az = acceleration_spherical_to_cartesian(a_r, a_theta, a_phi, pos_theta, pos_phi)
        
        acc_vec_cart = np.array([ax, ay, az])


        v_new = v + acc_vec_cart * dt #MUST BE CARTESIAN

        #Leapfrog
        r_pos = r_pos + v * dt + 0.5 * acc_vec_cart * dt**2

        v = v_new


        r_pos_sph = Cartesian_to_sph(r_pos[0], r_pos[1], r_pos[2])
        v_sph = Cartesian_to_sph_vel(r_pos[0], r_pos[1], r_pos[2], v[0], v[1], v[2])

        print(r_pos_sph*u.to_Kpc)

        #Update wavefunction potential

        radial_eigen_function_time_stepped = R_j_r * jnp.exp(-1j * k * dt * eigen_energies  / hbar.value)

        R_modes = radial_eigen_function_time_stepped[:, parent_j]
        aj_modes = aj[parent_j]

        full_psi_rtp = jnp.einsum('k,rk,ktp->rtp', aj_modes, R_modes, Y_lm) #Use this rather than np.sum as causes memory error due to limited vram in GPU

        psi_abs2 = jnp.abs(full_psi_rtp) ** 2         # (Nr, n_theta, n_phi)
        rho_rtp  = total_mass * psi_abs2              # (Nr, n_theta, n_phi)

        Phi_rtp, r_vals = Calculating_Phi_from_rho_in_3d_optimized_at_r_orbit(l, rho_rtp, r, dtheta, dphi, theta, phi, r_pos_sph[0])

        Phi_rtp = Phi_rtp.real

        acc_vec = Acc_from_pot_3d(Phi_rtp, r_vals, theta, phi) # (3, 3, n_theta, n_phi) = (dim, Nr, n_theta, n_phi)

        # Reshape to (r, theta, phi, 3) for a single interpolator
        acc_transposed = np.moveaxis(acc_vec, 0, -1)  # shape (3, 24, 47, 3)

        interp = RegularGridInterpolator(
            (r_vals, theta, phi),
            acc_transposed,
            method='linear'
        )

        acc_vec = interp(list(r_pos_sph))[0]

        r_mag = r_pos_sph[0]

        point.set_data([r_pos[0] * u.to_Kpc], [r_pos[1] * u.to_Kpc]) 


        avg_r_new = (avg_r * frame + r_mag) / (frame + 1)
        avg_radial_pos.set_radius(avg_r_new * u.to_Kpc)
        avg_r = avg_r_new

        return point, avg_radial_pos

    matplotlib.rcParams['animation.embed_limit'] = 2**128

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=10, blit=True)

    return ani


def Enclosed_mass_3d(r, theta, phi, rho_rtp, r_orbit):

    sin_theta = np.sin(theta)  # shape (N_theta,)



    dphi = np.diff(phi)
    int_phi = 0.5 * np.sum((rho_rtp[:, :, :-1] + rho_rtp[:, :, 1:]) * dphi[None, None, :], axis=2)  # (N_r, N_theta)

    dtheta = np.diff(theta)
    sin_theta_mid = 0.5 * (sin_theta[:-1] + sin_theta[1:])
    rho_r = np.sum(0.5 * (int_phi[:, :-1] + int_phi[:, 1:]) * sin_theta_mid[None, :] * dtheta[None, :], axis=1)  # (N_r,)

    # Cumulative trapezoid rule over r
    g = rho_r * r**2
    dr = np.diff(r)
    M_enc = np.zeros(len(r))
    M_enc[1:] = np.cumsum(0.5 * (g[:-1] + g[1:]) * dr)

    M_enc_func = interp1d(r, M_enc, kind='cubic', fill_value="extrapolate")

    return M_enc_func(r_orbit)










    