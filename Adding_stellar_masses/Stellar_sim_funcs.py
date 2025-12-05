import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

import jaxsp as jsp

from scipy import constants as const

import matplotlib.animation as animation
from IPython.display import HTML

    
from jaxsp.constants import h, om, hbar, Msun, GN, c, m22

import matplotlib.pyplot as plt


import matplotlib

from scipy.interpolate import interp1d

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
            M_enc[i] = np.trapz(integrand[:i+1], r[:i+1])

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
    I = I_rev[::-1]

    # base potential with V(r_max) = 0
    V = -G * I

    return -V

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
        M_enc[i] = np.trapz(integrand[:i+1], r[:i+1])

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

    vel_disp = vel_disp_r


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

    vel_disp_r = np.std(velocities_arr[:,0])

    vel_disp = vel_disp_r

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

    vel_disp_r = np.std(velocities_arr[:, 0])
    vel_disp   = vel_disp_r

    return r_pos, v, vel_disp, avg_r_new, r_mag, velocities



def Make_animation_t_indep(r_orbit, init_pos, init_vel, dt, num_steps, acc_mag):

    global r_pos, v

    
    orbit = plt.Circle((0, 0), r_orbit * u.to_Kpc, color='black', fill=False, linestyle='--', label='Star Orbit')
    fig, ax = plt.subplots() 

    ax.add_patch(orbit)
    ax.set_xlim(-10 , 10)
    ax.set_ylim(-10 , 10)

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

    ax.set_xlim(-10 , 10)
    ax.set_ylim(-10 , 10)

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

        vel_disp_r = np.std(velocities_arr[:,0])

        vel_disp = vel_disp_r

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
