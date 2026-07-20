import jax.numpy as jnp
import numpy as np


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


def Enclosed_mass(r, rho):

    dr = jnp.diff(r)

    integrand = 4 * jnp.pi * r**2 * rho

    avg_int = 0.5 * (integrand[1:] + integrand[:-1])

    M_enc = jnp.cumsum(avg_int * dr)

    M_enc = jnp.insert(M_enc, 0, 0.0)

    return M_enc