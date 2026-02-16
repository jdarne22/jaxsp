import unittest

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
# Force JAX to use CPU if GPU memory is limited
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

import Stellar_sim_funcs as SSF
import jaxsp as jsp

from scipy.interpolate import interp1d

from jaxsp.constants import h, om, hbar, Msun, GN, c, m22

import s2fft


class TestStellarSimFuncs(unittest.TestCase):

    def test_rho_to_pot_1d(self):

        m22 = 1
        u = jsp.set_schroedinger_units(m22)

        G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

        r_min = 0.01 * u.from_Kpc
        r_max = 10 * u.from_Kpc


        r = np.logspace(np.log10(r_min), np.log10(r_max), 10000)

        rho_0 = 1.0  # Example central density
        rho = rho_0 / (r)

        V = SSF.Obtain_pot(r_min, r_max, rho, r)

        v_known = 2 * np.pi * rho_0 * G * (r - r_max + r_min**2 / r - r_min**2/r_max)

        ratio = V / v_known

        mask = ~np.isnan(ratio)

        ratio = ratio[mask]

        self.assertTrue(np.allclose(ratio, 1.0, rtol=1e-3))
    
    def test_enclosed_mass(self):

        m22 = 1
        u = jsp.set_schroedinger_units(m22)

        G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

        r_min = 0.01 * u.from_Kpc
        r_max = 10 * u.from_Kpc

        r = np.logspace(np.log10(r_min), np.log10(r_max), 10000)

        rho_0 = 1.0  # Example central density
        rho = rho_0 / (r)

        M_enc = SSF.Enclosed_mass(r, rho)

        M_enc_known = 2 * np.pi * rho_0 * (r**2 - r_min**2)

        ratio = M_enc / M_enc_known

        ratio = ratio[ratio > 0]

        self.assertTrue(np.allclose(ratio, 1.0, rtol=1e-2))

    
    def test_S2fft(self):

        # Define sampled signal to transform and harmonic bandlimit

        n_theta = 24
        n_phi = 47
        L = 24

        i = np.arange(n_theta)

        theta = (np.pi * (2*i+1))/ (2*L - 1)

        j = np.arange(n_phi)

        phi = (2 * np.pi * j) / (2*L - 1)

        func_tp = []

        for i in theta:
            f_phi = []
            for j in phi:
                f_phi.append(1 + np.sin(i) * np.cos(j))
            func_tp.append(f_phi)

        func_tp = np.array(func_tp)

        L = 24
        # Compute harmonic coefficients
        flm = s2fft.forward(func_tp, L, sampling="mw", method='jax')

        # Map back to pixel-space signal
        f_recov = s2fft.inverse(flm, L, sampling="mw", method='jax')

        Mean_abs_err = np.nanmean(np.abs(f_recov - func_tp))

        self.assertTrue(Mean_abs_err < 1e-6)

    
    def test_rho_to_pot_3d(self):

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

        r_min = 0.01 * u.from_Kpc
        r_max = 10 * u.from_Kpc


        r = np.logspace(np.log10(r_min), np.log10(r_max), 1000)

        #Function is rho(r, theta, phi) = 1/r * (1 + sin(theta) * cos(phi))

        # Construct ρ(r, θ, φ)

        rho_rtp = (1 / r[:, None, None]) * (1 + jnp.sin(Theta)[None, :, :] * jnp.cos(Phi)[None, :, :])  # (Nr, n_theta, n_phi)

        # #Compute angle-averaged radial profile

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
        rho_psi_r = jnp.sum(rho_rtp * w, axis=(1, 2)) / norm  # (Nr,)

        Phi_rtp, Phi_r_dt = SSF.Calculating_Phi_from_rho_in_3d_optimized(l, rho_rtp, r, dtheta, dphi, theta, phi)

        R = r[:, None, None]
        theta = Theta[None, :, :]
        phi = Phi[None, :, :]

        Phi_rtp_maths = -4*np.pi*G*(r_max - 1/2*R - 1/2*r_min**2/R) - 4*np.pi*G/3*(R*np.log(r_max/R) + 1/3*R - 1/3*r_min**3/R**2)*np.sin(theta)*np.cos(phi)

        rms_error = np.sqrt(np.mean(abs(((Phi_rtp-Phi_rtp_maths)/Phi_rtp_maths)**2)))

        self.assertEqual(round(rms_error, 5), 0)


    def test_cartesian_to_sph(self):
        # Test coordinate transformation with known values

        # Test point along x-axis
        x, y, z = 1.0, 0.0, 0.0
        result = SSF.Cartesian_to_sph(x, y, z)
        expected = np.array([1.0, 0, 0.0])
        self.assertTrue(np.allclose(result, expected, rtol=1e-10))

        # Test point along y-axis
        x, y, z = 0.0, 1.0, 0.0
        result = SSF.Cartesian_to_sph(x, y, z)
        expected = np.array([1.0, 0, np.pi/2])
        self.assertTrue(np.allclose(result, expected, rtol=1e-10))

        # Test diagonal point
        x, y, z = 1.0, 1.0, 0.0
        result = SSF.Cartesian_to_sph(x, y, z)
        expected = np.array([np.sqrt(2), 0, np.pi/4])
        self.assertTrue(np.allclose(result, expected, rtol=1e-10))


    def test_cartesian_to_sph_vel(self):
        # Test velocity transformation for circular motion in xy-plane

        # Point on x-axis moving in +y direction (circular orbit)
        x, y, z = 1.0, 0.0, 0.0
        vx, vy, vz = 0.0, 1.0, 0.0

        result = SSF.Cartesian_to_sph_vel(x, y, z, vx, vy, vz)

        # For circular motion at this point: v_r = 0, v_theta = 0, v_phi = 1
        expected = np.array([0.0, 0, 1.0])

        self.assertTrue(np.allclose(result, expected, rtol=1e-10))

        # Test radial motion along x-axis
        x, y, z = 1.0, 0.0, 0.0
        vx, vy, vz = 2.0, 0.0, 0.0

        result = SSF.Cartesian_to_sph_vel(x, y, z, vx, vy, vz)
        expected = np.array([2.0, 0, 0.0])

        self.assertTrue(np.allclose(result, expected, rtol=1e-10))


    def test_find_acc_mag_from_rho_point_mass(self):
        # Test acceleration from point mass density profile

        m22 = 1
        u = jsp.set_schroedinger_units(m22)

        G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

        # Create a very concentrated mass profile (approximates point mass)
        r_min = 0.001 * u.from_Kpc
        r_max = 10 * u.from_Kpc

        r = np.logspace(np.log10(r_min), np.log10(r_max), 10000)

        # Total mass
        M_total = 1e10

        # Concentrated density profile that integrates to M_total
        # Use a steep profile: rho ~ 1/r^3 with cutoff
        rho = M_total * r_min / (4 * np.pi * r**3)

        # Test at a specific radius
        r_test = 1.0 * u.from_Kpc

        acc_mag = SSF.Find_acc_mag_from_rho(r, rho, r_test)

        # Expected acceleration: a = G*M/r^2
        acc_expected = G * M_total * r_min * np.log(r_test / r_min) / r_test**2

        ratio = acc_mag / acc_expected

        self.assertTrue(np.allclose(ratio, 1.0, rtol=1e-2))


    def test_find_acc_mag_from_Phi_keplerian(self):
        # Test acceleration from Keplerian potential

        m22 = 1
        u = jsp.set_schroedinger_units(m22)

        G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

        r_min = 0.01 * u.from_Kpc
        r_max = 10 * u.from_Kpc

        r = np.logspace(np.log10(r_min), np.log10(r_max), 10000)

        # Keplerian potential: Phi = -GM/r
        M_total = 1e10
        Phi_psi = -G * M_total / r

        # Test at specific radius
        r_test = 1.0 * u.from_Kpc

        acc_mag = SSF.Find_acc_mag_from_Phi(r, Phi_psi, r_test)

        # Expected: a = dPhi/dr = -GM/r^2
        acc_expected = G * M_total / r_test**2

        ratio = acc_mag / acc_expected

        self.assertTrue(np.allclose(ratio, 1.0, rtol=1e-3))


    def test_potential_boundary_condition(self):
        # Test that V(r_max) ≈ 0

        m22 = 1
        u = jsp.set_schroedinger_units(m22)

        r_min = 0.01 * u.from_Kpc
        r_max = 10 * u.from_Kpc

        r = np.logspace(np.log10(r_min), np.log10(r_max), 10000)

        rho_0 = 1.0
        rho = rho_0 / r**2

        V = SSF.Obtain_pot(r_min, r_max, rho, r)

        # Check that V at r_max is very close to zero
        self.assertTrue(np.abs(V[-1]) < 1e-10)


    def test_leapfrog_energy_conservation(self):
        # Test that leapfrog integrator conserves energy for circular orbit

        m22 = 1
        u = jsp.set_schroedinger_units(m22)

        G = GN.value * (u.from_cm**3) / (u.from_g * u.from_s**2)

        # Point mass potential
        M = 1e10
        r_orbit = 1.0 * u.from_Kpc

        # Circular orbit velocity
        v_circ = np.sqrt(G * M / r_orbit)

        # Initial conditions
        r_pos = np.array([r_orbit, 0.0, 0.0])
        v = np.array([0.0, v_circ, 0.0])

        # Acceleration magnitude
        acc_mag = G * M / r_orbit**2

        # Time step (small for accuracy)
        dt = 0.001 * u.from_s

        velocities = []
        avg_r = r_orbit

        # Initial energy
        E_initial = 0.5 * np.linalg.norm(v)**2 - G * M / np.linalg.norm(r_pos)

        # Evolve for several steps
        for i in range(100):
            r_pos, v, vel_disp, avg_r, r_mag, velocities = SSF.Time_step_t_indep_leapfrog(
                r_pos, v, dt, acc_mag, velocities, avg_r, i
            )
            # Update acceleration for new position
            acc_mag = G * M / np.linalg.norm(r_pos)**2

        # Final energy
        E_final = 0.5 * np.linalg.norm(v)**2 - G * M / np.linalg.norm(r_pos)

        # Energy should be conserved to high precision
        relative_error = np.abs((E_final - E_initial) / E_initial)

        self.assertTrue(relative_error < 1e-3)



if __name__ == '__main__':
    unittest.main()

