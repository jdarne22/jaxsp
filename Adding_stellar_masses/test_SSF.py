import unittest
import Stellar_sim_funcs as SSF

import numpy as np
import jaxsp as jsp

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

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

        self.assertTrue(np.allclose(ratio, 1.0, rtol=1e-3))

    
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

        Phi_rtp, Phi_r_dt, phi_lm = SSF.Calculating_Phi_from_rho_in_3d(l, rho_rtp, r, dtheta, dphi, theta, phi)


        Phi_r_maths = -4*np.pi*G*(r_max - 1/2*r - 1/2 * r_min**2 / r)


        ratio = Phi_r_dt / Phi_r_maths

        self.assertTrue(np.allclose(ratio, 1.0, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()

