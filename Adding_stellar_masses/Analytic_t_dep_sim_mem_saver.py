
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)


import os
import pickle
from time import time
import matplotlib.pyplot as plt
import rebound
import s2fft

import sys
sys.path.append('/home/joshua/PhD_year_1/jaxsp/Adding_stellar_masses')

import jaxsp as jsp

import numpy as np
import jax.numpy as jnp
from jaxsp.constants import GN

import gaunt_funcs as gf
import Stellar_sim_funcs as SSF

import Poisson_solver as PS
import Sharding_manager as SM
import Memory_speed_savers as MSS



import importlib
importlib.reload(SSF)
importlib.reload(gf)
importlib.reload(PS)
importlib.reload(SM)
importlib.reload(MSS)

class Simulation_Particle:
    """
    Stores the state (position + velocity) and history for a single stellar particle.
    """

    def __init__(self, particle_id, init_pos_cart, init_vel_cart, u):

        self.id = particle_id
        self.u = u

        # Current Cartesian state
        self.r_pos = np.array(init_pos_cart)   # (3,)
        self.v     = np.array(init_vel_cart)    # (3,)

        # Convert to spherical for initial record
        self.r_pos_sph = SSF.Cartesian_to_sph(self.r_pos[0], self.r_pos[1], self.r_pos[2])
        self.v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2],self.v[0], self.v[1], self.v[2])


        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]
        self.stellar_v_disp = [0]
        self.r_values       = [float(self.r_pos_sph[0])]
        self.average_r      = [float(self.r_pos_sph[0])]
        self.positions_xyz  = [[float(self.r_pos[0]), float(self.r_pos[1]), float(self.r_pos[2])]]

        self.potential_energy = []
        self.kinetic_energy = [1/2 * np.sum(self.v**2)]
        self.ang_mom = [np.linalg.norm(np.cross(self.r_pos, self.v))]

        self.time_step = 0


    def Change_to_new_vel(self, v_corrected):

        self.v = np.array(v_corrected)
        self.v_sph = SSF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2], v_corrected[0], v_corrected[1], v_corrected[2])
        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]

        self.kinetic_energy = [1/2 * np.sum(self.v**2)]
        self.ang_mom = [np.linalg.norm(np.cross(self.r_pos, self.v))]

    def Create_V_array(self, no_time_steps):
        # Preallocate (no_time_steps + 1, 3) so row 0 holds the initial v_sph
        # and rows 1..no_time_steps hold the values written by update_state.
        self.velocities_arr = np.zeros((no_time_steps + 1, 3))
        self.velocities_arr[0] = np.asarray(self.v_sph)


    def update_state(self, new_pos_cart, new_vel_cart):
        """
        Called after each rebound integration step to update this particle's
        Cartesian and spherical state and append to history arrays.

        """
        x, y, z    = float(new_pos_cart[0]), float(new_pos_cart[1]), float(new_pos_cart[2])
        vx, vy, vz = float(new_vel_cart[0]), float(new_vel_cart[1]), float(new_vel_cart[2])

        self.r_pos = np.array([x, y, z])
        self.v     = np.array([vx, vy, vz])

        r, theta, phi      = SSF.Cartesian_to_sph_np(x, y, z)
        vr, vtheta, vphi   = SSF.Cartesian_to_sph_vel_np(x, y, z, vx, vy, vz)
        self.r_pos_sph     = np.array([r, theta, phi])
        self.v_sph         = np.array([vr, vtheta, vphi])

        self.velocities.append(self.v_sph)
        self.velocities_cart.append(self.v)

        # In-place write into preallocated array; row 0 is the initial v_sph,
        # so the k-th update writes at row k.
        self.velocities_arr[self.time_step + 1] = self.v_sph
        valid = self.velocities_arr[:self.time_step + 2]

        new_vel_disp = (
            np.std(valid[:, 0])**2
            + np.std(valid[:, 1])**2
            + np.std(valid[:, 2])**2
        ) ** 0.5

        self.stellar_v_disp.append(new_vel_disp)

        self.r_values.append(r)
        self.positions_xyz.append([x, y, z])
        self.kinetic_energy.append(0.5 * (vx*vx + vy*vy + vz*vz))
        self.ang_mom.append(np.linalg.norm(np.cross(self.r_pos, self.v)))

        self.time_step += 1


#--------------------------------------------------------------------------------------------------------------------


class StellarSimTDep:

    '''
    Stellar simulation which controls how everything is done and calls the particle
    class to update particle states.
    '''

    def __init__(self, m22, r_half, r_half_width, no_of_particles, no_time_steps, total_evolve_time, r_min, r_max_enclosing_frac,
                 no_radius_bins, SphHT, integrator, plot, dt_override, ramp_time, sparse_k_batch, r_chunk_size, l_band_size,
                 compute_dtype, use_multi_gpu=True, L_out_frac=1.0):

        self.stellar_v_disp = []
        self.average_r = []
        self.time_step = 0
        self.SphHT = SphHT
        self.integrator = integrator
        self.plot = plot
        self.dt_override = dt_override
        self.ramp_time = ramp_time
        self.L_out_frac = L_out_frac

        # ---------- Memory-saving knobs ----------
        # complex64 / float32 for the heavy density / R_j_r path. Eigenenergies
        # and the per-step phase stay in float64 — they multiply by t and need
        # the precision for long-time stability.
        self.compute_dtype = jnp.dtype(compute_dtype)
        if self.compute_dtype == jnp.complex64:
            self.compute_real_dtype = jnp.float32
        elif self.compute_dtype == jnp.complex128:
            self.compute_real_dtype = jnp.float64
        else:
            raise ValueError(f"compute_dtype must be complex64 or complex128, got {compute_dtype}")

        # Chunk size for streaming the (l,m) integration loop in
        # `compute_phi_lm_and_deriv` — caps the per-particle intermediate at
        # (l_band_size, Nr+1) instead of (L_max_out**2, Nr+1).
        self.l_band_size = int(l_band_size)

        # Chunk size for the streamed sparse-a_u_j scatter-add. The dense
        # `(N_unique, Nj)` matrix scales as m22**5 and is ~99% zeros at
        # high m22; we never build it. Instead we stream `sparse_k_batch`
        # k-modes per scan iteration. Lower = less memory but more scan
        # iterations; higher = fewer iterations but larger per-batch peak.
        self.sparse_k_batch = int(sparse_k_batch)

        # Chunk size for streaming the SHT round-trip over the radial
        # axis. Caps the in-flight transient at (r_chunk, L_out, 2L_out-1)
        # complex per chunk instead of the full (Nr, ...) tensor. At m22=10
        # with c64 and Nr=1000, r_chunk=64 gives ~240 MB per chunk vs
        # 3.7 GB without chunking — plus s2fft's internal working set.
        self.r_chunk_size = int(r_chunk_size)

        # Shard the big (Nr, Nj) and (N_unique, Nj) arrays across all visible
        # CUDA devices when more than one is present.
        self.use_multi_gpu = bool(use_multi_gpu)
        self.sharding = SM.ShardingManager(self.use_multi_gpu)

        if self.sharding.shard_l is not None:
            n_dev = len(self.sharding.devices)
            if self.r_chunk_size % n_dev != 0:
                raise ValueError(
                    f"r_chunk_size ({self.r_chunk_size}) must be divisible by "
                    f"number of devices ({n_dev}): shard_map splits the SHT "
                    f"chunk along the r axis."
                )

        self.m22 = m22
        self.u = jsp.set_schroedinger_units(self.m22)

        self.no_of_particles = no_of_particles

        # List of Simulation_Particle instances — populated in initialising_simulation()
        self.particles = []

        self.r_half = r_half
        self.r_half_width = r_half_width
        self.no_time_steps = no_time_steps
        self.total_evolve_time = total_evolve_time
        self.dt = (self.total_evolve_time * self.u.from_Gyr) / self.no_time_steps

        self.r_min = r_min
        self.r_max_enclosing_frac = r_max_enclosing_frac

        self.no_radius_bins = no_radius_bins

        self.G = GN.value * (self.u.from_cm**3) / (self.u.from_g * self.u.from_s**2)


        self.current_phase = None
        self.R_j_r_phased = None
        self.eigen_energies = None
        self.lm_pairs_np = None


    def initialising_simulation(self):

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_wf")
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"m22_{float(self.m22):.6g}_rbins_{int(self.no_radius_bins)}"
        r_j_r_fname = os.path.join(cache_dir, f"precomputed_R_j_r_{cache_suffix}.npz")
        pkl_fname   = os.path.join(cache_dir, f"precomputed_objs_{cache_suffix}.pkl")

        cache_params = {
            'm22': float(self.m22),
            'r_min': float(self.r_min),
            'r_max_enclosing_frac': float(self.r_max_enclosing_frac),
            'no_radius_bins': int(self.no_radius_bins),
        }

        def _cache_valid(data, expected):
            for k, v in expected.items():
                if k not in data.files:
                    return False
                cached = data[k].item() if data[k].shape == () else data[k]
                if isinstance(v, float):
                    if not np.isclose(cached, v):
                        return False
                elif cached != v:
                    return False
            return True

        R_j_r = None
        if os.path.isfile(r_j_r_fname) and os.path.isfile(pkl_fname):
            data = np.load(r_j_r_fname)
            if _cache_valid(data, cache_params):
                print(f"Loading precomputed R_j_r from {r_j_r_fname}...")
                R_j_r = data['R_j_r']
                rmin = data['rmin'].item()
                rmax = data['rmax'].item()
                with open(pkl_fname, 'rb') as f:
                    objs = pickle.load(f)
                eigenstate_lib    = objs['eigenstate_lib']
                wavefunction_params = objs['wavefunction_params']
                eval_library = jax.vmap(jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0, None))

            else:
                print(f"Cached {r_j_r_fname} stale (parameter mismatch); recomputing.")

        if R_j_r is None:

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


            rmin = self.r_min * self.u.from_pc

            tol = 1e-7
            wavefunction_params = jsp.init_wavefunction_params(eigenstate_lib, density_params, rmin, rmax, tol)


            r = jnp.logspace(jnp.log10(rmin), jnp.log10(rmax), self.no_radius_bins)
            R_j_r = eval_library(r, eigenstate_lib.radial_eigenmode_params)  # (Nr, Nj)

            np.savez(r_j_r_fname, R_j_r=np.array(R_j_r), rmin=rmin, rmax=rmax, **cache_params)
            with open(pkl_fname, 'wb') as f:
                pickle.dump({'eigenstate_lib': eigenstate_lib, 'wavefunction_params': wavefunction_params}, f)


                
        l = eigenstate_lib.radial_eigenmode_params.l
        self.l = l

        print('l max from jaxsp:', max(l))
        L = int(max(l) + 1)

        self.L = L

        # Natural bandwidth of rho = |psi|^2: squaring two band-L expansions
        # gives content up to l = 2(L-1), so the lossless SHT bandwidth is
        # 2L-1. In SphHT mode we let the user truncate this with
        # `L_out_frac` to trade aliasing of the high-l rho modes for memory
        # — rho_lms, output_lm_pairs, the Y_lm table, and the SHT round-trip
        # all scale with L_max_out. Floor at L so the (l, m) scatter from
        # the input psi modes (max l = L-1) stays in-range; otherwise we'd
        # silently drop eigenmode contributions, not just truncate rho.
        L_max_out_full = 2 * L - 1
        if self.SphHT and self.L_out_frac < 1.0:
            L_sht = max(int(round(self.L_out_frac * L_max_out_full)), L)
            self.L_max_out = L_sht
            print(f"SphHT bandwidth truncated by L_out_frac={self.L_out_frac}: "
                  f"L_max_out = {self.L_max_out} (natural 2L-1 = {L_max_out_full}, floor L = {L})")
        else:
            self.L_max_out = L_max_out_full

        self.rmin = rmin
        self.rmax = rmax
        

        r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)
        self.r = r

        total_mass = wavefunction_params.total_mass
        self.total_mass = total_mass
        aj_2 = wavefunction_params.aj_2

        # Eigen energies stay in float64 — they multiply by t at every
        # timestep and small phase errors compound badly over a long sim.
        self.eigen_energies = eigenstate_lib.radial_eigenmode_params.E

        # Heavy array — cast to compute_real_dtype (float32 by default) and
        # shard along Nj across all visible GPUs. At m22=50, R_j_r is ~30 GB
        # in float64; this halves it and splits it across devices.
        R_j_r_cast = jnp.asarray(R_j_r, dtype=self.compute_real_dtype)
        
        self.nj_pad = 0
        if self.sharding.shard_nj is not None:
            n_dev = len(self.sharding.devices)
            pad = (-R_j_r_cast.shape[1]) % n_dev
            if pad:
                R_j_r_cast = jnp.pad(R_j_r_cast, ((0, 0), (0, pad)))
                # Keep per-j arrays aligned with the padded Nj axis. The padded
                # slots are never referenced by parent_j, so their values are
                # inert — pad l with 0 and eigen_energies with 0.0. R arrays
                # evaluated at particle radii (R_j_at_particles) must be padded
                # too at the call site.
                self.l = jnp.pad(self.l, (0, pad))
                self.eigen_energies = jnp.pad(self.eigen_energies, (0, pad))
                self.nj_pad = int(pad)


        self.R_j_r_fixed = self.sharding.shard_nj_arr(R_j_r_cast)
        del R_j_r_cast, R_j_r

        # NOTE: R_j_r_phased is no longer materialised. Downstream code uses
        # `R_j_r_fixed` and a per-step `phase` (or pre-phased aj) directly.

        (parent_j, lm_pairs, lm_l_per_mode, lm_m_per_mode, theta, phi, lm_idx_per_mode) = MSS.precompute_lm_pairs(l)

        Nmodes = len(parent_j)
        rand_phase_per_mode = jax.random.uniform(jax.random.PRNGKey(42), shape=(Nmodes,), minval=0.0, maxval=2 * jnp.pi)
        aj = (jnp.sqrt(aj_2[parent_j]) * jnp.exp(1j * rand_phase_per_mode)).astype(self.compute_dtype)


        self.parent_j = parent_j
        self.lm_l = lm_pairs[:, 0]
        self.lm_m = lm_pairs[:, 1]
        self.lm_l_per_mode = lm_l_per_mode
        self.lm_m_per_mode = lm_m_per_mode
        self.lm_idx_per_mode = lm_idx_per_mode
        self.lm_pairs_jax = jnp.asarray(lm_pairs, dtype=jnp.int32)
        self.theta = theta
        self.phi = phi

        # Sparse representation of a_u_j: we keep the three (Nmodes_k,)
        # triplet arrays — `aj`, `parent_j`, `lm_idx_per_mode` — instead of
        # ever materialising the dense `(N_unique, Nj)` matrix. The dense
        # form scales as m22**5 and is ~99% zeros; the sparse form scales
        # as ~m22**4. At m22=10 that swaps 13 GB for ~100 MB.
        # The (sparse) "a_u_j" is now just `(self.aj, self.parent_j, self.lm_idx_per_mode)`.
        self.N_unique_sphht = int(self.lm_pairs_jax.shape[0])
        self.Nj_total = int(len(self.eigen_energies))

        # Set self.aj here so `construct_rho_rtp` (called next) can access
        # it via the sparse triplet. run_simulation will overwrite this with
        # the returned aj — the value is identical.
        self.aj = jnp.asarray(aj)

        # Constructing initial conditions based on Andrew paper

        # The static background is the time-averaged (diagonal) density — the
        # smooth profile the halo is built to reproduce. Both arXiv:2510.17079
        # (Eq. 8) and arXiv:2604.26393 (§III) set the orbit ICs in this mean
        # field and treat the granular fluctuations as a perturbation ramped
        # on top; the instantaneous granule snapshot is NOT the equilibrium.
        # compute_diagonal_rho_expansion also sets self.weight_j (reused
        # per-particle during the ramp).
        rho_diag = self.compute_diagonal_rho_expansion()

        # (l=0, m=0) coefficient of the same diagonal density — the ramp
        # baseline consumed by Build_rho_lms_for_timestep. Y00 = 1/sqrt(4π)
        # so the coefficient is rho_diag · sqrt(4π).
        self.rho_static_r_l00 = (
            rho_diag * jnp.sqrt(4.0 * jnp.pi)).astype(self.compute_dtype)

        # Cumulative enclosed mass M_enc(r) on the radial grid; interpolated
        # per particle below. SSF.Enclosed_mass applies the 4π r² factor.
        M_enc_arr = SSF.Enclosed_mass(self.r, rho_diag)

        # M_enc_tot = M_enc_arr[-1]

        # print(f"Total enclosed mass at rmax: {M_enc_tot:.3e}")
        # print(f"Total mass from wavefunction: {total_mass:.3e}")

        # multiply_factor = total_mass / M_enc_tot

        # print(f"Scaling density and mass by factor {multiply_factor} to match total mass")

        # self.total_mass *= multiply_factor

        if self.plot:

            plt.plot(self.r * self.u.to_Kpc, rho_diag * self.u.to_Msun / (self.u.to_Kpc)**3)

            plt.xlabel('r (kpc)')
            plt.ylabel(r'$\rho$ [$M_\odot / kpc^3$]')
            plt.title(f'Time-averaged (diagonal) density profile with m22 = {self.m22}')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid()
            plt.show()



        #------------------------------------------------------------------
        # SIMULATION

        sim = rebound.Simulation()


        if self.integrator == 'ias15':
        
            sim.integrator = "ias15"
            sim.force_is_velocity_dependent = False
            sim.integrator.ri_ias15.epsilon = 1e-5
        
        elif self.integrator == 'leapfrog':

            sim.integrator = "leapfrog"
            sim.dt = self.dt

        init_vels = []
        r_orbit_mean = self.r_half * self.u.from_Kpc
        r_orbit_min = r_orbit_mean - self.r_half_width/2 * self.u.from_Kpc
        r_orbit_max = r_orbit_mean + self.r_half_width/2 * self.u.from_Kpc


        self.particles = []
        for i in range(self.no_of_particles):

            r_orbit = jax.random.uniform(jax.random.PRNGKey(i), shape=(), minval=r_orbit_min, maxval=r_orbit_max)


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
            M_enc_at_r = jnp.interp(r_orbit, self.r, M_enc_arr)
            v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / r_orbit)

            init_pos = r_i
            init_vel = v_circ_mag * v_i_unit

            init_vels.append(v_circ_mag)

            print(f"Particle {i}: v_circ = {v_circ_mag * self.u.to_kms:.3f} km/s")

            particle = Simulation_Particle(i, init_pos, init_vel, self.u)
            self.particles.append(particle) # Adding instances of particles to simulation class

            sim.add(
                m=0.0,
                x=float(init_pos[0]), y=float(init_pos[1]), z=float(init_pos[2]),
                vx=float(init_vel[0]), vy=float(init_vel[1]), vz=float(init_vel[2])
            )

        sim_particles = sim.particles

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
            N = self.no_of_particles

            # Pull rebound Cartesian state in one pass and do the Cartesian->spherical
            # transform batched in numpy. Avoids N+1 separate jnp.array dispatches.
            xyz = np.empty((N, 3))
            for i in range(N):
                p = sim_particles[i]
                xyz[i, 0] = p.x
                xyz[i, 1] = p.y
                xyz[i, 2] = p.z
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            r, theta, phi = SSF.Cartesian_to_sph_np(x, y, z)

            positions_sph = jnp.asarray(np.stack([r, theta, phi], axis=1))

            self._force_call_count += 1

            # Single batched acceleration computation — parallel over all particles
            a_r_all, a_theta_all, a_phi_all = self.construct_acc_master_func(
                positions_sph,
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib']
            )

            # Pull accs back to host once, then do the spherical->Cartesian
            # rotation batched in numpy.
            a_x, a_y, a_z = SSF.acceleration_spherical_to_cartesian_np(
                np.asarray(a_r_all), np.asarray(a_theta_all), np.asarray(a_phi_all),
                theta, phi,
            )


            for i in range(N):
                sim_particles[i].ax += float(a_x[i])
                sim_particles[i].ay += float(a_y[i])
                sim_particles[i].az += float(a_z[i])

        sim.additional_forces = additional_forces_step

        self.sim = sim
        self.sim_particles = sim_particles

        r_orbits = jnp.array([p.r_values[0] for p in self.particles])

        r_orbit_mean = jnp.mean(r_orbits)

        print(f"Mean r: {r_orbit_mean * self.u.to_Kpc:.3f} kpc")

        if self.dt_override is not None:

            orbital_P = 2 * jnp.pi * r_orbits / jnp.array(init_vels)

            min_orbital_P = jnp.min(orbital_P)

            mean_init_vel = jnp.mean(jnp.array(init_vels))

            lambda_db_kpc = 19.15 / (self.m22 * mean_init_vel * self.u.to_kms)
            T_c = lambda_db_kpc / (mean_init_vel * self.u.to_Kpc) 

            print(f"Min T_orb: {min_orbital_P * self.u.to_Myr:.3f} Myr")
            print(f"T_c: {T_c * self.u.to_Myr:.3f} Myr")

            new_dt_orb = min_orbital_P / self.dt_override

            new_dt_c = T_c / self.dt_override

            new_dt = min(new_dt_orb, new_dt_c)

            self.sim.dt = float(new_dt)

            self.dt = new_dt

            self.no_time_steps = int(self.total_evolve_time * self.u.from_Gyr / new_dt)

            print(f"dt: {self.dt * self.u.to_Gyr:.3f} Gyr")
            print(f"Number of time steps: {self.no_time_steps}")

        return aj


    def ramp_frac_for_step(self, time_step):

        """Scalar in [0, 1] giving the fraction of the off-diagonal (j != j') 
        cross-terms switched on.
        Linear from ~0 to 1 over n_ramp_steps
        """

        if time_step < self.n_ramp_steps:
            return (time_step + 1) / self.n_ramp_steps
        else:
            return 1.0

    def compute_diagonal_rho_expansion(self):

        """Time-averaged (diagonal) density ρ_static(r) — the smooth static
        profile the halo is built to reproduce, and the background that BOTH
        the circular-velocity ICs and the ramp baseline are built from
        (cf. arXiv:2510.17079 Eq. 8, arXiv:2604.26393 §III).

        Because |a_{j,m}|² is m-independent (isotropic random phases), the
        addition theorem Σ_m |Y_{lm}(θ,φ)|² = (2l+1)/4π collapses the angular
        sum to a constant, giving a spherically symmetric static density:

            ρ_static(r) = total_mass · Σ_j  weight_j · |R_j(r)|²
            weight_j     = |a_j|² · (2l_j + 1) / (4π)

        Returns the (Nr,) real density ρ_static(r). Side effect: sets
        self.weight_j, reused by `compute_rho_lm_at_particles_diagonal_only`.
        The (l=0, m=0) spherical-harmonic coefficient — the only nonzero slot,
        used as the ramp baseline — is ρ_static(r)·sqrt(4π).
        """

        Nj = self.R_j_r_fixed.shape[1]

        # Recover |a|² per j-mode: all k-modes sharing the same j carry the same value.
        aj_sq_k = jnp.abs(self.aj) ** 2                                       # (Nk,)
        aj_sq_j = jnp.zeros(Nj, dtype=jnp.float64).at[self.parent_j].set(aj_sq_k)  # (Nj,)

        weight_j = aj_sq_j * (2.0 * self.l.astype(jnp.float64) + 1.0) / (4.0 * jnp.pi)
        self.weight_j = weight_j                                               # (Nj,) — reused by static rho_lm calls

        R_sq = (jnp.abs(self.R_j_r_fixed) ** 2).astype(jnp.float64)
        rho_static_r = self.total_mass * (R_sq @ weight_j)                    # (Nr,)

        return rho_static_r

    def compute_rho_lm_at_particles_diagonal_only(self, R_j_at_particles):
        """Time-averaged (diagonal-only) rho_lm at each particle's exact r.

        Because ρ_static is spherically symmetric, only (l=0, m=0) is nonzero:
            ρ_lm(r_p) = ρ_static(r_p) · sqrt(4π) · δ_{l0} δ_{m0}
        No SHT needed — just a dot product against weight_j.
        """
        R_sq_all = (jnp.abs(R_j_at_particles) ** 2).astype(self.compute_real_dtype)  # (N_p, Nj)
        rho_r_p  = self.total_mass * (R_sq_all @ self.weight_j.astype(self.compute_real_dtype))  # (N_p,)
        N_p      = rho_r_p.shape[0]
        L_out    = self.L_max_out
        out = jnp.zeros((N_p, L_out, 2 * L_out - 1), dtype=self.compute_dtype)
        return out.at[:, 0, L_out - 1].set(rho_r_p * jnp.sqrt(4.0 * jnp.pi))

    def construct_rho_lms_gaunt(self, aj, parent_j, R_j_r_phased):

        '''
        Construct rho_lms using the Gaunt kernel.
        '''

        rho_lm_gaunt = gf.compute_rho_lm_gaunt(
            aj, R_j_r_phased, parent_j, self.lm_idx_sorted_per_mode,
            self.total_mass,
            L_max_out=self.L_max_out,
            gaunt_table=self.gaunt_table,
            batch_size = 100_000
        )

        return rho_lm_gaunt


    def compute_rho_lms_s2fft(self, rho_rtp):

        '''Forward s2fft of 3d density to get rho_lm(r)
        '''

        #Parallel forward SHT over all radii
        def forward_sht_single_r(rho_at_r):
            return s2fft.forward(rho_at_r, self.L_max_out, sampling='mw', method='jax')

        flm_r = jax.vmap(forward_sht_single_r)(rho_rtp)  # (Nr, L, 2*L-1)


        return flm_r

    def Build_rho_lms_for_timestep(self, time_step):
        """Build rho_lms at `time_step`, applying the two-phase schedule.

        Phase 1 (ramp, time_step < n_ramp_steps):
            rho = rho_static + ramp_frac * (rho_full(t) - rho_static).
            `rho_static` is the time-averaged (diagonal) density — the same
            smooth background the circular-velocity ICs were built from. The
            interference terms (the full time-dependent piece, both the
            monopole's breathing and the l>=1 asphericity) are linearly
            switched on from 0 to 1 over `ramp_time`.

        Phase 2 (main, time_step >= n_ramp_steps):
            rho = rho_full(t). Full instantaneous ULDM density, all terms.
        """

        # Phase stays in float64/complex128 for long-time stability; cast to
        # the compute dtype only when multiplying R_j_r (which is c64/f32).
        phase = jnp.exp(-1j * self.eigen_energies * time_step * self.dt / 1)
        self.current_phase = phase
        phase_c = phase.astype(self.compute_dtype)

        ramp_frac = self.ramp_frac_for_step(time_step)
        self.current_ramp_frac = jnp.float64(ramp_frac)
        # Cast ramp_frac to compute_real_dtype so multiplying it against a
        # c64 array doesn't promote to c128 (which then breaks the
        # c64 sharded output contract downstream).
        ramp_c = jnp.asarray(ramp_frac, dtype=self.compute_real_dtype)

        if self.SphHT:
            # Fused JIT: sparse-matmul -> scatter -> inv-SHT -> |.|^2 -> fwd-SHT
            # -> per-chunk ramp blend + static (l=0,m=0) add. The ramp / static
            # combine happens inside the scan so no second full-size
            # (Nr, L, 2L-1) tensor is ever materialised. XLA frees flm_r /
            # psi_rtp / rho_rtp transiently rather than holding them all at once.
            return MSS.build_sphht_rho_lms_jit(
                self.R_j_r_fixed, phase_c,
                self.aj, self.parent_j, self.lm_idx_per_mode,
                self.lm_pairs_jax, self.total_mass,
                ramp_c, self.rho_static_r_l00,
                int(self.L_max_out),
                int(self.N_unique_sphht), int(self.sparse_k_batch),
                int(self.r_chunk_size),
                out_sharding=self.sharding.shard_l,
            )

        # Gaunt path uses an external helper (`gf.compute_rho_lm_gaunt`)
        # that expects an already-phased R_j_r; keep that contract but
        # build the phased array transiently.
        R_j_r_phased = self.R_j_r_fixed * phase_c[None, :]
        rho_lms_full = self.construct_rho_lms_gaunt(self.aj, self.parent_j, R_j_r_phased)
        del R_j_r_phased

        # rho_lms = (1 - ramp_frac) * rho_static + ramp_frac * rho_full.
        L_out = self.L_max_out
        rho_lms = ramp_c * rho_lms_full
        rho_lms = rho_lms.at[:, 0, L_out - 1].add(
            ((1.0 - ramp_c) * self.rho_static_r_l00).astype(rho_lms.dtype)
        )
        return rho_lms


    def construct_rho_rtp(self, R_j_r_fixed, phase_c, lm_pairs):

        '''
        Construct rho_rtp without using Y_lms.
        Inverse SHT to get psi on the dense grid, then square.

        Delegates to the module-level JIT'd helper using the SPARSE a_u_j
        representation — `(self.aj, self.parent_j, self.lm_idx_per_mode)` —
        so the `(N_unique, Nj)` dense matrix is never materialised. The
        scatter-add streams k-modes in batches of `self.sparse_k_batch`.
        '''
        return MSS.build_sphht_rho_rtp_jit(
            R_j_r_fixed, phase_c,
            self.aj, self.parent_j, self.lm_idx_per_mode,
            lm_pairs, self.total_mass, int(self.L_max_out),
            int(self.N_unique_sphht), int(self.sparse_k_batch),
            int(self.r_chunk_size),
        )


    def compute_rho_lm_at_particles_gaunt(self, R_j_at_parts, phase_c, a_u_j, all_i, all_j, all_G, all_Lf):

        """Batched Gaunt path: compute rho_lm at every particle's radius in
        ONE call instead of once per particle inside a vmap.

        Change 9: fold the phase into the matmul rather than materialising
        an (N_particles, Nj) R_j_phased_at_parts copy.
        """

        # F_all[p, u] = Σ_j a_u_j[u, j] · phase[j] · R_j_at_parts[p, j]
        F_all = jnp.einsum('uj,j,pj->pu', a_u_j, phase_c, R_j_at_parts,
                           optimize='optimal')
        return gf.compute_rho_lm_gaunt_F(
            F_all, self.total_mass, self.L_max_out,
            all_i, all_j, all_G, all_Lf,
            batch_size=100_000,
        )
    

    def compute_rho_lm_at_particles_sphht(self, R_j_at_parts, phase_c):

        """SphHT path: rho_lm per particle via s2fft round-trip, using the
        sparse a_u_j scatter-add (no dense `(N_unique, Nj)` matrix).
        """
        return MSS.compute_rho_lm_at_particles_sphht_jit(
            R_j_at_parts, phase_c,
            self.aj, self.parent_j, self.lm_idx_per_mode,
            self.lm_pairs_jax, self.total_mass, int(self.L_max_out),
            int(self.N_unique_sphht), int(self.sparse_k_batch),
        )

    def insert_particle_rholm_and_get_philm(self, r_pos_sph, rho_lm_at_particle, rho_lms):

        """
        Per-particle insertion into the background radial grid + call to
        _compute_all_phi. Safe to vmap. rho_lm_at_particle is supplied by
        the caller.

        Change 2: instead of carrying two pre-padded `(Nr+1, L, 2L-1)`
        copies of rho_lms (`rho_lms_below` and `rho_lms_above`), we gather
        from the unpadded `(Nr, L, 2L-1)` array using an index shift —
        2x less standing rho_lms memory.

        Pattern: for i in [0, Nr]:
            if i  < insert_idx: take rho_lms[i]
            if i == insert_idx: insert rho_lm_at_particle
            if i  > insert_idx: take rho_lms[i-1]  (shifted; particle sits between)
        which is rho_lms[i - (i > insert_idx)] with override at i == insert_idx.
        """

        particle_r = r_pos_sph[0]

        insert_idx = jnp.searchsorted(self.r, particle_r)

        # gather_idx: for i  < insert_idx -> i;   for i  >= insert_idx -> i-1
        # (clipped so the i == insert_idx slot is safe; that slot is then
        # overwritten by particle_r / rho_lm_at_particle in the where below).
        gather_idx = jnp.clip(
            self.all_idx - (self.all_idx > insert_idx).astype(jnp.int32),
            0,
            self.Nr - 1,
        )

        r_updated = jnp.where(
            self.all_idx == insert_idx,
            particle_r,
            self.r[gather_idx],
        )

        # Change 3: do NOT materialise `rho_lm_updated = (Nr+1, L, 2L-1)`.
        # `compute_phi_lm_and_deriv` builds each `(Nr+1,)` (l,m) slice
        # lazily from `rho_lms + rho_lm_at_particle + gather_idx + insert_mask`,
        # saving a ~3.7 GiB per-particle transient at m22~5, Nr=1000, L_out=481.
        insert_mask = self.all_idx == insert_idx

        mask_int = jnp.arange(self.Nr) < insert_idx
        mask_ext = jnp.arange(self.Nr) < (self.Nr - insert_idx)

        dphi_lm_dr_at_r, phi_lm_at_r = PS.compute_phi_lm_and_deriv(
            rho_lms, rho_lm_at_particle, gather_idx, insert_mask,
            r_updated, self.output_lm_pairs,
            mask_int, mask_ext, int(self.L_max_out), self.G, particle_r,
            int(self.l_band_size),
        )

        if self.plot and not isinstance(particle_r, jax.core.Tracer):

            if self.current_ramp_frac == 1.0:
                # Materialise the inserted (Nr+1, L, 2L-1) rho_lms just for
                # plotting — the JIT path avoids this transient (Change 3),
                # but here we're eager and need it to line up with r_updated.
                rho_lms_updated = jnp.where(
                    insert_mask[:, None, None],
                    rho_lm_at_particle[None, :, :],
                    rho_lms[gather_idx],
                )

                for l in range(3):
                    plt.plot(r_updated * self.u.to_Kpc, jnp.abs(rho_lms_updated[:, l, self.L_max_out - 1]) * self.u.to_Msun / (self.u.to_Kpc)**3, label=f'l={l}')

                plt.xlabel('r (kpc)')
                plt.ylabel(r'$\rho_{lm}$ [$M_\odot / kpc^3$]')
                plt.title(f'Updated rho_lm with inserted particle at r={particle_r * self.u.to_Kpc:.3f} kpc')
                plt.xscale('log')
                plt.yscale('log')
                plt.grid()
                plt.legend()
                plt.show()

                def inverse_s2fft(rho_lm):
                    return s2fft.inverse(rho_lm, int(self.L_max_out), sampling='mw', method='jax')

                rho_rtp = jax.vmap(inverse_s2fft)(rho_lms_updated)  # (Nr+1, n_theta, n_phi)

                plotting_theta = int(self.theta.shape[0] / 2)
                plotting_phi = 0

                rho_r = rho_rtp[:, plotting_theta, plotting_phi]
                plt.plot(r_updated * self.u.to_Kpc, jnp.abs(rho_r) * self.u.to_Msun / (self.u.to_Kpc)**3, label='Updated rho(r)')

                plt.xlabel('r (kpc)')
                plt.ylabel(r'$\rho$ [$M_\odot / kpc^3$]')
                plt.title(f'Updated rho(r) with inserted particle at r={particle_r * self.u.to_Kpc:.3f} kpc')
                plt.xscale('log')
                plt.yscale('log')
                plt.grid()
                plt.legend()
                plt.show()


        return dphi_lm_dr_at_r, phi_lm_at_r  # (Nmodes,), (Nmodes,)

    def calc_rho_lm_at_parts_and_call_insert(self, positions_sph, current_phase, radial_eigenmode_params, a_u_j, all_i, all_j,
                              all_G, all_Lf, rho_lms, ramp_frac):

        """JIT-compilable: radial basis evaluation + batched rho_lm + vmap
        over the per-particle insertion step.

        Changes 2 & 9: takes unpadded `rho_lms` (gather is done inside
        `insert_particle_rholm_and_get_philm`) and passes the per-step
        phase through to the SphHT/Gaunt density helpers without
        materialising a phased R_j table.
        """

        # Cast phase down to compute_dtype just before contracting with R_j.
        # `current_phase` is c128 (precision needed in the exp); cast loses
        # negligible info for the contraction.
        phase_c = current_phase.astype(self.compute_dtype)

        particle_rs = positions_sph[:, 0]
        R_j_at_particles = self._eval_library(particle_rs, radial_eigenmode_params)
        R_j_at_particles = R_j_at_particles.astype(self.compute_real_dtype)

        # Match the Nj-axis padding applied to R_j_r_fixed / eigen_energies so
        # all per-j contractions line up. Padded slots are zero (inert).
        if self.nj_pad:
            R_j_at_particles = jnp.pad(R_j_at_particles, ((0, 0), (0, self.nj_pad)))

        # R_j_at_particles : (N_particles, Nj)  (real)

        # Compute rho_lm at each particle's radius OUTSIDE the vmap so the
        # expensive a_u_j matmul + Gaunt reduction happen once per call,
        # not once per particle.

        #print('computing rho_lm at particle positions...')
        if self.SphHT:
            # `a_u_j_sphht` arg is ignored — the sparse JIT pulls
            # (aj, parent_j, lm_idx_per_mode) directly from `self`.
            rho_lm_at_particles_full = self.compute_rho_lm_at_particles_sphht(
                R_j_at_particles, phase_c)
            
       
        else:
            rho_lm_at_particles_full = self.compute_rho_lm_at_particles_gaunt(
                R_j_at_particles, phase_c, a_u_j, all_i, all_j, all_G, all_Lf,
            )

        # Time-averaged (diagonal-only) rho_lm at the same radii. Used as
        # the baseline for the ramp.
        rho_lm_at_particles_static = self.compute_rho_lm_at_particles_diagonal_only(R_j_at_particles)

        rho_lm_at_particles = (rho_lm_at_particles_static +
                               ramp_frac * (rho_lm_at_particles_full - rho_lm_at_particles_static))
        # rho_lm_at_particles : (N_particles, L_max_out, 2*L_max_out-1)

        #print('computing per-particle insertions and phi_lm...')

        dphi_lm_dr_at_parts, phi_lm_at_parts = jax.lax.map(
            lambda inp: self.insert_particle_rholm_and_get_philm(
                inp[0], inp[1], rho_lms),
            (positions_sph, rho_lm_at_particles)
        )
        # dphi_dr_all : (N_particles, Nmodes)
        # phi_lm_all  : (N_particles, Nmodes)


        return dphi_lm_dr_at_parts, phi_lm_at_parts

    @staticmethod
    def combine_acc(dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi, particle_r, particle_theta):

        """
        JIT-compilable: contract radial outputs with angular terms to get accelerations.
        """

        dphi_lm_dr_T = dphi_lm_dr_at_parts.T   # (Nmodes, N_particles)
        phi_lm_T  = phi_lm_at_parts.T   # (Nmodes, N_particles)

        a_r     = jnp.sum(-dphi_lm_dr_T * Ylm_all, axis=0).real                                                      # (N_particles,)
        a_theta = jnp.sum(-phi_lm_T  * dY_dtheta / particle_r[None, :], axis=0).real                             # (N_particles,)
        a_phi   = jnp.sum(-phi_lm_T  * dY_dphi   / (particle_r[None, :] * jnp.sin(particle_theta[None, :])), axis=0).real  # (N_particles,)

        return a_r, a_theta, a_phi

    def construct_acc_master_func(self, positions_sph, eval_library, eigenstate_lib, poten = False):

        '''
        Constructing acc vectors for all particles master function.
        With JIT-compilable radial eval + rho_lm + insertion, followed by a non-JIT angular contraction to get accs.
        '''

        if not hasattr(self, 'calc_rho_lm_at_parts_and_call_insert_jit'):
            self._eval_library = eval_library
            self.calc_rho_lm_at_parts_and_call_insert_jit = jax.jit(self.calc_rho_lm_at_parts_and_call_insert)
            self.combine_acc_jit = jax.jit(StellarSimTDep.combine_acc)

        #print("Compiling JIT for per-particle rho_lm insertion and acc combination...")
        dphi_lm_dr_at_parts, phi_lm_at_parts = self.calc_rho_lm_at_parts_and_call_insert_jit(
            positions_sph,
            self.current_phase,
            eigenstate_lib.radial_eigenmode_params,
            self.a_u_j,
            self._jit_all_i,
            self._jit_all_j,
            self._jit_all_G,
            self._jit_all_Lf,
            self.rho_lms,
            self.current_ramp_frac,
        )


        #print('got dphi_lm_dr_at_parts and phi_lm_at_parts from JIT-compiled function')

        if self.plot:
            # Eagerly recompute rho_lm for particle 0 so matplotlib gets concrete
            # arrays. insert_particle_rholm_and_get_philm is called directly here
            # (outside JIT / lax.map) so its plot branch — guarded by
            # `not isinstance(particle_r, jax.core.Tracer)` — fires.
            _p0_r = positions_sph[0:1, 0]
            _R_j_p0 = self._eval_library(_p0_r, eigenstate_lib.radial_eigenmode_params)

            phase_c = self.current_phase.astype(self.compute_dtype)

            if self.SphHT:
                _rho_full_p0 = self.compute_rho_lm_at_particles_sphht(_R_j_p0, phase_c)
            else:
                _rho_full_p0 = self.compute_rho_lm_at_particles_gaunt(
                    _R_j_p0, phase_c, self.a_u_j, self._jit_all_i, self._jit_all_j,
                    self._jit_all_G, self._jit_all_Lf,
                )
            _rho_static_p0 = self.compute_rho_lm_at_particles_diagonal_only(_R_j_p0)
            _rho_p0 = (_rho_static_p0[0]
                       + self.current_ramp_frac * (_rho_full_p0[0] - _rho_static_p0[0]))
            self.insert_particle_rholm_and_get_philm(
                positions_sph[0], _rho_p0, self.rho_lms,
            )




        #print('computing Y_lm and derivatives with jax.scipy (GPU)...')

        # On-GPU Y_lm + dY_lm/dθ via jax.scipy.special.sph_harm_y + JVP.
        # No more GPU↔host round-trip or single-threaded scipy CPU call.
        # n_max must be static for sph_harm_y under JIT; l ranges over
        # [0, L_max_out-1] in output_lm_pairs.
        Ylm_all, dY_dtheta = MSS.compute_Ylm_and_dtheta_jit(
            self.output_lm_pairs,
            positions_sph[:, 1],
            positions_sph[:, 2],
            int(self.L_max_out - 1),
        )                                            # (Nmodes, N_particles) each

        m_vals  = self.output_lm_pairs[:, 1, None]  # (Nmodes, 1)
        dY_dphi = 1j * m_vals * Ylm_all             # (Nmodes, N_particles)

        #print('combining accs with JIT-compiled contraction...')
        if poten:
            accs = self.combine_acc_jit(
                dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
                positions_sph[:, 0], positions_sph[:, 1],
            )
            return *accs, phi_lm_at_parts, Ylm_all


        return self.combine_acc_jit(
            dphi_lm_dr_at_parts, phi_lm_at_parts, Ylm_all, dY_dtheta, dY_dphi,
            positions_sph[:, 0], positions_sph[:, 1],
        )


    def time_step_particle(self):

        """
        Synchronise all Simulation_Particle states into rebound, integrate
        one macro timestep, then read back and update each particle instance.
        """

        # Write current state of every particle into the rebound simulation
        for i, particle in enumerate(self.particles):
            p = self.sim_particles[i]
            p.x,  p.y,  p.z  = float(particle.r_pos[0]), float(particle.r_pos[1]), float(particle.r_pos[2])
            p.vx, p.vy, p.vz = float(particle.v[0]),     float(particle.v[1]),     float(particle.v[2])

        self._force_call_count = 0   # reset counter for this macro step
        target_time = self.sim.t + self.dt
        self.sim.integrate(target_time)
        print(f"  Force calls this timestep: {self._force_call_count}")

        # Read back and update each Simulation_Particle
        for i, particle in enumerate(self.particles):
            p = self.sim_particles[i]
            particle.update_state(
                [p.x,  p.y,  p.z],
                [p.vx, p.vy, p.vz]
            )
            #print(f"  Particle {i}: r = {float(particle.r_pos_sph[0]) * self.u.to_Kpc:.4f} kpc")


    def run_simulation(self):

        start = time()
        aj = self.initialising_simulation()
        end = time()
        self.aj = aj


        Nr = len(self.r)        # number of background radial bins
        all_idx = jnp.arange(Nr + 1)   # indices 0 .. Nr
        self.Nr = Nr
        self.all_idx = all_idx


        # r_below / r_above are no longer needed — `self.r[gather_idx]`
        # inside the JIT'd insert handles the radial-grid shift directly.

        # self.L_max_out was set in initialising_simulation — possibly truncated
        # by L_out_frac in the SphHT path. Just alias it for local use here.
        L_max_out = self.L_max_out


        if self.SphHT == False:

            # Precompute Gaunt table ONCE — reuse this across all time steps
            gaunt_table = gf.precompute_gaunt_table(self.lm_l, self.lm_m, L_max_out)
            self.gaunt_table = gaunt_table

            _, _, _, _, unique_lm = gaunt_table

            self.lm_idx_sorted_per_mode = gf.make_lm_idx_sorted_per_mode(
                self.lm_l_per_mode, self.lm_m_per_mode, unique_lm)


            Nj = len(self.eigen_energies)
            N_unique = len(unique_lm)
            # CPU-numpy scatter (avoids GPU peak-doubling) + optional Nj shard.
            self.a_u_j = self.sharding.build_sparse_au_j(
                self.lm_idx_sorted_per_mode, self.parent_j, self.aj,
                N_unique, Nj,
            )


            self._jit_all_i, self._jit_all_j, self._jit_all_G, self._jit_all_Lf, _ = gaunt_table

        else:

            # SphHT branch is fully sparse: (aj, parent_j, lm_idx_per_mode)
            # live on `self` directly. Gaunt-path placeholders still need to
            # be present because they appear in the JIT signature.
            self.a_u_j       = jnp.zeros((1, 1), dtype=self.aj.dtype)
            self._jit_all_i  = jnp.zeros(1, dtype=jnp.int32)
            self._jit_all_j  = jnp.zeros(1, dtype=jnp.int32)
            self._jit_all_G  = jnp.zeros(1, dtype=jnp.float64)
            self._jit_all_Lf = jnp.zeros(1, dtype=jnp.int32)


        # Pre-convert lm_pairs to numpy once so scipy sph_harm_y receives a
        # plain numpy array and avoids a GPU to CPU device transfer every sub-step.
        out_lm = [(L, M) for L in range(L_max_out) for M in range(-L, L+1)]
        output_lm_pairs = jnp.array(out_lm)

        self.output_lm_pairs = output_lm_pairs
        self.lm_pairs_np = np.array(output_lm_pairs)

        # Ramp phase: linearly switch on the off-diagonal cross-terms — i.e.
        # the time-dependent piece — over `ramp_time` so particles are not
        # abruptly exposed to the full fluctuating spectrum. After the ramp
        # the system evolves for `total_evolve_time` in the full ULDM
        # potential.
        ramp_time = self.ramp_time * self.u.from_Gyr
        self.n_ramp_steps = int(jnp.ceil(ramp_time / self.dt).item())

        # Update no_time_steps to include ramp steps on top of the original
        # main-phase steps.
        self.no_time_steps = self.n_ramp_steps + self.no_time_steps
        print(f"Ramp phase: {self.n_ramp_steps} steps "
              f"({self.n_ramp_steps * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Main phase: {self.no_time_steps - self.n_ramp_steps} steps "
              f"({(self.no_time_steps - self.n_ramp_steps) * self.dt * self.u.to_Gyr:.3f} Gyr)")
        print(f"Total: {self.no_time_steps} steps")


        # rho_static_r_l00 (the time-averaged diagonal monopole, used as the
        # ramp baseline) was precomputed in initialising_simulation;
        # Build_rho_lms_for_timestep needs it in self. Result is cast to
        # compute_dtype and sharded along L.
        self.rho_lms = self.sharding.shard_l_arr(self.Build_rho_lms_for_timestep(0).astype(self.compute_dtype))
        print('completed rho_lms precomputation')
        # Change 2: rho_lms_below / rho_lms_above are no longer pre-built.
        # The gather inside `insert_particle_rholm_and_get_philm` consumes
        # `self.rho_lms` directly, saving 2x the standing rho_lms memory.


        ###################################

        # if self.plot == True:

        #     def inverse_s2fft(single_rho_lm_r):
        #         return s2fft.inverse(single_rho_lm_r, self.L_max_out, sampling='mw', method='jax')

        #     rho_rtp = jax.vmap(inverse_s2fft)(self.rho_lms)  # (Nr, L, 2*L-1)


        #     import yt
        #     from yt.visualization.volume_rendering.api import (
        #         Scene, 
        #         Camera, 
        #         TransferFunctionHelper, 
        #         create_volume_source
        #     )

        #     def rho_rtp_to_cart(rho_rtp, r, theta, phi, Ncart=None):
        #         import numpy as np
        #         from scipy.interpolate import RegularGridInterpolator

        #         if Ncart is None:
        #             Ncart = len(r)

        #         r = np.asarray(r)
        #         theta = np.asarray(theta)
        #         phi = np.asarray(phi)
        #         rho_rtp = np.asarray(rho_rtp)

        #         r_max = r[-1]
        #         r_min = r[0]

        #         # Interpolate in log-r space since r is logspaced
        #         log_r = np.log10(r)
        #         interp = RegularGridInterpolator(
        #             (log_r, theta, phi), rho_rtp,
        #             bounds_error=False, fill_value=0.0
        #         )

        #         # Cartesian grid
        #         x = np.linspace(-r_max, r_max, Ncart)
        #         y = np.linspace(-r_max, r_max, Ncart)
        #         z = np.linspace(-r_max, r_max, Ncart)
        #         X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        #         # Cartesian -> spherical
        #         R = np.sqrt(X**2 + Y**2 + Z**2)
        #         Theta = np.arccos(np.clip(Z / np.clip(R, 1e-30, None), -1, 1))
        #         Phi = np.arctan2(Y, X) % (2 * np.pi)

        #         # Interpolate in log-r space
        #         log_R = np.log10(np.clip(R, r_min, None))
        #         pts = np.stack([log_R.ravel(), Theta.ravel(), Phi.ravel()], axis=-1)
        #         rho_xyz = interp(pts).reshape(X.shape)

        #         return rho_xyz, x, y, z

        #     rho_xyz, x, y, z = rho_rtp_to_cart(rho_rtp, self.r, self.theta, self.phi)


        #     ds = yt.load_uniform_grid(
        #     dict(density=np.asarray(rho_xyz) * float(self.u.to_Msun)/float(self.u.to_Kpc)**3),
        #     [1000,1000,1000],
        #     bbox=np.array([[-self.rmax, self.rmax], [-self.rmax, self.rmax], [-self.rmax, self.rmax]]) * float(self.u.to_Kpc),
        #     length_unit="kpc",
        #     mass_unit="Msun"
        #     )

        #     ds_section = ds.sphere(ds.domain_center,((self.rmax * self.u.to_Kpc).item(),"kpc"))
        #     sc = yt.create_scene(ds_section, ("stream", "density"), "perspective")
        #     source = sc.get_source()
        #     source.set_log(True)
        #     bounds=(1e-2, 3e5)
                
        #     tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=False)

        #     def quadramp(vals, minval, maxval):
        #         return ((vals - vals.min()) / (vals.max() - vals.min()))**0.5

        #     tf.map_to_colormap(
        #         np.log10(bounds[0]), np.log10(bounds[1]), 
        #         colormap="gist_stern", 
        #         scale_func=quadramp
        #     )

            
        #     tf.add_layers(8,
        #                 colormap="gist_stern", 
        #                 alpha=np.geomspace(1, 6, 8))

        #     source.tfh.tf = tf
        #     source.tfh.bounds = bounds

        #     camera = sc.camera
        #     camera.position = [1.,0,0]
        #     camera.resolution = (900,900)
        #     camera.zoom(1.)

        #     camera.switch_orientation()
        #     import matplotlib.pyplot as plt
        #     import matplotlib.colors as mcolors

        #     # Render the scene to an image array
        #     im = sc.render()

        #     # Plot with matplotlib so we can add a colorbar
        #     fig, ax = plt.subplots(1, 1, figsize=(9, 9))
        #     ax.imshow(im[:, :, :3] / im[:, :, :3].max(), origin="lower")
        #     ax.set_axis_off()

        #     # Add colorbar matching your transfer function bounds
        #     norm = mcolors.LogNorm(vmin=bounds[0], vmax=bounds[1])
        #     sm = plt.cm.ScalarMappable(cmap="gist_stern", norm=norm)
        #     sm.set_array([])
        #     cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        #     cbar.set_label(r"Density [M$_\odot$ / kpc$^3$]")

        #     plt.tight_layout()
        #     plt.show()

        ######################################


        # Compute initial potential energy for each particle
                
        r_pos_sphs = jnp.array([particle.r_pos_sph for particle in self.particles])  # (N_particles, 3)

        a_r, _, _, phi_lm_at_r, Y_lm_all = self.construct_acc_master_func(
            r_pos_sphs,
            self.autodiff_data['eval_library'],
            self.autodiff_data['eigenstate_lib'],
            poten = True
        )


        #v_circ_true = jnp.sqrt(jnp.abs(a_r) * r_pos_sphs[:, 0])

        phi_at_parts = jnp.sum(phi_lm_at_r * Y_lm_all.T, axis=1)  # (N_particles,)

        for i, particle in enumerate(self.particles):

            #v_old = particle.v
            #v_dir = v_old / jnp.linalg.norm(v_old)
            #v_new = v_dir * v_circ_true[i]
            #p = self.ps_step[i]
            #p.vx, p.vy, p.vz = float(v_new[0]), float(v_new[1]), float(v_new[2])
            particle.potential_energy.append(phi_at_parts[i].real)

            #particle.Change_to_new_vel(v_new)

        # Preallocate the per-particle velocity history array now that
        # no_time_steps is final (after dt_override + ramp adjustments).
        for particle in self.particles:
            particle.Create_V_array(self.no_time_steps)
        
        self.maximum_rho_00 = [jnp.max(jnp.abs(self.rho_lms[:, 0, self.L_max_out - 1]))]

        while self.time_step < self.no_time_steps:

            print(f"Time step {self.time_step + 1} / {self.no_time_steps}")


            # Cast to compute_dtype and (optionally) shard along L. The
            # gather-based insert in `insert_particle_rholm_and_get_philm`
            # consumes this directly — no rho_lms_below / rho_lms_above.
            # Free the previous step's rho_lms *before* building the new one —
            # otherwise the RHS allocates the full new (Nr, L, 2L-1) array
            # while the old one is still live, doubling peak memory.
            print('Building rho_lms for this timestep...')
            start = time()
            self.rho_lms = None
            self.rho_lms = self.sharding.shard_l_arr(
                self.Build_rho_lms_for_timestep(self.time_step).astype(self.compute_dtype)
            )
            end = time()
            print(f"rho_lms built in {end - start:.2f} seconds")

            self.maximum_rho_00.append(jnp.max(jnp.abs(self.rho_lms[:, 0, self.L_max_out - 1])))


            # Time step all particles (IAS15 calls additional_forces_step ~8× internally,
            # which loops over every particle each call)
            start = time()
            self.time_step_particle()
            end = time()
            print(f"Time stepping all particles completed in {end - start:.2f} seconds")


            self.current_phase = jnp.exp(-1j * self.eigen_energies * (self.time_step + 1) * self.dt / 1)

            # Compute phi at updated particle positions for potential energy tracking
            r_pos_sphs_new = jnp.array([p.r_pos_sph for p in self.particles])
            _, _, _, phi_lm_new, Ylm_new = self.construct_acc_master_func(
                r_pos_sphs_new,
                self.autodiff_data['eval_library'],
                self.autodiff_data['eigenstate_lib'],
                poten=True
            )

            

            phi_at_parts = jnp.sum(phi_lm_new * Ylm_new.T, axis=1).real  # (N_particles,)
            for i, particle in enumerate(self.particles):
                particle.potential_energy.append(float(phi_at_parts[i]))


            self.time_step += 1


    def run_simulation_profiled(self, time_output='profile_time.prof',
                                memory_output='profile_memory.txt', top_n=30):
        """Run `run_simulation` under cProfile (time) and tracemalloc (memory).

        cProfile output goes to `time_output` (binary). Inspect with e.g.
            python -m pstats profile_time.prof
            # or interactively: snakeviz profile_time.prof
        A console summary of the top `top_n` functions is printed for both
        cumulative-time and self-time sorts.

        tracemalloc takes a single snapshot at the end of the run and writes
        the top `top_n` allocation sites (by total size) to `memory_output`.
        Peak/current allocated bytes are also reported.

        Caveat for JAX: most heavy work here is JIT'd and dispatched
        asynchronously, so cProfile timings reflect Python-side overhead +
        time blocking on async results. For pure kernel timing, use
        `jax.profiler` or insert `jax.block_until_ready(...)` at key points.
        """
        import cProfile
        import pstats
        import tracemalloc

        tracemalloc.start()
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            self.run_simulation()
        finally:
            profiler.disable()
            snapshot = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        profiler.dump_stats(time_output)

        stats = pstats.Stats(profiler).strip_dirs()
        print(f"\n===== cProfile: top {top_n} by cumulative time =====")
        stats.sort_stats('cumulative').print_stats(top_n)
        print(f"\n===== cProfile: top {top_n} by self (tottime) =====")
        stats.sort_stats('tottime').print_stats(top_n)

        top_stats = snapshot.statistics('lineno')
        header = (f"tracemalloc: peak = {peak / 1e6:.2f} MB, "
                  f"current = {current / 1e6:.2f} MB")
        print(f"\n===== {header} =====")
        print(f"Top {top_n} allocation sites (by size):")
        with open(memory_output, 'w') as f:
            f.write(header + "\n\n")
            f.write(f"Top {top_n} allocation sites (by size):\n")
            for stat in top_stats[:top_n]:
                line = str(stat)
                print(line)
                f.write(line + "\n")

        print(f"\nFull cProfile data written to: {time_output}")
        print(f"Memory summary written to:     {memory_output}")
