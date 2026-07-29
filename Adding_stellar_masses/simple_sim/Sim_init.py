import os
import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import rebound

import Maths_funcs as MF
import Particles as Part



class SimInit:

    def __init__(self, m22, r_min, r_max_enclosing_frac, no_radius_bins):

        self.m22 = m22
        self.r_min = r_min
        self.r_max_enclosing_frac = r_max_enclosing_frac
        self.no_radius_bins = no_radius_bins

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "precomputed_wf")
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"m22_{float(self.m22):.6g}_rbins_{int(self.no_radius_bins)}"
        self.r_j_r_fname = os.path.join(cache_dir, f"precomputed_R_j_r_{cache_suffix}.npz")
        self.pkl_fname   = os.path.join(cache_dir, f"precomputed_objs_{cache_suffix}.pkl")

        self.cache_params = {
            'm22': float(self.m22),
            'r_min': float(self.r_min),
            'r_max_enclosing_frac': float(self.r_max_enclosing_frac),
            'no_radius_bins': int(self.no_radius_bins),
        }


    @staticmethod
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


    def Check_if_exists(self):
        if os.path.isfile(self.r_j_r_fname) and os.path.isfile(self.pkl_fname):
            data = np.load(self.r_j_r_fname)
            if self._cache_valid(data, self.cache_params):
                return True
        raise FileNotFoundError(f"Precomputed files {self.r_j_r_fname} and/or {self.pkl_fname} not found or invalid. Please run the precomputation script first.")


    def Load_files(self):
        print(f"Loading precomputed R_j_r from {self.r_j_r_fname}...")
        data = np.load(self.r_j_r_fname)
        self.rmin = data['rmin'].item()
        self.rmax = data['rmax'].item()
        self.l = data['l']
        self.aj_2 = data['aj_2']
        self.total_mass = data['total_mass'].item()
        R_j_r = data['R_j_r']

        # Eigen energies stay in float64 — they multiply by t at every timestep.
        self.eigen_energies = jnp.asarray(data['E'], dtype=jnp.float64)

        # Background radial grid the rest of the initialisation interpolates on.
        self.r = jnp.logspace(jnp.log10(self.rmin), jnp.log10(self.rmax), self.no_radius_bins)

        objs = pd.read_pickle(self.pkl_fname)
        self.radial_eigenmode_params = objs['eigenstate_lib'].radial_eigenmode_params

        self.R_j_r = R_j_r

        return R_j_r, self.radial_eigenmode_params

    def precompute_lm_pairs(self, l):

        l = np.asarray(l)
        n_j = l.shape[0]
        n_per_j = (2 * l + 1).astype(np.int64)
        total_k = int(n_per_j.sum())

        # j-index and l for every k-mode (each j contributes 2l+1 k-modes,
        # m = -l..l).
        ind_for_jmode_over_all_k = np.repeat(np.arange(n_j, dtype=np.int32), n_per_j)
        l_for_kmode = np.repeat(l, n_per_j).astype(np.int32)

        block_start = np.cumsum(n_per_j) - n_per_j
        within_block = np.arange(total_k, dtype=np.int64) - np.repeat(block_start, n_per_j)
        m_for_kmode = (within_block - l_for_kmode).astype(np.int32)

        # Unique (l, m) pairs only depend on which l values occur (<= l_max+1
        # of them), not on how many j's share each l, so build them directly
        # instead of de-duplicating the full total_k-length k-mode table.
        unique_l = np.unique(l).astype(np.int64)
        n_per_unique_l = 2 * unique_l + 1
        pair_block_start = np.cumsum(n_per_unique_l) - n_per_unique_l
        lm_pairs_l = np.repeat(unique_l, n_per_unique_l)
        within_pair_block = np.arange(lm_pairs_l.shape[0], dtype=np.int64) - np.repeat(pair_block_start, n_per_unique_l)
        lm_pairs_m = within_pair_block - lm_pairs_l
        lm_pairs_np = np.stack([lm_pairs_l, lm_pairs_m], axis=1).astype(np.int32)

        # Index into lm_pairs_np for every k-mode: that l's block start, plus
        # position within the block (m - (-l) = m + l).
        l_to_block_start = np.zeros(int(unique_l.max()) + 1, dtype=np.int64)
        l_to_block_start[unique_l] = pair_block_start
        lm_pairs_idx_for_kmode = (
            l_to_block_start[l_for_kmode] + (m_for_kmode.astype(np.int64) + l_for_kmode)
        ).astype(np.int32)

        L = int(l.max()) + 1
        L_max_out = 2 * L - 1
        n_theta = L_max_out
        n_phi = 2 * L_max_out - 1

        i = np.arange(n_theta)
        theta_np = (np.pi * (2 * i + 1)) / (2 * L_max_out - 1)
        j = np.arange(n_phi)
        phi_np = (2 * np.pi * j) / (2 * L_max_out - 1)

        return (jnp.asarray(ind_for_jmode_over_all_k), jnp.asarray(lm_pairs_np),
                jnp.asarray(l_for_kmode), jnp.asarray(m_for_kmode),
                jnp.asarray(theta_np), jnp.asarray(phi_np),
                jnp.asarray(lm_pairs_idx_for_kmode))


    def Truncating_L(self):

        print('l max from jaxsp:', max(self.l))
        L = int(max(self.l) + 1)
        self.L = L

        L_max_out_full = 2 * L - 1

        if self.L_out_frac < 1.0:
            L_max_out = int(round(self.L_out_frac * L_max_out_full))
            print(f"SphHT bandwidth truncated by L_out_frac={self.L_out_frac}: "
                  f"L_max_out = {L_max_out} (natural 2L-1 = {L_max_out_full}, floor L = {L})")
        else:
            L_max_out = L_max_out_full



        # L-sharding requires L_max_out divisible by the number of devices.
        if self.sharding.shard_l is not None:
            n_dev = len(self.sharding.devices)

            if L_max_out % n_dev != 0:
                L_aligned = (L_max_out // n_dev) * n_dev
                print(f"L_max_out {L_max_out} not divisible by {n_dev} devices; "
                      f"rounding down to {L_aligned} for L-sharding.")
                L_max_out = L_aligned

        self.L_max_out = L_max_out


    def Truncate_modes(self, L_max_out, L, parent_j, lm_l_per_mode, lm_m_per_mode, lm_idx_per_mode, lm_pairs):

        if L_max_out < L:
            lm_l_np = np.array(lm_l_per_mode)
            k_mask = lm_l_np < L_max_out          # bool over k-modes

            parent_j = parent_j[k_mask]
            lm_l_per_mode = lm_l_per_mode[k_mask]
            lm_m_per_mode = lm_m_per_mode[k_mask]
            lm_idx_old = np.array(lm_idx_per_mode)[k_mask]

            # Filter unique (l,m) pairs and rebuild the index mapping.
            lm_pairs_np = np.array(lm_pairs)
            pair_mask = lm_pairs_np[:, 0] < L_max_out
            lm_pairs = lm_pairs[pair_mask]
            remap = np.full(len(pair_mask), -1, dtype=np.int32)
            remap[np.where(pair_mask)[0]] = np.arange(int(pair_mask.sum()), dtype=np.int32)
            lm_idx_per_mode = jnp.array(remap[lm_idx_old], dtype=jnp.int32)

            # Recompute the MW grid for the truncated bandwidth.
            n_theta = L_max_out
            n_phi = 2 * L_max_out - 1
            theta = jnp.asarray((np.pi * (2 * np.arange(n_theta) + 1)) / n_phi)
            phi = jnp.asarray((2 * np.pi * np.arange(n_phi)) / n_phi)

            print(f"Mode mask (L_max_out={L_max_out} < L={L}): "
                    f"dropped {int((~k_mask).sum())} k-modes and "
                    f"{int((~pair_mask).sum())} unique (l,m) pairs with l >= {L_max_out}. "
                    f"Remaining: {len(parent_j)} k-modes, {lm_pairs.shape[0]} unique pairs.")

            self.theta = theta
            self.phi = phi

        self.parent_j = parent_j
        self.lm_l_per_mode = lm_l_per_mode
        self.lm_m_per_mode = lm_m_per_mode
        self.lm_idx_per_mode = lm_idx_per_mode
        self.lm_pairs = lm_pairs

        return parent_j, lm_l_per_mode, lm_m_per_mode, lm_idx_per_mode, lm_pairs

    def Setup_rebound(self):
        
        sim = rebound.Simulation()

        sim.integrator = "leapfrog"
        # dt isn't known yet - set_dt() computes it from the particles'
        # orbital periods, which don't exist until Particle_ICs runs after
        # this, and assigns it onto sim.dt directly once it does.

        # Live view onto sim.particles — reflects the particles added later in
        # Particle_ICs, so the force callback below can close over it now.
        sim_particles = sim.particles

        r_orbit_mean = self.r_half * self.u.from_Kpc
        self.r_orbit_min = r_orbit_mean - self.r_half_width/2 * self.u.from_Kpc
        self.r_orbit_max = r_orbit_mean + self.r_half_width/2 * self.u.from_Kpc

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
            r, theta, phi = MF.Cartesian_to_sph_np(x, y, z)

            positions_sph = jnp.asarray(np.stack([r, theta, phi], axis=1))

            self._force_call_count += 1

            # Single batched acceleration computation — parallel over all particles
            a_r_all, a_theta_all, a_phi_all = self.construct_acc_master_func(positions_sph)

            # Pull accs back to host once, then do the spherical->Cartesian
            # rotation batched in numpy.
            a_x, a_y, a_z = MF.acceleration_spherical_to_cartesian_np(
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


    def Particle_ICs(self):


        rho_diag = self.Rho_lm_builder.initialise()

        # Cumulative enclosed mass M_enc(r) on the radial grid; interpolated
        # per particle below. SSF.Enclosed_mass applies the 4π r² factor.
        M_enc_arr = MF.Enclosed_mass(self.r, rho_diag)

        self.particles = []


        r_orbit = jax.random.uniform(jax.random.PRNGKey(42), shape=(self.no_of_particles,), minval=self.r_orbit_min, maxval=self.r_orbit_max)


        X1 = jax.random.normal(jax.random.PRNGKey(43), shape=(self.no_of_particles,), dtype=jnp.float64)
        X2 = jax.random.normal(jax.random.PRNGKey(44), shape=(self.no_of_particles,), dtype=jnp.float64)
        X3 = jax.random.normal(jax.random.PRNGKey(45), shape=(self.no_of_particles,), dtype=jnp.float64)

        mag = jnp.sqrt(X1**2 + X2**2 + X3**2)

        # Particle-major (N, 3) layout throughout, so jnp.cross's default
        # last-axis convention and Add_particles_to_sim's init_pos[i]/init_vel[i]
        # indexing both apply per-particle rather than per-component.
        r_i_unit = jnp.stack([X1, X2, X3], axis=1) / mag[:, None]
        r_i = r_orbit[:, None] * r_i_unit

        #avoid degeneracy near z-axis
        ref = jnp.where(jnp.abs(r_i_unit[:, 2])[:, None] < 0.9,
                        jnp.array([0., 0., 1.]),
                        jnp.array([1., 0., 0.]))
        o_i_unit = jnp.cross(r_i_unit, ref)
        o_i_unit = o_i_unit / jnp.linalg.norm(o_i_unit, axis=1, keepdims=True)


        t_i_unit = jnp.cross(r_i_unit, o_i_unit)

        b_i_unit = jnp.cross(t_i_unit, r_i_unit)

        rand_theta = jax.random.uniform(jax.random.PRNGKey(46), shape=(self.no_of_particles,), minval=0.0, maxval=2 * jnp.pi,)

        v_i_unit = t_i_unit * jnp.sin(rand_theta)[:, None] + b_i_unit * jnp.cos(rand_theta)[:, None]

        # Compute circular velocity from spherically-averaged enclosed mass
        M_enc_at_r = jnp.interp(r_orbit, self.r, M_enc_arr)
        v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / r_orbit)

        init_pos = r_i
        init_vel = v_circ_mag[:, None] * v_i_unit

        return init_pos, init_vel
    
    def Add_particles_to_sim(self, init_pos, init_vel):

        self.init_vels = []

        v_mags = jnp.linalg.norm(init_vel, axis=1)

        for i in range(self.no_of_particles):

            self.init_vels.append(v_mags[i])

            print(f"Particle {i}: v_circ = {v_mags[i] * self.u.to_kms:.3f} km/s")

            particle = Part.Simulation_Particle(i, init_pos[i], init_vel[i], self.u)
            self.particles.append(particle)

            self.sim.add(
                m=0.0,
                x=float(init_pos[i, 0]), y=float(init_pos[i, 1]), z=float(init_pos[i, 2]),
                vx=float(init_vel[i, 0]), vy=float(init_vel[i, 1]), vz=float(init_vel[i, 2])
            )

        self.r_orbits = jnp.array([p.r_values[0] for p in self.particles])

        r_orbit_mean = jnp.mean(self.r_orbits)

        print(f"Mean r: {r_orbit_mean * self.u.to_Kpc:.3f} kpc")
    
    def Particle_ICs_Plummer(self, M_plummer, a_plummer):

        key1 = jax.random.PRNGKey(42)

        # Positions

        X = jax.random.uniform(key1, shape=(3, self.no_of_particles), minval=0.0, maxval=1.0)

        X1 = X[0]
        X2 = X[1]
        X3 = X[2]

        star_radii = a_plummer * (X1 ** (-2/3) - 1)**(-1/2)

        star_z = star_radii * (1 - 2 * X2)
        star_x = jnp.sqrt(star_radii**2 - star_z**2) * jnp.cos(2 * jnp.pi * X3)
        star_y = jnp.sqrt(star_radii**2 - star_z**2) * jnp.sin(2 * jnp.pi * X3)

        # Velocities

        v_esc = jnp.sqrt(2 * self.G * M_plummer / jnp.sqrt(star_radii**2 + a_plummer**2))

        get_x4 = []

        i = 0

        while len(get_x4) < self.no_of_particles:

            key2 = jax.random.PRNGKey(43 + i)

            X = jax.random.uniform(key2, shape=(2,self.no_of_particles - len(get_x4)), minval=0.0, maxval=1.0)

            X4 = X[0]
            X5 = X[1]

            def g(X):
                return X**2 * (1 - X**2)**(7/2)

            accepted = g(X4) > 0.1 * X5

            get_x4.append(X4[accepted])

            i += 1

        q = jnp.concatenate(get_x4)[:self.no_of_particles]
        v = q * v_esc

        key3 = jax.random.PRNGKey(44 + i)

        X = jax.random.uniform(key3, shape=(2,self.no_of_particles), minval=0.0, maxval=1.0)

        X6 = X[0]
        X7 = X[1]

        vel_z = v * (1 - 2 * X6)
        vel_x = jnp.sqrt(v**2 - vel_z**2) * jnp.cos(2 * jnp.pi * X7)
        vel_y = jnp.sqrt(v**2 - vel_z**2) * jnp.sin(2 * jnp.pi * X7)

        rho_diag = self.Rho_lm_builder.initialise()

        # Cumulative enclosed mass M_enc(r) on the radial grid; interpolated
        # per particle below. SSF.Enclosed_mass applies the 4π r² factor.
        M_enc_arr = MF.Enclosed_mass(self.r, rho_diag)

        M_enc_at_r = jnp.interp(star_radii, self.r, M_enc_arr)
        v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / star_radii)

        r_unit_vec = jnp.stack([star_x, star_y, star_z], axis=1) / star_radii[:, None]

        ref = jnp.where(jnp.abs(r_unit_vec[:, 2])[:, None] < 0.9,
                        jnp.array([0., 0., 1.]),
                        jnp.array([1., 0., 0.]))
        o_i_unit = jnp.cross(r_unit_vec, ref)
        o_i_unit = o_i_unit / jnp.linalg.norm(o_i_unit, axis=1, keepdims=True)

        t_i_unit = jnp.cross(r_unit_vec, o_i_unit)

        b_i_unit = jnp.cross(t_i_unit, r_unit_vec)

        rand_theta = jax.random.uniform(jax.random.PRNGKey(45 + i), shape=(self.no_of_particles,), minval=0.0, maxval=2 * jnp.pi,)

        v_i_unit = t_i_unit * jnp.sin(rand_theta)[:, None] + b_i_unit * jnp.cos(rand_theta)[:, None]

        init_pos = jnp.stack([star_x, star_y, star_z], axis=1)

        init_vel = v_circ_mag[:, None] * v_i_unit

        return init_pos, init_vel



    def set_dt(self):

        orbital_P = 2 * jnp.pi * self.r_orbits / jnp.array(self.init_vels)

        min_orbital_P = jnp.min(orbital_P)

        alive_j = jnp.unique(self.parent_j)          # eigenstates still in psi after the cut
        min_psi_t = jnp.min(-1 / self.eigen_energies[alive_j])


        print(f"Min T_orb: {min_orbital_P * self.u.to_Myr:.3f} Myr")
        print(f"T_c: {min_psi_t * self.u.to_Myr:.3f} Myr")

        new_dt_orb = min_orbital_P / self.dt_override

        new_dt_c = min_psi_t / self.dt_override

        new_dt = min(new_dt_orb, new_dt_c)

        self.sim.dt = float(new_dt)

        self.dt = new_dt

        self.no_time_steps = int(self.total_evolve_time * self.u.from_Gyr / new_dt)

        print(f"dt: {self.dt * self.u.to_Gyr:.3f} Gyr")
        print(f"Number of time steps: {self.no_time_steps}")
    
    def Number_of_ramp_steps(self):
        
        ramp_time = self.ramp_time * self.u.from_Gyr

        self.no_ramp_steps = int(ramp_time / self.dt)

        print(f"Ramp time: {self.ramp_time:.3f} Gyr, no_ramp_steps: {self.no_ramp_steps}")


    def Run_initialisation(self):

        self.Check_if_exists()

        self.R_j_r, self.radial_eigenmode_params = self.Load_files()

        (parent_j, lm_pairs, lm_l_per_mode, lm_m_per_mode, theta, phi, lm_idx_per_mode) = self.precompute_lm_pairs(self.l)

        self.parent_j = parent_j
        self.lm_pairs = lm_pairs
        self.lm_l_per_mode = lm_l_per_mode
        self.lm_m_per_mode = lm_m_per_mode
        self.lm_idx_per_mode = lm_idx_per_mode
        self.theta = theta
        self.phi = phi

        self.Truncating_L()

        self.Truncate_modes(self.L_max_out, self.L, self.parent_j, self.lm_l_per_mode, self.lm_m_per_mode, self.lm_idx_per_mode, self.lm_pairs)

        self.Setup_rebound()

        init_pos, init_vel = self.Particle_ICs()

        self.Add_particles_to_sim(init_pos, init_vel)


        self.set_dt()

        self.Number_of_ramp_steps()

        
