import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import rebound

import Maths_funcs as MF
import Memory_speed_savers as MSS
import Particles as Part


class SimInit:

    def __init__(self, m22, r_min, r_max_enclosing_frac, no_radius_bins, r_cut_kpc=None):

        self.m22 = m22
        self.r_min = r_min
        self.r_max_enclosing_frac = r_max_enclosing_frac
        self.no_radius_bins = no_radius_bins

        # Outer radius, in kpc, beyond which the background rho_lm / phi_lm
        # grid is simply not built. None keeps the full r_max_enclosing_frac
        # grid. See Truncate_radial_grid for why this is cheap and safe.
        self.r_cut_kpc = r_cut_kpc

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

        # Unpickling a jax.Array is a device_put to the *default* device, so
        # every leaf of the eigenstate library would land on GPU 0. At
        # m22 = 100 the two spline tables are float64 and 23.4 GB each - 47 GB
        # on one GPU, before Prune_radial_modes has dropped the 40% of radial
        # modes the L cut kills, and none of it is ever freed back to the
        # driver because XLA_PYTHON_CLIENT_PREALLOCATE is false. Read it onto
        # the CPU device instead: the prune below is a host-side gather
        # anyway, so only the pruned, sharded result needs to reach a GPU.
        with jax.default_device(jax.devices('cpu')[0]):
            objs = pd.read_pickle(self.pkl_fname)
        self.radial_eigenmode_params = objs['eigenstate_lib'].radial_eigenmode_params

        self.R_j_r = R_j_r

        # Drop the outer radii before anything downstream sizes itself off
        # len(self.r) / R_j_r.shape[0].
        self.Truncate_radial_grid()

        return self.R_j_r, self.radial_eigenmode_params

    def Truncate_radial_grid(self):
        """
        Cuts the background radial grid (and R_j_r with it) at r_cut_kpc.
        """

        if self.r_cut_kpc is None:
            print(f"No r_cut set - keeping the full radial grid out to {self.rmax * self.u.to_Kpc:.3g} kpc ({self.no_radius_bins} bins).")
            self.n_radii = int(self.r.shape[0])
            return

        r_cut = float(self.r_cut_kpc) * self.u.from_Kpc

        if r_cut <= self.rmin:
            raise ValueError(
                f"r_cut_kpc={self.r_cut_kpc} is at or inside the grid's inner radius "
                f"({self.rmin * self.u.to_Kpc:.3g} kpc) - nothing would be left.")

        if r_cut >= self.rmax:
            print(f"r_cut_kpc={self.r_cut_kpc} is beyond the grid's outer radius ({self.rmax * self.u.to_Kpc:.3g} kpc) - no truncation applied.")
            self.n_radii = int(self.r.shape[0])
            return

        r_np = np.asarray(self.r)
        n_keep = int(np.searchsorted(r_np, r_cut, side='right'))

        # build_sphht_rho_lms_jit pads the radial axis up to a whole number of
        # r_chunk_size chunks anyway, so rounding up to that boundary buys a
        # few extra *real* radii for exactly the memory and compute the
        # padding would otherwise waste on zero rows.
        r_chunk = int(getattr(self, 'r_chunk_size', 0) or 0)
        if r_chunk > 0:
            n_keep = int(np.ceil(n_keep / r_chunk) * r_chunk)

        n_keep = int(np.clip(n_keep, 2, r_np.shape[0]))

        self.r = self.r[:n_keep]
        # Materialise the slice so the full-length array can be freed rather
        # than kept alive by a view.
        self.R_j_r = np.ascontiguousarray(np.asarray(self.R_j_r)[:n_keep])
        self.n_radii = n_keep

        print(f"Radial grid truncated at r_cut = {self.r_cut_kpc:g} kpc: "
              f"{n_keep} / {r_np.shape[0]} bins kept "
              f"({100.0 * n_keep / r_np.shape[0]:.1f}%), "
              f"outermost radius now {float(self.r[-1]) * self.u.to_Kpc:.3g} kpc "
              f"(full grid reached {self.rmax * self.u.to_Kpc:.3g} kpc).")

    def Prune_radial_modes(self):
        """
        Drops every radial eigenmode j whose angular momentum l_j is at or
        above L_max_out.
        """

        l_full = np.asarray(self.l)
        n_full = int(l_full.shape[0])

        keep = np.flatnonzero(l_full < self.L_max_out)
        n_keep = int(keep.shape[0])

        if n_keep == 0:
            raise ValueError(
                f"Radial-mode pruning would drop every mode: L_max_out={self.L_max_out} "
                f"is at or below the smallest l in the library ({int(l_full.min())}).")

        dropping_modes = n_keep != n_full

        if dropping_modes:
            self.l = l_full[keep]
            self.aj_2 = np.asarray(self.aj_2)[keep]
            self.eigen_energies = self.eigen_energies[keep]
            self.R_j_r = np.ascontiguousarray(np.asarray(self.R_j_r)[:, keep])
        else:
            print(f"Radial-mode pruning: nothing to drop, all {n_full} modes have l < L_max_out.")

        # Every leaf of radial_eigenmode_params is indexed by radial mode on
        # its leading axis - that is exactly what R_j_at_radii's
        # vmap(..., in_axes=(None, 0)) maps over - so pruning them all keeps
        # the whole (nested) NamedTuple consistent.
        #
        # The library arrives on the host (see Load_files), so the gather runs
        # here and only the kept modes are uploaded - sharded across the
        # devices on that same radial-mode axis, so each one evaluates its own
        # modes. At m22 = 100 that is 28 GB of float64 spline tables split two
        # ways rather than all of it on GPU 0. This loop runs even when
        # nothing is dropped, because it is also what gets the library off the
        # CPU device in the first place.
        #
        # Upload first, delete second: on the CPU backend np.asarray(leaf) can
        # alias the array's own buffer, so releasing the leaf any earlier
        # would leave the gather - or the device_put - reading freed memory.
        # The old order (delete, then allocate) was there to keep the *device*
        # peak down, which no longer applies now that the source is host RAM.
        leaves, treedef = jax.tree_util.tree_flatten(self.radial_eigenmode_params)
        pruned_leaves = []
        for leaf in leaves:
            host_leaf = np.asarray(leaf)
            if dropping_modes:
                host_leaf = host_leaf[keep]
            pruned_leaves.append(self.sharding.shard_leading_axis_arr(host_leaf))
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
                leaf.delete()
            del host_leaf
        # Safe to drop the old tuple only now: nothing else holds a reference
        # to it (Load_files' `objs` is local and already out of scope).
        self.radial_eigenmode_params = jax.tree_util.tree_unflatten(treedef, pruned_leaves)

        if dropping_modes:
            print(f"Radial-mode pruning (l >= L_max_out={self.L_max_out}): "
                  f"dropped {n_full - n_keep} of {n_full} radial modes "
                  f"({100.0 * (n_full - n_keep) / n_full:.1f}%), {n_keep} kept.")

        # The single largest thing on the GPUs, so say where it landed - a
        # silent fallback to one device is what an OOM later looks like.
        leaf_bytes = sum(leaf.nbytes for leaf in pruned_leaves)
        n_dev = len(pruned_leaves[0].sharding.device_set)
        placement = (f"{n_dev}-way sharded on the radial-mode axis, "
                     f"{leaf_bytes / n_dev / 1e9:.2f} GB per device"
                     if n_dev > 1 else "on a single device")
        print(f"Eigenmode library uploaded: {leaf_bytes / 1e9:.2f} GB, {placement}.")

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


    def Particle_ICs_Massless(self):


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

            print(f"Particle {i}: v_circ = {v_mags[i] * self.u.to_kms:.6f} km/s")

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

        self.particles = []

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

        # v_esc = jnp.sqrt(2 * self.G * M_plummer / jnp.sqrt(star_radii**2 + a_plummer**2))

        # get_x4 = []

        # i = 0

        # while len(get_x4) < self.no_of_particles:

        #     key2 = jax.random.PRNGKey(43 + i)

        #     X = jax.random.uniform(key2, shape=(2,self.no_of_particles - len(get_x4)), minval=0.0, maxval=1.0)

        #     X4 = X[0]
        #     X5 = X[1]

        #     def g(X):
        #         return X**2 * (1 - X**2)**(7/2)

        #     accepted = g(X4) > 0.1 * X5

        #     get_x4.append(X4[accepted])

        #     i += 1

        # q = jnp.concatenate(get_x4)[:self.no_of_particles]
        # v = q * v_esc

        i = 0
        v = jnp.ones(self.no_of_particles)

        key3 = jax.random.PRNGKey(44 + i)

        X = jax.random.uniform(key3, shape=(2,self.no_of_particles), minval=0.0, maxval=1.0)

        X6 = X[0]
        X7 = X[1]

        vel_z = v * (1 - 2 * X6)
        vel_x = jnp.sqrt(v**2 - vel_z**2) * jnp.cos(2 * jnp.pi * X7)
        vel_y = jnp.sqrt(v**2 - vel_z**2) * jnp.sin(2 * jnp.pi * X7)

        init_vels = jnp.stack([vel_x, vel_y, vel_z], axis=1)

        init_vels_unit = init_vels / jnp.linalg.norm(init_vels, axis=1, keepdims=True)

        rho_diag = self.Rho_lm_builder.initialise()

        # # Cumulative enclosed mass M_enc(r) on the radial grid; interpolated
        # # per particle below. SSF.Enclosed_mass applies the 4π r² factor.
        M_enc_arr = MF.Enclosed_mass(self.r, rho_diag)

        r_inner_edge = self.r[0]
        rho_core = rho_diag[0]
        M_core = 4/3 * jnp.pi * rho_core * r_inner_edge**3

        M_enc_at_r = jnp.interp(star_radii, self.r, M_enc_arr + M_core)
        M_enc_at_r = jnp.where(star_radii < r_inner_edge,
                               4/3 * jnp.pi * rho_core * star_radii**3,
                               M_enc_at_r)

        v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / star_radii)

        init_vel = v_circ_mag[:, None] * init_vels_unit

        init_pos = jnp.stack([star_x, star_y, star_z], axis=1)

        # rho_diag = self.Rho_lm_builder.initialise()

        # # Cumulative enclosed mass M_enc(r) on the radial grid; interpolated
        # # per particle below. SSF.Enclosed_mass applies the 4π r² factor.
        # M_enc_arr = MF.Enclosed_mass(self.r, rho_diag)

        # M_enc_at_r = jnp.interp(star_radii, self.r, M_enc_arr)
        # v_circ_mag = jnp.sqrt(self.G * M_enc_at_r / star_radii)

        # r_unit_vec = jnp.stack([star_x, star_y, star_z], axis=1) / star_radii[:, None]

        # ref = jnp.where(jnp.abs(r_unit_vec[:, 2])[:, None] < 0.9,
        #                 jnp.array([0., 0., 1.]),
        #                 jnp.array([1., 0., 0.]))
        # o_i_unit = jnp.cross(r_unit_vec, ref)
        # o_i_unit = o_i_unit / jnp.linalg.norm(o_i_unit, axis=1, keepdims=True)

        # t_i_unit = jnp.cross(r_unit_vec, o_i_unit)

        # b_i_unit = jnp.cross(t_i_unit, r_unit_vec)

        # rand_theta = jax.random.uniform(jax.random.PRNGKey(1000), shape=(self.no_of_particles,), minval=0.0, maxval=2 * jnp.pi,)

        # v_i_unit = t_i_unit * jnp.sin(rand_theta)[:, None] + b_i_unit * jnp.cos(rand_theta)[:, None]

        # init_pos = jnp.stack([star_x, star_y, star_z], axis=1)

        # init_vel = v_circ_mag[:, None] * v_i_unit

        return init_pos, init_vel



    def set_dt(self):

        init_vels = jnp.asarray(self.init_vels)

        v_max = jnp.max(init_vels)

        # The de Broglie wavelength is therefore just 2*pi/v — m22 is already absorbed into
        # the unit system and must not appear again here.
        lambda_db = 2 * jnp.pi / jnp.mean(init_vels)

        # Time for the fastest particle to cross one granule.
        t_cross = lambda_db / v_max

        # Fastest beat in psi: the largest pairwise |E_j - E_k| over the
        # eigenstates still alive after the L cut, which is just the spectral
        # range of that subset. Prune_radial_modes has already dropped
        # everything above the cut, so that is every mode still here.
        energies = jnp.asarray(self.eigen_energies)
        max_dE = jnp.max(energies) - jnp.min(energies)
        t_beat = 2 * jnp.pi / max_dE

        print(f"lambda_dB crossing time: {t_cross * self.u.to_Myr:.3f} Myr")
        print(f"Fastest beat period T_c: {t_beat * self.u.to_Myr:.3f} Myr")

        new_dt = jnp.minimum(t_cross, t_beat) / self.dt_override

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

        # Truncating_L reads the *original* l max to set L_max_out, so it has
        # to run before the radial modes are pruned against that cut.
        self.Truncating_L()

        # Has to run before lm_pairs is built, so the (l, m) table only covers
        # the l values that actually survive the cut.
        self.Prune_radial_modes()

        self.lm_pairs = jnp.asarray(MSS.build_lm_pairs(self.l))
        print(f"(l, m) pairs to solve for: {self.lm_pairs.shape[0]} "
              f"(l = 0 .. {int(self.L_max_out) - 1})")

        self.Setup_rebound()

        #init_pos, init_vel = self.Particle_ICs_Massless()

        M_plummer = 1e6 * self.u.from_Msun
        a_plummer = 0.0385 * self.u.from_Kpc

        init_pos, init_vel = self.Particle_ICs_Plummer(M_plummer, a_plummer)

        self.Add_particles_to_sim(init_pos, init_vel)

        self.set_dt()

        self.Number_of_ramp_steps()

        
