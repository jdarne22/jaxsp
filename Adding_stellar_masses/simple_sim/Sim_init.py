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

    def __init__(self, m22, r_half, r_min, r_max_enclosing_frac, no_radius_bins, gal_name, r_cut_kpc=None):

        self.m22 = m22
        self.r_half = r_half
        self.r_min = r_min
        self.r_max_enclosing_frac = r_max_enclosing_frac
        self.no_radius_bins = no_radius_bins

        # Outer radius, in kpc, beyond which the background rho_lm / phi_lm
        # grid is simply not built. None keeps the full r_max_enclosing_frac
        # grid. See Truncate_radial_grid for why this is cheap and safe.
        self.r_cut_kpc = r_cut_kpc

        self.gal_name = gal_name

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "precomputed_wf", self.gal_name)
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


    def Teodori_ICs(self, M_plummer, a_plummer):

        self.particles = []

        def sample_plummer_positions(a_plummer):

            key1 = jax.random.PRNGKey(42)

            # Positions

            X = jax.random.uniform(key1, shape=(3, self.no_of_particles), minval=0.0, maxval=1.0)

            X1 = X[0]
            X2 = X[1]
            X3 = X[2]

            # Truncated at the grid's outer radius: psi_r below is not tabulated
            # past it, and the untruncated Plummer tail runs to infinity. With
            # s = r / sqrt(r^2 + a^2) the enclosed mass fraction is just s^3.
            s_max = self.r[-1] / jnp.sqrt(self.r[-1]**2 + a_plummer**2)

            s = (X1 * s_max**3)**(1/3)

            star_radii = a_plummer * s / jnp.sqrt(1 - s**2)

            star_z = star_radii * (1 - 2 * X2)
            star_x = jnp.sqrt(star_radii**2 - star_z**2) * jnp.cos(2 * jnp.pi * X3)
            star_y = jnp.sqrt(star_radii**2 - star_z**2) * jnp.sin(2 * jnp.pi * X3)

            return star_x, star_y, star_z, star_radii

        def build_f1():

            # initialise() still has to run - it builds the amplitudes, the
            # radial mode table and the static background the ramp blends
            # against - but its return value, the time-averaged density, is
            # NOT the potential to invert in. The stars are integrated in the
            # monopole of the realised wavefunction at step 0, which differs
            # from the time average by the eigenmode interference terms; see
            # Rho_lm_Builder.initial_rho_monopole.
            self.Rho_lm_builder.initialise()

            rho_mono = self.Rho_lm_builder.initial_rho_monopole()

            # # Cumulative enclosed mass M_enc(r) on the radial grid; interpolated
            # # per particle below. SSF.Enclosed_mass applies the 4π r² factor.
            M_enc_arr = MF.Enclosed_mass(self.r, rho_mono)

            # Enclosed_mass starts its cumsum at self.r[0], so M_enc_arr[0] is
            # exactly zero and dpsi_dr[0] would divide by it. Fold the core in here
            # rather than at the particles: everything below runs off M_enc_arr.
            r_inner_edge = self.r[0]
            rho_core = rho_mono[0]
            M_core = 4/3 * jnp.pi * rho_core * r_inner_edge**3

            M_enc_arr = M_enc_arr + M_core

            M_tot = M_enc_arr[-1]

            # psi on the simulation grid, with psi(self.r[-1]) = 0.
            integrand_in = self.G * M_enc_arr / self.r**2

            dA_in = 0.5 * (integrand_in[1:] + integrand_in[:-1]) * jnp.diff(self.r)

            psi_in = jnp.concatenate([jnp.cumsum(dA_in[::-1])[::-1], jnp.array([0.0])])

            # --- Keplerian tail -------------------------------------------------
            # psi(self.r[-1]) = 0 is only the right zero point if self.r[-1] is
            # infinity. All the halo mass is inside it, so outside the potential
            # is Keplerian and psi(r_max) = G M_tot / r_max - at m22 = 10 that is
            # 20.87 against psi_max = 237.6, a 9% offset, not a constant that
            # cancels. Zeroing it declares every star with v > sqrt(2 psi(r))
            # unbound and drops it, which is exactly the mass the DF then cannot
            # put back: the round trip rho -> f -> rho came back 94% LOW at
            # a_plummer = 0.1, and that error is structural - refining the
            # quadrature to 128000 nodes does not move it.
            #
            # So tabulate out to where the Plummer tail is negligible, with
            # M_enc frozen at M_tot and rho_dm = 0 beyond the grid. This assumes
            # the halo is isolated out there, which is the same assumption the
            # psi -> 0 boundary condition already makes, only now applied
            # consistently. Stars are still SAMPLED only inside self.r[-1] -
            # the sim has no forces beyond it - this just fixes their energies.
            #
            # Costs nothing where the old convention was already adequate: at
            # a_plummer = 0.01 sigma_3D moves by +0.00% and no star gains enough
            # energy to leave the grid.
            s_out = (1.0 - 1e-10)**(1/3)
            r_out = jnp.maximum(a_plummer * s_out / jnp.sqrt(1 - s_out**2),
                                self.r[-1] * 10.0)

            r_tail = jnp.logspace(jnp.log10(self.r[-1] * 1.0001), jnp.log10(r_out), 2000)

            r_tab = jnp.concatenate([self.r, r_tail])

            M_tab = jnp.concatenate([M_enc_arr, jnp.full_like(r_tail, M_tot)])

            rho_bg = jnp.concatenate([rho_mono, jnp.zeros_like(r_tail)])

            psi_r = jnp.concatenate([psi_in + self.G * M_tot / self.r[-1],
                                     self.G * M_tot / r_tail])
            # --------------------------------------------------------------------

            dpsi_dr = - self.G * M_tab / r_tab**2

            drho_dr = -15 * M_plummer * r_tab / (4 * jnp.pi * a_plummer**5) * (1 + r_tab**2 / a_plummer**2)**(-7/2)

            drho_dpsi = drho_dr / dpsi_dr

            d2rho_dr2 = -15 * M_plummer / (4 * jnp.pi * a_plummer**5) * ((1 + r_tab**2 / a_plummer**2)**(-7/2) - r_tab * 7 * (1 + r_tab**2/a_plummer**2)**(-9/2) * r_tab/a_plummer**2)

            d2psi_dr2 = 2 * self.G * M_tab / r_tab**3 - 4 * jnp.pi * self.G * rho_bg

            d2rho_dpsi2 = (d2rho_dr2 * dpsi_dr - drho_dr * d2psi_dr2) / dpsi_dr**3

            # Tabulate f on psi(r) itself, NOT on logspace(log10(psi_max) - 8,
            # log10(psi_max), 1000). The stars sit deep inside the soliton core
            # where psi is nearly flat - at m22 = 10, a_plummer = 0.01, the whole
            # region r < a_plummer spans psi = 236.16 .. 237.59, i.e. 0.0026 dex,
            # while a log grid of 1000 nodes over 8 decades steps 0.008 dex. That
            # is ONE node across the radii holding a third of the stellar mass,
            # and the round trip rho -> f -> rho came back 222% high there.
            # Matching the eps nodes to the radial grid puts them where psi
            # actually has structure and makes the interp below exact at the
            # nodes; it beats brute-forcing the log grid to 64000 nodes (0.08%
            # vs 0.47% error) without the 512 MB (n_eps, n_u) array.
            #
            # psi_r is strictly decreasing - integrand = G M_enc / r^2 > 0 - so
            # the reverse is strictly increasing and needs no unique() (which
            # would make the shape dynamic). Drop psi_r[-1], which is exactly 0:
            # f ~ eps^(-1/2) diverges there and the velocity sampler takes
            # log(epsilons).
            epsilons = psi_r[::-1][1:]

            # psi_r decreases outwards, jnp.interp needs an ascending abscissa.
            psi_asc = psi_r[::-1]
            d2rho_dpsi2_asc = d2rho_dpsi2[::-1]

            # Eq. (A5)'s Q = sqrt(E - psi) removes the 1/sqrt(E - psi) singularity;
            # writing Q = sqrt(E) * u then puts every energy on the same u nodes, so
            # the whole inversion is one (n_eps, n_u) array rather than a per-energy
            # integral with its own upper limit.
            #
            # u is clustered as t^2, NOT uniform. The integrand is sampled at
            # eps (1 - u^2), so a uniform u steps 1 - u^2 by only ~1/n_u^2 = 1e-6
            # near u = 0 - and the stars sit where psi is flat, so the whole
            # stellar body can span less than that in relative psi and simply not
            # be resolved. At a_plummer = 5e-5 the round trip was 94% off with a
            # uniform grid; t^2 brings it to 0.19% at the same 1000 nodes, which
            # is what a uniform grid needs 16000 nodes to reach. Identical to
            # four decimals wherever the uniform grid was already adequate.
            u_q = jnp.linspace(0.0, 1.0, 1000)**2

            integrand_Q = jnp.interp(epsilons[:, None] * (1 - u_q[None, :]**2), psi_asc, d2rho_dpsi2_asc)

            dA_Q = 0.5 * (integrand_Q[:, 1:] + integrand_Q[:, :-1]) * jnp.diff(u_q)

            second_term = jnp.sqrt(epsilons) * jnp.sum(dA_Q, axis=1)

            # drho_dpsi[-1] is the boundary term, now at r_tab[-1] where the
            # Plummer tail has 1e-10 of its mass left outside.
            f_eps = (drho_dpsi[-1] / jnp.sqrt(epsilons) + 2 * second_term) / (jnp.sqrt(8.0) * jnp.pi**2)

            # A NEGATIVE f is not always quadrature noise. Where it appears at
            # high eps - the most bound orbits, i.e. the core - it is Eddington's
            # non-negativity condition failing: no isotropic DF reproduces this
            # (rho_star, psi) pair at all, and no grid or precision fixes that.
            # For a Plummer tracer as extended as the halo it is emphatic: at
            # a_plummer = 0.6, 622 of 2999 nodes go negative, reaching
            # eps / psi_max = 1. Clipping silently turned "these ICs are
            # impossible" into "here are some quietly wrong ICs", so refuse them
            # instead. Only the far tail, eps < 1e-2 psi_max, is treated as noise.
            core_negative = (f_eps < 0) & (epsilons > 1e-2 * epsilons[-1])

            if bool(jnp.any(core_negative)):
                worst = float(jnp.max(epsilons[core_negative]) / epsilons[-1])
                raise ValueError(
                    f"No isotropic distribution function exists for this tracer in "
                    f"this potential: the Eddington f(eps) is negative at "
                    f"{int(core_negative.sum())} of {epsilons.size} energies, up to "
                    f"eps / psi_max = {worst:.3g}. This is Eddington's non-negativity "
                    f"condition failing, not a resolution problem - refining the "
                    f"quadrature will not help. a_plummer = {a_plummer} is too "
                    f"extended for this halo; reduce it, or move to an anisotropic "
                    f"DF (Osipkov-Merritt) which has the freedom to realise it."
                )

            # Far-tail noise only, by the check above.
            f_eps = jnp.clip(f_eps, 0.0, None)

            return r_tab, psi_r, epsilons, f_eps

        def sample_plummer_velocities(star_radii, r_tab, psi_r, epsilons, f_eps):

            psi_star = jnp.interp(star_radii, r_tab, psi_r)

            v_esc = jnp.sqrt(2 * psi_star)

            # One shared w = v / v_esc grid, so every star's CDF sits on the same
            # abscissa and the inversion is a single batched interp. Clustered as
            # t^2 for the same reason as u_q above: eps_w = psi_star (1 - w^2)
            # has to resolve the top of f, where the stars are.
            w = jnp.linspace(0.0, 1.0, 1000)**2

            eps_w = psi_star[:, None] * (1 - w[None, :]**2)

            # f spans many decades and epsilons is log-spaced, so interpolate in log.
            f_w = jnp.exp(jnp.interp(jnp.log(jnp.clip(eps_w, epsilons[0], epsilons[-1])),
                                    jnp.log(epsilons), jnp.log(jnp.clip(f_eps, 1e-300, None))))

            # Eq. (A7). This has to be a cumulative INTEGRAL, not jnp.cumsum of
            # the integrand: with the uniform w this used to be, the constant dw
            # cancelled against the normalisation below and the bare cumsum was
            # right, but w is clustered now and dropping dw biases the CDF
            # towards the dense end (it cost 13% in sigma_3D at a_plummer = 0.01).
            integrand_w = f_w * w[None, :]**2

            dcdf = 0.5 * (integrand_w[:, 1:] + integrand_w[:, :-1]) * jnp.diff(w)

            cdf = jnp.concatenate([jnp.zeros((integrand_w.shape[0], 1)),
                                   jnp.cumsum(dcdf, axis=1)], axis=1)

            cdf = cdf / cdf[:, -1:]

            key2 = jax.random.PRNGKey(43)

            X = jax.random.uniform(key2, shape=(3, self.no_of_particles), minval=0.0, maxval=1.0)

            X4 = X[0]
            X5 = X[1]
            X6 = X[2]

            # vmapped rather than batched because the abscissa differs row to row.
            v = jax.vmap(lambda c, q: jnp.interp(q, c, w))(cdf, X4) * v_esc

            # beta0 = 0 collapses Eqs. (A8)-(A10) to an isotropic direction,
            # independent of r_hat.
            vel_z = v * (1 - 2 * X5)
            vel_x = jnp.sqrt(v**2 - vel_z**2) * jnp.cos(2 * jnp.pi * X6)
            vel_y = jnp.sqrt(v**2 - vel_z**2) * jnp.sin(2 * jnp.pi * X6)

            return vel_x, vel_y, vel_z

        star_x, star_y, star_z, star_radii = sample_plummer_positions(a_plummer)

        r_tab, psi_r, epsilons, f_eps = build_f1()

        vel_x, vel_y, vel_z = sample_plummer_velocities(star_radii, r_tab, psi_r, epsilons, f_eps)

        # The Kepler tail gives stars near the grid edge a real, non-zero escape
        # speed, so unlike under psi(r_max) = 0 some can now be energetic enough
        # to leave self.r[-1] - where there is no tabulated force. Sampling still
        # truncates positions at the grid, so this is only a warning, but a large
        # count means a_plummer is pushing past what the grid can integrate.
        v_sq = vel_x**2 + vel_y**2 + vel_z**2
        psi_edge = jnp.interp(self.r[-1], r_tab, psi_r)
        n_escaping = int(jnp.sum(0.5 * v_sq > jnp.interp(star_radii, r_tab, psi_r) - psi_edge))

        if n_escaping:
            print(f"Warning: {n_escaping} of {self.no_of_particles} stars have enough "
                  f"energy to pass r_max = {float(self.r[-1]):.4g}, where no forces are "
                  f"tabulated. Consider reducing a_plummer = {a_plummer}.")

        init_pos = jnp.stack([star_x, star_y, star_z], axis=1)

        init_vel = jnp.stack([vel_x, vel_y, vel_z], axis=1)

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
        a_plummer = self.r_half * 0.7 * self.u.from_Kpc

        init_pos, init_vel = self.Teodori_ICs(M_plummer, a_plummer)

        self.Add_particles_to_sim(init_pos, init_vel)

        self.set_dt()

        self.Number_of_ramp_steps()

        
