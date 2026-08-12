import numpy as np




class Dt_manager:
    def __init__(self, sim, evolve_time_left, dt_override, m22, u, init_vels, parent_j, eigen_energies):
        self.sim = sim
        self.evolve_time_left = evolve_time_left
        self.dt_override = dt_override
        self.m22 = m22
        self.u = u
        self.init_vels = init_vels
        self.parent_j = parent_j
        self.eigen_energies = eigen_energies  
    
    def new_dt(self):

            V0 = np.mean(self.init_vels)

            lambda_db = 1.0 / (self.m22 * V0)

            min_dt_orbit = lambda_db / np.max(self.init_vels)

            #min_orbital_P = jnp.min(orbital_P)

            alive_j = np.unique(self.parent_j)          # eigenstates still in psi after the cut
            dE = np.abs(self.eigen_energies[alive_j][None, :] - self.eigen_energies[alive_j][:, None])
            max_dE = np.max(dE)
            min_psi_t = 2 * np.pi / max_dE

            print(f"Min T_orb: {min_dt_orbit * self.u.to_Myr:.3f} Myr")
            print(f"T_c: {min_psi_t * self.u.to_Myr:.3f} Myr")

            new_dt_orb = min_dt_orbit / self.dt_override

            new_dt_psi = min_psi_t / self.dt_override

            new_dt = min(new_dt_orb, new_dt_psi)

            self.sim.dt = float(new_dt)

            self.dt = new_dt

            self.no_time_steps = int(self.evolve_time_left * self.u.from_Gyr / new_dt)

            print(f"dt: {self.dt * self.u.to_Gyr:.3f} Gyr")
            print(f"Number of time steps left: {self.no_time_steps}")

    