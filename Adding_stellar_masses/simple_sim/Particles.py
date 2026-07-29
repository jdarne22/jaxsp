import numpy as np
import Maths_funcs as MF


class Simulation_Particle:
    """
    Stores the state (position + velocity) and history for a single stellar particle.
    """

    def __init__(self, particle_id, init_pos_cart, init_vel_cart, u):
        
        # Particle ID
        self.id = particle_id

        # Units
        self.u = u

        # Starting Cartesian state
        self.r_pos = np.array(init_pos_cart)   
        self.v     = np.array(init_vel_cart)  

        # Convert to spherical for initial record
        self.r_pos_sph = MF.Cartesian_to_sph(self.r_pos[0], self.r_pos[1], self.r_pos[2])
        self.v_sph = MF.Cartesian_to_sph_vel(self.r_pos[0], self.r_pos[1], self.r_pos[2],self.v[0], self.v[1], self.v[2])

        # History of velocities, stellar dispersions, radii, positions, energies and angular momenta
        self.velocities      = [self.v_sph]
        self.velocities_cart = [self.v]
        self.stellar_v_disp = [0]
        self.r_values       = [float(self.r_pos_sph[0])]
        self.average_r      = [float(self.r_pos_sph[0])]
        self.positions_xyz  = [[float(self.r_pos[0]), float(self.r_pos[1]), float(self.r_pos[2])]]

        self.potential_energy = []
        self.kinetic_energy = [1/2 * np.sum(self.v**2)]
        self.ang_mom = [np.linalg.norm(np.cross(self.r_pos, self.v))]


        # Keep record of current timestep
        self.time_step = 0


    def Create_V_array(self, no_time_steps):
        """
        Preallocates the per-timestep velocity history array now that
        no_time_steps is known; row 0 is the initial v_sph, update_state
        writes each subsequent step in place at row time_step + 1.
        """
        self.velocities_arr = np.empty((no_time_steps + 1, 3))
        self.velocities_arr[0] = self.v_sph

    def update_state(self, new_pos_cart, new_vel_cart):
        """
        Called after each rebound integration step to update this particle's
        Cartesian and spherical state and append to history arrays.
        """
        
        x, y, z    = float(new_pos_cart[0]), float(new_pos_cart[1]), float(new_pos_cart[2])
        vx, vy, vz = float(new_vel_cart[0]), float(new_vel_cart[1]), float(new_vel_cart[2])

        self.r_pos = np.array([x, y, z])
        self.v     = np.array([vx, vy, vz])

        r, theta, phi      = MF.Cartesian_to_sph_np(x, y, z)
        vr, vtheta, vphi   = MF.Cartesian_to_sph_vel_np(x, y, z, vx, vy, vz)
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

