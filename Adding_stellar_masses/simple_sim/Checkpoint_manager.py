"""
Checkpoint_manager saves and restores the parts of the simulation state
that change every timestep: each particle's history arrays, the current
step count, and rebound's own clock. Everything that's fixed for the
whole run (wavefunction data, rho_lm setup, ...) is rebuilt from scratch
on resume instead of being checkpointed - it's cheap to recompute and
would be wasteful to serialise.
"""

import os
import pickle


class Checkpoint_manager:

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def save(self, particles, sim, time_step, no_time_steps, n_ramp_steps, final=False):
        """
        Saves every particle's full history plus the current step count
        and rebound's clock. final=True names the file distinctly, so a
        finished run's checkpoint is never mistaken for a mid-run one.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        file_name = 'checkpoint_final.pkl' if final else f'checkpoint_step_{time_step}.pkl'
        path = os.path.join(self.checkpoint_dir, file_name)

        state = {
            'time_step': time_step,
            'no_time_steps': no_time_steps,
            'n_ramp_steps': n_ramp_steps,
            'sim_time': float(sim.t),
            'particle_states': [self._particle_to_dict(p) for p in particles],
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        label = "Final checkpoint" if final else f"Checkpoint at step {time_step}"
        print(f"{label} saved: {path}", flush=True)

    def load(self, particles, sim):
        """
        Looks for the latest checkpoint in checkpoint_dir (preferring a
        final one, since that means the run already completed). If found,
        restores every particle and rebound's clock in place and returns
        (time_step, no_time_steps, n_ramp_steps). Returns None if no
        checkpoint exists, so the caller knows to start from step 0.
        """
        path = self._find_checkpoint_to_load()
        if path is None:
            return None

        print(f"Resuming from checkpoint: {path}", flush=True)
        with open(path, 'rb') as f:
            state = pickle.load(f)

        sim.t = state['sim_time']
        for particle, particle_state in zip(particles, state['particle_states']):
            self._restore_particle(particle, particle_state)

        return state['time_step'], state['no_time_steps'], state['n_ramp_steps']

    def _find_checkpoint_to_load(self):
        if not os.path.isdir(self.checkpoint_dir):
            return None

        final_path = os.path.join(self.checkpoint_dir, 'checkpoint_final.pkl')
        if os.path.isfile(final_path):
            print("Final checkpoint found - simulation already complete.", flush=True)
            return final_path

        step_checkpoints = sorted(
            (f for f in os.listdir(self.checkpoint_dir)
             if f.startswith('checkpoint_step_') and f.endswith('.pkl')),
            key=lambda f: int(f[len('checkpoint_step_'):-len('.pkl')])
        )
        if not step_checkpoints:
            return None

        return os.path.join(self.checkpoint_dir, step_checkpoints[-1])

    @staticmethod
    def _particle_to_dict(p):
        return {
            'r_pos': p.r_pos.copy(),
            'v': p.v.copy(),
            'r_pos_sph': p.r_pos_sph.copy(),
            'v_sph': p.v_sph.copy(),
            'r_values': list(p.r_values),
            'positions_xyz': list(p.positions_xyz),
            'velocities': list(p.velocities),
            'velocities_cart': list(p.velocities_cart),
            'velocities_arr': p.velocities_arr.copy(),
            'kinetic_energy': list(p.kinetic_energy),
            'potential_energy': list(p.potential_energy),
            'ang_mom': list(p.ang_mom),
            'stellar_v_disp': list(p.stellar_v_disp),
            'average_r': list(p.average_r),
            'time_step': p.time_step,
        }

    @staticmethod
    def _restore_particle(particle, state):
        particle.r_pos = state['r_pos']
        particle.v = state['v']
        particle.r_pos_sph = state['r_pos_sph']
        particle.v_sph = state['v_sph']
        particle.r_values = state['r_values']
        particle.positions_xyz = state['positions_xyz']
        particle.velocities = state['velocities']
        particle.velocities_cart = state['velocities_cart']
        particle.velocities_arr = state['velocities_arr']
        particle.kinetic_energy = state['kinetic_energy']
        particle.potential_energy = state['potential_energy']
        particle.ang_mom = state['ang_mom']
        particle.stellar_v_disp = state['stellar_v_disp']
        particle.average_r = state['average_r']
        particle.time_step = state['time_step']
