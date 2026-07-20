import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"   # or: "cuda_async"

#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"


import os
import pickle
from time import time

from Sharding_manager import ShardingManager

import Stellar_sim_funcs as SSF

import importlib

import jaxsp as jsp

import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp


from scipy.special import sph_harm_y

from collections import defaultdict

import gc

import gaunt_funcs as gf

importlib.reload(SSF)
importlib.reload(gf)



sm = ShardingManager(use_multi_gpu=True)

m22_list = [80, 85, 90, 95, 100]
R_bins = [1000]


for m22 in m22_list:
    for R_bin in R_bins:

        print(f"Precomputing for m22={m22}, R_bin={R_bin}...")

        u = jsp.set_schroedinger_units(m22)

        r_min = 20
        r_max_enclosing_frac = 0.99


        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_wf")
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"m22_{float(m22):.6g}_rbins_{int(R_bin)}"
        r_j_r_fname = os.path.join(cache_dir, f"precomputed_R_j_r_{cache_suffix}.npz")
        pkl_fname   = os.path.join(cache_dir, f"precomputed_objs_{cache_suffix}.pkl")
        y_lm_fname  = os.path.join(cache_dir, f"precomputed_Y_lm_{cache_suffix}.npz")

        cache_params = {
            'm22': float(m22),
            'r_min': float(r_min),
            'r_max_enclosing_frac': float(r_max_enclosing_frac),
            'no_radius_bins': int(R_bin),
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

        Compute = True

        if os.path.isfile(r_j_r_fname):
            data = np.load(r_j_r_fname)
            if _cache_valid(data, cache_params) and 'l' in data.files:
                print(f"Cache already v2 for m22={m22}, R_bin={R_bin}; skipping.")
                Compute = False
            elif _cache_valid(data, cache_params):
                print(f"Cache is v1 (float64, no embedded arrays); recomputing as v2...")
            else:
                print(f"Cached {r_j_r_fname} stale (parameter mismatch); recomputing.")

        if Compute:

            cNFWtides_params = jnp.array([
            357964808.148399 * u.from_Msun,
            25.690207,
            0.407461,
            0.012670 * u.from_Kpc,
            1.857991 * u.from_Kpc,
            3.729259
            ])

            print("Initializing density parameters...")

            density_params = jsp.init_core_NFW_tides_params_from_sample(cNFWtides_params)

            N = 512
            rmin = .1 * u.from_pc
            rmax = jsp.enclosing_radius(0.999, density_params)

            print("Initializing potential parameters...")
            potential_params = jsp.init_potential_params(density_params, rmin, rmax, N)



            eval_library = jax.vmap(jax.vmap(jsp.eval_radial_eigenmode, in_axes=(None, 0)), in_axes=(0,None))

            N = 1024
            a = 1
            b = 10

            rmax = jsp.enclosing_radius(r_max_enclosing_frac, density_params)

            print("Initializing eigenstate library...")
            cpu = jax.devices('cpu')[0]
            with jax.default_device(cpu):
                eigenstate_lib = jsp.init_eigenstate_library(potential_params, rmin, rmax, a, b, N)
            print(f"  eigenstate_lib.J={eigenstate_lib.J}, J % {len(jax.devices())} devices = {eigenstate_lib.J % len(jax.devices())}")

            rmin = r_min * u.from_pc

            tol = 1e-7
            print("Initializing wavefunction parameters...")
            wavefunction_params = jsp.init_wavefunction_params(eigenstate_lib, density_params, rmin, rmax, tol, sharding_manager=sm)


            r = jnp.logspace(jnp.log10(rmin), jnp.log10(rmax), R_bin)

            Nj = eigenstate_lib.J
            chunk = 256
            R_j_r_chunks = []
            for i in range(0, Nj, chunk):
                chunk_result = eval_library(r, jax.tree.map(lambda x: x[i:i+chunk], eigenstate_lib.radial_eigenmode_params))
                R_j_r_chunks.append(np.array(chunk_result))

            R_j_r = np.concatenate(R_j_r_chunks, axis=1)

            l          = np.array(eigenstate_lib.radial_eigenmode_params.l)
            E          = np.array(eigenstate_lib.radial_eigenmode_params.E)
            aj_2       = np.array(wavefunction_params.aj_2)
            total_mass = float(wavefunction_params.total_mass)

            np.savez(r_j_r_fname,
                     R_j_r=R_j_r.astype(np.float32),
                     l=l, E=E, aj_2=aj_2, total_mass=np.float64(total_mass),
                     rmin=rmin, rmax=rmax, **cache_params)
            with open(pkl_fname, 'wb') as f:
                pickle.dump({'eigenstate_lib': eigenstate_lib, 'wavefunction_params': wavefunction_params}, f)

            del eigenstate_lib, wavefunction_params, R_j_r, potential_params, density_params, r
            gc.collect()
            jax.clear_caches()
            gc.collect()


