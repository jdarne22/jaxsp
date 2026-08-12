#!/bin/bash
#PBS -N det_probe2
#PBS -l select=1:ncpus=8:mem=256gb:ngpus=1:gpu_type=A100
#PBS -l walltime=02:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism2_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism2_error.log

# Round 1 took 18 minutes of an 8 h request; initialisation turned out to be
# ~60 s, not the 27 min the reproduc run's cold GPFS read suggested. Two
# instances at 5 macro steps each is well inside 2 h, and the smaller ask
# should schedule sooner.

ulimit -s 524288
echo "[job] stack limit: $(ulimit -s) kb"

WORKDIR=/gpfs/home/jd925/Adding_stellar_masses/simple_sim

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR
mkdir -p logs
mkdir -p /gpfs/home/jd925/jax_cache

# Identical environment to job_Reproduc.sh - no XLA_FLAGS, preallocate off.
# Round 1 showed neither of those knobs changes the answer, so this probe
# holds them at exactly what the runs being explained used.
export JAX_COMPILATION_CACHE_DIR=/gpfs/home/jd925/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -u $WORKDIR/Testing/determinism_probe2.py 5
