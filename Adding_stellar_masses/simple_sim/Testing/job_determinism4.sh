#!/bin/bash
#PBS -N det_probe4
#PBS -l select=1:ncpus=8:mem=256gb:ngpus=1:gpu_type=A100
#PBS -l walltime=03:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism4_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism4_error.log

# 100 macro steps of churn at ~18 s/step is ~30 min, plus a 60 s init and one
# cached compile. 3 h is ample.

ulimit -s 524288
WORKDIR=/gpfs/home/jd925/Adding_stellar_masses/simple_sim

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR
mkdir -p logs

# Identical environment to job_Reproduc.sh. PREALLOCATE=false in particular is
# the condition under test - it is what lets the pool grow and fragment as the
# run proceeds.
export JAX_COMPILATION_CACHE_DIR=/gpfs/home/jd925/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_FLAGS

python -u $WORKDIR/Testing/determinism_probe4.py
