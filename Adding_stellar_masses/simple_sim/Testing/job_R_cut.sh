#!/bin/bash
#PBS -N 10_R_cut
#PBS -l select=1:ncpus=8:mem=256gb:ngpus=1:gpu_type=A100
#PBS -l walltime=72:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_10_R_cut_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_10_R_cut_error.log



ulimit -s 524288
echo "[job] stack limit: $(ulimit -s) kb"

WORKDIR=/gpfs/home/jd925/Adding_stellar_masses/simple_sim

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR


mkdir -p logs
mkdir -p /gpfs/home/jd925/jax_cache


export JAX_COMPILATION_CACHE_DIR=/gpfs/home/jd925/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# One line per XLA compile, with the function name and how long it took. The
# m22 = 100 segfault left no compilation-cache entry for the merged Poisson
# solver, so it died mid-compile - this says which compile, and is the only
# way to tell "still compiling" apart from "compiled and then died running".
#export JAX_LOG_COMPILES=1


python -u $WORKDIR/Testing/Running_sims_R_cutoff.py 2>&1 | tee $WORKDIR/logs/live_10_R_cut_output.log
