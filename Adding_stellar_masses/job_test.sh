#!/bin/bash
#PBS -N Tests
#PBS -l select=1:ncpus=1:mem=256gb:ngpus=2:gpu_type=A100
#PBS -l walltime=72:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/logs/job_test_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/logs/job_test_error.log

WORKDIR=/gpfs/home/jd925/Adding_stellar_masses

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR


mkdir -p logs
mkdir -p /gpfs/home/jd925/jax_cache


export JAX_COMPILATION_CACHE_DIR=/gpfs/home/jd925/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false


python -u $WORKDIR/Running_tests.py 2>&1 | tee $WORKDIR/logs/live_test_output.log
