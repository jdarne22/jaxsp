#!/bin/bash
#PBS -N IC_test
#PBS -l select=1:ncpus=8:mem=128gb:ngpus=1:gpu_type=A100
#PBS -l walltime=24:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_IC_test_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_IC_test_error.log


ulimit -s 524288
echo "[job] stack limit: $(ulimit -s) kb"

WORKDIR=/gpfs/home/jd925/Adding_stellar_masses/simple_sim/Testing

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR


mkdir -p logs
mkdir -p /gpfs/home/jd925/jax_cache


export JAX_COMPILATION_CACHE_DIR=/gpfs/home/jd925/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false


#export JAX_LOG_COMPILES=1


export XLA_FLAGS="--xla_gpu_deterministic_ops=true"


python -u $WORKDIR/Running_sims_IC_test.py 2>&1 | tee $WORKDIR/../logs/live_IC_test_output.log
