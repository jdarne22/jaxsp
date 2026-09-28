#!/bin/bash
#PBS -N m22_10_anal
#PBS -l select=1:ncpus=8:mem=512gb:ngpus=2:gpu_type=A100
#PBS -l walltime=72:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_m22_10_anal_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_m22_10_anal_error.log


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


#export JAX_LOG_COMPILES=1


export XLA_FLAGS="--xla_gpu_deterministic_ops=true"




python -u $WORKDIR/Running_sims.py 2>&1 | tee $WORKDIR/logs/live_m22_10_anal_output.log
