#!/bin/bash
#PBS -N Precomp
#PBS -l select=1:ncpus=4:mem=128gb:ngpus=4:gpu_type=RTX6000
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_precomp_output.log
#PBS -e /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_precomp_error.log

WORKDIR=/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses

module load Python/3.11.3-GCCcore-12.3.0

source /rds/general/user/jd925/home/venvs/jaxsp_env/bin/activate

cd $WORKDIR

mkdir -p logs
mkdir -p /rds/general/user/jd925/home/.jax_cache

export JAX_COMPILATION_CACHE_DIR=/rds/general/user/jd925/home/.jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export TF_GPU_ALLOCATOR=cuda_malloc_async


python -u $WORKDIR/Generating_m22_functions.py 2>&1 | tee $WORKDIR/logs/live_precomp_output.log
