#!/bin/bash
#PBS -N Running_sims
#PBS -l select=1:ncpus=1:mem=16gb:ngpus=4:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_output.log
#PBS -e /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_error.log

WORKDIR=/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses

module load Python/3.11.3-GCCcore-12.3.0

source /rds/general/user/jd925/home/venvs/jaxsp_env/bin/activate

cd $WORKDIR


mkdir -p logs
mkdir -p /rds/general/user/jd925/ephemeral/jax_cache


export JAX_COMPILATION_CACHE_DIR=/rds/general/user/jd925/ephemeral/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false


python -u $WORKDIR/Running_sims.py 2>&1 | tee $WORKDIR/logs/live_output.log
