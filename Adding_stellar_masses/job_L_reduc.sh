#!/bin/bash
#PBS -N Run_L_reduc
#PBS -l select=1:ncpus=1:mem=16gb:ngpus=2:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -o /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_L_reduc_output.log
#PBS -e /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_L_reduc_error.log

WORKDIR=/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses

module load Python/3.11.3-GCCcore-12.3.0

source /rds/general/user/jd925/home/venvs/jaxsp_env/bin/activate

cd $WORKDIR

mkdir -p logs
mkdir -p /rds/general/user/jd925/home/.jax_cache

export JAX_COMPILATION_CACHE_DIR=/rds/general/user/jd925/home/.jax_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export XLA_PYTHON_CLIENT_PREALLOCATE=false


python -u $WORKDIR/Running_sims_L_reduc.py 2>&1 | tee $WORKDIR/logs/live_output_L_reduc.log