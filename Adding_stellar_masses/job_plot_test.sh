#!/bin/bash
#PBS -N Plot_test
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:30:00
#PBS -o /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_plot_test_output.log
#PBS -e /rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses/logs/job_plot_test_error.log

WORKDIR=/rds/general/user/jd925/home/PhD_first_year/jaxsp/Adding_stellar_masses

module load Python/3.11.3-GCCcore-12.3.0

source /rds/general/user/jd925/home/venvs/jaxsp_env/bin/activate

cd $WORKDIR

mkdir -p logs
mkdir -p /rds/general/user/jd925/home/.jax_cache


python -u $WORKDIR/plot_test.py --m22 20 --r0 1 --Lout 0.5 2>&1 | tee $WORKDIR/logs/live_output_plot_test.log
