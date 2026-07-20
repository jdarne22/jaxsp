#!/bin/bash
#PBS -N Plot_t_dep_final
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:30:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/logs/job_plot_t_dep_final_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/logs/job_plot_t_dep_final_error.log

WORKDIR=/gpfs/home/jd925/Adding_stellar_masses

module load Python/3.11.3-GCCcore-12.3.0

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR

mkdir -p logs
mkdir -p /gpfs/home/jd925/.jax_cache


python -u $WORKDIR/plot_t_dep_final.py --m22 50 --r0 1 --Lout 0.2 2>&1 | tee $WORKDIR/logs/live_output_plot_t_dep_final.log
