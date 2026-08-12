#!/bin/bash
#PBS -N det_probe
#PBS -l select=1:ncpus=8:mem=256gb:ngpus=1:gpu_type=A100
#PBS -l walltime=08:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism_error.log

# Same resources as job_Reproduc.sh (1 A100, use_multi_gpu=False), so the
# kernels under test are the ones that produced the two runs being explained.
# 8 h is generous: four processes at ~30 min of initialisation each, plus one
# full compile for the `det` configuration, whose XLA flags give it a
# different compilation-cache key.

ulimit -s 524288
echo "[job] stack limit: $(ulimit -s) kb"

WORKDIR=/gpfs/home/jd925/Adding_stellar_masses/simple_sim
OUTDIR=$WORKDIR/Testing/determinism_out

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR
mkdir -p logs
mkdir -p /gpfs/home/jd925/jax_cache
mkdir -p $OUTDIR

# Anything left from a previous submission would be silently compared against
# this one's results by compare_determinism.py.
rm -f $OUTDIR/*.npz

export JAX_COMPILATION_CACHE_DIR=/gpfs/home/jd925/jax_cache


run_probe () {
    echo ""
    echo "############################################################"
    echo "# probe: $1"
    echo "############################################################"
    python -u $WORKDIR/Testing/determinism_probe.py "$1" "$OUTDIR"
}

# --- baseline: exactly job_Reproduc.sh's environment -----------------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_FLAGS
run_probe base

# --- baseline again, in a SECOND process ----------------------------------
# The cross-process control. Everything above is unchanged, so a difference
# between `base` and `base2` is the reported failure, reproduced in minutes
# instead of ten hours.
run_probe base2

# --- test 2: deterministic reduction emitters ------------------------------
# Forces XLA to avoid the atomics-based lowerings its reduction and scatter
# emitters can otherwise choose. Costs throughput. Different XLA flags mean a
# different compilation-cache key, so this one recompiles from scratch.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
run_probe det

# Second process under the same flag: `det` being self-consistent is not
# enough, it has to reproduce across processes too.
run_probe det2

# --- test 3: fixed allocator state -----------------------------------------
# PREALLOCATE=false lets the BFC pool grow differently in each process, which
# changes the workspace available to cuBLAS/cuFFT and hence, potentially, the
# algorithm they select. Preallocating removes that as a variable.
#
# Deliberately WITHOUT the deterministic-ops flag, so the two tests stay
# separable - if both are on and the result is clean, we would not know which
# one did it.
unset XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE=true
run_probe prealloc
run_probe prealloc2

# --- comparison ------------------------------------------------------------
unset XLA_FLAGS
echo ""
echo "############################################################"
echo "# comparison"
echo "############################################################"
python -u $WORKDIR/Testing/compare_determinism.py "$OUTDIR"
