#!/bin/bash
#PBS -N det_probe3
#PBS -l select=1:ncpus=8:mem=256gb:ngpus=1:gpu_type=A100
#PBS -l walltime=04:00:00
#PBS -o /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism3_output.log
#PBS -e /gpfs/home/jd925/Adding_stellar_masses/simple_sim/logs/job_determinism3_error.log

# Rounds 1 and 2 found the pipeline bit-reproducible under every condition
# tested. The one asymmetry they both missed is the compilation cache: in the
# reproduc job, run_0's first acceleration call took 232 s and run_1's took
# 15 s, so run_0 COMPILED the fused solver and run_1 DESERIALISED it from
# JAX_COMPILATION_CACHE_DIR. Rounds 1 and 2 ran against a cache that was
# already 2.4 GB warm, so every process took the deserialised path and the
# asymmetry never occurred.
#
# The checkpoints agree with that reading: the implied acceleration difference
# between the two runs over their first step peaks at 1.64e-10, and round 1
# measured an actual kernel swap (base vs --xla_gpu_deterministic_ops) moving
# accelerations by up to 1.11e-10. Same magnitude. And no particle had a
# bit-identical first-step acceleration, which is what a different executable
# looks like, not what a race looks like.
#
# Two arms, each two instances in one process:
#
#   coldcache - a brand-new empty cache dir. Instance 0 compiles and writes,
#               instance 1 reads back. Reproduces the reproduc job exactly.
#               A divergence here means the cache round trip changes kernels.
#
#   nocache   - persistent cache disabled entirely, so BOTH instances compile
#               from scratch in the same process. A divergence here means
#               autotuning itself is not deterministic, and the cache is
#               innocent (in fact protective).
#
# These separate two very different fixes, so they must not be run together.

ulimit -s 524288
echo "[job] stack limit: $(ulimit -s) kb"

WORKDIR=/gpfs/home/jd925/Adding_stellar_masses/simple_sim
SCRATCH=/gpfs/home/jd925/jax_cache_probe3

source /gpfs/home/jd925/miniforge3/etc/profile.d/conda.sh
conda activate keir_env

cd $WORKDIR
mkdir -p logs

export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset XLA_FLAGS

# 3 steps is plenty - the reproduc runs had already diverged by step 1.
N_STEPS=3

echo ""
echo "############################################################"
echo "# ARM 1: coldcache  (instance 0 compiles, instance 1 loads)"
echo "############################################################"
# A scratch dir of its own, wiped first. Emphatically NOT the production
# /gpfs/home/jd925/jax_cache - that is 2.4 GB of entries the real runs depend
# on, and deleting it would cost hours of recompilation.
rm -rf $SCRATCH
mkdir -p $SCRATCH
export JAX_COMPILATION_CACHE_DIR=$SCRATCH
# Without this, JAX only caches compiles longer than 1 s, and it is exactly
# the big fused solver whose cache round trip is under test.
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
python -u $WORKDIR/Testing/determinism_probe2.py $N_STEPS

echo ""
echo "############################################################"
echo "# ARM 2: nocache  (both instances compile from scratch)"
echo "############################################################"
unset JAX_COMPILATION_CACHE_DIR
unset JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
python -u $WORKDIR/Testing/determinism_probe2.py $N_STEPS

rm -rf $SCRATCH
