"""
Compares the .npz files determinism_probe.py wrote.

The probe answers "is one process self-consistent". This answers the two
questions that need more than one process:

  base   vs base2  - two processes, identical flags. This is the actual
                     condition Running_sims_reproduc.py ran under (its two
                     runs shared a process, but also shared everything else),
                     so a difference here is the failure being reproduced.

  base   vs det / prealloc
                   - does the flag change the answer at all? If `det` is
                     self-consistent AND agrees with a second `det` process,
                     the flag is the fix. If `det` merely differs from `base`,
                     that means nothing: any change of kernel changes the last
                     bits, which is the whole problem.

Reported against the 4.4e-13 relative difference measured between the two
reproduc runs at step 1 - that is the number a candidate fix has to drive to
exactly zero, not merely make small.

Usage:  python compare_determinism.py [out_dir]
"""

import glob
import os
import sys

import numpy as np

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    '/gpfs/home/jd925/Adding_stellar_masses/simple_sim/Testing/determinism_out')

# Relative position difference between run_0 and run_1 of the reproduc pair at
# timestep 1, measured from their final checkpoints. The scale a fix is judged
# against.
REPRODUC_STEP1_REL = 4.4e-13

FIELDS = ('a_r', 'a_theta', 'a_phi', 'phi')

paths = sorted(glob.glob(os.path.join(OUT_DIR, '*.npz')))
if not paths:
    sys.exit(f"no probe output in {OUT_DIR}")

runs = {}
for path in paths:
    data = np.load(path, allow_pickle=True)
    runs[str(data['tag'])] = data

print("=" * 78)
print("WITHIN EACH PROCESS  (same executable, same GPU, identical inputs)")
print("=" * 78)
print(f"{'tag':<12} {'pid':>8}  {'rho_lm':>13}  {'a_r':>13}  {'phi':>13}   flags")
for tag, d in runs.items():
    def mark(name):
        return 'identical' if bool(d[f'{name}_identical']) else 'DIFFERS'
    flags = f"{str(d['xla_flags'])} prealloc={str(d['preallocate'])}"
    print(f"{tag:<12} {int(d['pid']):>8}  {'identical' if bool(d['rho_lm_identical']) else 'DIFFERS':>13}"
          f"  {mark('a_r'):>13}  {mark('phi'):>13}   {flags}")

print()
print("=" * 78)
print("BETWEEN PROCESSES")
print("=" * 78)

tags = list(runs)
for i, tag_a in enumerate(tags):
    for tag_b in tags[i + 1:]:
        a, b = runs[tag_a], runs[tag_b]

        same_rho = str(a['rho_lm_digest']) == str(b['rho_lm_digest'])
        print(f"\n{tag_a}  vs  {tag_b}")
        print(f"  rho_lm digest: {'identical' if same_rho else 'DIFFERS'}"
              f"  ({str(a['rho_lm_digest'])} / {str(b['rho_lm_digest'])})")

        for name in FIELDS:
            x, y = a[name], b[name]
            exact = np.array_equal(x.view(np.uint8), y.view(np.uint8))
            scale = np.max(np.abs(x))
            worst = float(np.max(np.abs(x - y))) / scale if scale > 0 else 0.0
            note = ''
            if not exact and worst > 0:
                note = f"   ({worst / REPRODUC_STEP1_REL:.1f}x the reproduc step-1 difference)"
            print(f"  {name:<8} {'bit-identical' if exact else 'DIFFERS'}"
                  f"  worst rel {worst:.3e}{note}")

print()
print("=" * 78)
print("READING THIS")
print("=" * 78)
print("""
A candidate fix has to be bit-identical BOTH within a process and between two
processes carrying that flag. Within-process alone is not enough: the reproduc
runs shared a process and still diverged.

  rho_lm differs      -> the nondeterminism is in the matmuls or s2fft, in
                         complex64. Expect ~1e-7, not the observed 4e-13.
  rho_lm identical
  but acc differs     -> it is in the merged Poisson solve or combine_acc,
                         which accumulate in complex128. This matches the
                         observed 4.4e-13 and is the predicted outcome.
  everything identical
  under `det`         -> XLA-level atomics were the cause; the fix is one line
                         in job.sh.
""")
