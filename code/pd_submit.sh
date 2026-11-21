#!/bin/bash
# Phase-diagram sweeps (stage 1, coarse). Raw weights. g log-spaced 0.1-30 (32 points), fully active start, T=3000, B=16.
#   (g, r) plane, real connectome: r in 0 .25 .5 1 1.5 2 3 4 6
#   nulls: configuration (targets) at r = 0, 1, 4; weight-shuffled at r = 0, 1, 4
#   (g, h) plane at r = 1: h in 1e-5 1e-4 1e-3 1e-2 1e-1 (T=1000; no absorbing state), real + configuration null at 1e-3
# plus Griffiths-window characterisation: module-seeded spreading (real, null), long decay runs in the slow window.
cd /MALECNS_ROOT
G=$(/MALECNS_PYBIN/python -c "import numpy as np; print(' '.join('%.4g'%x for x in np.logspace(-1, np.log10(30), 32)))")
S="env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --parsable -p gpu,default_partition code/swc_job.sbatch"
sub() { name=$1; shift; echo "$name $(env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --parsable -p gpu,default_partition -J $name ${TL:+-t $TL} code/swc_job.sbatch "$@")"; }
for r in 0 0.25 0.5 1 1.5 2 3 4 6; do sub pd-r$r --exp decay --norm raw --r $r --g $G --T 3000 --B 16 --out results/pd/decay_r${r}_real.npz; done
for r in 0 1 4; do
  sub pd-null-r$r --exp decay --norm raw --r $r --g $G --T 3000 --B 16 --null --out results/pd/decay_r${r}_null.npz
  sub pd-wnull-r$r --exp decay --norm raw --r $r --g $G --T 3000 --B 16 --null --null_kind weights --out results/pd/decay_r${r}_wnull.npz
done
for h in 1e-5 1e-4 1e-3 1e-2 1e-1; do sub pd-h$h --exp decay --norm raw --r 1 --h $h --g $G --T 1000 --B 16 --out results/pd/drive_h${h}_real.npz; done
sub pd-null-h1e-3 --exp decay --norm raw --r 1 --h 1e-3 --g $G --T 1000 --B 16 --null --out results/pd/drive_h1e-3_null.npz
TL=05:00:00
sub gw-mspread-a --exp spread --norm raw --r 1 --g 0.20 0.24 0.28 0.32 --T 3000 --B 512 --seed_set results/seeds_module_real.npz --out results/pd/mspread_real_a.npz
sub gw-mspread-b --exp spread --norm raw --r 1 --g 0.36 0.42 0.50 0.60 --T 3000 --B 512 --seed_set results/seeds_module_real.npz --out results/pd/mspread_real_b.npz
sub gw-mspread-null --exp spread --norm raw --r 1 --g 0.95 1.05 1.11 1.15 1.25 --T 3000 --B 512 --null --seed_set results/seeds_module_null.npz --out results/pd/mspread_null.npz
sub gw-long-real --exp decay --norm raw --r 1 --g 0.26 0.28 0.30 0.32 0.34 --T 30000 --B 32 --out results/pd/long_real.npz
sub gw-long-null --exp decay --norm raw --r 1 --g 0.98 1.02 1.06 1.10 --T 30000 --B 32 --null --out results/pd/long_null.npz
