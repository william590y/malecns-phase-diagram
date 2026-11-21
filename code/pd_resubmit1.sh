#!/bin/bash
# Resubmit stage-1 jobs that landed on pre-Turing GPUs (no kernel image) after the GPU constraint was dropped.
cd /MALECNS_ROOT
G=$(/MALECNS_PYBIN/python -c "import numpy as np; print(' '.join('%.4g'%x for x in np.logspace(-1, np.log10(30), 32)))")
sub() { name=$1; sb=$2; shift 2; echo "$name $(env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --parsable -J $name ${TL:+-t $TL} $sb "$@")"; }
S=code/swc_job.sbatch
for r in 4 6; do sub pd-r$r $S --exp decay --norm raw --r $r --g $G --T 3000 --B 16 --out results/pd/decay_r${r}_real.npz; done
for r in 0 1; do sub pd-null-r$r $S --exp decay --norm raw --r $r --g $G --T 3000 --B 16 --null --out results/pd/decay_r${r}_null.npz; done
for r in 0 1 4; do sub pd-wnull-r$r $S --exp decay --norm raw --r $r --g $G --T 3000 --B 16 --null --null_kind weights --out results/pd/decay_r${r}_wnull.npz; done
for h in 1e-5 1e-4; do sub pd-h$h $S --exp decay --norm raw --r 1 --h $h --g $G --T 1000 --B 16 --out results/pd/drive_h${h}_real.npz; done
R="0 0.25 0.5 1 1.5 2 3 4 6"
for k in real null wnull; do sub th-qmf-$k code/theory_job.sbatch --kind $k --r $R --out results/pd/qmf_$k.npz; done
sub th-qmf-drive code/theory_job.sbatch --kind real --r 1 --h 1e-5 1e-4 1e-3 1e-2 1e-1 --T 1000 --out results/pd/qmf_drive_real.npz
TL=05:00:00
sub gw-mspread-null $S --exp spread --norm raw --r 1 --g 0.95 1.05 1.11 1.15 1.25 --T 3000 --B 512 --null --seed_set results/seeds_module_null.npz --out results/pd/mspread_null.npz
sub gw-long-real $S --exp decay --norm raw --r 1 --g 0.26 0.28 0.30 0.32 0.34 --T 30000 --B 32 --out results/pd/long_real.npz
sub gw-long-null $S --exp decay --norm raw --r 1 --g 0.98 1.02 1.06 1.10 --T 30000 --B 32 --null --out results/pd/long_null.npz
