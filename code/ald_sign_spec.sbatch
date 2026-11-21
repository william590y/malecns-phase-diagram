#!/bin/bash
#SBATCH -J ald-e2-spec
#SBATCH -p thickstun
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# E2 transmitter signs: excitatory eigenmodes under sign changes (ald_sign_spec.py), CPU ARPACK, 10 tasks.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /MALECNS_ROOT/code
P=/MALECNS_PYBIN/python
case $SLURM_ARRAY_TASK_ID in
 0) $P ald_sign_spec.py det ;;
 1) $P ald_sign_spec.py ens type ignore 1 200 0 50 24 ;;
 2) $P ald_sign_spec.py ens type ignore 1 200 50 100 ;;
 3) $P ald_sign_spec.py ens type ignore 1 200 100 150 ;;
 4) $P ald_sign_spec.py ens type ignore 1 200 150 200 ;;
 5) $P ald_sign_spec.py ens neuron ignore 2 100 0 50 16 ;;
 6) $P ald_sign_spec.py ens neuron ignore 2 100 50 100 ;;
 7) $P ald_sign_spec.py ens type fix 3 100 0 50 ;;
 8) $P ald_sign_spec.py ens type fix 3 100 50 100 ;;
 9) $P ald_sign_spec.py ens neuron fix 4 50 0 50 ;;
esac
