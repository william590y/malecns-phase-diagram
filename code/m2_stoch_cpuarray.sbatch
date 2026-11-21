#!/bin/bash
#SBATCH -J m2-stoch-cpuarr
#SBATCH -p thickstun
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# lane m2/theory-stochastic, CPU torch: line SLURM_ARRAY_TASK_ID+1 of <argsfile> = arguments of one m2_stoch_run.py call
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
export M2_DEV=cpu OMP_NUM_THREADS=8
eval /MALECNS_PYBIN/python -W ignore /MALECNS_ROOT/code/m2_stoch_run.py $ARGS
