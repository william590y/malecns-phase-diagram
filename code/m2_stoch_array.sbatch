#!/bin/bash
#SBATCH -J m2-stoch
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# lane m2/theory-stochastic: line SLURM_ARRAY_TASK_ID+1 of <argsfile> = arguments of one code/m2_stoch_run.py call
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}: ${ARGS}"
nvidia-smi --query-gpu=name --format=csv,noheader
eval /MALECNS_PYBIN/python -W ignore /MALECNS_ROOT/code/m2_stoch_run.py $ARGS
