#!/bin/bash
#SBATCH -J ald-verify-sim
#SBATCH -p gpu,ellis
#SBATCH --gres=gpu:1
#SBATCH --constraint="ampere|ada|hopper|blackwell"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow,sun-compute-01
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH --nice=500
#SBATCH -o /MALECNS_ROOT/logs/%x_%A_%a.out
# verify lane: independent simulator checks, one task per array index (see code/ald_verify_simcheck.py)
echo "host $(hostname) task ${SLURM_ARRAY_TASK_ID}"; nvidia-smi --query-gpu=name --format=csv,noheader
cd /MALECNS_ROOT/code
/MALECNS_PYBIN/python ald_verify_simcheck.py ${SLURM_ARRAY_TASK_ID}
